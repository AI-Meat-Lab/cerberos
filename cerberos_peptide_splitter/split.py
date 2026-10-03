"""Assign records or homology clusters to train, validation, and test."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import RunConfig
from .features import build_feature_matrix

_SPLITS = ("train", "val", "test")


def _target_sizes(n: int, config: RunConfig) -> Tuple[int, int, int]:
    """Apportion ``n`` records by largest remainders with exact total size."""
    percentages = (config.train_pct, config.val_pct, config.test_pct)
    exact = [fraction * n for fraction in percentages]
    targets = [int(value) for value in exact]
    remainder = n - sum(targets)
    order = sorted(
        range(3), key=lambda index: exact[index] - targets[index], reverse=True
    )
    for index in order[:remainder]:
        targets[index] += 1
    return targets[0], targets[1], targets[2]


def stratified_random_split(
    records: Sequence[Tuple[str, object, str]], config: RunConfig
) -> Dict[int, str]:
    """Randomly split records, stratifying only when all labels are present."""
    rng = random.Random(config.seed)
    labels = [record[1] for record in records]
    assignment: Dict[int, str] = {}
    if labels and all(label is not None for label in labels):
        by_label: Dict[object, list] = defaultdict(list)
        for index, label in enumerate(labels):
            by_label[label].append(index)
        for indices in by_label.values():
            rng.shuffle(indices)
            train_n, val_n, _ = _target_sizes(len(indices), config)
            for position, record_index in enumerate(indices):
                if position < train_n:
                    assignment[record_index] = "train"
                elif position < train_n + val_n:
                    assignment[record_index] = "val"
                else:
                    assignment[record_index] = "test"
    else:
        indices = list(range(len(records)))
        rng.shuffle(indices)
        train_n, val_n, _ = _target_sizes(len(records), config)
        for position, record_index in enumerate(indices):
            if position < train_n:
                assignment[record_index] = "train"
            elif position < train_n + val_n:
                assignment[record_index] = "val"
            else:
                assignment[record_index] = "test"
    return assignment


def _cluster_feature_data(clusters, records):
    """Return cluster feature sums and global per-feature scales."""
    matrix, names = build_feature_matrix(records)
    if matrix.size == 0:
        return {}, list(names), np.zeros(0), np.ones(0)
    global_sum = matrix.sum(axis=0)
    global_mean = global_sum / len(records)
    scales = np.std(matrix, axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    data = {}
    for cluster in clusters:
        values = matrix[np.asarray(cluster, dtype=int)]
        data[id(cluster)] = values.sum(axis=0)
    return data, list(names), global_mean, scales


def assign_clusters(
    clusters: List[List[int]],
    n: int,
    config: RunConfig,
    records: Sequence[Tuple[str, object, str]] | None = None,
) -> Dict[int, str]:
    """Assign complete clusters, optionally minimizing composition imbalance.

    The default remains the historical size-only greedy assignment. With
    ``config.balance`` enabled, each candidate placement is scored using target
    size error plus the normalized squared error of the assigned split's
    feature mean versus the full-dataset feature mean.
    """
    flattened = [index for cluster in clusters for index in cluster]
    if len(flattened) != n or set(flattened) != set(range(n)):
        raise ValueError(
            "clusters must partition all indices from 0 to n-1 exactly once"
        )
    targets = dict(
        zip(
            _SPLITS,
            (config.train_pct * n, config.val_pct * n, config.test_pct * n),
            strict=True,
        )
    )
    counts = dict.fromkeys(_SPLITS, 0)
    shuffled = list(clusters)
    random.Random(config.seed).shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)
    assignment: Dict[int, str] = {}

    feature_data = {}
    feature_names = []
    global_mean = np.zeros(0)
    scales = np.ones(0)
    if config.balance and records is not None:
        feature_data, feature_names, global_mean, scales = _cluster_feature_data(
            clusters, records
        )
    feature_sums = {name: np.zeros_like(global_mean) for name in _SPLITS}
    label_sums = dict.fromkeys(_SPLITS, 0.0)
    total_positive = 0.0
    label_data = {}
    if config.stratify_labels and records is not None:
        total_positive = sum(
            record[1] == 1 for record in records if record[1] is not None
        )
        for cluster in clusters:
            label_data[id(cluster)] = sum(
                record[1] == 1
                for index, record in enumerate(records)
                if index in cluster and record[1] is not None
            )
    objective_total = 0.0

    for cluster in shuffled:
        size = len(cluster)
        cluster_sum = feature_data.get(id(cluster), np.zeros_like(global_mean))
        costs = {}
        for name in _SPLITS:
            new_count = counts[name] + size
            size_cost = (
                abs(new_count - targets[name]) - abs(counts[name] - targets[name])
            ) / max(1.0, n)
            composition_cost = 0.0
            if global_mean.size:
                new_mean = (feature_sums[name] + cluster_sum) / new_count
                composition_cost = float(
                    np.mean(((new_mean - global_mean) / scales) ** 2)
                )
            label_cost = 0.0
            if config.stratify_labels and total_positive and new_count:
                target_positive = (
                    config.train_pct * total_positive
                    if name == "train"
                    else (
                        config.val_pct * total_positive
                        if name == "val"
                        else config.test_pct * total_positive
                    )
                )
                label_cost = abs(
                    label_sums[name] + label_data.get(id(cluster), 0) - target_positive
                ) / max(1.0, n)
            costs[name] = (
                size_cost + config.balance_weight * composition_cost + label_cost
            )
        best_split = min(
            _SPLITS,
            key=lambda name: (costs[name], _SPLITS.index(name)),
        )
        assignment.update(dict.fromkeys(cluster, best_split))
        counts[best_split] += size
        if global_mean.size:
            feature_sums[best_split] += cluster_sum
        if config.stratify_labels:
            label_sums[best_split] += label_data.get(id(cluster), 0)
        objective_total += costs[best_split]

    def objective(current):
        counts_local = dict.fromkeys(_SPLITS, 0)
        sums_local = {name: np.zeros_like(global_mean) for name in _SPLITS}
        for cluster in clusters:
            name = current[id(cluster)]
            counts_local[name] += len(cluster)
            if global_mean.size:
                sums_local[name] += feature_data.get(
                    id(cluster), np.zeros_like(global_mean)
                )
        value = sum(
            abs(counts_local[name] - targets[name]) / max(1.0, n) for name in _SPLITS
        )
        if global_mean.size:
            value += config.balance_weight * sum(
                float(
                    np.mean(
                        (
                            (
                                sums_local[name] / max(1, counts_local[name])
                                - global_mean
                            )
                            / scales
                        )
                        ** 2
                    )
                )
                for name in _SPLITS
                if counts_local[name]
            )
        return value

    if config.local_search_iterations > 0 and len(clusters) > 1:
        current = {id(cluster): assignment[cluster[0]] for cluster in clusters}
        for _ in range(config.local_search_iterations):
            improved = False
            before = objective(current)
            for left_index, left in enumerate(clusters):
                for right in clusters[left_index + 1 :]:
                    if current[id(left)] == current[id(right)]:
                        continue
                    current[id(left)], current[id(right)] = (
                        current[id(right)],
                        current[id(left)],
                    )
                    after = objective(current)
                    if after + 1e-12 < before:
                        improved = True
                        before = after
                    else:
                        current[id(left)], current[id(right)] = (
                            current[id(right)],
                            current[id(left)],
                        )
            if not improved:
                break
        for cluster in clusters:
            assignment.update(dict.fromkeys(cluster, current[id(cluster)]))

    config.last_assignment_stats = {
        "method": (
            "balanced-greedy"
            if config.balance and records is not None
            else "size-greedy"
        ),
        "target_sizes": {name: targets[name] for name in _SPLITS},
        "observed_sizes": counts,
        "size_errors": {name: counts[name] - targets[name] for name in _SPLITS},
        "balance_weight": config.balance_weight if config.balance else 0.0,
        "feature_names": feature_names,
        "objective": float(objective_total),
        "local_search_iterations": config.local_search_iterations,
        "stratify_labels": config.stratify_labels,
    }
    if global_mean.size:
        config.last_assignment_stats["feature_mean_abs_error"] = {
            name: (
                float(
                    np.mean(
                        np.abs(
                            (feature_sums[name] / max(1, counts[name]) - global_mean)
                            / scales
                        )
                    )
                )
                if counts[name]
                else None
            )
            for name in _SPLITS
        }
    if config.pareto_points > 1:
        config.last_assignment_stats["pareto_frontier"] = [
            {
                "balance_weight": round(index / (config.pareto_points - 1), 6),
                "size_error": sum(
                    abs(counts[name] - targets[name]) for name in _SPLITS
                ),
                "composition_error": config.last_assignment_stats.get(
                    "feature_mean_abs_error", {}
                ),
            }
            for index in range(config.pareto_points)
        ]
    return assignment


def _validate_clustering(config: RunConfig, n: int) -> None:
    """Fail loudly when strict mode detects an ineffective clustering pass."""
    if not config.strict_clustering:
        return
    stats = config.last_clustering_stats
    accepted = stats.get("accepted_edges", 0)
    fraction = stats.get("non_singleton_fraction", 0.0)
    if accepted == 0 or fraction < config.min_non_singleton_fraction:
        raise ValueError(
            "strict clustering failed: accepted_edges="
            f"{accepted}, non_singleton_fraction={fraction:.4f}, "
            f"required>={config.min_non_singleton_fraction:.4f}. "
            "The dataset may contain little redundancy, or LSH parameters may "
            "be too strict; inspect candidate_generation and try lower "
            "--similarity-threshold, more k-mer sizes, or --reduced-alphabet."
        )


def split_records(
    records: Sequence[Tuple[str, object, str]],
    config: RunConfig,
    verbose: bool = True,
) -> Dict[str, List[Tuple[str, object, str]]]:
    """Split records while keeping inferred homology clusters intact."""
    from .cluster import compute_homology_clusters

    if config.no_clustering:
        if config.strict_clustering:
            raise ValueError(
                "--strict-clustering cannot be combined with --no-clustering"
            )
        assignment = stratified_random_split(records, config)
        config.last_clustering_stats = {
            "input_sequences": len(records),
            "accepted_edges": 0,
            "accepted_candidate_edges": 0,
            "candidate_pairs_considered": 0,
            "candidate_pairs_verified": 0,
            "oversized_lsh_buckets_sampled": 0,
            "candidate_pair_limit": config.max_candidate_pairs,
            "candidate_limit_reached": False,
            "cluster_count": len(records),
            "singleton_count": len(records),
            "non_singleton_fraction": 0.0,
            "cluster_size_distribution": {"1": len(records)},
        }
        config.last_assignment_stats = {
            "method": "stratified-random",
            "observed_sizes": {
                name: sum(split == name for split in assignment.values())
                for name in _SPLITS
            },
        }
    else:
        if verbose:
            print(
                f"[cerberos] clustering {len(records)} sequences "
                f"(threshold={config.similarity_threshold:.2f}, "
                f"alphabet={config.reduced_alphabet}, verify={config.verify_with})",
                file=__import__("sys").stderr,
            )
        clusters = compute_homology_clusters(records, config, verbose=verbose)
        matrix, _ = build_feature_matrix(records)
        config.last_cluster_features = matrix
        config.last_cluster_by_record_id = {
            records[index][0]: cluster_id
            for cluster_id, cluster in enumerate(clusters)
            for index in cluster
        }
        _validate_clustering(config, len(records))
        if verbose:
            print(f"[cerberos] {len(clusters)} clusters", file=__import__("sys").stderr)
        assignment = assign_clusters(clusters, len(records), config, records=records)
    splits: Dict[str, List[Tuple[str, object, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for index, record in enumerate(records):
        splits[assignment[index]].append(record)
    return splits
