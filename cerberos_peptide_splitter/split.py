"""Assign records or homology clusters to train, validation, and test."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
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


def _wasserstein(left, right):
    if not len(left) or not len(right):
        return 0.0
    quantiles = np.linspace(0.0, 1.0, max(len(left), len(right)))
    return float(
        np.mean(np.abs(np.quantile(left, quantiles) - np.quantile(right, quantiles)))
    )


def _ks_distance(left, right):
    if not len(left) or not len(right):
        return 0.0
    values = np.sort(np.unique(np.concatenate([left, right])))
    left_sorted, right_sorted = np.sort(left), np.sort(right)
    left_cdf = np.searchsorted(left_sorted, values, side="right") / len(left)
    right_cdf = np.searchsorted(right_sorted, values, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


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
    target_counts = dict(zip(_SPLITS, _target_sizes(n, config), strict=True))
    lower_bounds = {
        name: (
            max(1, int(np.floor(targets[name] * 0.95)))
            if target_counts[name] > 0 and len(clusters) >= 3
            else int(np.floor(targets[name] * 0.95))
        )
        for name in _SPLITS
    }
    upper_bounds = {name: int(np.ceil(targets[name] * 1.05)) for name in _SPLITS}
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
    feature_matrix = (
        build_feature_matrix(records)[0] if records is not None else np.zeros((0, 0))
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
    balance_constraint_fallback = False

    for position, cluster in enumerate(shuffled):
        size = len(cluster)
        cluster_sum = feature_data.get(id(cluster), np.zeros_like(global_mean))
        costs = {}
        remaining_clusters = len(shuffled) - position - 1
        unfilled = [
            name for name in _SPLITS if counts[name] == 0 and lower_bounds[name] > 0
        ]
        reserved = set(unfilled) if remaining_clusters < len(unfilled) else set()
        for name in _SPLITS:
            new_count = counts[name] + size
            if config.balance and records is not None:
                if new_count > upper_bounds[name] or name in reserved:
                    continue
                remaining_records = n - sum(counts.values()) - size
                other_need = sum(
                    max(
                        0,
                        lower_bounds[other]
                        - (new_count if other == name else counts[other]),
                    )
                    for other in _SPLITS
                )
                if remaining_records < other_need:
                    continue
            size_cost = (
                abs(new_count - targets[name]) - abs(counts[name] - targets[name])
            ) / max(1.0, n)
            composition_cost = 0.0
            if global_mean.size:
                if config.balance_metric == "mean":
                    new_mean = (feature_sums[name] + cluster_sum) / new_count
                    composition_cost = float(
                        np.mean(((new_mean - global_mean) / scales) ** 2)
                    )
                else:
                    current_indices = [
                        index
                        for cluster_indices in clusters
                        if cluster_indices[0] in assignment
                        and assignment[cluster_indices[0]] == name
                        for index in cluster_indices
                    ]
                    candidate_indices = current_indices + list(cluster)
                    for column in range(feature_matrix.shape[1]):
                        values = feature_matrix[candidate_indices, column]
                        target = feature_matrix[:, column]
                        distance = (
                            _wasserstein(values, target)
                            if config.balance_metric == "wasserstein"
                            else _ks_distance(values, target)
                        )
                        composition_cost += distance / max(scales[column], 1.0)
                    composition_cost /= max(1, feature_matrix.shape[1])
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
        if not costs:
            balance_constraint_fallback = True
            costs = {name: abs(counts[name] + size - targets[name]) for name in _SPLITS}
        best_split = min(
            costs,
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
                    if config.balance:
                        trial_counts = dict.fromkeys(_SPLITS, 0)
                        for trial_cluster in clusters:
                            trial_counts[current[id(trial_cluster)]] += len(
                                trial_cluster
                            )
                        if any(
                            trial_counts[name] < lower_bounds[name]
                            or trial_counts[name] > upper_bounds[name]
                            for name in _SPLITS
                        ):
                            current[id(left)], current[id(right)] = (
                                current[id(right)],
                                current[id(left)],
                            )
                            continue
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
        "hard_size_bounds": {
            name: {"min": lower_bounds[name], "max": upper_bounds[name]}
            for name in _SPLITS
        },
        "size_constraint_fallback": balance_constraint_fallback,
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
        frontier = []
        for index in range(config.pareto_points):
            weight = index / (config.pareto_points - 1)
            options = config.as_dict()
            options.update(
                {
                    "prefilter_threshold": config.prefilter_threshold,
                    "balance_weight": weight,
                    "pareto_points": 0,
                }
            )
            alternative = RunConfig(**options)
            assign_clusters(clusters, n, alternative, records=records)
            alternative_stats = alternative.last_assignment_stats
            size_error = sum(
                abs(value)
                for value in alternative_stats.get("size_errors", {}).values()
            )
            feature_error = sum(
                value
                for value in alternative_stats.get(
                    "feature_mean_abs_error", {}
                ).values()
                if value is not None
            )
            frontier.append(
                {
                    "balance_weight": round(weight, 6),
                    "size_error": float(size_error),
                    "composition_error": float(feature_error),
                    "observed_sizes": alternative_stats.get("observed_sizes", {}),
                }
            )
        config.last_assignment_stats["pareto_frontier"] = [
            point
            for point in frontier
            if not any(
                other["size_error"] <= point["size_error"]
                and other["composition_error"] <= point["composition_error"]
                and (
                    other["size_error"] < point["size_error"]
                    or other["composition_error"] < point["composition_error"]
                )
                for other in frontier
            )
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

    if config.external_clusters:
        from .advanced import load_cluster_assignments

        clusters = load_cluster_assignments(config.external_clusters, len(records))
        config.last_clustering_stats = {
            "input_sequences": len(records),
            "source": config.external_clusters,
            "external": True,
            "accepted_edges": sum(max(0, len(cluster) - 1) for cluster in clusters),
            "cluster_count": len(clusters),
            "singleton_count": sum(len(cluster) == 1 for cluster in clusters),
            "non_singleton_fraction": sum(
                len(cluster) for cluster in clusters if len(cluster) > 1
            )
            / max(1, len(records)),
            "cluster_size_distribution": dict(
                Counter(str(len(cluster)) for cluster in clusters)
            ),
        }
        config.last_cluster_by_record_id = {
            records[index][0]: cluster_id
            for cluster_id, cluster in enumerate(clusters)
            for index in cluster
        }
        assignment = assign_clusters(clusters, len(records), config, records=records)
    elif config.no_clustering:
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
