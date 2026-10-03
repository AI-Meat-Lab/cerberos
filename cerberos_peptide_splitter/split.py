"""Assign records or homology clusters to train, validation, and test."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .config import RunConfig


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


def assign_clusters(
    clusters: List[List[int]], n: int, config: RunConfig
) -> Dict[int, str]:
    """Greedily assign each complete cluster to minimize target-size error."""
    flattened = [index for cluster in clusters for index in cluster]
    if len(flattened) != n or set(flattened) != set(range(n)):
        raise ValueError(
            "clusters must partition all indices from 0 to n-1 exactly once"
        )
    targets = {
        "train": config.train_pct * n,
        "val": config.val_pct * n,
        "test": config.test_pct * n,
    }
    counts = dict.fromkeys(targets, 0)
    shuffled = list(clusters)
    random.Random(config.seed).shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)
    assignment: Dict[int, str] = {}
    for cluster in shuffled:
        best_split = min(
            ("train", "val", "test"),
            key=lambda name: (
                abs(counts[name] + len(cluster) - targets[name])
                - abs(counts[name] - targets[name])
            ),
        )
        assignment.update(dict.fromkeys(cluster, best_split))
        counts[best_split] += len(cluster)
    return assignment


def split_records(
    records: Sequence[Tuple[str, object, str]],
    config: RunConfig,
    verbose: bool = True,
) -> Dict[str, List[Tuple[str, object, str]]]:
    """Split records while keeping inferred homology clusters intact."""
    from .cluster import compute_homology_clusters

    if config.no_clustering:
        assignment = stratified_random_split(records, config)
    else:
        if verbose:
            print(
                f"[cerberos] clustering {len(records)} sequences "
                f"(threshold={config.similarity_threshold:.2f}, "
                f"alphabet={config.reduced_alphabet}, verify={config.verify_with})",
                file=__import__("sys").stderr,
            )
        clusters = compute_homology_clusters(records, config, verbose=verbose)
        if verbose:
            print(f"[cerberos] {len(clusters)} clusters", file=__import__("sys").stderr)
        assignment = assign_clusters(clusters, len(records), config)
    splits: Dict[str, List[Tuple[str, object, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for index, record in enumerate(records):
        splits[assignment[index]].append(record)
    return splits
