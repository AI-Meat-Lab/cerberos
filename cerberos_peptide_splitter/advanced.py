"""Optional advanced utilities for roadmap workflows."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Sequence

from .config import RunConfig
from .kmers import exact_kmer_jaccard


def cross_validation_assignments(
    clusters: Sequence[Sequence[int]],
    n: int,
    folds: int,
    seed: int = 42,
    records=None,
    config: RunConfig | None = None,
) -> list[dict[int, str]]:
    """Assign intact clusters to deterministic K-fold train/test partitions."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    flattened = [index for cluster in clusters for index in cluster]
    if len(flattened) != n or set(flattened) != set(range(n)):
        raise ValueError(
            "clusters must partition all indices from 0 to n-1 exactly once"
        )
    ordered = list(clusters)
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    fold_sizes = [0] * folds
    target_size = n / folds
    fold_clusters: list[list[Sequence[int]]] = [[] for _ in range(folds)]
    for cluster in ordered:
        fold = min(
            range(folds),
            key=lambda index: (
                abs(fold_sizes[index] + len(cluster) - target_size),
                index,
            ),
        )
        fold_clusters[fold].append(cluster)
        fold_sizes[fold] += len(cluster)
    output = []
    for fold_index in range(folds):
        assignment = {}
        for index, cluster_group in enumerate(fold_clusters):
            split = "test" if index == fold_index else "train"
            for cluster in cluster_group:
                assignment.update(dict.fromkeys(cluster, split))
        output.append(assignment)
    return output


def hierarchical_cluster_views(records, config: RunConfig, thresholds: Sequence[float]):
    """Return nested operational cluster views at several similarity thresholds."""
    from .cluster import compute_homology_clusters

    views = {}
    for threshold in sorted({float(value) for value in thresholds}, reverse=True):
        options = config.as_dict()
        options["prefilter_threshold"] = config.prefilter_threshold
        options["similarity_threshold"] = threshold
        child = RunConfig(**options)
        views[str(threshold)] = compute_homology_clusters(records, child, verbose=False)
    return views


def boundary_records(splits, k: int, threshold: float = 0.7) -> list[dict]:
    """Identify validation/test records near the leakage decision boundary."""
    train = splits.get("train", [])
    results = []
    for split in ("val", "test"):
        for record in splits.get(split, []):
            if not train:
                continue
            scores = [
                (exact_kmer_jaccard(record[2], ref[2], k), ref[0]) for ref in train
            ]
            score, nearest_id = max(scores)
            if abs(score - threshold) <= 0.10:
                results.append(
                    {
                        "id": record[0],
                        "split": split,
                        "nearest_train_id": nearest_id,
                        "similarity": score,
                    }
                )
    return sorted(results, key=lambda item: (-item["similarity"], item["id"]))


def load_cluster_assignments(path: str, n: int) -> list[list[int]]:
    """Load an external JSON/CSV id-to-cluster mapping for SpanSeq-style workflows."""
    source = Path(path)
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text())
        mapping = {int(key): int(value) for key, value in data.items()}
    else:
        with source.open(newline="") as handle:
            rows = csv.DictReader(handle)
            mapping = {int(row["index"]): int(row["cluster"]) for row in rows}
    if set(mapping) != set(range(n)):
        raise ValueError(
            "external cluster mapping must contain every record index exactly once"
        )
    groups: dict[int, list[int]] = {}
    for index, cluster in mapping.items():
        groups.setdefault(cluster, []).append(index)
    return [groups[key] for key in sorted(groups)]
