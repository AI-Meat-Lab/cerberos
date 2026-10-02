"""Assign records or clusters to train / validation / test."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from .config import RunConfig


def _target_sizes(n: int, config: RunConfig) -> Tuple[int, int, int]:
    n_train = int(round(config.train_pct * n))
    n_val = int(round(config.val_pct * n))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0
    return n_train, n_val, n_test


# ─────────────────────── stratified fallback ───────────────────────

def stratified_random_split(
    records: Sequence[Tuple[str, object, str]],
    config: RunConfig,
) -> Dict[int, str]:
    """Pure random split, stratified by label when labels exist."""
    rng = random.Random(config.seed)
    labels = [r[1] for r in records]
    labeled = all(lbl is not None for lbl in labels)
    assign: Dict[int, str] = {}

    if labeled:
        by_label: Dict[int, list] = defaultdict(list)
        for i, lbl in enumerate(labels):
            by_label[lbl].append(i)
        for _, idxs in by_label.items():
            rng.shuffle(idxs)
            n_tr, n_va, _ = _target_sizes(len(idxs), config)
            for i, ix in enumerate(idxs):
                assign[ix] = ("train" if i < n_tr else
                              "val"   if i < n_tr + n_va else "test")
    else:
        idx = list(range(len(records)))
        rng.shuffle(idx)
        n_tr, n_va, _ = _target_sizes(len(records), config)
        for i, ix in enumerate(idx):
            assign[ix] = ("train" if i < n_tr else
                          "val"   if i < n_tr + n_va else "test")
    return assign


# ─────────────────────── cluster assignment ───────────────────────

def assign_clusters(
    clusters: List[List[int]], n: int, config: RunConfig
) -> Dict[int, str]:
    """Greedily place whole clusters into the split furthest from target."""
    rng = random.Random(config.seed)
    target = {
        "train": config.train_pct * n,
        "val":   config.val_pct   * n,
        "test":  config.test_pct  * n,
    }
    counts = {"train": 0, "val": 0, "test": 0}

    clusters = list(clusters)
    rng.shuffle(clusters)
    clusters.sort(key=len, reverse=True)

    assign: Dict[int, str] = {}
    for cl in clusters:
        best_split, best_score = None, None
        for s in ("train", "val", "test"):
            before = abs(counts[s] - target[s])
            after = abs(counts[s] + len(cl) - target[s])
            score = after - before
            if best_score is None or score < best_score:
                best_score, best_split = score, s
        assign.update({i: best_split for i in cl})
        counts[best_split] += len(cl)
    return assign


# ─────────────────────── top-level split ───────────────────────

def split_records(
    records: Sequence[Tuple[str, object, str]],
    config: RunConfig,
    verbose: bool = True,
) -> Dict[str, List[Tuple[str, object, str]]]:
    """Split records into train / val / test."""
    from .cluster import compute_homology_clusters

    if config.no_clustering:
        assign = stratified_random_split(records, config)
    else:
        if verbose:
            print(f"[cerberos] clustering {len(records)} sequences "
                  f"(threshold={config.similarity_threshold:.2f}, "
                  f"alphabet={config.reduced_alphabet}, "
                  f"verify={config.verify_with})",
                  file=__import__("sys").stderr)
        clusters = compute_homology_clusters(records, config, verbose=verbose)
        if verbose:
            print(f"[cerberos] {len(clusters)} clusters",
                  file=__import__("sys").stderr)
        assign = assign_clusters(clusters, len(records), config)

    splits: Dict[str, list] = {"train": [], "val": [], "test": []}
    for i, rec in enumerate(records):
        splits[assign[i]].append(rec)
    return splits
