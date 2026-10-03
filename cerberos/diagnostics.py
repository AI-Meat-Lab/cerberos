"""Statistical diagnostics and human-readable report."""
from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from .config import RunConfig
from .features import build_feature_matrix
from .kmers import exact_kmer_jaccard


def _try_scipy():
    try:
        from scipy.stats import ks_2samp, wasserstein_distance
        return ks_2samp, wasserstein_distance
    except Exception:
        return None, None


def _sample(seqs, k, rng, max_n):
    if len(seqs) <= max_n:
        return seqs
    idx = rng.choice(len(seqs), size=max_n, replace=False)
    return [seqs[i] for i in idx]


def _mean_pairwise_jaccard(seqs_a, seqs_b, k, same):
    if same:
        sims = [exact_kmer_jaccard(seqs_a[i], seqs_a[j], k)
                for i in range(len(seqs_a))
                for j in range(i + 1, len(seqs_a))]
        n_pairs = len(seqs_a) * (len(seqs_a) - 1) // 2
    else:
        sims = [exact_kmer_jaccard(a, b, k)
                for a in seqs_a for b in seqs_b]
        n_pairs = len(seqs_a) * len(seqs_b)
    if not sims:
        return float("nan"), float("nan"), n_pairs
    return float(np.mean(sims)), float(np.std(sims)), n_pairs


def pairwise_similarity(
    splits: Dict[str, list], k: int, max_per_split: int, seed: int
) -> Dict[str, dict]:
    rng = np.random.RandomState(seed)
    seqs = {n: _sample([r[2] for r in recs], k, rng, max_per_split)
            for n, recs in splits.items()}
    names = list(splits.keys())
    out: Dict[str, dict] = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                continue
            same = (a == b)
            m, s, n = _mean_pairwise_jaccard(seqs[a], seqs[b], k, same=same)
            out[f"{a}-{b}"] = {"mean_jaccard": m, "std": s, "n_pairs": n}
    return out


def feature_ks(
    splits: Dict[str, list], feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    ks_2samp, wasserstein = _try_scipy()
    names = list(splits.keys())
    feats = {n: build_feature_matrix(splits[n])[0] for n in names}
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

    ks = np.zeros((len(feature_names), len(pairs)))
    wd = np.zeros((len(feature_names), len(pairs)))
    for pi, (a, b) in enumerate(pairs):
        for fi in range(len(feature_names)):
            x, y = feats[a][:, fi], feats[b][:, fi]
            if len(x) < 2 or len(y) < 2:
                continue
            if ks_2samp is not None:
                s, _ = ks_2samp(x, y)
                ks[fi, pi] = s
                wd[fi, pi] = wasserstein(x, y)
            else:
                sd = np.std(np.concatenate([x, y])) + 1e-12
                ks[fi, pi] = abs(np.mean(x) - np.mean(y)) / sd
                wd[fi, pi] = abs(np.mean(x) - np.mean(y))
    return ks, wd, pairs


def summarize(
    splits: Dict[str, list],
    config: RunConfig,
    kmer_k: int = 3,
    sim_plot_cap: int = 200,
) -> Tuple[dict, List[str]]:
    stats: dict = {
        "clustering_mode": config.as_dict(),
        "sizes": {}, "label_counts": {}, "length": {},
        "feature_means": {}, "similarity": {},
        "homology_warnings": [], "ood_warnings": [],
    }

    feature_names: List[str] = []
    labeled = any(r[1] is not None for recs in splits.values() for r in recs)

    for name, recs in splits.items():
        stats["sizes"][name] = len(recs)
        if labeled:
            stats["label_counts"][name] = {
                "0": sum(1 for r in recs if r[1] == 0),
                "1": sum(1 for r in recs if r[1] == 1),
                "unlabeled": sum(1 for r in recs if r[1] is None),
            }
        lens = [len(r[2]) for r in recs] or [0]
        stats["length"][name] = {
            "mean": float(np.mean(lens)),
            "std": float(np.std(lens)),
            "min": int(np.min(lens)),
            "max": int(np.max(lens)),
        }
        if recs:
            mat, feature_names = build_feature_matrix(recs)
            stats["feature_means"][name] = {
                f: float(mat[:, i].mean()) for i, f in enumerate(feature_names)
            }

    stats["feature_names"] = feature_names
    stats["similarity"] = pairwise_similarity(
        splits, k=kmer_k, max_per_split=sim_plot_cap, seed=config.seed)

    # Homology warnings
    tr_tr = stats["similarity"].get("train-train", {}).get("mean_jaccard", np.nan)
    tr_te = stats["similarity"].get("test-train", {}).get("mean_jaccard", np.nan)
    tr_va = stats["similarity"].get("train-val", {}).get("mean_jaccard", np.nan)
    if not np.isnan(tr_tr) and not np.isnan(tr_te) and tr_te >= 0.75 * tr_tr:
        stats["homology_warnings"].append(
            f"Test-train mean Jaccard ({tr_te:.3f}) is close to "
            f"train-train ({tr_tr:.3f})."
        )
    if not np.isnan(tr_tr) and not np.isnan(tr_va) and tr_va >= 0.85 * tr_tr:
        stats["homology_warnings"].append(
            f"Val-train mean Jaccard ({tr_va:.3f}) is close to "
            f"train-train ({tr_tr:.3f})."
        )

    ks, wd, pairs = feature_ks(splits, feature_names)
    stats["ks"] = {
        "pairs": [f"{a}-vs-{b}" for a, b in pairs],
        "features": feature_names,
        "values": ks.tolist(),
        "wasserstein": wd.tolist(),
    }
    for pi, (a, b) in enumerate(pairs):
        for fi, fn in enumerate(feature_names):
            if ks[fi, pi] > 0.20:
                stats["ood_warnings"].append(
                    f"Large KS shift ({ks[fi, pi]:.2f}) in '{fn}' "
                    f"between {a} and {b}."
                )
    return stats, feature_names


def build_report(stats: dict, mode: str) -> str:
    L = ["=" * 68,
         f" Cerberos diagnostics ({mode} mode)",
         "=" * 68, ""]

    cm = stats.get("clustering_mode", {})
    L.append("Clustering configuration:")
    L.append(f"  alphabet         : {cm.get('reduced_alphabet', 'none')}")
    L.append(f"  verify_with      : {cm.get('verify_with', 'none')}")
    L.append(f"  prefilter        : {cm.get('prefilter_threshold')}")
    L.append(f"  similarity_thresh: {cm.get('similarity_threshold')}")
    L.append(f"  kmer_sizes       : {cm.get('kmer_sizes')}")
    L.append(f"  num_hashes       : {cm.get('num_hashes')}")
    L.append(f"  seed             : {cm.get('seed')}")
    L.append("")

    L.append("Split sizes:")
    for n, c in stats["sizes"].items():
        L.append(f"  {n:<6s}: {c:>6d}")
    L.append("")

    if stats.get("label_counts"):
        L.append("Label counts:")
        for n, d in stats["label_counts"].items():
            L.append(f"  {n:<6s}: label0={d['0']:<5d} "
                     f"label1={d['1']:<5d} unlabeled={d['unlabeled']}")
        L.append("")

    L.append("Length stats:")
    for n, d in stats["length"].items():
        L.append(f"  {n:<6s}: mean={d['mean']:.2f} "
                 f"std={d['std']:.2f} min={d['min']} max={d['max']}")
    L.append("")

    L.append("Pairwise k-mer Jaccard (raw sequences, sampled):")
    for k, d in stats["similarity"].items():
        L.append(f"  {k:<14s}: mean={d['mean_jaccard']:.3f} "
                 f"std={d['std']:.3f} (pairs={d['n_pairs']})")
    L.append("")

    if stats.get("homology_warnings"):
        L.append("Homology warnings:")
        for w in stats["homology_warnings"]:
            L.append(f"  ! {w}")
        L.append("")

    if stats.get("ood_warnings"):
        L.append("OOD / distribution shift warnings:")
        for w in stats["ood_warnings"][:20]:
            L.append(f"  ! {w}")
        if len(stats["ood_warnings"]) > 20:
            L.append(f"  ... ({len(stats['ood_warnings']) - 20} more)")
        L.append("")

    if (not stats.get("homology_warnings")
            and not stats.get("ood_warnings")):
        L.append("No warnings: splits look well matched.")
    return "\n".join(L)
