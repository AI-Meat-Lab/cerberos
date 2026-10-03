"""Descriptive split diagnostics and human-readable report generation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import RunConfig
from .features import build_feature_matrix
from .kmers import exact_kmer_jaccard


def _sample(sequences, k, rng, max_n):
    """Sample at most ``max_n`` values; ``k`` is retained for compatibility."""
    del k
    if len(sequences) <= max_n:
        return sequences
    indices = rng.choice(len(sequences), size=max_n, replace=False)
    return [sequences[index] for index in indices]


def _mean_pairwise_jaccard(seqs_a, seqs_b, k, same):
    if same:
        similarities = [
            exact_kmer_jaccard(seqs_a[i], seqs_a[j], k)
            for i in range(len(seqs_a))
            for j in range(i + 1, len(seqs_a))
        ]
        n_pairs = len(seqs_a) * (len(seqs_a) - 1) // 2
    else:
        similarities = [exact_kmer_jaccard(a, b, k) for a in seqs_a for b in seqs_b]
        n_pairs = len(seqs_a) * len(seqs_b)
    if not similarities:
        return None, None, n_pairs
    return (
        float(np.mean(similarities)),
        float(np.std(similarities)),
        n_pairs,
    )


def pairwise_similarity(
    splits: Dict[str, list], k: int, max_per_split: int, seed: int
) -> Dict[str, dict]:
    """Summarize sampled within- and between-split k-mer Jaccard values."""
    if max_per_split < 1:
        raise ValueError("max_per_split must be positive")
    rng = np.random.RandomState(seed)
    sequences = {
        name: _sample([record[2] for record in records], k, rng, max_per_split)
        for name, records in splits.items()
    }
    names = list(splits)
    output: Dict[str, dict] = {}
    for i, split_a in enumerate(names):
        for split_b in names[i:]:
            mean, std, count = _mean_pairwise_jaccard(
                sequences[split_a], sequences[split_b], k, split_a == split_b
            )
            output[f"{split_a}-{split_b}"] = {
                "mean_jaccard": mean,
                "std": std,
                "n_pairs": count,
            }
    return output


def _ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Exact two-sample empirical Kolmogorov-Smirnov distance (D statistic)."""
    values = np.sort(np.concatenate((sample_a, sample_b)))
    cdf_a = np.searchsorted(np.sort(sample_a), values, side="right") / len(sample_a)
    cdf_b = np.searchsorted(np.sort(sample_b), values, side="right") / len(sample_b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _wasserstein_distance(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Exact one-dimensional first Wasserstein distance for empirical samples."""
    values = np.unique(np.concatenate((sample_a, sample_b)))
    if len(values) < 2:
        return 0.0
    sorted_a, sorted_b = np.sort(sample_a), np.sort(sample_b)
    cdf_a = np.searchsorted(sorted_a, values[:-1], side="right") / len(sorted_a)
    cdf_b = np.searchsorted(sorted_b, values[:-1], side="right") / len(sorted_b)
    return float(np.sum(np.abs(cdf_a - cdf_b) * np.diff(values)))


def feature_ks(
    splits: Dict[str, list], feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Return empirical KS D and 1-D Wasserstein distances for each feature.

    These descriptive distances are computed directly with NumPy; no SciPy
    approximation is substituted for the KS statistic when SciPy is absent.
    Empty split pairs receive zero distances and should be interpreted with
    their accompanying split sizes.
    """
    names = list(splits)
    matrices = {name: build_feature_matrix(splits[name])[0] for name in names}
    feature_names = list(feature_names)
    pairs = [
        (split_a, split_b)
        for index, split_a in enumerate(names)
        for split_b in names[index + 1 :]
    ]
    ks = np.zeros((len(feature_names), len(pairs)), dtype=np.float64)
    wasserstein = np.zeros_like(ks)
    for pair_index, (split_a, split_b) in enumerate(pairs):
        matrix_a, matrix_b = matrices[split_a], matrices[split_b]
        for feature_index, _feature_name in enumerate(feature_names):
            if (
                matrix_a.shape[0] == 0
                or matrix_b.shape[0] == 0
                or feature_index >= matrix_a.shape[1]
                or feature_index >= matrix_b.shape[1]
            ):
                continue
            values_a = matrix_a[:, feature_index]
            values_b = matrix_b[:, feature_index]
            ks[feature_index, pair_index] = _ks_statistic(values_a, values_b)
            wasserstein[feature_index, pair_index] = _wasserstein_distance(
                values_a, values_b
            )
    return ks, wasserstein, pairs


def summarize(
    splits: Dict[str, list],
    config: RunConfig,
    kmer_k: int = 3,
    sim_plot_cap: int = 200,
) -> Tuple[dict, List[str]]:
    """Build JSON-compatible descriptive statistics for the supplied splits."""
    stats: dict = {
        "clustering_mode": config.as_dict(),
        "sizes": {},
        "label_counts": {},
        "length": {},
        "feature_means": {},
        "similarity": {},
        "homology_warnings": [],
        "ood_warnings": [],
    }
    feature_names: List[str] = []
    labeled = any(
        record[1] is not None for records in splits.values() for record in records
    )
    for name, records in splits.items():
        stats["sizes"][name] = len(records)
        if labeled:
            stats["label_counts"][name] = {
                "0": sum(record[1] == 0 for record in records),
                "1": sum(record[1] == 1 for record in records),
                "unlabeled": sum(record[1] is None for record in records),
            }
        lengths = [len(record[2]) for record in records]
        stats["length"][name] = {
            "mean": float(np.mean(lengths)) if lengths else None,
            "std": float(np.std(lengths)) if lengths else None,
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
        }
        matrix, names = build_feature_matrix(records)
        if names:
            feature_names = names
            stats["feature_means"][name] = {
                feature: float(matrix[:, index].mean())
                for index, feature in enumerate(names)
            }
        else:
            stats["feature_means"][name] = {}
    stats["feature_names"] = feature_names
    stats["similarity"] = pairwise_similarity(
        splits, k=kmer_k, max_per_split=sim_plot_cap, seed=config.seed
    )

    train_train = stats["similarity"].get("train-train", {}).get("mean_jaccard")
    train_test = stats["similarity"].get("train-test", {}).get("mean_jaccard")
    train_val = stats["similarity"].get("train-val", {}).get("mean_jaccard")
    if (
        train_train is not None
        and train_test is not None
        and train_test >= 0.75 * train_train
    ):
        stats["homology_warnings"].append(
            f"Test-train mean Jaccard ({train_test:.3f}) is close to "
            f"train-train ({train_train:.3f})."
        )
    if (
        train_train is not None
        and train_val is not None
        and train_val >= 0.85 * train_train
    ):
        stats["homology_warnings"].append(
            f"Val-train mean Jaccard ({train_val:.3f}) is close to "
            f"train-train ({train_train:.3f})."
        )

    ks, wasserstein, pairs = feature_ks(splits, feature_names)
    stats["ks"] = {
        "method": "two-sample empirical Kolmogorov-Smirnov D statistic",
        "pairs": [f"{a}-vs-{b}" for a, b in pairs],
        "features": feature_names,
        "values": ks.tolist(),
        "wasserstein": wasserstein.tolist(),
    }
    for pair_index, (split_a, split_b) in enumerate(pairs):
        for feature_index, feature in enumerate(feature_names):
            distance = ks[feature_index, pair_index]
            if distance > 0.20:
                stats["ood_warnings"].append(
                    f"Large KS D ({distance:.2f}) in '{feature}' "
                    f"between {split_a} and {split_b}."
                )
    return stats, feature_names


def _format_number(value: Optional[float], format_spec: str) -> str:
    return "n/a" if value is None else format(value, format_spec)


def build_report(stats: dict, mode: str) -> str:
    """Format statistics as a plain-text report without implying validation."""
    lines = ["=" * 68, f" Cerberos diagnostics ({mode} mode)", "=" * 68, ""]
    config = stats.get("clustering_mode", {})
    lines.extend(
        [
            "Clustering configuration:",
            f"  alphabet         : {config.get('reduced_alphabet', 'none')}",
            f"  verify_with      : {config.get('verify_with', 'none')}",
            f"  prefilter        : {config.get('prefilter_threshold')}",
            f"  similarity_thresh: {config.get('similarity_threshold')}",
            f"  kmer_sizes       : {config.get('kmer_sizes')}",
            f"  num_hashes       : {config.get('num_hashes')}",
            f"  seed             : {config.get('seed')}",
            "",
            "Split sizes:",
        ]
    )
    for name, count in stats["sizes"].items():
        lines.append(f"  {name:<6s}: {count:>6d}")
    lines.append("")
    if stats.get("label_counts"):
        lines.append("Label counts:")
        for name, counts in stats["label_counts"].items():
            lines.append(
                f"  {name:<6s}: label0={counts['0']:<5d} "
                f"label1={counts['1']:<5d} unlabeled={counts['unlabeled']}"
            )
        lines.append("")
    lines.append("Length stats:")
    for name, values in stats["length"].items():
        lines.append(
            f"  {name:<6s}: mean={_format_number(values['mean'], '.2f')} "
            f"std={_format_number(values['std'], '.2f')} "
            f"min={values['min']} max={values['max']}"
        )
    lines.extend(["", "Pairwise k-mer Jaccard (raw sequences, sampled):"])
    for name, values in stats["similarity"].items():
        lines.append(
            f"  {name:<14s}: mean={_format_number(values['mean_jaccard'], '.3f')} "
            f"std={_format_number(values['std'], '.3f')} "
            f"(pairs={values['n_pairs']})"
        )
    lines.append("")
    if stats.get("homology_warnings"):
        lines.append("Homology warnings (heuristic):")
        lines.extend(f"  ! {warning}" for warning in stats["homology_warnings"])
        lines.append("")
    if stats.get("ood_warnings"):
        lines.append("Distribution-distance warnings (descriptive):")
        lines.extend(f"  ! {warning}" for warning in stats["ood_warnings"][:20])
        if len(stats["ood_warnings"]) > 20:
            lines.append(f"  ... ({len(stats['ood_warnings']) - 20} more)")
        lines.append("")
    if not stats.get("homology_warnings") and not stats.get("ood_warnings"):
        lines.append(
            "No heuristic warnings; this is not a guarantee of leakage-free splits."
        )
    lines.extend(
        [
            "",
            "Interpretation note: these diagnostics are descriptive heuristics,",
            "not a substitute for domain-specific validation or alignment-based homology analysis.",
        ]
    )
    return "\n".join(lines)
