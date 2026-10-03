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


def _nearest_neighbor_similarity(
    query_records, reference_records, k: int
) -> Dict[str, object]:
    """Return the exact-k-mer similarity distribution to the nearest reference."""
    if not query_records or not reference_records:
        return {"n_queries": len(query_records), "mean": None, "max": None, "values": []}
    values = [
        max(exact_kmer_jaccard(query[2], reference[2], k) for reference in reference_records)
        for query in query_records
    ]
    return {
        "n_queries": len(values),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
        "values": values,
    }


def _feature_balance_against_overall(splits: Dict[str, list], feature_names: List[str]):
    """Compare each split's feature distribution with the full dataset."""
    all_records = [record for records in splits.values() for record in records]
    overall = build_feature_matrix(all_records)[0]
    output = {}
    for split, records in splits.items():
        matrix = build_feature_matrix(records)[0]
        ks_values, wasserstein_values = [], []
        for index in range(len(feature_names)):
            if matrix.shape[0] == 0 or overall.shape[0] == 0:
                ks_values.append(None)
                wasserstein_values.append(None)
            else:
                left, right = matrix[:, index], overall[:, index]
                ks_values.append(_ks_statistic(left, right))
                wasserstein_values.append(_wasserstein_distance(left, right))
        output[split] = {"ks": ks_values, "wasserstein": wasserstein_values}
    return output


def _cluster_quality(matrix, labels, max_n=1000):
    """Compute NumPy-only silhouette, Davies-Bouldin, and CH scores."""
    if matrix is None or len(matrix) < 3 or len(set(labels)) < 2:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels)
    if len(matrix) > max_n:
        indices = np.linspace(0, len(matrix) - 1, max_n, dtype=int)
        matrix, labels = matrix[indices], labels[indices]
    unique = sorted(set(labels.tolist()))
    distances = np.sqrt(np.maximum(0.0, ((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2)))
    centroids, scatters = {}, {}
    for label in unique:
        members = matrix[labels == label]
        centroids[label] = members.mean(axis=0)
        scatters[label] = float(np.mean(np.linalg.norm(members - centroids[label], axis=1)))
    silhouettes = []
    for index, label in enumerate(labels):
        own = labels == label
        a = float(distances[index, own].sum() / max(1, own.sum() - 1))
        other_means = [float(distances[index, labels == other].mean()) for other in unique if other != label]
        b = min(other_means) if other_means else 0.0
        silhouettes.append((b - a) / max(a, b, 1e-12))
    db_terms = []
    for left in unique:
        ratios = []
        for right in unique:
            if left != right:
                separation = np.linalg.norm(centroids[left] - centroids[right])
                ratios.append((scatters[left] + scatters[right]) / max(separation, 1e-12))
        db_terms.append(max(ratios) if ratios else 0.0)
    overall = matrix.mean(axis=0)
    between = sum(np.sum(labels == label) * np.sum((centroids[label] - overall) ** 2) for label in unique)
    within = sum(np.sum((matrix[labels == label] - centroids[label]) ** 2) for label in unique)
    ch = between / max(within, 1e-12) * (len(matrix) - len(unique)) / max(1, len(unique) - 1)
    return {"silhouette": float(np.mean(silhouettes)), "davies_bouldin": float(np.mean(db_terms)), "calinski_harabasz": float(ch)}


def _size_confidence_intervals(sizes, config):
    total = sum(sizes.values())
    output = {}
    for name, proportion in zip(("train", "val", "test"), (config.train_pct, config.val_pct, config.test_pct)):
        standard_error = (proportion * (1 - proportion) / max(1, total)) ** 0.5
        output[name] = {
            "observed": sizes.get(name, 0),
            "target_proportion": proportion,
            "expected": proportion * total,
            "random_assignment_95ci_proportion": [max(0.0, proportion - 1.96 * standard_error), min(1.0, proportion + 1.96 * standard_error)],
        }
    return output


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
    stats["feature_balance"] = {
        "reference": "overall_dataset",
        "features": feature_names,
        "splits": _feature_balance_against_overall(splits, feature_names),
    }
    stats["nearest_neighbor_similarity"] = {
        "metric": f"maximum exact {kmer_k}-mer Jaccard to reference split",
        "test_to_train": _nearest_neighbor_similarity(
            splits.get("test", []), splits.get("train", []), kmer_k
        ),
        "val_to_train": _nearest_neighbor_similarity(
            splits.get("val", []), splits.get("train", []), kmer_k
        ),
    }
    stats["size_confidence_intervals"] = _size_confidence_intervals(stats["sizes"], config)
    stats["cluster_quality"] = _cluster_quality(
        config.last_cluster_features,
        config.last_cluster_assignments,
        max_n=config.diagnostics_sample_size,
    )
    if config.last_assignment_stats:
        stats["assignment"] = dict(config.last_assignment_stats)
    if config.last_clustering_stats:
        stats["clustering"] = dict(config.last_clustering_stats)
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
            f"  max_candidates   : {config.get('max_candidate_pairs')}",
            f"  seed             : {config.get('seed')}",
        ]
    )
    candidate_generation = stats.get("candidate_generation")
    candidate_warning = bool(
        candidate_generation
        and (
            candidate_generation.get("candidate_limit_reached", False)
            or candidate_generation.get("oversized_lsh_buckets_sampled", 0)
        )
    )
    if candidate_generation:
        lines.extend(
            [
                "",
                "Candidate generation:",
                f"  pairs considered : {candidate_generation.get('candidate_pairs_considered', 0)}",
                f"  pairs verified   : {candidate_generation.get('candidate_pairs_verified', 0)}",
                f"  dense buckets sampled: {candidate_generation.get('oversized_lsh_buckets_sampled', 0)}",
            ]
        )
        if candidate_generation.get("candidate_limit_reached", False):
            lines.append(
                "  WARNING: candidate-pair limit reached; this split may contain "
                "unmerged similar sequences."
            )
        elif candidate_generation.get("oversized_lsh_buckets_sampled", 0):
            lines.append(
                "  WARNING: dense LSH buckets were sampled; candidate recall is reduced."
            )
    lines.extend(["", "Split sizes:"])
    for name, count in stats["sizes"].items():
        lines.append(f"  {name:<6s}: {count:>6d}")
    clustering = stats.get("clustering", {})
    if clustering:
        lines.extend(
            [
                "",
                "Clustering observability:",
                f"  accepted edges       : {clustering.get('accepted_edges', 0)}",
                f"  clusters             : {clustering.get('cluster_count', 'n/a')}",
                f"  singleton clusters   : {clustering.get('singleton_count', 'n/a')}",
                f"  non-singleton fraction: {_format_number(clustering.get('non_singleton_fraction'), '.3f')}",
                f"  size distribution    : {clustering.get('cluster_size_distribution', {})}",
            ]
        )
        if clustering.get("candidate_recall_benchmark") is not None:
            lines.append(f"  exact benchmark recall: {clustering['candidate_recall_benchmark']:.3f}")
        lines.append(
            "  accepted/rejected score histograms: "
            f"{clustering.get('accepted_score_histogram', [])} / "
            f"{clustering.get('rejected_score_histogram', [])}"
        )
    assignment = stats.get("assignment", {})
    if assignment:
        lines.extend(
            [
                "",
                "Cluster assignment:",
                f"  method               : {assignment.get('method', 'n/a')}",
                f"  target sizes         : {assignment.get('target_sizes', {})}",
                f"  observed sizes       : {assignment.get('observed_sizes', {})}",
            ]
        )
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
    nn = stats.get("nearest_neighbor_similarity", {})
    if nn:
        lines.extend(["", "Cross-split nearest-neighbor similarity (leakage risk):"])
        for name in ("test_to_train", "val_to_train"):
            values = nn.get(name, {})
            lines.append(
                f"  {name:<14s}: mean={_format_number(values.get('mean'), '.3f')} "
                f"p95={_format_number(values.get('p95'), '.3f')} "
                f"max={_format_number(values.get('max'), '.3f')}"
            )
    quality = stats.get("cluster_quality", {})
    if quality:
        lines.extend(
            [
                "",
                "Cluster quality:",
                f"  silhouette          : {_format_number(quality.get('silhouette'), '.3f')}",
                f"  Davies-Bouldin      : {_format_number(quality.get('davies_bouldin'), '.3f')}",
                f"  Calinski-Harabasz   : {_format_number(quality.get('calinski_harabasz'), '.3f')}",
            ]
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
    if (
        not stats.get("homology_warnings")
        and not stats.get("ood_warnings")
        and not candidate_warning
    ):
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


def build_html_report(stats: dict, mode: str) -> str:
    """Render the plain-text report and machine-readable stats as standalone HTML."""
    import html
    import json

    text = html.escape(build_report(stats, mode))
    payload = html.escape(json.dumps(stats, indent=2, allow_nan=False))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Cerberos diagnostics</title><style>"
        "body{font:15px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#202124}"
        "pre{white-space:pre-wrap;background:#f6f8fa;padding:1rem;border-radius:8px}"
        "details{margin-top:1rem}</style></head><body>"
        f"<h1>Cerberos diagnostics ({html.escape(mode)} mode)</h1><pre>{text}</pre>"
        f"<details><summary>Raw stats.json</summary><pre>{payload}</pre></details>"
        "</body></html>"
    )
