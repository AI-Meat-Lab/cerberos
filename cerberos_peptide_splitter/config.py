"""Validated configuration shared by the Python API and command line."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

VALID_ALPHABETS = ("none", "groups5", "groups7")
VALID_VERIFIERS = ("none", "kmer-exact", "levenshtein", "containment", "cosine")
VALID_CLUSTER_METHODS = ("components", "label-propagation")
VALID_AGGREGATIONS = ("max", "mean", "vote", "weighted")
VALID_BALANCE_METRICS = ("mean", "wasserstein", "ks")
VALID_BACKENDS = ("auto", "numpy", "cupy")
_MAX_SEED = 2**32 - 1


@dataclass
class RunConfig:
    """Options controlling clustering, splitting, balancing, and diagnostics."""

    train_pct: float = 0.80
    val_pct: float = 0.10
    test_pct: float = 0.10
    kmer_sizes: List[int] = field(default_factory=lambda: [3])
    num_hashes: int = 128
    similarity_threshold: float = 0.50
    reduced_alphabet: str = "none"
    verify_with: str = "none"
    verification_metrics: List[str] = field(default_factory=list)
    metric_aggregation: str = "max"
    metric_weights: List[float] = field(default_factory=list)
    prefilter_threshold: float = 0.30
    seed: int = 42
    no_clustering: bool = False
    max_candidate_pairs: int = 250_000
    strict_clustering: bool = False
    min_non_singleton_fraction: float = 0.05
    balance: bool = False
    balance_weight: float = 1.0
    balance_metric: str = "wasserstein"
    balance_report: bool = False
    lsh_rows: List[int] = field(default_factory=lambda: [2, 4, 8])
    exact_mode: bool = False
    exact_mode_max_sequences: int = 50_000
    adaptive_threshold: bool = False
    cluster_method: str = "components"
    short_peptide_mode: str = "warn"
    local_search_iterations: int = 0
    stratify_labels: bool = False
    pareto_points: int = 0
    dry_run: bool = False
    diagnostics_sample_size: int = 10000
    backend: str = "auto"
    external_clusters: str | None = None
    last_clustering_stats: dict = field(default_factory=dict, init=False, repr=False)
    last_assignment_stats: dict = field(default_factory=dict, init=False, repr=False)
    last_cluster_assignments: List[int] = field(
        default_factory=list, init=False, repr=False
    )
    last_cluster_by_record_id: dict = field(
        default_factory=dict, init=False, repr=False
    )
    last_cluster_features: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.reduced_alphabet not in VALID_ALPHABETS:
            raise ValueError(f"reduced_alphabet must be one of {VALID_ALPHABETS}")
        if self.verify_with not in VALID_VERIFIERS:
            raise ValueError(f"verify_with must be one of {VALID_VERIFIERS}")
        if self.metric_aggregation not in VALID_AGGREGATIONS:
            raise ValueError(f"metric_aggregation must be one of {VALID_AGGREGATIONS}")
        if self.balance_metric not in VALID_BALANCE_METRICS:
            raise ValueError(f"balance_metric must be one of {VALID_BALANCE_METRICS}")
        if self.backend not in VALID_BACKENDS:
            raise ValueError(f"backend must be one of {VALID_BACKENDS}")
        metrics = self.verification_metrics or (
            [self.verify_with] if self.verify_with != "none" else []
        )
        if any(metric not in VALID_VERIFIERS or metric == "none" for metric in metrics):
            raise ValueError("verification_metrics contains an unsupported metric")
        self.verification_metrics = list(dict.fromkeys(metrics))
        if self.verification_metrics and self.verify_with == "none":
            self.verify_with = self.verification_metrics[0]
        if self.metric_weights and len(self.metric_weights) != len(
            self.verification_metrics
        ):
            raise ValueError("metric_weights must match verification_metrics")
        if any(
            weight < 0 or not math.isfinite(weight) for weight in self.metric_weights
        ):
            raise ValueError("metric_weights must be finite and non-negative")
        if self.cluster_method not in VALID_CLUSTER_METHODS:
            raise ValueError(f"cluster_method must be one of {VALID_CLUSTER_METHODS}")
        if self.short_peptide_mode not in ("ignore", "warn", "auto"):
            raise ValueError("short_peptide_mode must be ignore, warn, or auto")
        percentages = (self.train_pct, self.val_pct, self.test_pct)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in percentages
        ):
            raise ValueError("split percentages must be finite and non-negative")
        total = sum(percentages)
        if total <= 0:
            raise ValueError("at least one split percentage must be positive")
        self.train_pct, self.val_pct, self.test_pct = (
            value / total for value in percentages
        )
        if not self.kmer_sizes or any(
            isinstance(k, bool) or not isinstance(k, int) or k < 1
            for k in self.kmer_sizes
        ):
            raise ValueError("kmer_sizes must contain positive integers")
        if (
            isinstance(self.num_hashes, bool)
            or not isinstance(self.num_hashes, int)
            or self.num_hashes < 4
        ):
            raise ValueError("num_hashes must be an integer of at least 4")
        for name, value in (
            ("similarity_threshold", self.similarity_threshold),
            ("prefilter_threshold", self.prefilter_threshold),
            ("min_non_singleton_fraction", self.min_non_singleton_fraction),
            ("balance_weight", self.balance_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if (
            self.verify_with != "none"
            and self.prefilter_threshold > self.similarity_threshold
        ):
            raise ValueError(
                "prefilter_threshold must not exceed similarity_threshold when verification is enabled"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.no_clustering, bool):
            raise ValueError("no_clustering must be a boolean")
        if not isinstance(self.strict_clustering, bool):
            raise ValueError("strict_clustering must be a boolean")
        if not isinstance(self.balance, bool):
            raise ValueError("balance must be a boolean")
        if not isinstance(self.balance_report, bool):
            raise ValueError("balance_report must be a boolean")
        if not isinstance(self.stratify_labels, bool):
            raise ValueError("stratify_labels must be a boolean")
        if not isinstance(self.dry_run, bool):
            raise ValueError("dry_run must be a boolean")
        for name, value, minimum in (
            ("max_candidate_pairs", self.max_candidate_pairs, 1),
            ("exact_mode_max_sequences", self.exact_mode_max_sequences, 2),
            ("local_search_iterations", self.local_search_iterations, 0),
            ("pareto_points", self.pareto_points, 0),
            ("diagnostics_sample_size", self.diagnostics_sample_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not self.lsh_rows or any(
            isinstance(row, bool) or not isinstance(row, int) or row < 1
            for row in self.lsh_rows
        ):
            raise ValueError("lsh_rows must contain positive integers")
        if not 0 <= self.seed <= _MAX_SEED:
            raise ValueError(f"seed must be between 0 and {_MAX_SEED}")

    def as_dict(self) -> dict:
        """Return a JSON-friendly snapshot of the effective configuration."""
        return {
            "train_pct": self.train_pct,
            "val_pct": self.val_pct,
            "test_pct": self.test_pct,
            "kmer_sizes": list(self.kmer_sizes),
            "num_hashes": self.num_hashes,
            "similarity_threshold": self.similarity_threshold,
            "reduced_alphabet": self.reduced_alphabet,
            "verify_with": self.verify_with,
            "verification_metrics": list(self.verification_metrics),
            "metric_aggregation": self.metric_aggregation,
            "metric_weights": list(self.metric_weights),
            "prefilter_threshold": (
                self.prefilter_threshold if self.verify_with != "none" else None
            ),
            "seed": self.seed,
            "no_clustering": self.no_clustering,
            "max_candidate_pairs": self.max_candidate_pairs,
            "strict_clustering": self.strict_clustering,
            "min_non_singleton_fraction": self.min_non_singleton_fraction,
            "balance": self.balance,
            "balance_weight": self.balance_weight,
            "balance_metric": self.balance_metric,
            "balance_report": self.balance_report,
            "lsh_rows": list(self.lsh_rows),
            "exact_mode": self.exact_mode,
            "exact_mode_max_sequences": self.exact_mode_max_sequences,
            "adaptive_threshold": self.adaptive_threshold,
            "cluster_method": self.cluster_method,
            "short_peptide_mode": self.short_peptide_mode,
            "local_search_iterations": self.local_search_iterations,
            "stratify_labels": self.stratify_labels,
            "pareto_points": self.pareto_points,
            "dry_run": self.dry_run,
            "diagnostics_sample_size": self.diagnostics_sample_size,
            "backend": self.backend,
            "external_clusters": self.external_clusters,
        }
