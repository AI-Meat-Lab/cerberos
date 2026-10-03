"""Validated configuration shared by the Python API and command line."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

VALID_ALPHABETS = ("none", "groups5", "groups7")
VALID_VERIFIERS = ("none", "kmer-exact", "levenshtein")
_MAX_SEED = 2**32 - 1


@dataclass
class RunConfig:
    """Options controlling clustering, splitting, and diagnostics.

    Split percentages may use any common scale (for example, ``80, 10, 10``)
    and are normalized to sum to one. Thresholds and k-mer sizes are validated.
    """

    train_pct: float = 0.80
    val_pct: float = 0.10
    test_pct: float = 0.10
    kmer_sizes: List[int] = field(default_factory=lambda: [3])
    num_hashes: int = 128
    similarity_threshold: float = 0.70
    reduced_alphabet: str = "none"
    verify_with: str = "none"
    prefilter_threshold: float = 0.30
    seed: int = 42
    no_clustering: bool = False

    def __post_init__(self) -> None:
        if self.reduced_alphabet not in VALID_ALPHABETS:
            raise ValueError(f"reduced_alphabet must be one of {VALID_ALPHABETS}")
        if self.verify_with not in VALID_VERIFIERS:
            raise ValueError(f"verify_with must be one of {VALID_VERIFIERS}")
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
                "prefilter_threshold must not exceed similarity_threshold "
                "when verification is enabled"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.no_clustering, bool):
            raise ValueError("no_clustering must be a boolean")
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
            "prefilter_threshold": (
                self.prefilter_threshold if self.verify_with != "none" else None
            ),
            "seed": self.seed,
            "no_clustering": self.no_clustering,
        }
