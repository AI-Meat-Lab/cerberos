"""Single source of truth for every clustering / split option."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


VALID_ALPHABETS = ("none", "groups5", "groups7")
VALID_VERIFIERS = ("none", "kmer-exact", "levenshtein")


@dataclass
class RunConfig:
    """Everything the pipeline needs, independent of argparse."""

    # ── split proportions ────────────────────────────────────
    train_pct: float = 0.80
    val_pct: float = 0.10
    test_pct: float = 0.10

    # ── clustering ───────────────────────────────────────────
    kmer_sizes: List[int] = field(default_factory=lambda: [3])
    num_hashes: int = 128
    similarity_threshold: float = 0.70

    # ── Path A — reduced alphabet ────────────────────────────
    reduced_alphabet: str = "none"

    # ── Path B — deterministic verification ──────────────────
    verify_with: str = "none"
    prefilter_threshold: float = 0.30

    # ── misc ─────────────────────────────────────────────────
    seed: int = 42
    no_clustering: bool = False

    # ─────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        if self.reduced_alphabet not in VALID_ALPHABETS:
            raise ValueError(f"reduced_alphabet must be in {VALID_ALPHABETS}")
        if self.verify_with not in VALID_VERIFIERS:
            raise ValueError(f"verify_with must be in {VALID_VERIFIERS}")

        total = self.train_pct + self.val_pct + self.test_pct
        if abs(total - 1.0) > 1e-6:
            self.train_pct /= total
            self.val_pct /= total
            self.test_pct /= total

        if (self.verify_with != "none"
                and self.prefilter_threshold > self.similarity_threshold):
            # Prefilter must be looser than the final threshold.
            self.prefilter_threshold = max(
                0.0, self.similarity_threshold - 0.3
            )

    def as_dict(self) -> dict:
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
