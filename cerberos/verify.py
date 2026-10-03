"""Deterministic verification metrics for Path B.

Two metrics are provided:

* ``kmer-exact``  — exact k-mer Jaccard on the *transformed* sequence
                    (matches the space the MinHash sketch lived in).
* ``levenshtein`` — normalised edit-distance identity on the *raw*
                    sequence (a true per-residue measure).

Both are bounded so expensive comparisons exit early.
"""
from __future__ import annotations

from typing import Optional

from .kmers import exact_kmer_jaccard


# ─────────────────────── Levenshtein ───────────────────────

def levenshtein(a: str, b: str, max_dist: Optional[int] = None) -> int:
    """Banded edit distance.

    If `max_dist` is set and the true distance exceeds it, returns
    ``max_dist + 1`` immediately (cheap rejection).
    """
    if len(a) < len(b):
        a, b = b, a
    if max_dist is not None and len(a) - len(b) > max_dist:
        return max_dist + 1
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1,      # insertion
                          prev[j] + 1,           # deletion
                          prev[j - 1] + cost)    # substitution
            if curr[j] < row_min:
                row_min = curr[j]
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        prev = curr
    return prev[-1]


def levenshtein_identity(
    a: str, b: str, max_dist: Optional[int] = None
) -> float:
    """Normalised identity = 1 - dist / max_len, in [0, 1]."""
    if not a and not b:
        return 1.0
    m = max(len(a), len(b))
    d = levenshtein(a, b, max_dist)
    if max_dist is not None and d > max_dist:
        return 0.0
    return 1.0 - d / m


# ─────────────────────── dispatch ───────────────────────

def verify_pair(
    raw_a: str,
    raw_b: str,
    transformed_a: str,
    transformed_b: str,
    kmer_sizes: list,
    metric: str,
    threshold: float,
) -> float:
    """Return the verification score for a candidate pair.

    The return value is the score, not a boolean, so callers can log
    the distribution of accepted/rejected scores.  A return of 0.0
    means "rejected by an early-exit bound or a failed metric".
    """
    if metric == "kmer-exact":
        return max(
            exact_kmer_jaccard(transformed_a, transformed_b, k)
            for k in kmer_sizes
        )

    if metric == "levenshtein":
        max_len = max(len(raw_a), len(raw_b))
        cap = int((1.0 - threshold) * max_len)
        return levenshtein_identity(raw_a, raw_b, max_dist=cap)

    raise ValueError(f"unknown verify_with metric: {metric!r}")
