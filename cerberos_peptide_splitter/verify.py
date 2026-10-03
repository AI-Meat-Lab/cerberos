"""Deterministic pairwise verification metrics (not homology inference)."""

from __future__ import annotations

from typing import Optional

from .kmers import exact_kmer_jaccard, kmers


def levenshtein(a: str, b: str, max_dist: Optional[int] = None) -> int:
    """Return unit-cost global Levenshtein edit distance."""
    if max_dist is not None and (
        isinstance(max_dist, bool) or not isinstance(max_dist, int) or max_dist < 0
    ):
        raise ValueError("max_dist must be a non-negative integer or None")
    if len(a) < len(b):
        a, b = b, a
    if max_dist is not None and len(a) - len(b) > max_dist:
        return max_dist + 1
    if not b:
        return len(a)
    infinity = max_dist + 1 if max_dist is not None else len(a) + len(b)
    previous = list(range(len(b) + 1))
    if max_dist is not None:
        for column in range(max_dist + 1, len(previous)):
            previous[column] = infinity
    for row, char_a in enumerate(a, start=1):
        current = [infinity] * (len(b) + 1)
        if max_dist is None or row <= max_dist:
            current[0] = row
        start = 1 if max_dist is None else max(1, row - max_dist)
        end = len(b) if max_dist is None else min(len(b), row + max_dist)
        row_min = infinity
        for column in range(start, end + 1):
            cost = 0 if char_a == b[column - 1] else 1
            current[column] = min(
                current[column - 1] + 1,
                previous[column] + 1,
                previous[column - 1] + cost,
            )
            row_min = min(row_min, current[column])
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        previous = current
    distance = previous[-1]
    return min(distance, max_dist + 1) if max_dist is not None else distance


def levenshtein_identity(a: str, b: str, max_dist: Optional[int] = None) -> float:
    """Return normalized edit similarity ``1 - distance / max_length``."""
    distance = levenshtein(a, b, max_dist)
    if not a and not b:
        return 1.0
    denominator = max(len(a), len(b))
    if max_dist is not None and distance > max_dist:
        return 0.0
    return 1.0 - distance / denominator


def _containment(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return max(len(a & b) / len(a), len(a & b) / len(b))


def _cosine(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / (len(a) * len(b)) ** 0.5


def verify_pair(
    raw_a: str,
    raw_b: str,
    transformed_a: str,
    transformed_b: str,
    kmer_sizes: list,
    metric: str,
    threshold: float,
) -> float:
    """Score a candidate pair with one deterministic similarity metric."""
    if metric == "kmer-exact":
        if not kmer_sizes:
            raise ValueError("kmer_sizes must not be empty")
        return max(exact_kmer_jaccard(transformed_a, transformed_b, k) for k in kmer_sizes)
    if metric in ("containment", "cosine"):
        scores = []
        for k in kmer_sizes:
            left, right = kmers(transformed_a, k), kmers(transformed_b, k)
            scores.append(_containment(left, right) if metric == "containment" else _cosine(left, right))
        return max(scores, default=0.0)
    if metric == "levenshtein":
        max_length = max(len(raw_a), len(raw_b))
        if max_length == 0:
            return 1.0
        max_distance = int((1.0 - threshold) * max_length)
        return levenshtein_identity(raw_a, raw_b, max_dist=max_distance)
    raise ValueError(f"unknown verify_with metric: {metric!r}")
