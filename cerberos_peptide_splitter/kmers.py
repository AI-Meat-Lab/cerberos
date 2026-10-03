"""Unique k-mer sets and Jaccard similarity."""

from __future__ import annotations

from typing import Set


def kmers(seq: str, k: int) -> Set[str]:
    """Return the set of contiguous length-``k`` substrings of ``seq``."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if len(seq) < k:
        return set()
    return {seq[index : index + k] for index in range(len(seq) - k + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Return set Jaccard similarity (empty/empty set convention is 1)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def exact_kmer_jaccard(seq_a: str, seq_b: str, k: int) -> float:
    """Return exact unique-k-mer Jaccard, or 0 if either sequence lacks k-mers.

    Empty/empty sequence k-mer sets contain no evidence of similarity, so this
    wrapper returns 0.0 rather than applying :func:`jaccard`'s set convention.
    """
    kmers_a, kmers_b = kmers(seq_a, k), kmers(seq_b, k)
    if not kmers_a or not kmers_b:
        return 0.0
    return jaccard(kmers_a, kmers_b)
