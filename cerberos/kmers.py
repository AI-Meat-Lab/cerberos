"""k-mer sets and exact Jaccard similarity."""
from __future__ import annotations

from typing import Set


def kmers(seq: str, k: int) -> Set[str]:
    if len(seq) < k:
        return set()
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def exact_kmer_jaccard(seq_a: str, seq_b: str, k: int) -> float:
    """Exact (non-estimated) k-mer Jaccard — used by Path B's kmer-exact
    verifier and by diagnostics."""
    return jaccard(kmers(seq_a, k), kmers(seq_b, k))
