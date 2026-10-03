"""Deterministic MinHash sketches and locality-sensitive hashing.

CRC32 is stable across processes and platforms. The affine MinHash family is
practical, not a cryptographic or mathematically perfect random-hash process.
LSH proposes candidate pairs approximately; it cannot guarantee retrieval of
every pair above a target similarity.
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from itertools import combinations
from typing import Set, Tuple

import numpy as np

_PRIME = 2_147_483_647


def _hash_kmer(kmer: str) -> int:
    return zlib.crc32(kmer.encode("utf-8")) & 0xFFFFFFFF


def minhash_sketch(kmer_set: Set[str], num_hashes: int, seed: int) -> np.ndarray:
    """Return an int64 MinHash signature; empty sets use a sentinel value."""
    if (
        isinstance(num_hashes, bool)
        or not isinstance(num_hashes, int)
        or num_hashes < 1
    ):
        raise ValueError("num_hashes must be a positive integer")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    rng = np.random.RandomState(seed)
    a = rng.randint(1, _PRIME, size=num_hashes, dtype=np.int64)
    b = rng.randint(0, _PRIME, size=num_hashes, dtype=np.int64)
    result = np.full(num_hashes, _PRIME, dtype=np.int64)
    for kmer in kmer_set:
        hashed = _hash_kmer(kmer) % _PRIME
        values = (a * np.int64(hashed) + b) % _PRIME
        np.minimum(result, values, out=result)
    return result


def lsh_candidates(
    sketches: np.ndarray, num_hashes: int, rows: int = 4
) -> Set[Tuple[int, int]]:
    """Return pairs sharing at least one LSH band; include a partial final band."""
    if (
        isinstance(num_hashes, bool)
        or not isinstance(num_hashes, int)
        or num_hashes < 1
    ):
        raise ValueError("num_hashes must be a positive integer")
    if sketches.ndim != 2 or sketches.shape[1] != num_hashes:
        raise ValueError("sketches must be a 2D array with num_hashes columns")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError("rows must be a positive integer")
    buckets = defaultdict(list)
    band_count = (num_hashes + rows - 1) // rows
    for band_index in range(band_count):
        start = band_index * rows
        end = min(start + rows, num_hashes)
        for index, values in enumerate(sketches[:, start:end]):
            band_key = (band_index, values.tobytes())
            buckets[band_key].append(index)
    pairs: Set[Tuple[int, int]] = set()
    for indices in buckets.values():
        if len(indices) > 1:
            pairs.update(combinations(sorted(indices), 2))
    return pairs


def estimate_jaccard(sketch_a: np.ndarray, sketch_b: np.ndarray) -> float:
    """Estimate Jaccard similarity as the fraction of equal signature rows."""
    if sketch_a.shape != sketch_b.shape or sketch_a.size == 0:
        raise ValueError("sketches must have the same non-zero shape")
    return float(np.mean(sketch_a == sketch_b))
