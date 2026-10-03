"""MinHash sketches and LSH banding.

Hashes are computed via zlib.crc32, which is stable across processes
and independent of PYTHONHASHSEED — so runs are reproducible.
"""
from __future__ import annotations

import zlib
from collections import defaultdict
from itertools import combinations
from typing import Iterable, Set, Tuple

import numpy as np

_PRIME = 2147483647  # 2^31 - 1


def _hash_kmer(kmer: str) -> int:
    return zlib.crc32(kmer.encode("ascii")) & 0xFFFFFFFF


def minhash_sketch(kmer_set: Set[str], num_hashes: int, seed: int) -> np.ndarray:
    """Return an int64 array of `num_hashes` MinHash values.

    An empty k-mer set yields all-prime values (max sentinel).
    """
    rng = np.random.RandomState(seed)
    a = rng.randint(1, _PRIME, size=num_hashes, dtype=np.int64)
    b = rng.randint(0, _PRIME, size=num_hashes, dtype=np.int64)
    out = np.full(num_hashes, _PRIME, dtype=np.int64)
    for km in kmer_set:
        x = _hash_kmer(km) % _PRIME
        h = (a * np.int64(x) + b) % _PRIME
        np.minimum(out, h, out=out)
    return out


def lsh_candidates(
    sketches: np.ndarray, num_hashes: int, rows: int = 4
) -> Set[Tuple[int, int]]:
    """Return candidate index pairs sharing at least one LSH band."""
    n = sketches.shape[0]
    bands = max(1, num_hashes // rows)
    buckets: dict = defaultdict(list)
    for bi in range(bands):
        s, e = bi * rows, bi * rows + rows
        band = sketches[:, s:e]
        for i in range(n):
            key = (bi, *band[i].tolist())
            buckets[key].append(i)
    pairs: Set[Tuple[int, int]] = set()
    for idxs in buckets.values():
        if len(idxs) > 1:
            idxs.sort()
            for i, j in combinations(idxs, 2):
                pairs.add((i, j))
    return pairs


def estimate_jaccard(sketch_a: np.ndarray, sketch_b: np.ndarray) -> float:
    return float(np.mean(sketch_a == sketch_b))
