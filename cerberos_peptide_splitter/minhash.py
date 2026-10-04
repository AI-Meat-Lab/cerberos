"""Deterministic MinHash sketches and locality-sensitive hashing.

CRC32 is stable across processes and platforms. The affine MinHash family is
practical, not a cryptographic or mathematically perfect random-hash process.
LSH proposes candidate pairs approximately; it cannot guarantee retrieval of
every pair above a target similarity.
"""

from __future__ import annotations

import random
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from typing import Deque, Iterator, Optional, Set, Tuple

import numpy as np

_PRIME = 2_147_483_647


@dataclass
class CandidateStreamStats:
    """Counters describing a bounded LSH candidate stream."""

    band_count: int = 0
    bucket_count: int = 0
    sampled_bucket_count: int = 0
    emitted_pair_count: int = 0
    limit_reached: bool = False


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


def minhash_sketch_backend(
    kmer_set: Set[str], num_hashes: int, seed: int, backend: str = "numpy"
) -> np.ndarray:
    """Compute a sketch with optional CuPy acceleration and NumPy output."""
    if backend != "cupy":
        return minhash_sketch(kmer_set, num_hashes, seed)
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "backend='cupy' requires the optional cupy package; use backend='numpy'"
        ) from exc
    if not kmer_set:
        return np.full(num_hashes, _PRIME, dtype=np.int64)
    rng = np.random.RandomState(seed)
    a = cp.asarray(rng.randint(1, _PRIME, size=num_hashes, dtype=np.int64))
    b = cp.asarray(rng.randint(0, _PRIME, size=num_hashes, dtype=np.int64))
    result = cp.full(num_hashes, _PRIME, dtype=cp.int64)
    for kmer in kmer_set:
        hashed = _hash_kmer(kmer) % _PRIME
        values = (a * np.int64(hashed) + b) % _PRIME
        result = cp.minimum(result, values)
    return cp.asnumpy(result)


def _bucket_pair_iterator(
    members: list[int], max_bucket_size: int, max_neighbors: int, seed: int
) -> Iterator[Tuple[int, int]]:
    """Enumerate small buckets; sample a shuffled ring for dense buckets."""
    if len(members) <= max_bucket_size:
        yield from combinations(sorted(members), 2)
        return
    shuffled = list(members)
    random.Random(seed).shuffle(shuffled)
    neighbors = min(max_neighbors, (len(shuffled) - 1) // 2)
    for offset in range(1, neighbors + 1):
        for index, first in enumerate(shuffled):
            second = shuffled[(index + offset) % len(shuffled)]
            yield (first, second) if first < second else (second, first)


def iter_lsh_candidates(
    sketches: np.ndarray,
    num_hashes: int,
    rows: int = 4,
    *,
    max_pairs: Optional[int] = None,
    max_bucket_size: int = 128,
    max_bucket_neighbors: int = 8,
    seed: int = 42,
    stats: Optional[CandidateStreamStats] = None,
) -> Iterator[Tuple[int, int]]:
    """Yield deterministic candidates with bounded memory and optional work cap.

    Only one band's buckets are retained at a time. Small buckets are fully
    enumerated; dense buckets are sampled using a seeded shuffled-ring
    neighborhood. ``max_pairs`` caps unique yielded pairs. Sampling and any
    cap make candidate recall approximate and should be reported to users.
    """
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
    if max_pairs is not None and (
        isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 1
    ):
        raise ValueError("max_pairs must be a positive integer or None")
    if (
        isinstance(max_bucket_size, bool)
        or not isinstance(max_bucket_size, int)
        or max_bucket_size < 2
    ):
        raise ValueError("max_bucket_size must be an integer of at least 2")
    if (
        isinstance(max_bucket_neighbors, bool)
        or not isinstance(max_bucket_neighbors, int)
        or max_bucket_neighbors < 1
    ):
        raise ValueError("max_bucket_neighbors must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    band_count = (num_hashes + rows - 1) // rows
    stream_stats = stats if stats is not None else CandidateStreamStats()
    stream_stats.band_count = band_count
    seen: Set[int] = set()
    sequence_count = sketches.shape[0]

    for band_index in range(band_count):
        start = band_index * rows
        end = min(start + rows, num_hashes)
        buckets = defaultdict(list)
        for index, values in enumerate(sketches[:, start:end]):
            buckets[values.tobytes()].append(index)
        populated = [members for members in buckets.values() if len(members) > 1]
        stream_stats.bucket_count += len(populated)
        if not populated:
            continue

        band_rng = random.Random((seed + band_index * 1_000_003) & 0xFFFFFFFF)
        band_rng.shuffle(populated)
        iterators: Deque[Iterator[Tuple[int, int]]] = deque()
        for bucket_index, members in enumerate(populated):
            if len(members) > max_bucket_size:
                stream_stats.sampled_bucket_count += 1
            bucket_seed = (seed + band_index * 1_000_003 + bucket_index) & 0xFFFFFFFF
            iterators.append(
                _bucket_pair_iterator(
                    members,
                    max_bucket_size=max_bucket_size,
                    max_neighbors=max_bucket_neighbors,
                    seed=bucket_seed,
                )
            )

        remaining_bands = band_count - band_index
        if max_pairs is None:
            band_limit = None
        else:
            remaining_pairs = max_pairs - stream_stats.emitted_pair_count
            if remaining_pairs <= 0:
                stream_stats.limit_reached = True
                return
            band_limit = (remaining_pairs + remaining_bands - 1) // remaining_bands
        emitted_this_band = 0

        # Round-robin across buckets avoids allowing one dense bucket to starve
        # all other candidate groups when a per-band budget is reached.
        while iterators and (band_limit is None or emitted_this_band < band_limit):
            pair_iterator = iterators.popleft()
            try:
                first, second = next(pair_iterator)
            except StopIteration:
                continue
            iterators.append(pair_iterator)
            if first > second:
                first, second = second, first
            key = first * sequence_count + second
            if key in seen:
                continue
            seen.add(key)
            stream_stats.emitted_pair_count += 1
            emitted_this_band += 1
            yield first, second
            if max_pairs is not None and stream_stats.emitted_pair_count >= max_pairs:
                stream_stats.limit_reached = True
                return


def lsh_candidates(
    sketches: np.ndarray, num_hashes: int, rows: int = 4
) -> Set[Tuple[int, int]]:
    """Materialize all candidates; intended for small inputs and tests only.

    Large-data callers should use :func:`iter_lsh_candidates` with a pair cap;
    this compatibility helper necessarily allocates a set of all candidate
    pairs and can therefore require quadratic memory for dense buckets.
    """
    return set(iter_lsh_candidates(sketches, num_hashes, rows=rows))


def estimate_jaccard(sketch_a: np.ndarray, sketch_b: np.ndarray) -> float:
    """Estimate Jaccard similarity as the fraction of equal signature rows."""
    if sketch_a.shape != sketch_b.shape or sketch_a.size == 0:
        raise ValueError("sketches must have the same non-zero shape")
    return float(np.mean(sketch_a == sketch_b))
