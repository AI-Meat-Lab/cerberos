"""Union-Find clustering with Path A + Path B.

This is the file the rest of the package pivots around.  Everything
downstream (splits, diagnostics) treats it as a black box that turns
a list of records into a list of index clusters.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence, Tuple

import numpy as np

from .alphabet import apply_reduced_alphabet
from .config import RunConfig
from .kmers import kmers
from .minhash import estimate_jaccard, lsh_candidates, minhash_sketch
from .verify import verify_pair


# ─────────────────────── Union-Find ───────────────────────

class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ─────────────────────── candidate generation ───────────────────────

def _generate_candidates(
    transformed: Sequence[str], config: RunConfig, verbose: bool
) -> dict:
    """Return {(i, j) -> estimated Jaccard} for every LSH candidate.

    The estimate is the *max* over all requested k values.
    """
    candidates: dict = {}
    for k in config.kmer_sizes:
        if verbose:
            print(f"[cerberos]   sketching at k={k} "
                  f"(alphabet={config.reduced_alphabet})",
                  file=__import__("sys").stderr)
        kmer_sets = [kmers(s, k) for s in transformed]
        sketches = np.array(
            [minhash_sketch(ks, config.num_hashes, config.seed + k)
             for ks in kmer_sets],
            dtype=np.int64,
        )
        for i, j in lsh_candidates(sketches, config.num_hashes, rows=4):
            est = estimate_jaccard(sketches[i], sketches[j])
            key = (i, j) if i < j else (j, i)
            if est > candidates.get(key, -1.0):
                candidates[key] = est
    return candidates


# ─────────────────────── public entry point ───────────────────────

def compute_homology_clusters(
    records: Sequence[Tuple[str, object, str]],
    config: RunConfig,
    verbose: bool = True,
) -> List[List[int]]:
    """Cluster record indices into homology groups.

    Pipeline
    --------
    1. Transform every sequence through the reduced alphabet (Path A).
    2. MinHash-sketch the transformed sequences at every k; LSH-band.
    3. For every candidate pair, verify with Path B if enabled,
       otherwise accept based on the estimated Jaccard alone.
    4. Union accepted pairs; return the connected components.
    """
    n = len(records)
    uf = UnionFind(n)

    if config.no_clustering or n < 2:
        return [[i] for i in range(n)]

    # ── Step 1 — Path A ──
    transformed = [apply_reduced_alphabet(r[2], config.reduced_alphabet)
                   for r in records]

    # ── Step 2 — candidates ──
    candidates = _generate_candidates(transformed, config, verbose)

    # ── Steps 3 & 4 — verify and union ──
    n_prefilter = n_verified = 0
    for (i, j), est in candidates.items():
        if config.verify_with == "none":
            if est >= config.similarity_threshold:
                uf.union(i, j)
            continue

        # Two-stage: MinHash prefilter, then deterministic verify.
        if est < config.prefilter_threshold:
            continue
        n_prefilter += 1

        score = verify_pair(
            raw_a=records[i][2], raw_b=records[j][2],
            transformed_a=transformed[i], transformed_b=transformed[j],
            kmer_sizes=config.kmer_sizes,
            metric=config.verify_with,
            threshold=config.similarity_threshold,
        )
        n_verified += 1
        if score >= config.similarity_threshold:
            uf.union(i, j)

    if verbose and config.verify_with != "none":
        print(f"[cerberos]   verification: {n_prefilter} candidates "
              f"above prefilter, {n_verified} scored, "
              f"metric='{config.verify_with}'",
              file=__import__("sys").stderr)

    groups: dict = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return list(groups.values())
