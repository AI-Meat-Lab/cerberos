"""Union-Find clustering over approximate k-mer similarity candidates."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .alphabet import apply_reduced_alphabet
from .config import RunConfig
from .kmers import kmers
from .minhash import estimate_jaccard, lsh_candidates, minhash_sketch
from .verify import verify_pair


class UnionFind:
    """Disjoint-set forest with path compression and union by rank."""

    __slots__ = ("parent", "rank")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1


def _generate_candidates(
    transformed: Sequence[str], config: RunConfig, verbose: bool
) -> Dict[Tuple[int, int], float]:
    """Return LSH candidate pairs and their maximum valid-k estimate.

    Empty k-mer sets are omitted: matching empty signatures are not evidence
    of similarity. Two-row bands are used with Path B to improve recall near
    its configured prefilter, but candidate recall remains approximate.
    """
    candidates: Dict[Tuple[int, int], float] = {}
    rows = 2 if config.verify_with != "none" else 4
    for k in config.kmer_sizes:
        if verbose:
            print(
                f"[cerberos]   sketching at k={k} (alphabet={config.reduced_alphabet})",
                file=__import__("sys").stderr,
            )
        sets = [kmers(sequence, k) for sequence in transformed]
        sketches = np.asarray(
            [
                minhash_sketch(kmer_set, config.num_hashes, (config.seed + k) % (2**32))
                for kmer_set in sets
            ],
            dtype=np.int64,
        )
        active = [index for index, kmer_set in enumerate(sets) if kmer_set]
        if len(active) < 2:
            continue
        for local_i, local_j in lsh_candidates(
            sketches[active], config.num_hashes, rows=rows
        ):
            index_a, index_b = active[local_i], active[local_j]
            estimate = estimate_jaccard(sketches[index_a], sketches[index_b])
            key = (index_a, index_b) if index_a < index_b else (index_b, index_a)
            candidates[key] = max(candidates.get(key, -1.0), estimate)
    return candidates


def compute_homology_clusters(
    records: Sequence[Tuple[str, object, str]],
    config: RunConfig,
    verbose: bool = True,
) -> List[List[int]]:
    """Cluster record indices connected by accepted candidate-pair edges.

    MinHash/LSH is approximate; pairs missed by candidate generation cannot be
    recovered by verification. Connected components can include endpoints
    whose direct pairwise score is below the edge threshold.
    """
    count = len(records)
    union_find = UnionFind(count)
    if config.no_clustering or count < 2:
        return [[index] for index in range(count)]
    transformed = [
        apply_reduced_alphabet(record[2], config.reduced_alphabet) for record in records
    ]
    candidates = _generate_candidates(transformed, config, verbose)
    verified = 0
    for (index_a, index_b), estimate in candidates.items():
        if config.verify_with == "none":
            if estimate >= config.similarity_threshold:
                union_find.union(index_a, index_b)
            continue
        if estimate < config.prefilter_threshold:
            continue
        score = verify_pair(
            raw_a=records[index_a][2],
            raw_b=records[index_b][2],
            transformed_a=transformed[index_a],
            transformed_b=transformed[index_b],
            kmer_sizes=config.kmer_sizes,
            metric=config.verify_with,
            threshold=config.similarity_threshold,
        )
        verified += 1
        if score >= config.similarity_threshold:
            union_find.union(index_a, index_b)
    if verbose and config.verify_with != "none":
        print(
            f"[cerberos]   verification: {verified} candidates scored, "
            f"metric='{config.verify_with}'",
            file=__import__("sys").stderr,
        )
    groups = defaultdict(list)
    for index in range(count):
        groups[union_find.find(index)].append(index)
    return list(groups.values())
