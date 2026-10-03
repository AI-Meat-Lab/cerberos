"""Union-Find clustering over approximate k-mer similarity candidates."""

from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import List, Sequence, Tuple

import numpy as np

from .alphabet import apply_reduced_alphabet
from .config import RunConfig
from .kmers import kmers
from .minhash import (
    CandidateStreamStats,
    estimate_jaccard,
    iter_lsh_candidates,
    minhash_sketch,
)
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
    config.last_clustering_stats = {}
    if config.no_clustering or count < 2:
        return [[index] for index in range(count)]
    transformed = [
        apply_reduced_alphabet(record[2], config.reduced_alphabet) for record in records
    ]

    # Identical sequences with usable k-mers are certainly identical under
    # every configured verification metric; merge them before sketching.
    first_by_sequence: dict[str, int] = {}
    representatives = []
    duplicate_count = 0
    for index, sequence in enumerate(transformed):
        if not any(len(sequence) >= k for k in config.kmer_sizes):
            continue
        raw_sequence = records[index][2]
        first = first_by_sequence.get(raw_sequence)
        if first is None:
            first_by_sequence[raw_sequence] = index
            representatives.append(index)
        else:
            union_find.union(first, index)
            duplicate_count += 1

    verified = 0
    accepted = 0
    total_candidates = 0
    sampled_buckets = 0
    k_sizes = config.kmer_sizes
    rows = 4
    for k_position, k in enumerate(k_sizes):
        active = [index for index in representatives if len(transformed[index]) >= k]
        if len(active) < 2:
            continue
        if verbose:
            print(
                f"[cerberos]   sketching at k={k} (alphabet={config.reduced_alphabet})",
                file=__import__("sys").stderr,
            )

        # Fill one compact signature matrix directly. Avoid retaining every
        # sequence's Python k-mer set or a list of temporary signature arrays.
        sketches = np.empty((len(active), config.num_hashes), dtype=np.int64)
        sketch_seed = (config.seed + k) % (2**32)
        for local_index, record_index in enumerate(active):
            sketches[local_index] = minhash_sketch(
                kmers(transformed[record_index], k), config.num_hashes, sketch_seed
            )

        remaining_budget = config.max_candidate_pairs - total_candidates
        remaining_k_values = len(k_sizes) - k_position
        if remaining_budget <= 0:
            break
        k_budget = ceil(remaining_budget / remaining_k_values)
        stream_stats = CandidateStreamStats()
        for local_i, local_j in iter_lsh_candidates(
            sketches,
            config.num_hashes,
            rows=rows,
            max_pairs=k_budget,
            max_bucket_size=128,
            max_bucket_neighbors=8,
            seed=sketch_seed,
            stats=stream_stats,
        ):
            total_candidates += 1
            index_a, index_b = active[local_i], active[local_j]
            estimate = estimate_jaccard(sketches[local_i], sketches[local_j])
            if config.verify_with == "none":
                if estimate >= config.similarity_threshold and union_find.find(
                    index_a
                ) != union_find.find(index_b):
                    union_find.union(index_a, index_b)
                    accepted += 1
                continue
            if estimate < config.prefilter_threshold:
                continue
            if union_find.find(index_a) == union_find.find(index_b):
                continue
            score = verify_pair(
                raw_a=records[index_a][2],
                raw_b=records[index_b][2],
                transformed_a=transformed[index_a],
                transformed_b=transformed[index_b],
                kmer_sizes=k_sizes,
                metric=config.verify_with,
                threshold=config.similarity_threshold,
            )
            verified += 1
            if score >= config.similarity_threshold:
                union_find.union(index_a, index_b)
                accepted += 1
        sampled_buckets += stream_stats.sampled_bucket_count
        if verbose:
            print(
                f"[cerberos]   k={k}: streamed {stream_stats.emitted_pair_count:,} "
                f"candidates; sampled {stream_stats.sampled_bucket_count:,} dense buckets",
                file=__import__("sys").stderr,
            )

    limit_reached = total_candidates >= config.max_candidate_pairs
    config.last_clustering_stats = {
        "input_sequences": count,
        "unique_sequences_with_kmers": len(representatives),
        "duplicate_sequences_collapsed": duplicate_count,
        "candidate_pairs_considered": total_candidates,
        "candidate_pairs_verified": verified,
        "accepted_candidate_edges": accepted,
        "oversized_lsh_buckets_sampled": sampled_buckets,
        "candidate_pair_limit": config.max_candidate_pairs,
        "candidate_limit_reached": limit_reached,
        "lsh_rows_per_band": rows,
        "sampled_neighbors_per_dense_bucket": 8,
    }
    if verbose and (limit_reached or sampled_buckets):
        print(
            "[cerberos] WARNING: candidate search was capped/sampled; some "
            "similar pairs may not have been examined. See stats.json "
            "candidate_generation for details.",
            file=__import__("sys").stderr,
        )
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
