"""Deterministic leakage-oriented clustering over k-mer similarity candidates."""

from __future__ import annotations

import itertools
import random
from collections import Counter, defaultdict
from typing import Iterator, List, Sequence, Tuple

import numpy as np

from .alphabet import apply_reduced_alphabet
from .config import RunConfig
from .kmers import exact_kmer_jaccard, kmers
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


def _otsu_threshold(values: Sequence[float], fallback: float) -> float:
    """Select a data-driven threshold using Otsu's between-class variance."""
    if len(values) < 8 or max(values) - min(values) < 1e-12:
        return fallback
    hist, edges = np.histogram(values, bins=64, range=(0.0, 1.0))
    probabilities = hist.astype(float) / max(1, hist.sum())
    centers = (edges[:-1] + edges[1:]) / 2
    cumulative = np.cumsum(probabilities)
    means = np.cumsum(probabilities * centers)
    total_mean = means[-1]
    denominator = cumulative * (1.0 - cumulative)
    variance = np.where(
        denominator > 0,
        (total_mean * cumulative - means) ** 2 / denominator,
        -1.0,
    )
    return float(np.clip(centers[int(np.argmax(variance))], 0.0, 1.0))


def _label_propagation(
    n: int, edges: set[tuple[int, int]], seed: int
) -> List[List[int]]:
    """Deterministic lightweight community detection for sparse accepted graphs."""
    neighbors = defaultdict(set)
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    labels = list(range(n))
    order = list(range(n))
    random.Random(seed).shuffle(order)
    for _ in range(12):
        changed = False
        for node in order:
            if not neighbors[node]:
                continue
            counts = Counter(labels[other] for other in neighbors[node])
            best = min(counts, key=lambda label: (-counts[label], label))
            if best != labels[node]:
                labels[node] = best
                changed = True
        if not changed:
            break
    groups = defaultdict(list)
    for node, label in enumerate(labels):
        groups[label].append(node)
    return list(groups.values())


def _benchmark_recall(records, transformed, seen_pairs, threshold, k_sizes, seed):
    """Estimate retrieval recall on a deterministic small exact pair benchmark."""
    count = len(records)
    pairs = list(itertools.combinations(range(count), 2))
    if len(pairs) > 256:
        pairs = random.Random(seed).sample(pairs, 256)
    eligible = 0
    retrieved = 0
    for left, right in pairs:
        score = max(
            exact_kmer_jaccard(transformed[left], transformed[right], k)
            for k in k_sizes
        )
        if score >= threshold:
            eligible += 1
            if (left, right) in seen_pairs:
                retrieved += 1
    return None if eligible == 0 else retrieved / eligible


def compute_homology_clusters(
    records: Sequence[Tuple[str, object, str]],
    config: RunConfig,
    verbose: bool = True,
) -> List[List[int]]:
    """Cluster record indices using deterministic multi-LSH candidate generation."""
    count = len(records)
    union_find = UnionFind(count)
    config.last_clustering_stats = {}
    config.last_cluster_assignments = []
    if config.no_clustering or count < 2:
        config.last_clustering_stats = {
            "input_sequences": count,
            "accepted_edges": 0,
            "accepted_candidate_edges": 0,
            "cluster_count": count,
            "singleton_count": count,
            "non_singleton_fraction": 0.0,
            "cluster_size_distribution": {"1": count} if count else {},
        }
        return [[index] for index in range(count)]

    transformed = [
        apply_reduced_alphabet(record[2], config.reduced_alphabet) for record in records
    ]
    short_count = sum(
        len(sequence) < min(config.kmer_sizes) for sequence in transformed
    )
    if (
        short_count
        and config.short_peptide_mode == "auto"
        and config.reduced_alphabet == "none"
    ):
        transformed = [
            apply_reduced_alphabet(record[2], "groups5") for record in records
        ]
    if verbose and short_count and config.short_peptide_mode == "warn":
        print(
            f"[cerberos] WARNING: {short_count} sequences are shorter than the smallest k-mer; "
            "they provide no similarity evidence.",
            file=__import__("sys").stderr,
        )

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
    candidate_pairs_seen: set[tuple[int, int]] = set()
    accepted_edges: set[tuple[int, int]] = set()
    edge_candidates: list[tuple[int, int, float]] = []
    score_values = []
    rejected_values = []
    threshold = config.similarity_threshold
    rows_used = []
    for k_position, k in enumerate(config.kmer_sizes):
        active = [index for index in representatives if len(transformed[index]) >= k]
        if len(active) < 2:
            continue
        use_exact = config.exact_mode and len(active) <= config.exact_mode_max_sequences
        if verbose:
            print(
                f"[cerberos]   sketching at k={k} (alphabet={config.reduced_alphabet})",
                file=__import__("sys").stderr,
            )
        sketches = None
        if not use_exact:
            sketches = np.empty((len(active), config.num_hashes), dtype=np.int64)
            sketch_seed = (config.seed + k) % (2**32)
            for local_index, record_index in enumerate(active):
                sketches[local_index] = minhash_sketch(
                    kmers(transformed[record_index], k), config.num_hashes, sketch_seed
                )
        remaining_budget = config.max_candidate_pairs - total_candidates
        if remaining_budget <= 0:
            break
        k_budget = max(
            1, remaining_budget // max(1, len(config.kmer_sizes) - k_position)
        )
        if use_exact:
            rows_for_k = [0]
        else:
            rows_for_k = config.lsh_rows
        per_config_budget = max(1, k_budget // max(1, len(rows_for_k)))
        for row_position, rows in enumerate(rows_for_k):
            rows_used.append(rows)
            stream_stats = CandidateStreamStats()
            pairs: Iterator[tuple[int, int]]
            if use_exact:
                pairs = itertools.combinations(range(len(active)), 2)
            else:
                pairs = iter_lsh_candidates(
                    sketches,
                    config.num_hashes,
                    rows=rows,
                    max_pairs=per_config_budget,
                    max_bucket_size=128,
                    max_bucket_neighbors=8,
                    seed=(config.seed + k * 1009 + row_position) % (2**32),
                    stats=stream_stats,
                )
            for local_i, local_j in pairs:
                if total_candidates >= config.max_candidate_pairs:
                    break
                index_a, index_b = active[local_i], active[local_j]
                pair = (min(index_a, index_b), max(index_a, index_b))
                candidate_pairs_seen.add(pair)
                total_candidates += 1
                if use_exact:
                    estimate = exact_kmer_jaccard(
                        transformed[index_a], transformed[index_b], k
                    )
                else:
                    estimate = estimate_jaccard(sketches[local_i], sketches[local_j])
                score = estimate
                if config.verify_with != "none":
                    if estimate < config.prefilter_threshold:
                        rejected_values.append(estimate)
                        continue
                    if union_find.find(index_a) == union_find.find(index_b):
                        continue
                    score = verify_pair(
                        raw_a=records[index_a][2],
                        raw_b=records[index_b][2],
                        transformed_a=transformed[index_a],
                        transformed_b=transformed[index_b],
                        kmer_sizes=config.kmer_sizes,
                        metric=config.verify_with,
                        threshold=threshold,
                    )
                    verified += 1
                score_values.append(score)
                edge_candidates.append((index_a, index_b, score))
            sampled_buckets += stream_stats.sampled_bucket_count
            if verbose and not use_exact:
                print(
                    f"[cerberos]   k={k}, rows={rows}: streamed {stream_stats.emitted_pair_count:,} "
                    f"candidates; sampled {stream_stats.sampled_bucket_count:,} dense buckets",
                    file=__import__("sys").stderr,
                )

    if config.adaptive_threshold and score_values:
        threshold = _otsu_threshold(score_values, config.similarity_threshold)
    rejected_values.extend(score for score in score_values if score < threshold)
    for index_a, index_b, score in edge_candidates:
        if score >= threshold and union_find.find(index_a) != union_find.find(index_b):
            union_find.union(index_a, index_b)
            accepted += 1
            accepted_edges.add((min(index_a, index_b), max(index_a, index_b)))
    limit_reached = total_candidates >= config.max_candidate_pairs
    clusters = []
    if config.cluster_method == "label-propagation" and accepted_edges:
        clusters = _label_propagation(count, accepted_edges, config.seed)
    else:
        groups = defaultdict(list)
        for index in range(count):
            groups[union_find.find(index)].append(index)
        clusters = list(groups.values())
    cluster_size_distribution = Counter(str(len(cluster)) for cluster in clusters)
    non_singleton_sequences = sum(
        len(cluster) for cluster in clusters if len(cluster) > 1
    )
    config.last_cluster_assignments = [0] * count
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            config.last_cluster_assignments[index] = cluster_id
    config.last_clustering_stats = {
        "input_sequences": count,
        "unique_sequences_with_kmers": len(representatives),
        "duplicate_sequences_collapsed": duplicate_count,
        "candidate_pairs_considered": total_candidates,
        "candidate_pairs_verified": verified,
        "accepted_candidate_edges": accepted,
        "accepted_edges": accepted + duplicate_count,
        "oversized_lsh_buckets_sampled": sampled_buckets,
        "candidate_pair_limit": config.max_candidate_pairs,
        "candidate_limit_reached": limit_reached,
        "lsh_rows_per_band": sorted(set(rows_used)),
        "lsh_configurations": len(rows_used),
        "exact_mode": bool(
            config.exact_mode
            and len(representatives) <= config.exact_mode_max_sequences
        ),
        "short_sequences": short_count,
        "short_peptide_mode": config.short_peptide_mode,
        "adaptive_threshold": config.adaptive_threshold,
        "effective_similarity_threshold": threshold,
        "candidate_recall_benchmark": _benchmark_recall(
            records,
            transformed,
            candidate_pairs_seen,
            threshold,
            config.kmer_sizes,
            config.seed,
        ),
        "accepted_score_histogram": np.histogram(score_values, bins=20, range=(0, 1))[
            0
        ].tolist(),
        "rejected_score_histogram": np.histogram(
            rejected_values, bins=20, range=(0, 1)
        )[0].tolist(),
        "cluster_method": config.cluster_method,
        "cluster_count": len(clusters),
        "singleton_count": sum(len(cluster) == 1 for cluster in clusters),
        "non_singleton_fraction": non_singleton_sequences / count if count else 0.0,
        "cluster_size_distribution": dict(sorted(cluster_size_distribution.items())),
    }
    if verbose and (limit_reached or sampled_buckets):
        print(
            "[cerberos] WARNING: candidate search was capped/sampled; some similar pairs "
            "may not have been examined. See stats.json candidate_generation for details.",
            file=__import__("sys").stderr,
        )
    return clusters
