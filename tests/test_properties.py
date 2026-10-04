"""Property-style invariants for roadmap reproducibility and split integrity."""

from __future__ import annotations

from cerberos_peptide_splitter.cluster import compute_homology_clusters
from cerberos_peptide_splitter.config import RunConfig
from cerberos_peptide_splitter.split import split_records


def test_identical_sequences_always_share_a_cluster():
    for size in range(2, 8):
        records = [(str(index), None, "ACDEFGHIKLMN") for index in range(size)]
        config = RunConfig(exact_mode=True, kmer_sizes=[2, 3], seed=size)
        clusters = compute_homology_clusters(records, config, verbose=False)
        assert any(len(cluster) == size for cluster in clusters)


def test_split_invariants_hold_across_seeds(labeled_records):
    for seed in range(6):
        config = RunConfig(seed=seed, balance=True, exact_mode=True, kmer_sizes=[2])
        splits = split_records(labeled_records, config, verbose=False)
        identifiers = [record[0] for values in splits.values() for record in values]
        assert len(identifiers) == len(set(identifiers)) == len(labeled_records)
        assert sum(len(values) for values in splits.values()) == len(labeled_records)


def test_same_seed_is_reproducible(labeled_records):
    left = split_records(labeled_records, RunConfig(seed=17), verbose=False)
    right = split_records(labeled_records, RunConfig(seed=17), verbose=False)
    assert left == right
