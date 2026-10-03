"""Tests for cerberos.cluster — the Path A + Path B pipeline."""

from __future__ import annotations

from cerberos_peptide_splitter.cluster import UnionFind, compute_homology_clusters

# ─────────────────────── UnionFind ───────────────────────


def test_uf_basic():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(2, 3)
    assert uf.find(0) == uf.find(1)
    assert uf.find(2) == uf.find(3)
    assert uf.find(0) != uf.find(2)


def test_uf_transitive():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)


def test_uf_self():
    uf = UnionFind(3)
    assert uf.find(1) == 1


def test_uf_path_compression():
    uf = UnionFind(10)
    for i in range(9):
        uf.union(i, i + 1)
    root = uf.find(0)
    assert all(uf.find(i) == root for i in range(10))


# ─────────────────────── trivial cases ───────────────────────


def test_empty_records_returns_empty():
    clusters = compute_homology_clusters([], config=make_cfg(), verbose=False)
    assert clusters == []


def test_single_record_is_one_cluster():
    clusters = compute_homology_clusters(
        [("a", None, "ACDE")], config=make_cfg(), verbose=False
    )
    assert clusters == [[0]]


def test_no_clustering_flag_returns_singletons(labeled_records):
    cfg = make_cfg(no_clustering=True)
    clusters = compute_homology_clusters(labeled_records, cfg, verbose=False)
    assert sorted(len(c) for c in clusters) == [1] * len(labeled_records)


# ─────────────────────── baseline behaviour ───────────────────────


def test_identical_duplicates_merge(homologous_records):
    cfg = make_cfg(kmer_sizes=[3], similarity_threshold=0.9, verify_with="none")
    clusters = compute_homology_clusters(homologous_records, cfg, verbose=False)
    flat = sorted(i for c in clusters for i in c)
    assert flat == list(range(len(homologous_records)))
    # the 4 exact duplicates are indices 0..3 — must be in one cluster
    for c in clusters:
        if set(c) & {0, 1, 2, 3}:
            assert {0, 1, 2, 3}.issubset(c)


def test_unrelated_sequences_stay_separate(simple_records):
    cfg = make_cfg(kmer_sizes=[3], similarity_threshold=0.7, verify_with="none")
    clusters = compute_homology_clusters(simple_records, cfg, verbose=False)
    assert len(clusters) >= 5


def test_threshold_zero_clusters_everything():
    recs = [(f"p{i}", None, "ACDEFGHIKL") for i in range(3)]
    cfg = make_cfg(kmer_sizes=[3], similarity_threshold=0.0, verify_with="none")
    clusters = compute_homology_clusters(recs, cfg, verbose=False)
    assert len(clusters) == 1


# ─────────────────────── Path A ───────────────────────


def test_path_a_does_not_change_identical_pair():
    recs = [("a", None, "ACDEFGHIKL"), ("b", None, "ACDEFGHIKL")]
    cfg = make_cfg(
        reduced_alphabet="groups5",
        kmer_sizes=[3],
        similarity_threshold=0.9,
        verify_with="none",
    )
    clusters = compute_homology_clusters(recs, cfg, verbose=False)
    assert len(clusters) == 1


def test_path_a_merges_conservative_variants(conservative_variants):
    # without Path A they may or may not cluster at 0.8
    cfg_none = make_cfg(
        reduced_alphabet="none",
        kmer_sizes=[3],
        similarity_threshold=0.8,
        verify_with="none",
    )
    cfg_a = make_cfg(
        reduced_alphabet="groups5",
        kmer_sizes=[3],
        similarity_threshold=0.8,
        verify_with="none",
    )

    c_none = compute_homology_clusters(conservative_variants, cfg_none, verbose=False)
    c_a = compute_homology_clusters(conservative_variants, cfg_a, verbose=False)

    # Path A should produce a merged cluster or at least not more splits
    assert len(c_a) <= len(c_none)


def test_path_a_on_groups7():
    recs = [("a", None, "KWKLFKKIEK"), ("b", None, "RWRLFRRIER")]
    cfg = make_cfg(
        reduced_alphabet="groups7",
        kmer_sizes=[3],
        similarity_threshold=0.9,
        verify_with="none",
    )
    clusters = compute_homology_clusters(recs, cfg, verbose=False)
    assert len(clusters) == 1


# ─────────────────────── Path B ───────────────────────


def test_path_b_levenshtein_recovers_single_sub(single_sub_pair):
    """Two 21-mers differing by 1 residue: identity ≈ 0.95."""
    cfg = make_cfg(
        kmer_sizes=[3],
        similarity_threshold=0.90,
        prefilter_threshold=0.40,
        verify_with="levenshtein",
    )
    clusters = compute_homology_clusters(single_sub_pair, cfg, verbose=False)
    assert len(clusters) == 1


def test_path_b_kmer_exact_matches_single_stage_on_identical():
    recs = [(f"p{i}", None, "ACDEFGHIKL") for i in range(3)]
    cfg = make_cfg(
        kmer_sizes=[3],
        similarity_threshold=0.9,
        prefilter_threshold=0.4,
        verify_with="kmer-exact",
    )
    clusters = compute_homology_clusters(recs, cfg, verbose=False)
    assert len(clusters) == 1
    assert sorted(len(c) for c in clusters) == [3]


def test_path_b_verification_branching():
    """Verify that prefilter rejects unrelated pairs early."""
    recs = [("a", None, "AAAAAAAAAA"), ("b", None, "YYYYYYYYYY")]
    cfg = make_cfg(
        kmer_sizes=[3],
        similarity_threshold=0.5,
        prefilter_threshold=0.4,
        verify_with="levenshtein",
    )
    clusters = compute_homology_clusters(recs, cfg, verbose=False)
    assert len(clusters) == 2


# ─────────────────────── combined A + B ───────────────────────


def test_path_a_plus_b_on_conservative_variants(conservative_variants):
    cfg = make_cfg(
        reduced_alphabet="groups5",
        kmer_sizes=[3],
        similarity_threshold=0.75,
        prefilter_threshold=0.30,
        verify_with="levenshtein",
    )
    clusters = compute_homology_clusters(conservative_variants, cfg, verbose=False)
    # Their global unit-cost edit similarity is 16/21 (~0.762), so a 0.75
    # threshold groups them while a threshold of 0.85 correctly does not.
    assert len(clusters) == 1


def test_full_coverage_after_all_paths(homologous_records):
    cfg = make_cfg(
        reduced_alphabet="groups5",
        kmer_sizes=[2, 3],
        similarity_threshold=0.8,
        prefilter_threshold=0.3,
        verify_with="levenshtein",
    )
    clusters = compute_homology_clusters(homologous_records, cfg, verbose=False)
    flat = sorted(i for c in clusters for i in c)
    assert flat == list(range(len(homologous_records)))


# ─────────────────────── helper ───────────────────────


def make_cfg(**kw):
    """Local shorthand to avoid depending on the conftest fixture."""
    from cerberos_peptide_splitter.config import RunConfig

    base = dict(
        train_pct=0.8,
        val_pct=0.1,
        test_pct=0.1,
        kmer_sizes=[3],
        num_hashes=64,
        similarity_threshold=0.7,
        reduced_alphabet="none",
        verify_with="none",
        prefilter_threshold=0.3,
        seed=42,
        no_clustering=False,
    )
    base.update(kw)
    return RunConfig(**base)


def test_short_sequences_without_kmers_do_not_cluster_as_identical():
    records = [("a", None, "A"), ("b", None, "A")]
    cfg = make_cfg(kmer_sizes=[3], similarity_threshold=0.0)
    clusters = compute_homology_clusters(records, cfg, verbose=False)
    assert len(clusters) == 2


def test_exact_duplicates_are_collapsed_before_candidate_generation():
    records = [(f"p{i}", None, "ACDEFGHIKLMNPQ") for i in range(400)]
    cfg = make_cfg(max_candidate_pairs=10)
    clusters = compute_homology_clusters(records, cfg, verbose=False)
    assert clusters == [list(range(400))]
    assert cfg.last_clustering_stats["duplicate_sequences_collapsed"] == 399
    assert cfg.last_clustering_stats["candidate_pairs_considered"] == 0


def test_candidate_pair_budget_is_reported_when_reached(monkeypatch):
    from cerberos_peptide_splitter import cluster as cluster_module

    def one_candidate(*args, **kwargs):
        yield 0, 1

    monkeypatch.setattr(cluster_module, "iter_lsh_candidates", one_candidate)
    records = [("a", None, "ACDEFGHIKLMNPQ"), ("b", None, "ACDEFGIKLMNPQ")]
    cfg = make_cfg(max_candidate_pairs=1)
    compute_homology_clusters(records, cfg, verbose=False)
    assert cfg.last_clustering_stats["candidate_pairs_considered"] <= 1
    assert cfg.last_clustering_stats["candidate_limit_reached"]
