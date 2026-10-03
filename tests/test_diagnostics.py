"""Tests for cerberos.diagnostics."""

from __future__ import annotations

import json

import numpy as np

from cerberos_peptide_splitter.diagnostics import (
    _mean_pairwise_jaccard,
    _sample,
    build_report,
    feature_ks,
    pairwise_similarity,
    summarize,
)

# ─────────────────────── sampling ───────────────────────


def test_sample_returns_all_below_cap():
    rng = np.random.RandomState(0)
    assert _sample(["AAA", "BBB"], 3, rng, 10) == ["AAA", "BBB"]


def test_sample_caps_length():
    rng = np.random.RandomState(0)
    out = _sample([f"s{i}" for i in range(100)], 3, rng, 10)
    assert len(out) == 10


# ─────────────────────── pairwise ───────────────────────


def test_mean_pairwise_same():
    seqs = ["ACDEFGHIKL", "ACDEFGHIKM"]
    m, _s, n = _mean_pairwise_jaccard(seqs, seqs, k=3, same=True)
    assert n == 1
    assert 0.0 <= m <= 1.0


def test_mean_pairwise_different():
    a = ["ACDEFGHIKL"]
    b = ["YYYYYYYYYY"]
    m, _s, n = _mean_pairwise_jaccard(a, b, k=3, same=False)
    assert n == 1
    assert m == 0.0


def test_pairwise_similarity_structure(labeled_records):
    splits = {
        "train": labeled_records[:6],
        "val": labeled_records[6:9],
        "test": labeled_records[9:],
    }
    out = pairwise_similarity(splits, k=3, max_per_split=10, seed=0)
    for key in (
        "train-train",
        "train-val",
        "train-test",
        "val-val",
        "val-test",
        "test-test",
    ):
        assert key in out
    assert all(0.0 <= v["mean_jaccard"] <= 1.0 for v in out.values())


# ─────────────────────── feature_ks ───────────────────────


def test_feature_ks_shapes(labeled_records):
    from cerberos_peptide_splitter.features import build_feature_matrix

    splits = {
        "train": labeled_records[:6],
        "val": labeled_records[6:9],
        "test": labeled_records[9:],
    }
    _, names = build_feature_matrix(labeled_records)
    ks, wd, pairs = feature_ks(splits, names)
    assert ks.shape == (len(names), len(pairs))
    assert wd.shape == ks.shape
    assert (ks >= 0).all() and (ks <= 1).all()
    assert (wd >= 0).all()


# ─────────────────────── summarize ───────────────────────


def test_summarize_structure(labeled_records, config_factory):
    cfg = config_factory()
    splits = {
        "train": labeled_records[:6],
        "val": labeled_records[6:9],
        "test": labeled_records[9:],
    }
    stats, feats = summarize(splits, cfg, kmer_k=3, sim_plot_cap=20)
    assert stats["sizes"] == {"train": 6, "val": 3, "test": 3}
    assert "clustering_mode" in stats
    assert "similarity" in stats
    assert "ks" in stats
    assert "feature_names" in stats
    assert feats == stats["feature_names"]
    json.dumps(stats)


def test_summarize_detects_homology_leakage():
    recs = [(f"p{i}", 0, "ACDEFGHIKL") for i in range(20)]
    splits = {
        "train": recs[:10],
        "val": recs[:5],
        "test": recs[:10],
    }
    from cerberos_peptide_splitter.config import RunConfig

    stats, _ = summarize(splits, RunConfig(), kmer_k=3, sim_plot_cap=20)
    assert stats["homology_warnings"]


def test_summarize_reports_clustering_mode(labeled_records, config_factory):
    cfg = config_factory(
        reduced_alphabet="groups5",
        verify_with="levenshtein",
        prefilter_threshold=0.3,
        similarity_threshold=0.85,
    )
    splits = {
        "train": labeled_records[:6],
        "val": labeled_records[6:9],
        "test": labeled_records[9:],
    }
    stats, _ = summarize(splits, cfg, kmer_k=3, sim_plot_cap=20)
    cm = stats["clustering_mode"]
    assert cm["reduced_alphabet"] == "groups5"
    assert cm["verify_with"] == "levenshtein"
    assert cm["prefilter_threshold"] == 0.3


# ─────────────────────── report ───────────────────────


def test_build_report_sections(labeled_records, config_factory):
    cfg = config_factory()
    splits = {
        "train": labeled_records[:6],
        "val": labeled_records[6:9],
        "test": labeled_records[9:],
    }
    stats, _ = summarize(splits, cfg, kmer_k=3, sim_plot_cap=20)
    report = build_report(stats, mode="split")
    for token in (
        "Clustering configuration:",
        "Split sizes:",
        "Length stats:",
        "Pairwise k-mer Jaccard",
    ):
        assert token in report


def test_build_report_audit_mode(labeled_records, config_factory):
    cfg = config_factory()
    splits = {
        "train": labeled_records[:6],
        "val": labeled_records[6:9],
        "test": labeled_records[9:],
    }
    stats, _ = summarize(splits, cfg, kmer_k=3, sim_plot_cap=20)
    report = build_report(stats, mode="audit")
    assert "audit mode" in report


def test_report_warns_when_candidate_budget_was_reached(
    labeled_records, config_factory
):
    cfg = config_factory()
    splits = {
        "train": labeled_records[:6],
        "val": labeled_records[6:9],
        "test": labeled_records[9:],
    }
    stats, _ = summarize(splits, cfg, kmer_k=3, sim_plot_cap=20)
    stats["candidate_generation"] = {
        "candidate_pairs_considered": 250_000,
        "candidate_pairs_verified": 30_000,
        "oversized_lsh_buckets_sampled": 0,
        "candidate_limit_reached": True,
    }
    report = build_report(stats, mode="split")
    assert "candidate-pair limit reached" in report
    assert "No heuristic warnings" not in report


def test_empty_splits_produce_strict_json_without_nan(config_factory):
    stats, _ = summarize(
        {"train": [], "val": [], "test": []},
        config_factory(),
        kmer_k=3,
    )
    json.dumps(stats, allow_nan=False)
    assert stats["similarity"]["train-train"]["mean_jaccard"] is None


def test_empirical_distance_implementations_are_exact():
    from cerberos_peptide_splitter.diagnostics import (
        _ks_statistic,
        _wasserstein_distance,
    )

    left = np.asarray([0.0, 0.0])
    right = np.asarray([1.0, 1.0])
    assert _ks_statistic(left, right) == 1.0
    assert _wasserstein_distance(left, right) == 1.0
