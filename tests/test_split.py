"""Tests for cerberos.split."""

from __future__ import annotations

from collections import Counter

from cerberos_peptide_splitter.split import (
    _target_sizes,
    assign_clusters,
    split_records,
    stratified_random_split,
)

# ─────────────────────── _target_sizes ───────────────────────


def test_target_sizes_sum_to_n():
    from cerberos_peptide_splitter.config import RunConfig

    cfg = RunConfig()
    for n in (10, 100, 1001, 9999):
        t, v, te = _target_sizes(n, cfg)
        assert t + v + te == n


def test_target_sizes_roughly_correct():
    from cerberos_peptide_splitter.config import RunConfig

    cfg = RunConfig(train_pct=0.7, val_pct=0.15, test_pct=0.15)
    t, v, te = _target_sizes(1000, cfg)
    assert 690 <= t <= 710
    assert 140 <= v <= 160
    assert 140 <= te <= 160


# ─────────────────────── stratified ───────────────────────


def test_stratified_preserves_label_ratio(labeled_records, config_factory):
    cfg = config_factory(train_pct=0.6, val_pct=0.2, test_pct=0.2)
    assign = stratified_random_split(labeled_records, cfg)
    counts = Counter(assign.values())
    assert sum(counts.values()) == len(labeled_records)
    for split in ("train", "val", "test"):
        labels = [labeled_records[i][1] for i, s in assign.items() if s == split]
        if labels:
            c0, c1 = labels.count(0), labels.count(1)
            assert abs(c0 - c1) <= 1


def test_stratified_unlabeled(simple_records, config_factory):
    cfg = config_factory(train_pct=0.5, val_pct=0.25, test_pct=0.25)
    assign = stratified_random_split(simple_records, cfg)
    counts = Counter(assign.values())
    assert counts["train"] == 3
    assert counts["val"] + counts["test"] == 3


def test_stratified_covers_all(labeled_records, config_factory):
    cfg = config_factory()
    assign = stratified_random_split(labeled_records, cfg)
    assert set(assign.keys()) == set(range(len(labeled_records)))


def test_stratified_deterministic(labeled_records, config_factory):
    cfg = config_factory()
    a1 = stratified_random_split(labeled_records, cfg)
    a2 = stratified_random_split(labeled_records, cfg)
    assert a1 == a2


# ─────────────────────── assign_clusters ───────────────────────


def test_assign_clusters_covers_all(config_factory):
    cfg = config_factory()
    clusters = [[0, 1], [2], [3, 4, 5]]
    assign = assign_clusters(clusters, n=6, config=cfg)
    assert set(assign.keys()) == {0, 1, 2, 3, 4, 5}


def test_assign_clusters_keeps_groups_together(config_factory):
    cfg = config_factory()
    clusters = [[0, 1, 2, 3], [4, 5], [6]]
    assign = assign_clusters(clusters, n=7, config=cfg)
    for cl in clusters:
        assert len({assign[i] for i in cl}) == 1


def test_assign_clusters_roughly_matches_targets(config_factory):
    cfg = config_factory()
    clusters = [[i] for i in range(100)]
    assign = assign_clusters(clusters, n=100, config=cfg)
    c = Counter(assign.values())
    assert 75 <= c["train"] <= 85
    assert 5 <= c["val"] <= 15
    assert 5 <= c["test"] <= 15


def test_assign_clusters_deterministic(config_factory):
    cfg = config_factory()
    clusters = [[0, 1], [2, 3], [4], [5], [6]]
    a1 = assign_clusters(clusters, n=7, config=cfg)
    a2 = assign_clusters(clusters, n=7, config=cfg)
    assert a1 == a2


# ─────────────────────── split_records ───────────────────────


def test_split_records_no_clustering(labeled_records, config_factory):
    cfg = config_factory(no_clustering=True)
    splits = split_records(labeled_records, cfg, verbose=False)
    assert set(splits.keys()) == {"train", "val", "test"}
    assert sum(len(v) for v in splits.values()) == len(labeled_records)


def test_split_records_default_minhash(labeled_records, config_factory):
    cfg = config_factory()
    splits = split_records(labeled_records, cfg, verbose=False)
    total = sum(len(v) for v in splits.values())
    assert total == len(labeled_records)


def test_split_records_path_a(labeled_records, config_factory):
    cfg = config_factory(reduced_alphabet="groups5")
    splits = split_records(labeled_records, cfg, verbose=False)
    total = sum(len(v) for v in splits.values())
    assert total == len(labeled_records)


def test_split_records_path_b(labeled_records, config_factory):
    cfg = config_factory(
        verify_with="levenshtein", similarity_threshold=0.85, prefilter_threshold=0.4
    )
    splits = split_records(labeled_records, cfg, verbose=False)
    total = sum(len(v) for v in splits.values())
    assert total == len(labeled_records)


def test_split_records_path_ab(labeled_records, config_factory):
    cfg = config_factory(
        reduced_alphabet="groups5",
        verify_with="levenshtein",
        similarity_threshold=0.85,
        prefilter_threshold=0.3,
    )
    splits = split_records(labeled_records, cfg, verbose=False)
    total = sum(len(v) for v in splits.values())
    assert total == len(labeled_records)


def test_split_records_keeps_homolog_family_together(
    homologous_records, config_factory
):
    cfg = config_factory(similarity_threshold=0.7, kmer_sizes=[3])
    splits = split_records(homologous_records, cfg, verbose=False)
    family = {0, 1, 2, 3}
    for _name, recs in splits.items():
        idx = {i for i, r in enumerate(homologous_records) if r in recs}
        if family & idx:
            assert family.issubset(idx)


def test_split_records_deterministic(labeled_records, config_factory):
    cfg = config_factory()
    s1 = split_records(labeled_records, cfg, verbose=False)
    s2 = split_records(labeled_records, cfg, verbose=False)
    for k in s1:
        assert s1[k] == s2[k]
