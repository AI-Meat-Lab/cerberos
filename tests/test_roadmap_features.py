"""Regression tests for roadmap additions."""

from __future__ import annotations

import json
from itertools import chain

import pytest

from cerberos_peptide_splitter import (
    boundary_records,
    cross_validation_assignments,
    hierarchical_cluster_views,
    load_cluster_assignments,
)
from cerberos_peptide_splitter.config import RunConfig
from cerberos_peptide_splitter.diagnostics import summarize
from cerberos_peptide_splitter.split import assign_clusters, split_records


def test_strict_clustering_rejects_all_singletons(simple_records):
    config = RunConfig(strict_clustering=True, min_non_singleton_fraction=0.1)
    with pytest.raises(ValueError, match="strict clustering failed"):
        split_records(simple_records, config, verbose=False)
    assert config.last_clustering_stats["accepted_edges"] == 0
    assert config.last_clustering_stats["non_singleton_fraction"] == 0.0


def test_balanced_assignment_keeps_clusters_intact(labeled_records):
    config = RunConfig(balance=True, balance_weight=1.0, seed=7)
    clusters = [[0, 1, 2], [3, 4], [5], [6, 7], [8, 9, 10, 11]]
    assignment = assign_clusters(
        clusters, len(labeled_records), config, labeled_records
    )
    for cluster in clusters:
        assert len({assignment[index] for index in cluster}) == 1
    assert config.last_assignment_stats["method"] == "balanced-greedy"
    assert set(config.last_assignment_stats["observed_sizes"]) == {
        "train",
        "val",
        "test",
    }


def test_balanced_split_emits_balance_and_nearest_neighbor_diagnostics(labeled_records):
    config = RunConfig(balance=True, seed=3)
    splits = split_records(labeled_records, config, verbose=False)
    stats, _ = summarize(splits, config, kmer_k=3)
    assert stats["assignment"]["method"] == "balanced-greedy"
    assert set(stats["feature_balance"]["splits"]) == {"train", "val", "test"}
    assert "test_to_train" in stats["nearest_neighbor_similarity"]
    assert set(stats["cluster_quality"]) == {
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
    }


def test_advanced_cluster_utilities(labeled_records, tmp_path):
    clusters = [[0, 1], [2], [3, 4], [5, 6], [7, 8, 9, 10, 11]]
    folds = cross_validation_assignments(clusters, len(labeled_records), 3, seed=1)
    assert len(folds) == 3
    assert all(
        set(assignment) == set(range(len(labeled_records))) for assignment in folds
    )
    views = hierarchical_cluster_views(labeled_records, RunConfig(), [0.5, 0.8])
    assert set(views) == {"0.5", "0.8"}
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps({str(index): index % 2 for index in range(len(labeled_records))})
    )
    imported = load_cluster_assignments(str(path), len(labeled_records))
    assert sorted(chain.from_iterable(imported)) == list(range(len(labeled_records)))
    assert isinstance(
        boundary_records(
            {
                "train": labeled_records[:6],
                "val": labeled_records[6:9],
                "test": labeled_records[9:],
            },
            3,
        ),
        list,
    )
