"""Tests for cerberos.pipeline."""
from __future__ import annotations

import json
import os

import pytest

from cerberos.config import RunConfig
from cerberos.pipeline import _load_check_dir, run_audit, run_split


# ─────────────────────── _load_check_dir ───────────────────────

def test_load_check_dir_finds_standard_names(sample_splits_dir):
    splits = _load_check_dir(sample_splits_dir)
    assert set(splits.keys()) == {"train", "val", "test"}
    assert all(len(v) > 0 for v in splits.values())


def test_load_check_dir_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_check_dir(str(tmp_path / "nope"))


def test_load_check_dir_incomplete(tmp_path):
    d = tmp_path / "partial"
    d.mkdir()
    (d / "train.fasta").write_text(">a\nACDE\n")
    with pytest.raises(FileNotFoundError):
        _load_check_dir(str(d))


def test_load_check_dir_accepts_alt_extensions(tmp_path):
    d = tmp_path / "alt"
    d.mkdir()
    (d / "train.fa").write_text(">a\nACDE\n")
    (d / "valid.fasta").write_text(">b\nFGHI\n")
    (d / "test.faa").write_text(">c\nKLMN\n")
    splits = _load_check_dir(str(d))
    assert set(splits.keys()) == {"train", "val", "test"}


# ─────────────────────── run_split ───────────────────────

def test_run_split_creates_all_outputs(tmp_path, sample_labeled_fasta_path):
    out = tmp_path / "out"
    cfg = RunConfig(seed=1, kmer_sizes=[2, 3], num_hashes=64)
    stats = run_split(sample_labeled_fasta_path, str(out), cfg)

    assert (out / "train.fasta").exists()
    assert (out / "val.fasta").exists()
    assert (out / "test.fasta").exists()
    assert (out / "report.txt").exists()
    assert (out / "stats.json").exists()

    loaded = json.loads((out / "stats.json").read_text())
    assert loaded["sizes"] == stats["sizes"]


def test_run_split_raises_on_empty(tmp_path):
    empty = tmp_path / "empty.fasta"
    empty.write_text("")
    with pytest.raises(ValueError):
        run_split(str(empty), str(tmp_path / "out"), RunConfig())


def test_run_split_path_ab(tmp_path, sample_labeled_fasta_path):
    out = tmp_path / "out"
    cfg = RunConfig(
        reduced_alphabet="groups5",
        verify_with="levenshtein",
        similarity_threshold=0.85,
        prefilter_threshold=0.3,
        kmer_sizes=[2, 3],
        num_hashes=64,
        seed=42,
    )
    stats = run_split(sample_labeled_fasta_path, str(out), cfg)
    assert stats["clustering_mode"]["reduced_alphabet"] == "groups5"
    assert stats["clustering_mode"]["verify_with"] == "levenshtein"


def test_run_split_deterministic(tmp_path, sample_labeled_fasta_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    cfg = RunConfig(seed=42, kmer_sizes=[2, 3], num_hashes=64)
    run_split(sample_labeled_fasta_path, str(a), cfg)
    run_split(sample_labeled_fasta_path, str(b), cfg)
    for f in ("train.fasta", "val.fasta", "test.fasta"):
        assert (a / f).read_text() == (b / f).read_text()


# ─────────────────────── run_audit ───────────────────────

def test_run_audit_produces_report_and_stats(tmp_path, sample_splits_dir):
    out = tmp_path / "audit"
    stats = run_audit(sample_splits_dir, str(out), RunConfig())
    assert (out / "report.txt").exists()
    assert (out / "stats.json").exists()
    assert "sizes" in stats
    # no FASTAs should be written in audit mode
    assert not (out / "train.fasta").exists()
