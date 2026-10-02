"""Subprocess-level end-to-end tests for the cerberos CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args, timeout=300, cwd=None):
    cmd = [sys.executable, "-m", "cerberos", *args]
    return subprocess.run(
        cmd, cwd=cwd or str(ROOT),
        capture_output=True, text=True, timeout=timeout,
    )


# ─────────────────────── top-level ───────────────────────

def test_no_command_errors():
    r = run_cli()
    assert r.returncode != 0


def test_help_exits_clean():
    r = run_cli("--help")
    assert r.returncode == 0
    assert "cerberos" in (r.stdout + r.stderr).lower()


def test_split_help():
    r = run_cli("split", "--help")
    assert r.returncode == 0
    assert "--input" in r.stdout


def test_audit_help():
    r = run_cli("audit", "--help")
    assert r.returncode == 0
    assert "--dir" in r.stdout


# ─────────────────────── split ───────────────────────

def test_split_end_to_end(tmp_path, sample_labeled_fasta_path):
    out = tmp_path / "out"
    r = run_cli("split",
                "--input", sample_labeled_fasta_path,
                "--output-dir", str(out),
                "--seed", "1",
                "--kmer-sizes", "2,3",
                "--num-hashes", "64")
    assert r.returncode == 0, r.stderr
    for f in ("train.fasta", "val.fasta", "test.fasta",
              "report.txt", "stats.json"):
        assert (out / f).exists(), f"missing {f}"
    stats = json.loads((out / "stats.json").read_text())
    assert stats["sizes"]["train"] > 0


def test_split_path_a(tmp_path, sample_labeled_fasta_path):
    out = tmp_path / "out"
    r = run_cli("split",
                "--input", sample_labeled_fasta_path,
                "--reduced-alphabet", "groups5",
                "--kmer-sizes", "2,3",
                "--num-hashes", "64",
                "--output-dir", str(out))
    assert r.returncode == 0, r.stderr
    stats = json.loads((out / "stats.json").read_text())
    assert stats["clustering_mode"]["reduced_alphabet"] == "groups5"


def test_split_path_b(tmp_path, sample_labeled_fasta_path):
    out = tmp_path / "out"
    r = run_cli("split",
                "--input", sample_labeled_fasta_path,
                "--verify-with", "levenshtein",
                "--similarity-threshold", "0.85",
                "--prefilter-threshold", "0.4",
                "--kmer-sizes", "3",
                "--num-hashes", "64",
                "--output-dir", str(out))
    assert r.returncode == 0, r.stderr
    stats = json.loads((out / "stats.json").read_text())
    assert stats["clustering_mode"]["verify_with"] == "levenshtein"


def test_split_path_ab(tmp_path, sample_labeled_fasta_path):
    out = tmp_path / "out"
    r = run_cli("split",
                "--input", sample_labeled_fasta_path,
                "--reduced-alphabet", "groups5",
                "--verify-with", "levenshtein",
                "--similarity-threshold", "0.85",
                "--prefilter-threshold", "0.3",
                "--kmer-sizes", "2,3",
                "--num-hashes", "64",
                "--output-dir", str(out))
    assert r.returncode == 0, r.stderr
    stats = json.loads((out / "stats.json").read_text())
    cm = stats["clustering_mode"]
    assert cm["reduced_alphabet"] == "groups5"
    assert cm["verify_with"] == "levenshtein"


def test_split_no_clustering(tmp_path, sample_labeled_fasta_path):
    out = tmp_path / "out"
    r = run_cli("split",
                "--input", sample_labeled_fasta_path,
                "--no-clustering",
                "--output-dir", str(out))
    assert r.returncode == 0, r.stderr


def test_split_renormalizes_percentages(tmp_path, sample_fasta_path):
    out = tmp_path / "out"
    r = run_cli("split",
                "--input", sample_fasta_path,
                "--train-pct", "8", "--val-pct", "1", "--test-pct", "1",
                "--num-hashes", "32", "--kmer-sizes", "3",
                "--output-dir", str(out))
    assert r.returncode == 0, r.stderr


def test_split_seed_changes_output(tmp_path, sample_labeled_fasta_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    base = ["split",
            "--input", sample_labeled_fasta_path,
            "--kmer-sizes", "2,3", "--num-hashes", "64"]
    run_cli(*base, "--seed", "1", "--output-dir", str(a))
    run_cli(*base, "--seed", "2", "--output-dir", str(b))
    assert (a / "train.fasta").read_text() != (b / "train.fasta").read_text()


def test_split_same_seed_same_output(tmp_path, sample_labeled_fasta_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    base = ["split",
            "--input", sample_labeled_fasta_path,
            "--kmer-sizes", "2,3", "--num-hashes", "64", "--seed", "42"]
    run_cli(*base, "--output-dir", str(a))
    run_cli(*base, "--output-dir", str(b))
    for f in ("train.fasta", "val.fasta", "test.fasta"):
        assert (a / f).read_text() == (b / f).read_text()


# ─────────────────────── audit ───────────────────────

def test_audit_end_to_end(tmp_path, sample_splits_dir):
    out = tmp_path / "audit"
    r = run_cli("audit", "--dir", sample_splits_dir,
                "--output-dir", str(out))
    assert r.returncode == 0, r.stderr
    assert (out / "report.txt").exists()
    assert (out / "stats.json").exists()


def test_audit_missing_dir_errors(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    r = run_cli("audit", "--dir", str(empty))
    assert r.returncode != 0


# ─────────────────────── arg validation ───────────────────────

def test_split_requires_input():
    r = run_cli("split")
    assert r.returncode != 0


def test_audit_requires_dir():
    r = run_cli("audit")
    assert r.returncode != 0


def test_invalid_alphabet_rejected(tmp_path, sample_fasta_path):
    r = run_cli("split",
                "--input", sample_fasta_path,
                "--reduced-alphabet", "groups42",
                "--output-dir", str(tmp_path / "out"))
    assert r.returncode != 0


def test_invalid_verifier_rejected(tmp_path, sample_fasta_path):
    r = run_cli("split",
                "--input", sample_fasta_path,
                "--verify-with", "blosum62",
                "--output-dir", str(tmp_path / "out"))
    assert r.returncode != 0
