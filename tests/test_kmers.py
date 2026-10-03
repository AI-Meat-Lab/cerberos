"""Tests for cerberos.kmers."""

from __future__ import annotations

from cerberos_peptide_splitter.kmers import exact_kmer_jaccard, jaccard, kmers

# ─────────────────────── kmers ───────────────────────


def test_kmers_k2():
    assert kmers("ACDE", 2) == {"AC", "CD", "DE"}


def test_kmers_k_eq_len():
    assert kmers("ACDE", 4) == {"ACDE"}


def test_kmers_too_short():
    assert kmers("AC", 3) == set()


def test_kmers_empty():
    assert kmers("", 3) == set()


def test_kmers_repeats_collapse():
    assert kmers("AAAA", 2) == {"AA"}


def test_kmers_k1_is_amino_acid_set():
    assert kmers("ACDE", 1) == {"A", "C", "D", "E"}


# ─────────────────────── jaccard ───────────────────────


def test_jaccard_identical():
    assert jaccard({"A", "B"}, {"A", "B"}) == 1.0


def test_jaccard_disjoint():
    assert jaccard({"A"}, {"B"}) == 0.0


def test_jaccard_partial():
    assert jaccard({"A"}, {"A", "B"}) == 0.5


def test_jaccard_both_empty_is_one():
    assert jaccard(set(), set()) == 1.0


def test_jaccard_symmetric():
    a, b = {"A", "B", "C"}, {"B", "C", "D"}
    assert jaccard(a, b) == jaccard(b, a)


# ─────────────────────── exact_kmer_jaccard ───────────────────────


def test_exact_identical():
    assert exact_kmer_jaccard("ACDEFG", "ACDEFG", 3) == 1.0


def test_exact_single_sub_peptide():
    # 6-mers → 4 k-mers of size 3; one internal sub breaks 3 of them
    a = "ACDEFG"
    b = "ACDDFG"  # E->D
    j = exact_kmer_jaccard(a, b, 3)
    assert 0.0 < j < 1.0


def test_exact_disjoint():
    assert exact_kmer_jaccard("AAAA", "YYYY", 2) == 0.0


def test_exact_short_sequence_has_no_similarity_evidence():
    # No sequence k-mers exist when the sequence is shorter than k.
    assert exact_kmer_jaccard("AA", "AA", 3) == 0.0
