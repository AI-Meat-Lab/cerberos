"""Tests for cerberos.verify — Path B."""

from __future__ import annotations

import pytest

from cerberos_peptide_splitter.verify import (
    levenshtein,
    levenshtein_identity,
    verify_pair,
)

# ─────────────────────── levenshtein ───────────────────────


def test_levenshtein_identical():
    assert levenshtein("ACDEF", "ACDEF") == 0


def test_levenshtein_single_sub():
    assert levenshtein("ACDEF", "ACDDF") == 1


def test_levenshtein_single_ins():
    assert levenshtein("ACDEF", "ACDDEF") == 1


def test_levenshtein_single_del():
    assert levenshtein("ACDDEF", "ACDEF") == 1


def test_levenshtein_empty_a():
    assert levenshtein("", "ACDE") == 4


def test_levenshtein_both_empty():
    assert levenshtein("", "") == 0


def test_levenshtein_symmetric():
    a, b = "KWKLFKK", "RWRLFRR"
    assert levenshtein(a, b) == levenshtein(b, a)


def test_levenshtein_bounded_early_exit():
    d = levenshtein("AAAAAAAA", "YYYYYYYY", max_dist=2)
    assert d > 2


def test_levenshtein_bounded_no_false_positive():
    # true distance 0, max_dist 2 — should return 0
    assert levenshtein("ACDEF", "ACDEF", max_dist=2) == 0


def test_levenshtein_bounded_length_gap_short_circuits():
    d = levenshtein("A", "AAAAAAAA", max_dist=2)
    assert d > 2


def test_levenshtein_classic_kitten_sitting():
    # Textbook case: distance between "kitten" and "sitting" is 3
    assert levenshtein("kitten", "sitting") == 3


# ─────────────────────── levenshtein_identity ───────────────────────


def test_identity_identical_is_one():
    assert levenshtein_identity("ACDEF", "ACDEF") == 1.0


def test_identity_one_sub_in_five_is_0_8():
    assert levenshtein_identity("ACDEF", "ACDDF") == 0.8


def test_identity_empty_pair_is_one():
    assert levenshtein_identity("", "") == 1.0


def test_identity_symmetric():
    a, b = "KWKLFKK", "RWRLFRR"
    assert levenshtein_identity(a, b) == levenshtein_identity(b, a)


def test_identity_bounded_rejects_far_pair():
    # with max_dist=1, a distance-5 pair returns 0.0
    assert levenshtein_identity("ACDEF", "YYYYY", max_dist=1) == 0.0


def test_identity_short_pair_is_easily_inflated():
    """Two 4-mers differing by 1 sub: identity = 0.75."""
    assert levenshtein_identity("ACDE", "ACDF") == 0.75


# ─────────────────────── verify_pair dispatch ───────────────────────


def test_verify_pair_kmer_exact_identical():
    score = verify_pair(
        raw_a="ACDEFG",
        raw_b="ACDEFG",
        transformed_a="ACDEFG",
        transformed_b="ACDEFG",
        kmer_sizes=[3],
        metric="kmer-exact",
        threshold=0.9,
    )
    assert score == 1.0


def test_verify_pair_kmer_exact_uses_transformed():
    # raw sequences differ, transformed are identical → kmer-exact = 1.0
    score = verify_pair(
        raw_a="KWKLFKK",
        raw_b="RWRLFRR",
        transformed_a="+++H+++",
        transformed_b="+++H+++",
        kmer_sizes=[3],
        metric="kmer-exact",
        threshold=0.9,
    )
    assert score == 1.0


def test_verify_pair_kmer_exact_max_over_k():
    """Max over all requested k values should be returned."""
    score = verify_pair(
        raw_a="ACDEFG",
        raw_b="ACDEFG",
        transformed_a="ACDEFG",
        transformed_b="ACDEFG",
        kmer_sizes=[2, 3, 5],
        metric="kmer-exact",
        threshold=0.9,
    )
    assert score == 1.0


def test_verify_pair_levenshtein_uses_raw():
    # transformed sequences are identical; levenshtein uses raw
    score = verify_pair(
        raw_a="ACDEFG",
        raw_b="ACDDFG",  # 1 sub → identity 5/6
        transformed_a="H--HHH",
        transformed_b="H--HHH",
        kmer_sizes=[3],
        metric="levenshtein",
        threshold=0.5,
    )
    assert abs(score - (5 / 6)) < 1e-9


def test_verify_pair_levenshtein_rejects_when_below_threshold():
    score = verify_pair(
        raw_a="ACDEFG",
        raw_b="YYYYYY",
        transformed_a="ACDEFG",
        transformed_b="YYYYYY",
        kmer_sizes=[3],
        metric="levenshtein",
        threshold=0.9,
    )
    assert score == 0.0


def test_verify_pair_unknown_metric_raises():
    with pytest.raises(ValueError):
        verify_pair(
            raw_a="A",
            raw_b="A",
            transformed_a="A",
            transformed_b="A",
            kmer_sizes=[1],
            metric="blosum62",
            threshold=0.9,
        )
