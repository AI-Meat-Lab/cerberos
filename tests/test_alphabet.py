"""Tests for cerberos.alphabet — Path A."""
from __future__ import annotations

import pytest

from cerberos.alphabet import (
    apply_reduced_alphabet,
    get_mapping,
    is_reduced,
)


# ─────────────────────── get_mapping ───────────────────────

def test_get_mapping_none_returns_none():
    assert get_mapping("none") is None


def test_get_mapping_groups5_returns_dict():
    m = get_mapping("groups5")
    assert isinstance(m, dict)
    assert m["K"] == m["R"] == m["H"] == "+"
    assert m["D"] == m["E"] == "-"
    assert m["A"] == m["V"] == m["I"] == m["L"] == m["M"] == "H"


def test_get_mapping_groups7_returns_dict():
    m = get_mapping("groups7")
    assert m["A"] == "A"
    assert m["F"] == m["W"] == m["Y"] == "R"
    assert m["G"] == m["P"] == "G"
    assert m["C"] == "C"


def test_get_mapping_unknown_raises():
    with pytest.raises(ValueError):
        get_mapping("groups42")


# ─────────────────────── apply ───────────────────────

def test_apply_none_is_identity():
    assert apply_reduced_alphabet("ACDEF", "none") == "ACDEF"


def test_apply_groups5_maps_each_residue():
    # A→H, C→S, D→-, E→-, F→H
    assert apply_reduced_alphabet("ACDEF", "groups5") == "HS--H"


def test_apply_groups5_merges_positive_residues():
    assert apply_reduced_alphabet("KRH", "groups5") == "+++"


def test_apply_groups5_merges_hydrophobic_residues():
    # A V I L M F W Y → all "H"
    assert apply_reduced_alphabet("AVILMFWY", "groups5") == "HHHHHHHH"


def test_apply_groups7_distinguishes_aromatic_from_aliphatic():
    assert apply_reduced_alphabet("AVF", "groups7") == "AAR"


def test_apply_unknown_residue_becomes_X():
    # B, Z, X, U, O are not in the map
    assert apply_reduced_alphabet("ABZ", "groups5") == "HXX"


def test_apply_empty_sequence():
    assert apply_reduced_alphabet("", "groups5") == ""


def test_conservative_k_to_r_variants_match_under_groups5():
    a = apply_reduced_alphabet("KWKLFKK", "groups5")
    b = apply_reduced_alphabet("RWRLFRR", "groups5")
    assert a == b


def test_conservative_variants_differ_under_identity():
    a = apply_reduced_alphabet("KWKLFKK", "none")
    b = apply_reduced_alphabet("RWRLFRR", "none")
    assert a != b


# ─────────────────────── is_reduced ───────────────────────

def test_is_reduced_none_false():
    assert is_reduced("none") is False


def test_is_reduced_groups5_true():
    assert is_reduced("groups5") is True


def test_is_reduced_groups7_true():
    assert is_reduced("groups7") is True


def test_is_reduced_unknown_raises():
    with pytest.raises(ValueError):
        is_reduced("groups42")
