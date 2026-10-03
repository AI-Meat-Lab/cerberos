"""Tests for cerberos.features."""

from __future__ import annotations

import math

import numpy as np

from cerberos_peptide_splitter.features import (
    AA20,
    biochem_features,
    build_feature_matrix,
)


def test_has_expected_keys():
    f = biochem_features("ACDEFGHIKL")
    for key in (
        "length",
        "hydrophobicity",
        "charge",
        "aromaticity",
        "mw",
        "grp_aromatic",
        "grp_aliphatic",
        "grp_polar",
        "grp_positive",
        "grp_negative",
        "grp_special",
    ):
        assert key in f


def test_length():
    assert biochem_features("ACDEFGHIKL")["length"] == 10.0


def test_freq_sums_to_one():
    f = biochem_features("ACDEFGHIKL")
    total = sum(f[f"freq_{aa}"] for aa in AA20)
    assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_positive_charge():
    assert biochem_features("KKKKRRRR")["charge"] > 0


def test_negative_charge():
    assert biochem_features("DDDDEEEE")["charge"] < 0


def test_aromaticity():
    assert biochem_features("FFFF")["aromaticity"] == 1.0
    assert biochem_features("AAAA")["aromaticity"] == 0.0


def test_hydrophobic_positive():
    assert biochem_features("IIII")["hydrophobicity"] > 3.0


def test_hydrophilic_negative():
    assert biochem_features("KKKK")["hydrophobicity"] < 0.0


def test_unknown_residue_does_not_crash():
    f = biochem_features("ACDXFGX")
    assert f["length"] == 7.0


def test_feature_matrix_shapes(labeled_records):
    mat, names = build_feature_matrix(labeled_records)
    assert mat.shape[0] == len(labeled_records)
    assert mat.shape[1] == len(names)
    assert "length" in names and "freq_A" in names


def test_feature_matrix_empty_input():
    mat, names = build_feature_matrix([])
    assert mat.shape == (0, 0)
    assert names == []


def test_feature_matrix_dtype_and_no_nan(simple_records):
    mat, _ = build_feature_matrix(simple_records)
    assert mat.dtype == np.float64
    assert not np.isnan(mat).any()


def test_feature_matrix_consistent_column_order(labeled_records):
    _, n1 = build_feature_matrix(labeled_records)
    _, n2 = build_feature_matrix(labeled_records[:3])
    assert n1 == n2
