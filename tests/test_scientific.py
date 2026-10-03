"""Regression tests for scientific formulas and clearly named metrics."""

from itertools import product

import pytest

from cerberos_peptide_splitter.features import biochem_features
from cerberos_peptide_splitter.verify import levenshtein, levenshtein_identity


def test_peptide_mass_accounts_for_water_loss_per_peptide_bond():
    one = biochem_features("A")["mw"]
    two = biochem_features("AA")["mw"]
    assert one == pytest.approx(89.1)
    assert two == pytest.approx(2 * 89.1 - 18.01528)
    assert biochem_features("AAA")["mw"] == pytest.approx(3 * 89.1 - 2 * 18.01528)


def test_charge_is_pH_dependent_and_includes_termini():
    acidic = biochem_features("D", pH=2)["charge"]
    neutral = biochem_features("D", pH=7)["charge"]
    basic = biochem_features("K", pH=7)["charge"]
    assert acidic > neutral
    assert neutral < 0
    assert basic > 0


def test_unknown_residue_is_reported_and_excluded_from_hydropathy_mean():
    canonical = biochem_features("II")
    with_unknown = biochem_features("IXI")
    assert with_unknown["unknown_fraction"] == pytest.approx(1 / 3)
    assert with_unknown["hydrophobicity"] == pytest.approx(canonical["hydrophobicity"])


def test_pH_range_is_validated():
    with pytest.raises(ValueError):
        biochem_features("ACD", pH=14.1)


def test_banded_levenshtein_matches_unbounded_for_short_sequences():
    strings = [
        "".join(chars) for length in range(4) for chars in product("AC", repeat=length)
    ]
    for left in strings:
        for right in strings:
            exact = levenshtein(left, right)
            for bound in range(4):
                result = levenshtein(left, right, max_dist=bound)
                assert result == (exact if exact <= bound else bound + 1)


def test_edit_similarity_is_not_mislabelled_as_biological_identity():
    assert levenshtein_identity("ACDE", "ACDF") == pytest.approx(0.75)


def test_levenshtein_rejects_non_integer_cutoff_even_for_empty_strings():
    with pytest.raises(ValueError, match="non-negative integer"):
        levenshtein("A", "A", max_dist=1.0)
    with pytest.raises(ValueError, match="non-negative integer"):
        levenshtein_identity("", "", max_dist=-1)
