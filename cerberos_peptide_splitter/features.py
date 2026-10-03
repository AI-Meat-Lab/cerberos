"""Composition and biochemical descriptors computed on raw sequences.

Hydropathy uses the Kyte-Doolittle scale. ``charge`` is a Henderson-Hasselbalch
estimate of peptide net charge including free termini at pH 7.0. ``mw`` is an
average-mass estimate in Da for a linear peptide with free termini: free-amino-
acid masses less one water per peptide bond. These sequence descriptors are
approximations, not experimental measurements.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

_KD_HYDRO = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}
# Average masses of free amino acids in Da; subtract H2O for each peptide bond.
_FREE_AA_MASS = {
    "A": 89.1,
    "R": 174.2,
    "N": 132.1,
    "D": 133.1,
    "C": 121.2,
    "Q": 146.2,
    "E": 147.1,
    "G": 75.1,
    "H": 155.2,
    "I": 131.2,
    "L": 131.2,
    "K": 146.2,
    "M": 149.2,
    "F": 165.2,
    "P": 115.1,
    "S": 105.1,
    "T": 119.1,
    "W": 204.2,
    "Y": 181.2,
    "V": 117.1,
}
AA20 = "ACDEFGHIKLMNPQRSTVWY"
_GROUPS = {
    "aromatic": set("FWY"),
    "aliphatic": set("AVILM"),
    "polar": set("STNQ"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "special": set("GPC"),
}
_SIDECHAIN_PKA = {
    "D": (3.9, "acid"),
    "E": (4.1, "acid"),
    "C": (8.3, "acid"),
    "Y": (10.1, "acid"),
    "H": (6.0, "base"),
    "K": (10.5, "base"),
    "R": (12.5, "base"),
}
_N_TERMINUS_PKA = 8.0
_C_TERMINUS_PKA = 3.1
_WATER_MASS = 18.01528


def _ionization_charge(sequence: str, pH: float) -> float:
    """Estimate ionization charge from standard approximate pKa values."""
    charge = 1.0 / (1.0 + 10.0 ** (pH - _N_TERMINUS_PKA))
    charge -= 1.0 / (1.0 + 10.0 ** (_C_TERMINUS_PKA - pH))
    for residue in sequence:
        pka = _SIDECHAIN_PKA.get(residue)
        if pka is not None:
            value, kind = pka
            fraction = 1.0 / (1.0 + 10.0 ** (pH - value))
            charge += fraction if kind == "base" else fraction - 1.0
    return charge


def biochem_features(seq: str, pH: float = 7.0) -> Dict[str, float]:
    """Return sequence descriptors at ``pH`` (default 7.0).

    Frequencies and group fractions are per position. Non-canonical symbols
    contribute to ``unknown_fraction`` and length but are excluded from the
    hydropathy average. Unknown residues use a 110 Da free-residue estimate.
    """
    if not np.isfinite(pH) or not 0.0 <= pH <= 14.0:
        raise ValueError("pH must be finite and between 0 and 14")
    length = len(seq)
    if not length:
        return {}
    counts = Counter(seq)
    features: Dict[str, float] = {"length": float(length)}
    for residue in AA20:
        features[f"freq_{residue}"] = counts.get(residue, 0) / length
    canonical_count = sum(counts.get(residue, 0) for residue in AA20)
    features["unknown_fraction"] = (length - canonical_count) / length
    features["hydrophobicity"] = (
        sum(_KD_HYDRO[residue] * counts.get(residue, 0) for residue in AA20)
        / canonical_count
        if canonical_count
        else 0.0
    )
    features["charge"] = _ionization_charge(seq, float(pH))
    features["aromaticity"] = (
        sum(counts.get(residue, 0) for residue in _GROUPS["aromatic"]) / length
    )
    for group, residues in _GROUPS.items():
        features[f"grp_{group}"] = (
            sum(counts.get(residue, 0) for residue in residues) / length
        )
    free_mass = sum(_FREE_AA_MASS.get(residue, 110.0) for residue in seq)
    features["mw"] = free_mass - _WATER_MASS * (length - 1)
    return features


def build_feature_matrix(records) -> Tuple[np.ndarray, List[str]]:
    """Build a dense, consistently ordered feature matrix for records."""
    features = [biochem_features(record[2]) for record in records if record[2]]
    if not features:
        return np.zeros((0, 0), dtype=np.float64), []
    names = sorted(features[0])
    matrix = np.array(
        [[feature[name] for name in names] for feature in features],
        dtype=np.float64,
    )
    return matrix, names
