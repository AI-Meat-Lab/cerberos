"""Biochemical / composition / length features.

Always computed on the RAW sequence so numbers stay comparable
across Path A / Path B settings.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

_KD_HYDRO = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}
_CHARGE = {'D': -1.0, 'E': -1.0, 'K': 1.0, 'R': 1.0, 'H': 0.1}
_MW = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1,
}
_GROUPS = {
    "aromatic": set("FWY"),
    "aliphatic": set("AVILM"),
    "polar": set("STNQ"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "special": set("GPC"),
}

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def biochem_features(seq: str) -> Dict[str, float]:
    n = len(seq)
    if n == 0:
        return {}
    cnt = Counter(seq)
    f: Dict[str, float] = {"length": float(n)}
    for aa in AA20:
        f[f"freq_{aa}"] = cnt.get(aa, 0) / n
    f["hydrophobicity"] = sum(_KD_HYDRO.get(aa, 0.0) for aa in seq) / n
    f["charge"] = sum(_CHARGE.get(aa, 0.0) for aa in seq) / n
    f["aromaticity"] = sum(1 for aa in seq if aa in _GROUPS["aromatic"]) / n
    for g, aas in _GROUPS.items():
        f[f"grp_{g}"] = sum(1 for aa in seq if aa in aas) / n
    f["mw"] = sum(_MW.get(aa, 110.0) for aa in seq)
    return f


def build_feature_matrix(records) -> Tuple[np.ndarray, List[str]]:
    feats = [biochem_features(r[2]) for r in records if r[2]]
    if not feats:
        return np.zeros((0, 0)), []
    names = sorted(feats[0].keys())
    mat = np.array([[f[n] for n in names] for f in feats], dtype=np.float64)
    return mat, names
