"""Reduced amino-acid alphabets.

Collapsing 20 residues into 5 or 7 functional groups makes conservative
substitutions (K<->R, I<->V, D<->E, ...) invisible to k-mer sketches.
This drastically improves sensitivity for short peptides, at the cost
of specificity — that's why Path B exists.
"""
from __future__ import annotations

from typing import Optional

_GROUPS5 = {
    # hydrophobic (aliphatic + aromatic)
    "A": "H", "V": "H", "I": "H", "L": "H", "M": "H",
    "F": "H", "W": "H", "Y": "H",
    # polar uncharged
    "S": "P", "T": "P", "N": "P", "Q": "P",
    # positive
    "K": "+", "R": "+", "H": "+",
    # negative
    "D": "-", "E": "-",
    # special / flexible / cysteine
    "G": "S", "P": "S", "C": "S",
}

_GROUPS7 = {
    "A": "A", "V": "A", "I": "A", "L": "A", "M": "A",     # aliphatic
    "F": "R", "W": "R", "Y": "R",                          # aromatic
    "S": "P", "T": "P", "N": "P", "Q": "P",                # polar
    "K": "+", "R": "+", "H": "+",                          # positive
    "D": "-", "E": "-",                                    # negative
    "G": "G", "P": "G",                                    # gly/pro
    "C": "C",                                              # cysteine
}

_ALPHABETS = {
    "none": None,
    "groups5": _GROUPS5,
    "groups7": _GROUPS7,
}


def get_mapping(name: str) -> Optional[dict]:
    """Return the residue → group mapping, or None if identity."""
    if name not in _ALPHABETS:
        raise ValueError(f"unknown reduced alphabet: {name!r}")
    return _ALPHABETS[name]


def apply_reduced_alphabet(seq: str, name: str) -> str:
    """Transform a raw peptide into its reduced-alphabet representation.

    Unknown residues are mapped to 'X' so downstream k-mer ops keep
    working without special cases.
    """
    mapping = get_mapping(name)
    if mapping is None:
        return seq
    return "".join(mapping.get(aa, "X") for aa in seq)


def is_reduced(name: str) -> bool:
    return get_mapping(name) is not None
