"""FASTA parsing and writing with optional binary labels."""

from __future__ import annotations

from typing import List, Optional, Tuple

Record = Tuple[str, Optional[int], str]


def parse_fasta(path: str) -> List[Record]:
    """Parse FASTA records, accepting a terminal ``|0`` or ``|1`` label.

    Sequence lines are whitespace-stripped and uppercased. Empty records are
    ignored; data before the first header and empty headers are rejected.
    Other residue symbols (including X, gaps, and stop symbols) are retained.
    """
    records: List[Record] = []
    name: Optional[str] = None
    label: Optional[int] = None
    parts: List[str] = []

    def append_record() -> None:
        if name is not None:
            sequence = "".join(parts).upper()
            if sequence:
                records.append((name, label, sequence))

    with open(path, encoding="utf-8-sig") as fasta:
        for line_number, raw in enumerate(fasta, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                append_record()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"empty FASTA header at line {line_number}")
                name, label = header, None
                if "|" in header:
                    head, tail = header.rsplit("|", 1)
                    if tail in ("0", "1"):
                        name, label = head, int(tail)
                if not name:
                    raise ValueError(f"empty FASTA identifier at line {line_number}")
                parts = []
            else:
                if name is None:
                    raise ValueError(
                        f"sequence data before first FASTA header at line {line_number}"
                    )
                parts.append("".join(line.split()))
    append_record()
    return records


def write_fasta(records, path: str, wrap: int = 60) -> None:
    """Write records in FASTA format, wrapping sequences at ``wrap`` chars."""
    if isinstance(wrap, bool) or not isinstance(wrap, int) or wrap < 1:
        raise ValueError("wrap must be a positive integer")
    with open(path, "w", encoding="utf-8", newline="\n") as fasta:
        for name, label, sequence in records:
            header = f">{name}|{label}" if label is not None else f">{name}"
            fasta.write(header + "\n")
            for start in range(0, len(sequence), wrap):
                fasta.write(sequence[start : start + wrap] + "\n")
