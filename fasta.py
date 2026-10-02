"""FASTA parsing and writing (label-aware)."""
from __future__ import annotations

from typing import List, Optional, Tuple

Record = Tuple[str, Optional[int], str]


def parse_fasta(path: str) -> List[Record]:
    records: List[Record] = []
    name: Optional[str] = None
    label: Optional[int] = None
    parts: List[str] = []

    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, label, "".join(parts).upper()))
                hdr = line[1:].strip()
                if "|" in hdr:
                    head, tail = hdr.rsplit("|", 1)
                    if tail in ("0", "1"):
                        name, label = head, int(tail)
                    else:
                        name, label = hdr, None
                else:
                    name, label = hdr, None
                parts = []
            else:
                parts.append(line.strip())

    if name is not None:
        records.append((name, label, "".join(parts).upper()))
    return [r for r in records if r[2]]


def write_fasta(records, path: str, wrap: int = 60) -> None:
    with open(path, "w") as fh:
        for name, label, seq in records:
            hdr = f">{name}|{label}" if label is not None else f">{name}"
            fh.write(hdr + "\n")
            for i in range(0, len(seq), wrap):
                fh.write(seq[i:i + wrap] + "\n")
