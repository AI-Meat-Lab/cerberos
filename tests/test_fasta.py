"""Tests for cerberos.fasta."""

from __future__ import annotations

import pytest

import cerberos_peptide_splitter as cerberos
from cerberos_peptide_splitter.fasta import parse_fasta, write_fasta

# ─────────────────────── parsing ───────────────────────


def test_parse_plain(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">p1\nACDE\n>p2\nFGHI\n")
    assert parse_fasta(str(p)) == [("p1", None, "ACDE"), ("p2", None, "FGHI")]


def test_parse_labeled(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">p1|1\nACDE\n>p2|0\nFGHI\n")
    assert parse_fasta(str(p)) == [("p1", 1, "ACDE"), ("p2", 0, "FGHI")]


def test_parse_mixed_labels(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">a|1\nACDE\n>b\nFGHI\n>c|0\nKLMN\n")
    recs = parse_fasta(str(p))
    assert [r[1] for r in recs] == [1, None, 0]


def test_parse_multiline(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">p1\nACDE\nFGHI\nKLMN\n")
    assert parse_fasta(str(p)) == [("p1", None, "ACDEFGHIKLMN")]


def test_parse_uppercases_and_strips(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">p1\nacde  \nfghi\t\n")
    assert parse_fasta(str(p)) == [("p1", None, "ACDEFGHI")]


def test_parse_drops_empty(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">p1\n>p2\nACDE\n")
    assert parse_fasta(str(p)) == [("p2", None, "ACDE")]


def test_parse_handles_crlf(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_bytes(b">p1\r\nACDE\r\n>p2\r\nFGHI\r\n")
    assert [r[2] for r in parse_fasta(str(p))] == ["ACDE", "FGHI"]


def test_parse_invalid_label_kept_in_name(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">p1|2\nACDE\n")
    recs = parse_fasta(str(p))
    assert recs[0][0] == "p1|2" and recs[0][1] is None


def test_parse_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_fasta("/definitely/not/here.fasta")


def test_parse_trailing_newline(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">p1\nACDE\n\n\n")
    assert parse_fasta(str(p)) == [("p1", None, "ACDE")]


# ─────────────────────── writing ───────────────────────


def test_write_roundtrip(tmp_path, sample_labeled_fasta_path):
    recs = parse_fasta(sample_labeled_fasta_path)
    out = tmp_path / "out.fasta"
    write_fasta(recs, str(out))
    assert parse_fasta(str(out)) == recs


def test_write_wraps_at_60(tmp_path):
    recs = [("p1", None, "A" * 200)]
    out = tmp_path / "out.fasta"
    write_fasta(recs, str(out))
    lines = out.read_text().splitlines()
    assert len(lines) == 5
    assert all(len(line) <= 60 for line in lines[1:])


def test_write_preserves_label(tmp_path):
    recs = [("p1", 1, "ACDE"), ("p2", 0, "FGHI")]
    out = tmp_path / "out.fasta"
    write_fasta(recs, str(out))
    text = out.read_text()
    assert ">p1|1" in text and ">p2|0" in text


def test_write_no_label_no_pipe(tmp_path):
    out = tmp_path / "out.fasta"
    write_fasta([("p1", None, "ACDE")], str(out))
    assert ">p1\n" in out.read_text()
    assert "|" not in out.read_text().splitlines()[0]


def test_public_alias_reexports_functions():
    assert cerberos.parse_fasta is parse_fasta
    assert cerberos.write_fasta is write_fasta


def test_parse_rejects_sequence_before_header(tmp_path):
    path = tmp_path / "invalid.fasta"
    path.write_text("ACDE\n>p1\nFGHI\n")
    with pytest.raises(ValueError, match="before first FASTA header"):
        parse_fasta(str(path))


def test_parse_rejects_empty_header(tmp_path):
    path = tmp_path / "invalid.fasta"
    path.write_text(">   \nACDE\n")
    with pytest.raises(ValueError, match="empty FASTA header"):
        parse_fasta(str(path))


def test_write_rejects_invalid_wrap(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        write_fasta([("p", None, "ACDE")], str(tmp_path / "x.fasta"), wrap=0)
