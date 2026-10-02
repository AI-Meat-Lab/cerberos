"""Shared pytest fixtures for the Cerberos test suite."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# Make the package importable without `pip install -e .`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cerberos  # noqa: E402
from cerberos.config import RunConfig  # noqa: E402


# ────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir() -> str:
    return str(DATA_DIR)


@pytest.fixture
def sample_fasta_path() -> str:
    return str(DATA_DIR / "sample.fasta")


@pytest.fixture
def sample_labeled_fasta_path() -> str:
    return str(DATA_DIR / "sample_labeled.fasta")


@pytest.fixture
def sample_duplicates_fasta_path() -> str:
    return str(DATA_DIR / "sample_duplicates.fasta")


@pytest.fixture
def sample_conservative_fasta_path() -> str:
    """Two peptides differing only by conservative substitutions (K↔R)."""
    return str(DATA_DIR / "sample_conservative.fasta")


@pytest.fixture
def sample_splits_dir() -> str:
    return str(DATA_DIR / "splits")


# ────────────────────────────────────────────────────────────
# Synthetic records
# ────────────────────────────────────────────────────────────
@pytest.fixture
def simple_records():
    """Six unrelated short peptides, no labels."""
    return [
        ("p1", None, "ACDEFGHIKL"),
        ("p2", None, "MNPQRSTVWY"),
        ("p3", None, "GGGGSGGGGS"),
        ("p4", None, "YYYYYYYYYY"),
        ("p5", None, "KKKKRRRRHH"),
        ("p6", None, "DDDDEEEEEE"),
    ]


@pytest.fixture
def labeled_records():
    """Twelve peptides, 6 label-0, 6 label-1."""
    pos = [
        ("pos1", 1, "KWKLFKKIEKVGQNIRDGIIK"),
        ("pos2", 1, "KWKLFKKIEKVGQNIRDGIIK"),
        ("pos3", 1, "KWKLFKKIEKVGQNIRDGIIA"),
        ("pos4", 1, "GIGKFLHSAKKFGKAFVGEIMN"),
        ("pos5", 1, "GIGKFLHSAKKFGKAFVGEIMS"),
        ("pos6", 1, "LLGDFFRKSKEKIGKEFKRIV"),
    ]
    neg = [
        ("neg1", 0, "MKTIIALSYIFCLVFA"),
        ("neg2", 0, "MKTIIALSYIFCLVFA"),
        ("neg3", 0, "MKTIIALSYIFCLVFS"),
        ("neg4", 0, "ACDEFGHIKLMNPQRSTVWY"),
        ("neg5", 0, "YYYYYYYYYYYYYYYYYYYY"),
        ("neg6", 0, "DDDEEERRRKKKHHHNNNQQQ"),
    ]
    return pos + neg


@pytest.fixture
def homologous_records():
    """A tight family of 4 duplicates + 5 near-duplicates + 3 outliers."""
    fam = [(f"fam{i}", None, "KWKLFKKIEKVGQNIRDGIIK") for i in range(4)]
    near = [(f"near{i}", None, "KWKLFKKIEKVGQNIRDGII" + aa)
            for i, aa in enumerate("ACDEF")]
    out = [(f"out{i}", None, s) for i, s in enumerate(
        ["MKTIIALSYIFCLVFA", "YYYYYYYYYYYYYYY", "DDDEEERRRKKKHHH"])]
    return fam + near + out


@pytest.fixture
def conservative_variants():
    """Two peptides: identical except K↔R substitutions."""
    return [
        ("raw",  None, "KWKLFKKIEKVGQNIRDGIIK"),
        ("conservative", None, "RWRLFRRIERVGQNIRDGIIK"),
    ]


@pytest.fixture
def single_sub_pair():
    """Two 21-mers differing by exactly one residue at the end."""
    return [
        ("a", None, "KWKLFKKIEKVGQNIRDGIIK"),
        ("b", None, "KWKLFKKIEKVGQNIRDGIIR"),   # last K->R
    ]


# ────────────────────────────────────────────────────────────
# Config factory
# ────────────────────────────────────────────────────────────
def _default_config(**overrides):
    base = dict(
        train_pct=0.80,
        val_pct=0.10,
        test_pct=0.10,
        kmer_sizes=[2, 3],
        num_hashes=64,
        similarity_threshold=0.70,
        reduced_alphabet="none",
        verify_with="none",
        prefilter_threshold=0.30,
        seed=42,
        no_clustering=False,
    )
    base.update(overrides)
    return RunConfig(**base)


@pytest.fixture
def config_factory():
    return _default_config


@pytest.fixture
def default_config():
    return _default_config()


# ────────────────────────────────────────────────────────────
# Temp FASTA helper
# ────────────────────────────────────────────────────────────
@pytest.fixture
def write_fasta(tmp_path):
    def _write(records, name="input.fasta"):
        path = tmp_path / name
        cerberos.write_fasta(records, str(path))
        return str(path)
    return _write


@pytest.fixture
def write_split_dir(tmp_path):
    """Write train/val/test FASTAs into a subdir, return its path."""
    def _write(train, val, test, name="splits"):
        d = tmp_path / name
        d.mkdir()
        cerberos.write_fasta(train, str(d / "train.fasta"))
        cerberos.write_fasta(val, str(d / "val.fasta"))
        cerberos.write_fasta(test, str(d / "test.fasta"))
        return str(d)
    return _write


# ────────────────────────────────────────────────────────────
# Skip markers for optional dependencies
# ────────────────────────────────────────────────────────────
def _has_mpl() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def _has_scipy() -> bool:
    try:
        import scipy  # noqa: F401
        return True
    except Exception:
        return False


requires_mpl = pytest.mark.skipif(not _has_mpl(),
                                  reason="matplotlib not installed")
requires_scipy = pytest.mark.skipif(not _has_scipy(),
                                    reason="scipy not installed")


# ────────────────────────────────────────────────────────────
# Isolate cwd
# ────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield
