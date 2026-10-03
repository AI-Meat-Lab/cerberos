"""Cerberos — three heads, three splits, no leakage."""
from .config import RunConfig
from .fasta import parse_fasta, write_fasta
from .diagnostics import build_report, summarize
from .split import split_records, assign_clusters, stratified_random_split
from .pipeline import run_split, run_audit

__version__ = "2.0.0"

__all__ = [
    "RunConfig",
    "parse_fasta", "write_fasta",
    "split_records", "assign_clusters", "stratified_random_split",
    "summarize", "build_report",
    "run_split", "run_audit",
    "__version__",
]
