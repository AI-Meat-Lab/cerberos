"""Argparse front-end for the split and audit commands."""

from __future__ import annotations

import argparse
import sys

from .config import RunConfig
from .pipeline import run_audit, run_split


def _parse_kmer_sizes(value: str) -> list:
    try:
        sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "k-mer sizes must be comma-separated integers"
        ) from exc
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("k-mer sizes must be positive integers")
    return sizes


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="cerberos_out")
    parser.add_argument("--seed", type=int, default=42)


def _add_clustering(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-pct", type=float, default=0.80)
    parser.add_argument("--val-pct", type=float, default=0.10)
    parser.add_argument("--test-pct", type=float, default=0.10)
    parser.add_argument("--no-clustering", action="store_true")
    parser.add_argument("--similarity-threshold", type=float, default=0.70)
    parser.add_argument("--kmer-sizes", type=_parse_kmer_sizes, default=[3])
    parser.add_argument("--num-hashes", type=int, default=128)
    parser.add_argument(
        "--reduced-alphabet",
        choices=["none", "groups5", "groups7"],
        default="none",
        help="collapse residues into functional groups before k-mering (Path A)",
    )
    parser.add_argument(
        "--verify-with",
        choices=["none", "kmer-exact", "levenshtein"],
        default="none",
        help="verify LSH candidates using exact k-mer Jaccard or global edit similarity",
    )
    parser.add_argument("--prefilter-threshold", type=float, default=0.30)


def _build_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        kmer_sizes=args.kmer_sizes,
        num_hashes=args.num_hashes,
        similarity_threshold=args.similarity_threshold,
        reduced_alphabet=args.reduced_alphabet,
        verify_with=args.verify_with,
        prefilter_threshold=args.prefilter_threshold,
        seed=args.seed,
        no_clustering=args.no_clustering,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the public CLI parser."""
    parser = argparse.ArgumentParser(
        prog="cerberos-peptide-splitter",
        description="Approximate homology-aware peptide splitting and split diagnostics.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    split_parser = subcommands.add_parser("split", help="split a FASTA file")
    split_parser.add_argument("--input", required=True)
    _add_common(split_parser)
    _add_clustering(split_parser)
    audit_parser = subcommands.add_parser("audit", help="audit existing split files")
    audit_parser.add_argument("--dir", required=True)
    _add_common(audit_parser)
    audit_parser.add_argument("--kmer-sizes", type=_parse_kmer_sizes, default=[3])
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "split":
            run_split(args.input, args.output_dir, _build_config(args))
        else:
            config = RunConfig(kmer_sizes=args.kmer_sizes, seed=args.seed)
            run_audit(args.dir, args.output_dir, config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
