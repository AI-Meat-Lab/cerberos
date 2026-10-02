"""Argparse front-end. Two subcommands: split and audit."""
from __future__ import annotations

import argparse
import sys

from .config import RunConfig
from .pipeline import run_audit, run_split


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="cerberos_out",
                        help="where to write FASTA, report, stats, plots")
    parser.add_argument("--seed", type=int, default=42)


def _add_clustering(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-pct", type=float, default=0.80)
    parser.add_argument("--val-pct", type=float, default=0.10)
    parser.add_argument("--test-pct", type=float, default=0.10)

    parser.add_argument("--no-clustering", action="store_true",
                        help="skip homology clustering (random split)")
    parser.add_argument("--similarity-threshold", type=float, default=0.70,
                        help="final clustering threshold; 0.95 = dup-only")
    parser.add_argument("--kmer-sizes", default="3",
                        help="comma-separated k values, e.g. '2,3,4'")
    parser.add_argument("--num-hashes", type=int, default=128)

    # Path A
    parser.add_argument("--reduced-alphabet",
                        choices=["none", "groups5", "groups7"],
                        default="none",
                        help="collapse residues into functional groups "
                             "before computing k-mers (Path A)")

    # Path B
    parser.add_argument("--verify-with",
                        choices=["none", "kmer-exact", "levenshtein"],
                        default="none",
                        help="deterministic verification stage (Path B)")
    parser.add_argument("--prefilter-threshold", type=float, default=0.30,
                        help="MinHash prefilter threshold for Path B; "
                             "should be lower than --similarity-threshold")


def _build_config(args: argparse.Namespace) -> RunConfig:
    kmer_sizes = [int(x) for x in args.kmer_sizes.split(",") if x.strip()]
    return RunConfig(
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        kmer_sizes=kmer_sizes,
        num_hashes=args.num_hashes,
        similarity_threshold=args.similarity_threshold,
        reduced_alphabet=args.reduced_alphabet,
        verify_with=args.verify_with,
        prefilter_threshold=args.prefilter_threshold,
        seed=args.seed,
        no_clustering=args.no_clustering,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cerberos",
        description="Homology-aware, OOD-safe peptide splitting. "
                    "Three heads: train, val, test.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split", help="split a single FASTA")
    p_split.add_argument("--input", required=True)
    _add_common(p_split)
    _add_clustering(p_split)

    p_audit = sub.add_parser("audit", help="audit an existing split directory")
    p_audit.add_argument("--dir", required=True)
    _add_common(p_audit)
    p_audit.add_argument("--kmer-sizes", default="3")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "split":
        config = _build_config(args)
        run_split(args.input, args.output_dir, config)
    elif args.command == "audit":
        kmer_sizes = [int(x) for x in args.kmer_sizes.split(",") if x.strip()]
        config = RunConfig(kmer_sizes=kmer_sizes, seed=args.seed)
        run_audit(args.dir, args.output_dir, config)
    else:                                    # pragma: no cover
        parser.error(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":                   # pragma: no cover
    sys.exit(main())
