"""Argparse front-end for the split and audit commands."""

from __future__ import annotations

import argparse
import sys

from .config import RunConfig
from .pipeline import run_audit, run_split


def _parse_csv_ints(value: str, label: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(f"{label} must contain positive integers")
    return values


def _parse_kmer_sizes(value: str) -> list[int]:
    return _parse_csv_ints(value, "k-mer sizes")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="cerberos_out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostics-sample-size", type=int, default=1000)


def _add_clustering(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-pct", type=float, default=0.80)
    parser.add_argument("--val-pct", type=float, default=0.10)
    parser.add_argument("--test-pct", type=float, default=0.10)
    parser.add_argument("--no-clustering", action="store_true")
    parser.add_argument("--similarity-threshold", type=float, default=0.70)
    parser.add_argument("--kmer-sizes", type=_parse_kmer_sizes, default=[3])
    parser.add_argument("--num-hashes", type=int, default=128)
    parser.add_argument("--max-candidate-pairs", type=int, default=250_000)
    parser.add_argument("--reduced-alphabet", choices=["none", "groups5", "groups7"], default="none")
    parser.add_argument(
        "--verify-with",
        choices=["none", "kmer-exact", "levenshtein", "containment", "cosine"],
        default="none",
    )
    parser.add_argument("--prefilter-threshold", type=float, default=0.30)
    parser.add_argument("--strict-clustering", action="store_true")
    parser.add_argument("--min-non-singleton-fraction", type=float, default=0.05)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--lsh-rows", type=lambda value: _parse_csv_ints(value, "LSH rows"), default=[2, 4, 8])
    parser.add_argument("--exact-mode", action="store_true")
    parser.add_argument("--exact-mode-max-sequences", type=int, default=50_000)
    parser.add_argument("--adaptive-threshold", action="store_true")
    parser.add_argument("--cluster-method", choices=["components", "label-propagation"], default="components")
    parser.add_argument("--short-peptide-mode", choices=["ignore", "warn", "auto"], default="warn")
    parser.add_argument("--local-search-iterations", type=int, default=0)
    parser.add_argument("--stratify-labels", action="store_true")
    parser.add_argument("--pareto-points", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")


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
        max_candidate_pairs=args.max_candidate_pairs,
        strict_clustering=args.strict_clustering,
        min_non_singleton_fraction=args.min_non_singleton_fraction,
        balance=args.balance,
        balance_weight=args.balance_weight,
        lsh_rows=args.lsh_rows,
        exact_mode=args.exact_mode,
        exact_mode_max_sequences=args.exact_mode_max_sequences,
        adaptive_threshold=args.adaptive_threshold,
        cluster_method=args.cluster_method,
        short_peptide_mode=args.short_peptide_mode,
        local_search_iterations=args.local_search_iterations,
        stratify_labels=args.stratify_labels,
        pareto_points=args.pareto_points,
        dry_run=args.dry_run,
        diagnostics_sample_size=args.diagnostics_sample_size,
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
            config = RunConfig(
                kmer_sizes=args.kmer_sizes,
                seed=args.seed,
                diagnostics_sample_size=args.diagnostics_sample_size,
            )
            run_audit(args.dir, args.output_dir, config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
