"""High-level orchestration used by the CLI and by library users."""

from __future__ import annotations

import json
import os
import sys
from typing import Dict

from .config import RunConfig
from .diagnostics import build_html_report, build_report, summarize
from .fasta import parse_fasta, write_fasta
from .split import split_records


def _log(msg: str) -> None:
    print(f"[cerberos] {msg}", file=sys.stderr)


def _load_check_dir(path: str) -> Dict[str, list]:
    candidates = {
        "train": ["train.fasta", "train.fa", "train.faa"],
        "val": ["val.fasta", "val.fa", "val.faa", "valid.fasta", "valid.fa"],
        "test": ["test.fasta", "test.fa", "test.faa"],
    }
    splits: Dict[str, list] = {}
    for split, names in candidates.items():
        for fn in names:
            p = os.path.join(path, fn)
            if os.path.exists(p):
                splits[split] = parse_fasta(p)
                break
        if split not in splits:
            raise FileNotFoundError(
                f"no {split} FASTA found in {path} (tried {', '.join(names)})"
            )
    return splits


def _write_outputs(
    splits: Dict[str, list],
    output_dir: str,
    config: RunConfig,
    mode: str,
    write_fastas: bool,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    if write_fastas:
        for name, recs in splits.items():
            path = os.path.join(output_dir, f"{name}.fasta")
            write_fasta(recs, path)
            _log(f"wrote {len(recs):>6d} -> {path}")

    stats, _ = summarize(splits, config, kmer_k=config.kmer_sizes[0])
    leakage_max = (
        stats.get("nearest_neighbor_similarity", {}).get("test_to_train", {}).get("max")
    )
    if (
        mode == "split"
        and config.strict_clustering
        and leakage_max is not None
        and leakage_max > config.similarity_threshold
    ):
        raise ValueError(
            "Leakage detected: test_to_train max Jaccard = "
            f"{leakage_max:.3f} > threshold = {config.similarity_threshold:.3f}. "
            "Clustering failed to group these sequences."
        )
    if mode == "split" and config.last_clustering_stats:
        stats["candidate_generation"] = dict(config.last_clustering_stats)
    with open(os.path.join(output_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, allow_nan=False)
        fh.write("\n")

    report = build_report(stats, mode=mode)
    with open(os.path.join(output_dir, "report.txt"), "w") as fh:
        fh.write(report)
    with open(os.path.join(output_dir, "report.html"), "w") as fh:
        fh.write(build_html_report(stats, mode=mode))

    try:
        from .plots import make_plots

        make_plots(
            splits,
            output_dir,
            kmer_k=config.kmer_sizes[0],
            cluster_labels=config.last_cluster_by_record_id,
            clustering=config.last_clustering_stats,
            threshold=config.last_clustering_stats.get(
                "effective_similarity_threshold", config.similarity_threshold
            ),
        )
    except Exception as exc:  # pragma: no cover
        _log(f"plots skipped: {exc}")

    print()
    print(report)
    return stats


def run_split(input_path: str, output_dir: str, config: RunConfig) -> dict:
    """Split a single FASTA into train / val / test."""
    _log(f"reading {input_path}")
    records = parse_fasta(input_path)
    if not records:
        raise ValueError(f"no sequences in {input_path}")
    _log(f"{len(records)} sequences loaded")

    splits = split_records(records, config, verbose=True)
    return _write_outputs(
        splits,
        output_dir,
        config,
        mode="split",
        write_fastas=not config.dry_run,
    )


def run_audit(check_dir: str, output_dir: str, config: RunConfig) -> dict:
    """Audit an existing train/val/test directory."""
    _log(f"reading splits from {check_dir}")
    splits = _load_check_dir(check_dir)
    for n, recs in splits.items():
        _log(f"  {n}: {len(recs)} sequences")
    return _write_outputs(splits, output_dir, config, mode="audit", write_fastas=False)
