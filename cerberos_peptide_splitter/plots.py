"""Matplotlib visualisations — optional, imported lazily."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .features import build_feature_matrix
from .kmers import exact_kmer_jaccard


def _mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _hist_split_sizes(splits, path):
    plt = _mpl()
    if plt is None:
        return
    names = list(splits.keys())
    counts = [len(splits[n]) for n in names]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar(names, counts, color=["#4C72B0", "#DD8452", "#55A868"])
    for b, c in zip(bars, counts, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            c,
            str(c),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Sequences")
    ax.set_title("Split sizes")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _hist_lengths(splits, path):
    plt = _mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for n, recs in splits.items():
        lens = [len(r[2]) for r in recs]
        if lens:
            ax.hist(lens, bins=30, alpha=0.55, density=True, label=n)
    ax.set_xlabel("Peptide length")
    ax.set_ylabel("Density")
    ax.set_title("Length distribution by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _boxplots(splits, path):
    plt = _mpl()
    if plt is None:
        return
    selected = [
        "length",
        "hydrophobicity",
        "charge",
        "aromaticity",
        "grp_positive",
        "grp_negative",
    ]
    data = {}
    for n, recs in splits.items():
        mat, names = build_feature_matrix(recs)
        if mat.size == 0:
            continue
        idx = {fn: i for i, fn in enumerate(names)}
        data[n] = {f: mat[:, idx[f]] for f in selected if f in idx}
    if not data:
        return
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    axes = axes.flatten()
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    for i, f in enumerate(selected):
        ax = axes[i]
        vals, labs, cols = [], [], []
        for n in data:
            if f in data[n]:
                vals.append(data[n][f])
                labs.append(n)
                cols.append(colors.get(n, "gray"))
        if not vals:
            continue
        bp = ax.boxplot(vals, patch_artist=True, tick_labels=labs, showfliers=False)
        for p, c in zip(bp["boxes"], cols, strict=True):
            p.set_facecolor(c)
            p.set_alpha(0.7)
        ax.set_title(f, fontsize=10)
    fig.suptitle("Biochemical / composition feature distributions")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pca(splits, path, k=3, max_n=1000, max_features=512, seed=0, cluster_labels=None):
    plt = _mpl()
    if plt is None:
        return
    from collections import Counter

    from .kmers import kmers

    rng = np.random.RandomState(seed)
    seqs, labels, cluster_ids = [], [], []
    for split_name, records in splits.items():
        for record in records:
            seqs.append(record[2])
            labels.append(split_name)
            cluster_ids.append(None if cluster_labels is None else cluster_labels.get(record[0]))
    if len(seqs) < 2:
        return
    if len(seqs) > max_n:
        indices = rng.choice(len(seqs), max_n, replace=False)
        seqs = [seqs[index] for index in indices]
        labels = [labels[index] for index in indices]
        cluster_ids = [cluster_ids[index] for index in indices]
    per_sequence = [kmers(sequence, k) for sequence in seqs]
    frequencies = Counter(kmer for values in per_sequence for kmer in values)
    vocabulary = sorted(frequencies, key=lambda kmer: (-frequencies[kmer], kmer))[
        :max_features
    ]
    if len(vocabulary) < 2:
        return
    column = {kmer: index for index, kmer in enumerate(vocabulary)}
    matrix = np.zeros((len(seqs), len(vocabulary)), dtype=np.float32)
    for row, values in enumerate(per_sequence):
        for kmer in values:
            if kmer in column:
                matrix[row, column[kmer]] = 1.0
    matrix -= matrix.mean(axis=0, keepdims=True)
    try:
        left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return
    if len(singular_values) < 2:
        return
    projection = left[:, :2] * singular_values[:2]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if cluster_labels is not None and any(value is not None for value in cluster_ids):
        unique = sorted(set(value for value in cluster_ids if value is not None))
        palette = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique))))
        for index, cluster_id in enumerate(unique):
            mask = np.array([value == cluster_id for value in cluster_ids])
            ax.scatter(projection[mask, 0], projection[mask, 1], s=10, alpha=0.55, color=palette[index], label=f"cluster {cluster_id}")
        ax.set_title(f"PCA of full/sample k-mer profiles colored by cluster (k={k})")
    else:
        colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
        for split_name in splits:
            mask = np.array([name == split_name for name in labels])
            if mask.any():
                ax.scatter(projection[mask, 0], projection[mask, 1], s=10, alpha=0.55, c=colors.get(split_name, "gray"), label=f"{split_name} (n={mask.sum()})")
        ax.set_title(f"PCA of sampled {k}-mer profiles by split (top {len(vocabulary)} k-mers)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA of sampled {k}-mer profiles (top {len(vocabulary)} k-mers)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _similarity_hist(splits, path, k=3, max_per_split=300, seed=0):
    plt = _mpl()
    if plt is None:
        return
    rng = np.random.RandomState(seed)

    def sample(lst):
        if len(lst) <= max_per_split:
            return lst
        idx = rng.choice(len(lst), max_per_split, replace=False)
        return [lst[i] for i in idx]

    seqs = {n: sample([r[2] for r in recs]) for n, recs in splits.items()}
    names = list(splits.keys())
    dists = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                continue
            sims = []
            if a == b:
                for x in range(len(seqs[a])):
                    for y in range(x + 1, len(seqs[a])):
                        sims.append(exact_kmer_jaccard(seqs[a][x], seqs[a][y], k))
            else:
                for A in seqs[a]:
                    for B in seqs[b]:
                        sims.append(exact_kmer_jaccard(A, B, k))
            if sims:
                dists[f"{a}-{b}"] = sims
    if not dists:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for label, sims in dists.items():
        ax.hist(
            sims,
            bins=30,
            alpha=0.5,
            density=True,
            label=f"{label} (µ={np.mean(sims):.2f})",
        )
    ax.set_xlabel(f"Jaccard similarity (k={k})")
    ax.set_ylabel("Density")
    ax.set_title("Pairwise similarity within / between splits")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _label_balance(splits, path):
    plt = _mpl()
    if plt is None:
        return False
    if not any(r[1] is not None for recs in splits.values() for r in recs):
        return False
    names = list(splits.keys())
    pos = [sum(1 for r in splits[n] if r[1] == 1) for n in names]
    neg = [sum(1 for r in splits[n] if r[1] == 0) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar(x - 0.2, neg, 0.4, label="label 0", color="#4C72B0")
    ax.bar(x + 0.2, pos, 0.4, label="label 1", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Count")
    ax.set_title("Label balance per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def make_plots(splits: Dict[str, list], output_dir: str, kmer_k: int = 3, cluster_labels=None) -> None:
    import os

    plt = _mpl()
    if plt is None:
        return
    p = os.path.join(output_dir, "plots")
    os.makedirs(p, exist_ok=True)
    _hist_split_sizes(splits, os.path.join(p, "split_sizes.png"))
    _label_balance(splits, os.path.join(p, "label_balance.png"))
    _hist_lengths(splits, os.path.join(p, "length_distribution.png"))
    _boxplots(splits, os.path.join(p, "biochem_boxplots.png"))
    _pca(splits, os.path.join(p, "pca_projection.png"), k=kmer_k, cluster_labels=cluster_labels)
    _similarity_hist(splits, os.path.join(p, "pairwise_similarity.png"), k=kmer_k)
