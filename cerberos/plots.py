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
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, str(c),
                ha="center", va="bottom", fontsize=9)
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
    selected = ["length", "hydrophobicity", "charge",
                "aromaticity", "grp_positive", "grp_negative"]
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
        bp = ax.boxplot(vals, patch_artist=True, labels=labs, showfliers=False)
        for p, c in zip(bp["boxes"], cols):
            p.set_facecolor(c)
            p.set_alpha(0.7)
        ax.set_title(f, fontsize=10)
    fig.suptitle("Biochemical / composition feature distributions")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pca(splits, path, k=3, max_n=4000, seed=0):
    plt = _mpl()
    if plt is None:
        return
    from .kmers import kmers
    rng = np.random.RandomState(seed)
    seqs, labels = [], []
    for n, recs in splits.items():
        for r in recs:
            seqs.append(r[2])
            labels.append(n)
    if not seqs:
        return
    if len(seqs) > max_n:
        idx = rng.choice(len(seqs), max_n, replace=False)
        seqs = [seqs[i] for i in idx]
        labels = [labels[i] for i in idx]
    vocab, per = set(), []
    for s in seqs:
        ks = kmers(s, k)
        per.append(ks)
        vocab.update(ks)
    vocab = sorted(vocab)
    vi = {km: i for i, km in enumerate(vocab)}
    X = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for i, ks in enumerate(per):
        for km in ks:
            X[i, vi[km]] = 1.0
    X -= X.mean(axis=0, keepdims=True)
    try:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return
    proj = U[:, :2] * S[:2]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    for n in splits:
        mask = np.array([x == n for x in labels])
        if mask.sum():
            ax.scatter(proj[mask, 0], proj[mask, 1], s=10, alpha=0.55,
                       c=colors.get(n, "gray"), label=f"{n} (n={mask.sum()})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA of {k}-mer profiles")
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
                        sims.append(exact_kmer_jaccard(
                            seqs[a][x], seqs[a][y], k))
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
        ax.hist(sims, bins=30, alpha=0.5, density=True,
                label=f"{label} (µ={np.mean(sims):.2f})")
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


def make_plots(splits: Dict[str, list], output_dir: str, kmer_k: int = 3) -> None:
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
    _pca(splits, os.path.join(p, "pca_projection.png"), k=kmer_k)
    _similarity_hist(splits, os.path.join(p, "pairwise_similarity.png"),
                     k=kmer_k)
