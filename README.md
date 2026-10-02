# cerberos

> **Homology-aware, OOD-safe splitting of peptide FASTA files.**
> *Three heads. Three splits. No leakage.*

[![CI](https://github.com/AI-Meat-Lab/cerberos/actions/workflows/ci.yml/badge.svg)](https://github.com/AI-Meat-Lab/cerberos/actions/workflows/ci.yml)
[![Lint](https://github.com/AI-Meat-Lab/cerberos/actions/workflows/lint.yml/badge.svg)](https://github.com/AI-Meat-Lab/cerberos/actions/workflows/lint.yml)
[![CodeQL](https://github.com/AI-Meat-Lab/cerberos/actions/workflows/codeql.yml/badge.svg)](https://github.com/AI-Meat-Lab/cerberos/actions/workflows/codeql.yml)
[![codecov](https://codecov.io/gh/AI-Meat-Lab/cerberos/branch/main/graph/badge.svg)](https://codecov.io/gh/AI-Meat-Lab/cerberos)
[![PyPI](https://img.shields.io/pypi/v/cerberos.svg)](https://pypi.org/project/cerberos/)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made with NumPy](https://img.shields.io/badge/made%20with-NumPy-013243.svg)](https://numpy.org/)
[![No external tools](https://img.shields.io/badge/external%20tools-none-brightgreen.svg)](#why-this-exists)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**cerberos** is a pure-Python tool for machine-learning practitioners
working with peptides, antimicrobial peptides, signal peptides, or any
short-protein sequence dataset.

It solves two problems:

1. **"I have one big FASTA — how do I split it so my model actually
   generalizes?"**
2. **"I already have train / val / test files — are they leaky? Are they
   distributionally matched?"**

It does both from a single CLI, with **two complementary homology-safety
mechanisms** — a reduced-alphabet k-mer sketch (Path A) and a
deterministic per-residue verification stage (Path B).

> *Cerberus guarded the gates of the Underworld — three heads, one
> for each split. **cerberos** guards your train / val / test gates, on
> the lookout for homology leakage and distributional drift.*

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Features](#features)
- [The two paths](#the-two-paths)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Input formats](#input-formats)
- [Command-line reference](#command-line-reference)
- [Output files](#output-files)
- [How it works](#how-it-works)
- [Diagnostics and how to interpret them](#diagnostics-and-how-to-interpret-them)
- [Examples](#examples)
- [Recipes](#recipes)
- [Library API](#library-api)
- [Testing](#testing)
- [Development](#development)
- [CI/CD](#cicd)
- [Performance](#performance)
- [FAQ](#faq)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Why this exists

Random splitting of peptide data is almost always **wrong** because:

- Near-duplicate peptides (single point mutants, shifted windows,
  family members) end up in train **and** test, inflating metrics.
- Compositional features (charge, hydrophobicity, length) can drift
  heavily between random folds when the dataset is small, causing
  **OOD** between train and test without anyone noticing.
- Existing tools that fix this (CD-HIT, MMseqs2, BLASTClust) are heavy
  binaries that are hard to install in CI / containers / notebooks.

**cerberos** is deliberately **pure Python** (only NumPy is required) and
implements:

- k-mer **MinHash + LSH** clustering to keep homologs together,
- a **reduced alphabet** (Path A) that makes conservative substitutions
  invisible to the sketch,
- a **deterministic verification** step (Path B) that confirms every
  candidate with either exact k-mer Jaccard or normalised Levenshtein
  identity,
- full biochemical / composition / length feature extraction,
- statistical and visual diagnostics for OOD.

---

## Features

- **Label-aware FASTA parsing** — headers `>name|0` and `>name|1`
  are detected automatically; unlabeled FASTA is also supported.
- **Two-stage homology pipeline** — cheap MinHash prefilter, then
  deterministic verification (Path A + Path B).
- **Reduced alphabets** — 5-group and 7-group biochemical alphabets
  that improve sensitivity to conservative substitutions.
- **Deterministic verification** — exact k-mer Jaccard *or*
  normalised Levenshtein identity, both bounded for early exit.
- **Stratified** by label when labels are present, so class balance
  is preserved across splits.
- **Rich diagnostics** — KS statistic, Wasserstein distance, PCA
  projection of k-mer profiles, pairwise Jaccard distributions
  within/between splits, biochemical boxplots.
- **Publication-ready plots** — saved as PNGs at 150 dpi.
- **Two subcommands** — `split` to create, `audit` to validate.
- **Zero external binaries** — no CD-HIT, MMseqs2, BLAST needed.
- **Deterministic** — one `--seed` reproduces everything, across OSes.
- **Tested on 3 OSes × 5 Python versions** with a three-tier matrix
  (NumPy-only, full deps, Path A×B combinations).
- **Typed, linted, security-scanned** — ruff, black, isort, mypy,
  CodeQL all green on `main`.

---

## The two paths

Cerberos's homology clustering is **layered**. You can use either layer
alone, or combine them for maximum robustness.

### Path A — reduced alphabets

Before computing k-mers, each peptide is mapped to a **coarser
biochemical alphabet**:

| Alphabet  | Groups | Collapses |
|-----------|--------|-----------|
| `groups5` | 5      | K/R/H → `+`, D/E → `-`, hydrophobic (A V I L M F W Y) → `H`, polar (S T N Q) → `P`, special (G P C) → `S` |
| `groups7` | 7      | aliphatic / aromatic / polar / positive / negative / gly-pro / cysteine |

This makes conservative substitutions — K↔R, I↔V, D↔E — **invisible** to
the k-mer sketch, so chemically similar peptides cluster together even
when their raw sequences differ by several residues.

### Path B — deterministic verification

Every candidate pair above a low MinHash **prefilter threshold** is then
verified by one of two exact metrics:

| Metric        | Input                | What it measures | Best for |
|---------------|----------------------|-------------------|----------|
| `kmer-exact`  | transformed sequence | exact k-mer Jaccard | confirming Path A merges |
| `levenshtein` | **raw** sequence     | 1 − edit distance / max length | short peptides (< 25 aa) |

`levenshtein` runs on the **raw** sequences, not the transformed ones —
so it sees the residue-level truth even when Path A has already blurred
the distinctions.

### Combined pipeline

```
   raw FASTA
        │
        ▼
 ┌────────────────┐   Path A
 │ apply_reduced_ │   (or identity)
 │  alphabet      │
 └───────┬────────┘
         │ transformed
         ▼
 ┌────────────────┐
 │ MinHash + LSH  │   candidates + estimated Jaccard
 └───────┬────────┘
         │ est ≥ prefilter_threshold
         ▼
 ┌────────────────┐   Path B
 │ verify_pair()  │   kmer-exact  OR  levenshtein
 └───────┬────────┘
         │ score ≥ similarity_threshold
         ▼
 ┌────────────────┐
 │ Union-Find     │   one cluster per connected component
 └───────┬────────┘
         │
         ▼
   train / val / test
```

**Diagnostics always run on raw sequences**, so the Jaccard histograms,
KS statistics, and PCA projections remain comparable across every Path
A / Path B configuration.

---

## Installation

### Minimal install

Only **NumPy** is required. The tool runs, but skips plots and
KS/Wasserstein statistics (prints a warning to `stderr`).

```bash
pip install cerberos
```

### Full install

Adds `matplotlib` (plots) and `scipy` (KS, Wasserstein).

```bash
pip install "cerberos[full]"
```

### Developer install

Full + test + lint tooling.

```bash
pip install "cerberos[dev]"
```

Or from a clone:

```bash
git clone https://github.com/your-org/cerberos.git
cd cerberos
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

### Requirements

| Package       | Required? | Purpose                              |
|---------------|-----------|--------------------------------------|
| `numpy`       | **Yes**   | MinHash sketches, feature matrices   |
| `matplotlib`  | Optional  | PNG plots                            |
| `scipy`       | Optional  | KS test, Wasserstein distance        |

If `matplotlib` / `scipy` are missing, the tool still runs — it just
skips the corresponding outputs.

---

## Quick start

### Split one FASTA (defaults)

```bash
cerberos split --input peptides.fasta --output-dir splits/
```

Produces `splits/train.fasta`, `splits/val.fasta`, `splits/test.fasta`,
`report.txt`, `stats.json`, and a `plots/` directory.

### Split with Path A + Path B (recommended for short peptides)

```bash
cerberos split \
    --input peptides.fasta \
    --reduced-alphabet groups5 \
    --verify-with levenshtein \
    --similarity-threshold 0.85 \
    --prefilter-threshold 0.30 \
    --kmer-sizes 2,3 \
    --output-dir splits_ab/
```

### Audit an existing split

```bash
cerberos audit --dir splits/ --output-dir audit/
```

> If you haven't installed the package, replace `cerberos` with
> `python -m cerberos` in any of the commands above.

---

## Input formats

### FASTA, no labels

```
>pep_1
KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK
>pep_2
GIGKFLHSAKKFGKAFVGEIMNS
```

### FASTA with binary labels

The label is the last `|`-separated field of the header and must be `0`
or `1`:

```
>AMP_0001|1
KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK
>nonAMP_0001|0
MKTIIALSYIFCLVFA
```

Rules:

- Sequences are uppercased on read.
- Whitespace inside sequences is stripped.
- Headers without a `|0` / `|1` suffix are treated as unlabeled and are
  tolerated in mixed files.
- Empty sequences are dropped.
- CRLF line endings are handled.

### Directory layout for `cerberos audit`

```
my_splits/
├── train.fasta   (or train.fa / train.faa)
├── val.fasta     (or val.fa / valid.fasta / valid.fa)
└── test.fasta    (or test.fa / test.faa)
```

---

## Command-line reference

```
usage: cerberos [-h] {split,audit} ...

positional arguments:
  {split,audit}
    split        split a single FASTA
    audit        audit an existing split directory

options:
  -h, --help     show this help message and exit
```

### `cerberos split`

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | – | **Required.** FASTA file to split. |
| `--output-dir DIR` | `cerberos_out` | Where to write FASTAs, report, stats, plots. |
| `--seed N` | `42` | RNG seed. |
| `--train-pct F` | `0.80` | Train fraction. |
| `--val-pct F` | `0.10` | Validation fraction. |
| `--test-pct F` | `0.10` | Test fraction. Auto-renormalised if sum ≠ 1. |
| `--no-clustering` | off | Skip homology clustering — plain stratified random split. |
| `--similarity-threshold F` | `0.70` | Final clustering threshold. |
| `--kmer-sizes CSV` | `3` | Comma-separated k values, e.g. `2,3,4`. |
| `--num-hashes N` | `128` | MinHash sketch size. |
| `--reduced-alphabet {none,groups5,groups7}` | `none` | Path A: biochemical alphabet. |
| `--verify-with {none,kmer-exact,levenshtein}` | `none` | Path B: deterministic verifier. |
| `--prefilter-threshold F` | `0.30` | Path B: MinHash prefilter. Must be ≤ similarity threshold. |

### `cerberos audit`

| Flag | Default | Description |
|------|---------|-------------|
| `--dir DIR` | – | **Required.** Directory with train/val/test FASTAs. |
| `--output-dir DIR` | `cerberos_out` | Where to write report, stats, plots. |
| `--seed N` | `42` | RNG seed for sampled diagnostics. |
| `--kmer-sizes CSV` | `3` | k values used for Jaccard diagnostics. |

**Exit codes**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Input error (no sequences, missing files) |
| 2 | CLI usage error (argparse) |

---

## Output files

For `cerberos split`:

```
output_dir/
├── train.fasta
├── val.fasta
├── test.fasta
├── report.txt          # human-readable summary
├── stats.json          # machine-readable metrics
└── plots/
    ├── split_sizes.png
    ├── label_balance.png
    ├── length_distribution.png
    ├── biochem_boxplots.png
    ├── pca_projection.png
    └── pairwise_similarity.png
```

For `cerberos audit`, only `report.txt`, `stats.json`, and `plots/` are
written — the FASTAs are read-only inputs.

### `stats.json` schema (abridged)

```json
{
  "clustering_mode": {
    "train_pct": 0.8,
    "val_pct": 0.1,
    "test_pct": 0.1,
    "kmer_sizes": [2, 3],
    "num_hashes": 128,
    "similarity_threshold": 0.85,
    "reduced_alphabet": "groups5",
    "verify_with": "levenshtein",
    "prefilter_threshold": 0.3,
    "seed": 42,
    "no_clustering": false
  },
  "sizes":           {"train": 800, "val": 100, "test": 100},
  "label_counts":    {"train": {"0": 400, "1": 400, "unlabeled": 0}},
  "length":          {"train": {"mean": 21.3, "std": 5.4, "min": 5, "max": 60}},
  "feature_names":   ["length", "freq_A", "freq_C", "...", "mw"],
  "feature_means":   {"train": {"length": 21.3, "charge": 2.1}},
  "similarity": {
    "train-train": {"mean_jaccard": 0.31, "std": 0.14, "n_pairs": 319600},
    "train-val":   {"mean_jaccard": 0.28, "std": 0.13, "n_pairs": 80000},
    "train-test":  {"mean_jaccard": 0.27, "std": 0.12, "n_pairs": 80000},
    "val-test":    {"mean_jaccard": 0.26, "std": 0.12, "n_pairs": 10000}
  },
  "ks": {
    "pairs":       ["train-vs-val", "train-vs-test", "val-vs-test"],
    "features":    ["length", "freq_A", "..."],
    "values":      [[...], [...], [...]],
    "wasserstein": [[...], [...], [...]]
  },
  "homology_warnings": ["Test-train mean Jaccard (0.27) is close to ..."],
  "ood_warnings":      ["Large KS shift (0.24) in 'charge' between ..."]
}
```

---

## How it works

### 1. FASTA parsing

Headers are parsed to extract an optional `|0` / `|1` label. Sequences
are uppercased and whitespace-stripped.

### 2. Path A — alphabet transform

If `--reduced-alphabet groups5` (or `groups7`) is set, every residue is
mapped to its functional-group code before k-mering. Diagnostics still
see the raw sequences.

### 3. k-mer sketching

For each requested `k`, a k-mer set is built per transformed sequence. A
MinHash sketch of size `--num-hashes` is computed with deterministic
**CRC32** hashing — reproducible across runs, machines, and OSes,
independent of `PYTHONHASHSEED`.

### 4. LSH candidate generation

Sketches are split into bands of 4 rows; sequences sharing a band are
candidate neighbours. This avoids O(n²) all-pairs comparison.

### 5. Path B — deterministic verification

When `--verify-with` is set, each candidate whose MinHash estimate is at
least `--prefilter-threshold` is checked with either:

- `kmer-exact` — exact Jaccard on the **transformed** sequences, max over
  all k. Cheap and consistent with Path A.
- `levenshtein` — normalised edit-distance identity on the **raw**
  sequences, with a length-based early-exit bound.

Only pairs whose verification score ≥ `--similarity-threshold` are
merged.

### 6. Union-Find clustering

Accepted pairs are merged. The process repeats per `k`; because the
union graph is shared, sequences similar at *any* k end up together.

### 7. Cluster assignment

Clusters are sorted by size (largest first) and greedily assigned to
whichever split is furthest below its target. This keeps target
proportions while never splitting a cluster.

### 8. Diagnostics

- **Pairwise Jaccard** within and between splits (sampled for speed).
- **Biochemical features** per peptide: length, aa frequencies,
  Kyte–Doolittle hydrophobicity, net charge, aromaticity, group
  fractions, molecular weight.
- **KS test** + **Wasserstein distance** per feature, per split pair.
- **PCA** of k-mer indicator matrices, colored by split.
- **Warnings** emitted when `test-train ≈ train-train` similarity
  (homology leakage) or when any feature's KS > 0.20 (distribution
  shift / OOD).

---

## Diagnostics and how to interpret them

### Jaccard similarity table

| Pattern | Meaning | Action |
|---------|---------|--------|
| `train-test ≈ train-train` | Test is basically in-distribution | Enable `--exclude-homologs` equivalent (Path A / B) |
| `train-test << train-train` | Good separation ✅ | None |
| All values near 0 | Very diverse dataset | Fine, but check dataset size |

### KS statistic heatmap

- **0 – 0.10** — features match, excellent.
- **0.10 – 0.20** — mild shift, usually acceptable.
- **> 0.20** — OOD risk. Inspect `plots/biochem_boxplots.png`.

### PCA projection

Splits should **overlap** in PCA space. If test forms its own cluster,
the split is biased — rerun with a different seed, a stricter
similarity threshold, or Path A/B enabled.

### Warnings

Example output:

```
! Test-train mean Jaccard (0.42) is close to train-train (0.48).
! Large KS shift (0.24) in 'charge' between train and test.
```

Treat any warning as a **blocker** for publication-grade results.

---

## Examples

### Example 1 — Long proteins, speed matters

```bash
cerberos split --input proteins.fasta --output-dir splits/
```

### Example 2 — Short peptides, K↔R variants present

```bash
cerberos split \
    --input peptides.fasta \
    --reduced-alphabet groups5 \
    --verify-with levenshtein \
    --similarity-threshold 0.80 \
    --prefilter-threshold 0.25 \
    --kmer-sizes 2,3 \
    --output-dir splits_ab/
```

### Example 3 — Balanced AMP dataset

```bash
cerberos split \
    --input amp_dataset.fasta \
    --reduced-alphabet groups7 \
    --verify-with levenshtein \
    --similarity-threshold 0.70 \
    --prefilter-threshold 0.30 \
    --kmer-sizes 2,3,4 \
    --train-pct 0.70 --val-pct 0.15 --test-pct 0.15 \
    --seed 7 \
    --output-dir amp_splits/
```

### Example 4 — Deduplication only

```bash
cerberos split \
    --input raw.fasta \
    --similarity-threshold 0.98 \
    --verify-with kmer-exact \
    --output-dir dedup_splits/
```

### Example 5 — Audit a colleague's splits

```bash
cerberos audit --dir their_splits/ --output-dir audit/
cat audit/report.txt
```

---

## Recipes

### Reproducible splits in GitHub Actions

```yaml
- name: Split peptides reproducibly
  run: |
    cerberos split \
      --input data/peptides.fasta \
      --seed 1234 \
      --reduced-alphabet groups5 \
      --verify-with levenshtein \
      --output-dir /tmp/splits

- name: Fail if splits are leaky
  run: |
    grep -q "No warnings" /tmp/splits/report.txt
```

### Compare two strategies

```bash
cerberos split --input data.fasta --output-dir s_minhash
cerberos split --input data.fasta --reduced-alphabet groups5 \
    --verify-with levenshtein --similarity-threshold 0.85 \
    --output-dir s_ab
diff <(grep "mean=" s_minhash/report.txt) \
     <(grep "mean=" s_ab/report.txt)
```

### Verify that Path A actually helps on your data

```bash
for alpha in none groups5 groups7; do
  cerberos split --input data.fasta \
      --reduced-alphabet $alpha \
      --similarity-threshold 0.80 \
      --output-dir "s_$alpha"
  echo "$alpha: $(grep 'train-train' s_$alpha/report.txt)"
done
```

---

## Library API

The public API is deliberately small — eight names from the top-level
package.

```python
import cerberos

# ── Config ────────────────────────────────────────────
from cerberos import RunConfig

cfg = RunConfig(
    train_pct=0.80,
    val_pct=0.10,
    test_pct=0.10,
    kmer_sizes=[2, 3],
    num_hashes=128,
    similarity_threshold=0.85,
    reduced_alphabet="groups5",       # Path A
    verify_with="levenshtein",        # Path B
    prefilter_threshold=0.30,
    seed=42,
)

# ── Split ─────────────────────────────────────────────
from cerberos import parse_fasta, split_records

records = parse_fasta("peptides.fasta")
splits = split_records(records, cfg, verbose=False)
print({k: len(v) for k, v in splits.items()})
# {'train': 800, 'val': 100, 'test': 100}

# ── Summarize ─────────────────────────────────────────
from cerberos import summarize, build_report

stats, feature_names = summarize(splits, cfg, kmer_k=3)
print("Homology warnings:", stats["homology_warnings"])
print("OOD warnings:     ", stats["ood_warnings"])
print(build_report(stats, mode="split"))

# ── High-level pipelines ──────────────────────────────
from cerberos import run_split, run_audit

run_split("peptides.fasta", "splits/", cfg)
run_audit("splits/", "audit/", cfg)

# ── Low-level pieces ──────────────────────────────────
from cerberos.alphabet import apply_reduced_alphabet
from cerberos.kmers import kmers, jaccard, exact_kmer_jaccard
from cerberos.minhash import minhash_sketch, lsh_candidates
from cerberos.verify import levenshtein, levenshtein_identity
from cerberos.cluster import UnionFind, compute_homology_clusters

transformed = apply_reduced_alphabet("KWKLFKKIEK", "groups5")
print(transformed)                       # '+++H+++H+H'
print(exact_kmer_jaccard("ACDEFG", "ACDDFG", 3))
print(levenshtein_identity("ACDEF", "ACDDF"))
```

### Public exports

```python
from cerberos import (
    RunConfig,             # dataclass with all pipeline options
    parse_fasta,           # → List[(name, label, sequence)]
    write_fasta,           # writes records back out
    split_records,         # records + config → {train, val, test}
    assign_clusters,       # clusters + config → {index: split}
    stratified_random_split,  # no clustering
    summarize,             # → (stats_dict, feature_names)
    build_report,          # → human-readable string
    run_split,             # input path + output dir + config
    run_audit,             # check dir + output dir + config
)
```

---

## Testing

The project ships with a full `pytest` suite covering every module,
including dedicated tests for Path A, Path B, and their combination at
both the library and CLI level.

### Run locally

```bash
# Minimal environment (NumPy only)
pytest -q -m "not optional"

# Full environment with coverage
pytest -q --cov=cerberos --cov-report=term-missing

# Parallel
pytest -q -n auto

# A single module
pytest -q tests/test_cluster.py
pytest -q tests/test_verify.py
```

### Test layout

```
tests/
├── conftest.py                 # shared fixtures + skip markers
├── test_fasta.py               # FASTA parsing / writing
├── test_alphabet.py            # Path A
├── test_kmers.py               # k-mer sets, exact Jaccard
├── test_minhash.py             # MinHash sketches + LSH
├── test_verify.py              # Path B
├── test_cluster.py             # Path A + B pipeline
├── test_split.py               # stratified + cluster assignment
├── test_features.py            # biochemical features
├── test_diagnostics.py         # summarize, KS, report
├── test_pipeline.py            # run_split / run_audit
├── test_cli.py                 # subprocess end-to-end
└── data/
    ├── sample.fasta
    ├── sample_labeled.fasta
    ├── sample_duplicates.fasta
    ├── sample_conservative.fasta
    └── splits/{train,val,test}.fasta
```

### Coverage map

| Module            | Test file              | Notable cases |
|-------------------|------------------------|---------------|
| `fasta.py`        | `test_fasta.py`        | CRLF, mixed labels, empty seqs |
| `alphabet.py`     | `test_alphabet.py`     | K↔R merging, unknown residues → `X` |
| `kmers.py`        | `test_kmers.py`        | Empty, short, repeat-collapse |
| `minhash.py`      | `test_minhash.py`      | Determinism, Jaccard tolerance, LSH |
| `verify.py`       | `test_verify.py`       | Bounded Levenshtein, raw vs transformed dispatch |
| `cluster.py`      | `test_cluster.py`      | Path A merge, Path B recovery, A+B combo |
| `split.py`        | `test_split.py`        | Stratification, cluster assignment, determinism |
| `features.py`     | `test_features.py`     | All feature keys, unknown residues |
| `diagnostics.py`  | `test_diagnostics.py`  | JSON round-trip, warnings, mode echo |
| `pipeline.py`     | `test_pipeline.py`     | Split vs audit, missing files, determinism |
| `cli.py`          | `test_cli.py`          | Subcommands, arg validation, seed reproducibility |

---

## Development

### Common tasks

```bash
make install          # pip install -e ".[dev]"
make test             # pytest -q
make test-minimal     # pytest -q -m "not optional"
make lint             # ruff + black --check + isort --check
make format           # black + isort (writes)
make typecheck        # mypy cerberos/
make build            # python -m build
make clean            # remove caches / build artifacts
```

### Coding standards

- PEP 8, 4-space indentation.
- Type hints on public functions.
- One concern per module — see the layout above.
- **No new required dependencies.** Optional deps must be imported
  lazily (inside functions), never at module top.
- Everything must be seedable through `RunConfig.seed`.

### Running the test matrix locally

```bash
pip install tox
tox                          # everything
tox -e py311-full            # one environment
tox -e lint                  # just the linters
```

---

## CI/CD

Every push and pull request runs:

| Job | Matrix | Purpose |
|-----|--------|---------|
| `test-minimal` | 3 OSes × py3.8/3.10/3.12, **NumPy only** | Enforce the "NumPy-only" promise |
| `test-full` | 3 OSes × py3.9/3.11/3.12, scipy+matplotlib | Full diagnostics + Codecov |
| `path-matrix` | **9 combinations** of `{none,groups5,groups7} × {none,kmer-exact,levenshtein}` | Directly exercises Path A, Path B, and A+B |
| `smoke-cli` | Ubuntu, py3.11 | Installs the wheel and runs `split` / `audit` |
| `determinism` | Ubuntu, py3.11 | Runs twice with `--seed 42`, `diff`s outputs |
| `build` | Ubuntu, py3.11 | Builds sdist + wheel, installs, tests entry point |
| `Lint` | Ubuntu, py3.11 | ruff, black `--check`, isort `--check`, mypy |
| `CodeQL` | Ubuntu, py3.11 | Python security scan |

The `path-matrix` job is what makes this project trustworthy for
homology-sensitive work: it proves that **every combination of Path A
and Path B runs end-to-end**, not just the ones you happen to use.

### Release flow

```bash
git tag v0.0.1b
git push origin v0.0.1b
```

The `Release` workflow runs the test suite, builds, publishes to PyPI,
and creates a GitHub Release — all from the tag.

---

## Performance

Approximate timings on a 2020 MacBook Pro (M1):

| Dataset | Mode | Time |
|---------|------|------|
| 5 000 peptides | default (`--kmer-sizes 3`) | ~2 s |
| 5 000 peptides | Path A only (`groups5`, k=2,3) | ~4 s |
| 5 000 peptides | Path B only (`levenshtein`, k=3) | ~8 s |
| 5 000 peptides | Path A + B (`groups5` + `levenshtein`, k=2,3) | ~10 s |
| 50 000 peptides | Path A + B | ~55 s |
| 200 000 peptides | Path A + B | ~4 min |

Scaling with `--num-hashes` is roughly linear. Path B adds cost
proportional to the number of candidate pairs surviving the prefilter —
lower `--prefilter-threshold` ⇒ more verifications.

---

## FAQ

**Q: Do I need CD-HIT or MMseqs2?**
No. Clustering is pure Python (MinHash + LSH + optional deterministic
verification). No external binaries.

**Q: Which Path should I use?**
- Long proteins (> 100 aa), speed matters → neither (defaults).
- Peptides 20–50 aa → `--reduced-alphabet groups7`.
- Peptides 10–25 aa → `--reduced-alphabet groups5`.
- Peptides < 15 aa, publication-grade → **both**:
  `--reduced-alphabet groups5 --verify-with levenshtein`.

**Q: How do I choose `--prefilter-threshold`?**
Rule of thumb: `prefilter_threshold ≤ similarity_threshold − 0.3` for
peptides of 15–30 aa. Lower = higher recall (more verifier work); the
verifier sorts out false positives anyway.

**Q: Does `levenshtein` verification use the raw or the reduced
alphabet?**
**Raw.** That's the point — Path A makes the sketch insensitive to
conservative substitutions, but Path B sees the residue-level truth.

**Q: How big a FASTA can it handle?**
Comfortably up to ~1M peptides on a laptop. The all-pairs Jaccard for
diagnostics is capped via internal sampling.

**Q: Does it work with non-canonical amino acids?**
Yes — unknown residues map to `X` in the reduced alphabet and contribute
0 to their biochemical feature sums. Length and k-mers still work.

**Q: Does it need `scipy` / `matplotlib`?**
No. If missing, the tool prints a warning and skips those outputs.

**Q: Is the output reproducible across OSes?**
Yes. Everything is seeded through `--seed`, and MinHash hashing uses
`zlib.crc32`, which is stable across Python builds and platforms.

**Q: How do I change the k-mer alphabet to something custom?**
Not exposed via the CLI. Import `cerberos.alphabet.apply_reduced_alphabet`
directly, or add a new mapping to `REDUCED_ALPHABETS` — the pipeline
picks it up automatically.

**Q: Can I use it as a library?**
Yes — see the [Library API](#library-api) section. The full CLI is a
thin shell over `run_split` and `run_audit`.

---

## Contributing

Bug reports, feature requests, and PRs are welcome.
Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

- 🐛 [Open an issue](https://github.com/yAI-Meat-Lab/cerberos/issues/new)
- 💡 [Request a feature](https://github.com/AI-Meat-Lab/cerberos/issues/new)
- 🔧 [Submit a pull request](https://github.com/AI-Meat-Lab/cerberos/pulls)

### Release history

See [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use **cerberos** in academic work, please cite:

```bibtex
@software{cerberos,
  title  = {cerberos: Homology-aware and OOD-safe splitting of peptide FASTA files},
  author = {Celio Dias Santos-Junior},
  year   = {2026},
  url    = {https://github.com/AI-Meat-Lab/cerberos/},
  note   = {Version 0.0.1b}
}
```

## License

MIT — see [LICENSE](LICENSE).
