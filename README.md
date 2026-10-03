# cerberos

**A local, Python-based tool for approximate homology-aware splitting and descriptive diagnostics of peptide/protein FASTA datasets.**

[![CI](https://github.com/AI-Meat-Lab/cerberos/actions/workflows/ci.yml/badge.svg)](https://github.com/AI-Meat-Lab/cerberos/actions/workflows/ci.yml)
[![Lint](https://github.com/AI-Meat-Lab/cerberos/actions/workflows/lint.yml/badge.svg)](https://github.com/AI-Meat-Lab/cerberos/actions/workflows/lint.yml)
[![CodeQL](https://github.com/AI-Meat-Lab/cerberos/actions/workflows/codeql.yml/badge.svg)](https://github.com/AI-Meat-Lab/cerberos/actions/workflows/codeql.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Cerberos reads a FASTA file, generates approximate sequence-similarity clusters, and assigns complete clusters to train, validation, and test splits. It can also describe an existing set of three splits. It is designed to help identify leakage; **however, it is not a substitute for a validated alignment-based homology search**.

The package requires NumPy. Plotting is an optional Matplotlib extra; KS and Wasserstein descriptive distances are computed with NumPy and do not require SciPy. It runs locally and does not call external services or external sequence-search binaries.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Input and output](#input-and-output)
- [CLI reference](#cli-reference)
- [Method and interpretation](#method-and-interpretation)
- [Scientific definitions and limitations](#scientific-definitions-and-limitations)
- [Python API](#python-api)
- [Development and tests](#development-and-tests)
- [References](#references)

## Installation

Python 3.10 or later is supported. The PyPI distribution name is `cerberos-peptide-splitter`, the Python import name is `cerberos_peptide_splitter`, and the installed command is `cerberos-peptide-splitter`. This project has not yet been published to PyPI; install from the source checkout below until the first release.

```bash
# After the first release to PyPI:
python -m pip install cerberos-peptide-splitter
```

Optional plotting support (after the first PyPI release):

```bash
python -m pip install 'cerberos-peptide-splitter[plots]'
```

Optional plotting and the full developer toolchain (after the first PyPI release):

```bash
python -m pip install 'cerberos-peptide-splitter[full]'
python -m pip install 'cerberos-peptide-splitter[dev]'
```

To work from a source checkout:

```bash
git clone https://github.com/AI-Meat-Lab/cerberos.git
cd cerberos
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:    .venv\Scripts\activate
python -m pip install -e '.[dev]'
```

## Quick start

Split one FASTA file using defaults:

```bash
cerberos-peptide-splitter split --input peptides.fasta --output-dir splits/
```

For a sensitivity-oriented reduced-alphabet pass followed by exact raw-sequence edit-distance verification:

```bash
cerberos-peptide-splitter split \
  --input peptides.fasta \
  --reduced-alphabet groups5 \
  --verify-with levenshtein \
  --similarity-threshold 0.80 \
  --prefilter-threshold 0.30 \
  --kmer-sizes 2,3 \
  --output-dir splits/
```

Audit an existing train/validation/test directory:

```bash
cerberos-peptide-splitter audit --dir splits/ --output-dir audit/
```

Use `python -m cerberos_peptide_splitter` instead of `cerberos-peptide-splitter` if running from an unpacked source tree without installing the console entry point.

## Input and output

### FASTA input

Records may be unlabeled or have a binary label as the final header field:

```fasta
>peptide_1|1
KWKLFKKIEKVGQNIRDGIIK
>peptide_2|0
MKTIIALSYIFCLVFA
```

Rules:

- A final `|0` or `|1` is parsed as a label; other header text is retained as the identifier.
- Sequence lines are uppercased and all whitespace within them is removed.
- Empty records are ignored. Sequence text before the first `>` header and empty headers are errors.
- Other symbols (for example `X`, `-`, or `*`) are retained; they are not validated against a particular alphabet. With a reduced alphabet enabled, every non-canonical symbol is mapped to the same `X` group.
- Mixed labeled and unlabeled files are accepted. Label stratification is used only when every input record has a label.

### Outputs

`split` writes `train.fasta`, `val.fasta`, `test.fasta`, `report.txt`, `stats.json`, and—when Matplotlib is installed—a `plots/` directory. `audit` reads FASTAs and writes only report/statistics/plots; it does not overwrite the input FASTAs.

The audit command recognizes `train.fasta`/`.fa`/`.faa`, `val.fasta`/`.fa`/`.faa` or `valid.fasta`/`.fa`, and `test.fasta`/`.fa`/`.faa`.

`stats.json` contains effective configuration, split sizes and label counts, length/feature summaries, sampled pairwise k-mer Jaccard summaries, empirical KS D statistics, Wasserstein distances, and heuristic warnings. Empty samples are represented with JSON `null`, never non-standard `NaN` literals.

## CLI reference

### `cerberos-peptide-splitter split`

| Option | Default | Meaning |
|---|---:|---|
| `--input PATH` | required | Input FASTA. |
| `--output-dir DIR` | `cerberos_out` | Output directory. |
| `--seed N` | `42` | Integer seed in NumPy's supported 32-bit range. |
| `--train-pct F` | `0.80` | Train proportion. The three non-negative proportions are normalized to sum to one. |
| `--val-pct F` | `0.10` | Validation proportion. |
| `--test-pct F` | `0.10` | Test proportion. |
| `--no-clustering` | off | Use random splitting instead of grouping candidate homologs. |
| `--similarity-threshold F` | `0.70` | MinHash estimate threshold if no verifier is selected; final score threshold otherwise. Must be in [0, 1]. |
| `--kmer-sizes CSV` | `3` | Positive comma-separated k-mer sizes. |
| `--num-hashes N` | `128` | MinHash signature size (at least 4). |
| `--reduced-alphabet NAME` | `none` | `none`, `groups5`, or `groups7`; transform residues before candidate generation. |
| `--verify-with NAME` | `none` | `none`, `kmer-exact`, or `levenshtein`. |
| `--prefilter-threshold F` | `0.30` | MinHash estimate threshold before deterministic verification; cannot exceed final threshold. |

### `cerberos-peptide-splitter audit`

| Option | Default | Meaning |
|---|---:|---|
| `--dir DIR` | required | Directory containing the three split FASTAs. |
| `--output-dir DIR` | `cerberos_out` | Diagnostics output directory. |
| `--seed N` | `42` | Seed for reproducible diagnostic sampling. |
| `--kmer-sizes CSV` | `3` | Positive k-mer sizes; diagnostics use the first value. |

Invalid CLI values and missing input files are reported as concise command-line errors. The process exits non-zero on failure.

## Method and interpretation

1. **Optional alphabet transform.** `groups5` maps aliphatic and aromatic residues together, polar residues together, basic residues together, acidic residues together, and G/P/C together. `groups7` keeps aliphatic, aromatic, polar, basic, acidic, G/P, and C groups separate. These are heuristic groupings, not a substitution matrix or evolutionary model.
2. **K-mer sketches.** For each configured k, unique contiguous k-mers are hashed into seeded MinHash signatures. CRC32 provides stable input hashes. The observed sketch agreement estimates set Jaccard similarity.
3. **LSH candidate generation.** Signatures are partitioned into bands; pairs sharing a band are considered. Candidate retrieval is approximate and depends on signature length, band size, sequence composition, k, and threshold. A low prefilter threshold does **not** make candidate recall exhaustive.
4. **Optional deterministic scoring.** `kmer-exact` calculates exact unique-k-mer Jaccard on transformed sequences and takes the maximum across configured k values. `levenshtein` calculates global unit-cost edit similarity on raw sequences, `1 - edit_distance / max(sequence_length)`. It is not alignment-based percent identity. Verification only scores pairs already proposed by LSH.
5. **Connected components.** Accepted pair edges are merged with Union-Find. A connected component may include a pair of endpoints whose direct score is below the edge threshold (transitivity).
6. **Whole-cluster split assignment.** Clusters are greedily assigned to minimize deviation from target split sizes; preserving clusters takes priority over exact proportions. Clusters are not stratified by label.
7. **Descriptive diagnostics.** Raw sequences are used for pairwise k-mer distances and biochemical features. Pairwise similarity is sampled per split for cost control. Feature distances are descriptive, not inferential tests of model generalization.

## Scientific definitions and limitations

- **Jaccard:** unique-k-mer set intersection divided by union. Repeated occurrences do not receive extra weight. A sequence shorter than k has no k-mers; Cerberos treats that as no similarity evidence rather than declaring two empty sets a match.
- **Reduced alphabets:** grouping changes the sequence representation and can increase similarity among chemically grouped residues. It can also merge unrelated sequences. It is a heuristic sensitivity/specificity tradeoff; evaluate settings for the target domain.
- **Levenshtein score:** global edit distance gives substitutions, insertions, and deletions unit cost. The normalized score is length-dependent, does not use BLOSUM/PAM scores, does not model local alignment, and should not be reported as biological percent identity.
- **MinHash and LSH:** MinHash agreement approximates Jaccard under assumptions about the hash family; LSH is a probabilistic candidate filter. A missed candidate cannot be recovered by verification. For high-assurance leakage control, use an alignment-based clustering/search method and inspect thresholds on a representative validation set.
- **Clusters and thresholds:** threshold edges are transitively closed; therefore maximum pairwise within-cluster similarity is not guaranteed to meet the threshold. The cluster is an operational grouping for split assignment, not a statement that all members are homologous.
- **Hydropathy:** mean Kyte-Doolittle score over canonical residues, using the original residue scale. This is a sequence summary, not a membrane topology prediction.
- **Charge:** approximate peptide net charge at pH 7.0, including free termini and standard approximate side-chain pKa values via Henderson-Hasselbalch fractions. Local environment, terminal modifications, pKa shifts, and non-canonical residues are not modeled.
- **Molecular mass:** average free-amino-acid masses are summed and 18.01528 Da is subtracted per peptide bond. Unknown symbols are assigned an approximate 110 Da free-residue mass. This is not monoisotopic mass and does not account for modifications, cyclization, disulfide formation, or unusual termini.
- **Unknown symbols:** canonical frequencies and group fractions are per total sequence length; `unknown_fraction` reports positions outside the 20 canonical amino acids. Hydropathy averages canonical residues only.
- **KS and Wasserstein:** the report stores empirical two-sample KS D and 1-D first Wasserstein distance per feature. No p-values or multiple-testing correction are computed. The `>0.20` warning is a heuristic and depends on sample size, feature scaling, and data context.
- **No leakage guarantee:** no warning is not proof of independence, no distribution test establishes OOD safety, and this software has not been validated as a clinical or regulatory assay.

## Python API

```python
from cerberos_peptide_splitter import RunConfig, parse_fasta, split_records

records = parse_fasta("peptides.fasta")
config = RunConfig(
    train_pct=0.8,
    val_pct=0.1,
    test_pct=0.1,
    kmer_sizes=[2, 3],
    reduced_alphabet="groups5",
    verify_with="levenshtein",
    similarity_threshold=0.8,
    prefilter_threshold=0.3,
    seed=42,
)
splits = split_records(records, config, verbose=False)
print({name: len(items) for name, items in splits.items()})
```

Top-level exports also include `write_fasta`, `assign_clusters`, `stratified_random_split`, `summarize`, `build_report`, `run_split`, and `run_audit`. Lower-level modules expose the reduced-alphabet, k-mer, MinHash, and edit-distance helpers.

## Development and tests

```bash
python -m pip install -e '.[dev]'
pytest -q
pytest -q -m 'not optional'  # NumPy-only CI subset
ruff check cerberos_peptide_splitter tests
ruff format --check cerberos_peptide_splitter tests
black --check cerberos_peptide_splitter tests
isort --check-only cerberos_peptide_splitter tests
mypy cerberos_peptide_splitter --ignore-missing-imports --no-strict-optional
python -m build
python -m twine check dist/*
```

GitHub Actions checks a NumPy-only test path and a full-dependency test path across Ubuntu, macOS, and Windows, exercises all 3 × 3 alphabet/verifier combinations, runs CLI and reproducibility smoke tests, builds distributions, and runs formatting/lint/type checks. CI results—not this README—are the source of truth for currently passing environments.

## References

1. Kyte, J. & Doolittle, R. F. (1982). *A simple method for displaying the hydropathic character of a protein.* Journal of Molecular Biology 157(1), 105–132. [doi:10.1016/0022-2836(82)90515-0](https://doi.org/10.1016/0022-2836(82)90515-0); [PubMed](https://pubmed.ncbi.nlm.nih.gov/7108955/).
2. LibreTexts, *What is a protein?*—peptide bond formation, residue masses, and water loss. [Section 1A](https://chem.libretexts.org/Bookshelves/Analytical_Chemistry/Supplemental_Modules_(Analytical_Chemistry)/Analytical_Sciences_Digital_Library/In_Class_Activities/Biological_Mass_Spectrometry%3A_Proteomics/Instructors_Manual/Section_1%3A_Proteins_and_Proteomics/Section_1A._What_is_a_protein).
3. Grimsley, G. R., Scholtz, J. M. & Pace, C. N. (2009). *A summary of the measured pK values of the ionizable groups in folded proteins.* Protein Science 18, 247–251. [doi:10.1002/pro.19](https://doi.org/10.1002/pro.19); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2708032/).

## Release maintainers

The release workflow publishes through PyPI Trusted Publishing (OIDC). Before the first release, configure the PyPI project `cerberos-peptide-splitter` with a trusted publisher for the GitHub repository `AI-Meat-Lab/cerberos` and workflow `.github/workflows/release.yml`. Releases use `v*` tags and fail unless the tag exactly matches `project.version` in `pyproject.toml`.

## License and project links

MIT licensed; see [LICENSE](LICENSE). Report issues or contribute through the [GitHub repository](https://github.com/AI-Meat-Lab/cerberos), [issue tracker](https://github.com/AI-Meat-Lab/cerberos/issues), and [pull requests](https://github.com/AI-Meat-Lab/cerberos/pulls).
