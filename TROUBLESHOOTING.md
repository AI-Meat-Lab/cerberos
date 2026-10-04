# Cerberos troubleshooting guide

## `strict clustering failed`

This means the run found no accepted edges or fewer than the configured minimum fraction of sequences in multi-sequence clusters. It is intentionally a hard failure, not a data-quality judgment.

- Inspect `candidate_generation` and `clustering` in `stats.json`.
- Use `--exact-mode` for datasets below 50,000 sequences.
- Try more sensitive candidate generation: `--lsh-rows 2,4,8`, more k-mer sizes, or `--reduced-alphabet groups5`.
- Lower `--similarity-threshold` only after inspecting accepted/rejected score histograms.
- If the dataset is genuinely non-redundant, omit strict mode and record that result.

## `candidate_limit_reached` or sampled dense buckets

The search budget or dense-bucket cap reduced candidate recall. Increase `--max-candidate-pairs`, use `--exact-mode` for small data, or compare multiple deterministic `--lsh-rows` configurations. The `candidate_recall_benchmark` is an estimate on a small exact sample, not a proof of exhaustive recall.

## All sequences are singletons

Check whether sequences are shorter than the smallest k-mer. `--short-peptide-mode warn` reports this; `--short-peptide-mode auto` applies the `groups5` reduced alphabet as a sensitivity pass and adds k=2 evidence when needed. Very short sequences may not contain enough evidence for k-mer methods.

## Splits are compositionally different

Enable `--balance` and inspect `feature_balance`, normalized Wasserstein/KS distances, and `feature_mean_abs_error`. Increase `--balance-weight` or `--local-search-iterations`. Whole clusters cannot be split, so large compositionally biased families can make perfect balance infeasible.

## Split sizes are not exact

Balanced assignment uses hard approximately 95–105% target bounds where feasible, while cluster integrity remains mandatory. Inspect `assignment.hard_size_bounds`, `assignment.size_errors`, and `assignment.size_constraint_fallback`. If a small number of very large clusters dominates, exact proportions may be impossible without splitting clusters—which Cerberos deliberately refuses to do.

## Exact mode refuses to run

Exact mode is intentionally never truncated. It requires `--max-candidate-pairs >= N*(N-1)/2` for the active dataset and is O(N²). Remove the candidate cap or use LSH mode for larger datasets.

## `candidate_pairs_verified` is zero or unexpectedly small

The counter includes accepted candidate edges even when no deterministic verifier is configured. If it remains low, inspect `candidate_pairs_considered`, `candidate_recall_benchmark`, dense-bucket sampling, and the candidate cap.

## High nearest-neighbor leakage risk

Inspect `nearest_neighbor_similarity.test_to_train` and `val_to_train`. High values indicate that the configured similarity method still sees close cross-boundary neighbors. Use stricter clustering, a lower threshold, additional verification metrics, or an alignment-based external method for high-assurance work.

Strict split runs abort when the maximum exact k-mer Jaccard from test to train exceeds the configured threshold. Lower the threshold, increase k-mer evidence, use exact mode for small datasets, or investigate duplicate identifiers and external cluster imports.

## Adaptive thresholds behave unexpectedly

Use `--adaptive-threshold` only when the candidate score distribution has enough observations. Compare `effective_similarity_threshold` with the requested threshold and review the accepted/rejected histograms. Adaptive thresholds are data-dependent and should be fixed in a reproducibility record.

## Need cross-validation or external clusters

Use the Python API:

```python
from cerberos_peptide_splitter import (
    cross_validation_assignments,
    hierarchical_cluster_views,
    load_cluster_assignments,
)
```

`load_cluster_assignments` accepts JSON mappings or CSV files with `index,cluster` columns. Imported assignments should be independently validated before use.
