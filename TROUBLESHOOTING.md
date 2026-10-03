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

Check whether sequences are shorter than the smallest k-mer. `--short-peptide-mode warn` reports this; `--short-peptide-mode auto` applies the `groups5` reduced alphabet as a sensitivity pass. Very short sequences may not contain enough evidence for k-mer methods.

## Splits are compositionally different

Enable `--balance` and inspect `feature_balance`, KS/Wasserstein distances, and `feature_mean_abs_error`. Increase `--balance-weight` or `--local-search-iterations`. Whole clusters cannot be split, so large compositionally biased families can make perfect balance infeasible.

## Split sizes are not exact

Cluster integrity has priority over target sizes. Inspect `assignment.size_errors` and `size_confidence_intervals`. If a small number of very large clusters dominates, exact proportions may be impossible without splitting clusters—which Cerberos deliberately refuses to do.

## High nearest-neighbor leakage risk

Inspect `nearest_neighbor_similarity.test_to_train` and `val_to_train`. High values indicate that the configured similarity method still sees close cross-boundary neighbors. Use stricter clustering, a lower threshold, additional verification metrics, or an alignment-based external method for high-assurance work.

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
