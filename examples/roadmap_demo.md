# Cerberos dual-objective example

```bash
cerberos-peptide-splitter split \
  --input peptides.fasta \
  --output-dir splits \
  --strict-clustering \
  --lsh-rows 2,4,8 \
  --kmer-sizes 2,3,4 \
  --reduced-alphabet groups5 \
  --verify-with levenshtein \
  --similarity-threshold 0.80 \
  --balance \
  --balance-weight 0.75 \
  --local-search-iterations 5 \
  --stratify-labels \
  --pareto-points 5
```

Review these fields before using the split:

- `stats.json.clustering.accepted_edges`
- `stats.json.clustering.non_singleton_fraction`
- `stats.json.clustering.candidate_recall_benchmark`
- `stats.json.assignment.size_errors`
- `stats.json.feature_balance`
- `stats.json.nearest_neighbor_similarity`
- `stats.json.cluster_quality`

For a non-destructive preview, add `--dry-run`. The preview writes statistics and reports but no split FASTA files.
