# Efficiency and resumability notes

The publication workflow uses a persistent, source-fingerprinted discovery
cache, parallelizes independent configurations rather than BLAS internals, and
writes long-running additive-Gaussian and K-sensitivity results incrementally.
These mechanisms do not change the declared DE budgets, order ranges,
normalized selection criterion, or model-selection rules.

## Smoke mode

`reproduce/run_all.sh --smoke` is deliberately separate from the publication
workflow. It runs the complete tests and one tiny deterministic Burgers
weak/strong discovery, then explicitly skips precomputation, publication tables
and figures, K-sensitivity, forward validation, representative equations, and
ablation. `set -euo pipefail` ensures any failed smoke command returns a nonzero
status.

## Full revised campaign

```bash
bash run_branch_aware_campaign.sh results/normalized_metric_final
```

The optional second argument is the number of independent worker processes;
the default is two. Re-running the same command resumes the persistent cache
and incremental experiment files. The cache is stored under
`results/normalized_metric_final/main/.discovery_cache`.

The final test count and syntax/build results are recorded in `TEST_RESULTS.md`.
