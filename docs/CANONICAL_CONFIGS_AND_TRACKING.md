# Canonical configs and tracking

Each dataset has one canonical candidate search space in `dataset_configs.py`. All methods use this same search space:

```text
weak_pareto
vanilla_pareto
weak_grid_stridge
weak_fixed_stability
```

This avoids unfair comparisons where one method receives a narrower hypothesis class than another.

## What the config may contain

The config may contain public benchmark knowledge such as:

- the problem is linear in the RHS terms for the packaged datasets;
- the plausible range of temporal orders `alpha`;
- the plausible range of spatial orders `beta`;
- whether the spatial operator is periodic Riesz or one-sided finite-domain.

## What the config does not contain

The config does not provide:

- the true coefficients;
- an oracle support mask;
- a single forced derivative order;
- a result-dependent tuning after seeing the output.

## Tracking files

Each run writes:

```text
truth_spec.json
config_search_space.json
run_tracking.json
summary.json
selected_fde.json        # for Pareto methods
```

The full benchmark writes:

```text
method_comparison.csv
method_comparison.json
experiment_manifest.json
```

These files record selected structures, alpha/beta errors, coefficient diagnostics, and recovery labels.
