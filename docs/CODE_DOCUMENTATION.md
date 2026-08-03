# Code documentation and public API

The repository separates reusable library code from experiment drivers.

## Recommended public API

- `dataset_configs.benchmark_spec(...)`: load one named benchmark, its discovery
  configuration, and truth metadata.
- `weak_pareto_fde_discovery.build_weak_candidate_library(...)`: construct and
  precompute weak-form regression rows without running model selection.
- `weak_pareto_fde_discovery.build_best_subset_pareto_problem(...)`: create the
  deterministic train/validation split and Pareto optimizer.
- `weak_pareto_fde_discovery.run_best_subset_pareto_de(...)`: run the support-size
  sweep and select the validation-error--complexity elbow.
- `weak_pareto_fde_discovery.run_weak_pareto_discovery(...)`: one-call wrapper for
  the complete proposed method.
- `fractional_weak_form`: lower-level fractional operators, test functions,
  discrete adjoints, and least-squares helpers.

The reusable modules use NumPy-style docstrings and type annotations. Experiment
and reproduction scripts provide a module-level description and command-line
help (`python SCRIPT.py --help`). Internal helpers beginning with `_` are not
part of the stable public API.

Run the documentation audit with:

```bash
python scripts/check_documentation.py
```

The audit checks that every non-test Python file has a module docstring and that
all top-level public objects in the supported API modules have docstrings.
