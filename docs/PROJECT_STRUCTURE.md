# Project structure

```text
fractional_weak_form.py        weak fractional adjoint operators
pareto_fde_discovery.py        original best-subset Pareto-DE engine and vanilla feature bank
weak_pareto_fde_discovery.py   weak feature bank, split weak/Pareto-DE helpers, proposed weak_pareto runner
baselines.py                   weak Grid-STRidge and fixed-library stability baselines
dataset_configs.py             one canonical config and truth metadata per dataset
fpde_datasets.py               built-in benchmark datasets
scripts/                       reproducible benchmark entry points, including Yu comparison
external_baselines/yu2025/     reproducible adapter, notice, and upstream acquisition instructions
notebooks/                     step-by-step learning notebooks
examples/tutorial_weak_pareto.py concise script tutorial
docs/PAPER_RESULTS_GUIDE.md     manuscript object-to-script index
```

The proposed method is implemented by `run_weak_pareto_discovery(...)` in `weak_pareto_fde_discovery.py`. The command-line name is `weak_pareto`.


Key teaching notebook: `notebooks/04_split_weak_library_and_pareto_de.ipynb` separates weak-library construction from the best-subset Pareto-DE optimizer and shows `support_size_progress.csv`.

The contemporary neural fractional-discovery framework comparison entry points are `scripts/run_yu2025_comparison.py` and `scripts/run_yu2025_comparison.sh`. The audit and fairness protocol are in `docs/YU2025_BASELINE_INTEGRATION.md`.

## GitHub entry points

Use `examples/tutorial_weak_pareto.py` or the matching Notebook 05 for a first run. Use `docs/PAPER_RESULTS_GUIDE.md` to locate the exact generator for any paper figure or table. Run `python scripts/check_documentation.py` before a release.
