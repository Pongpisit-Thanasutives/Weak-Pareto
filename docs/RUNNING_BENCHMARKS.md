## Final branch-aware campaign

Run the complete affected publication campaign with:

```bash
bash run_branch_aware_campaign.sh results/normalized_metric_final
```

This includes the internal five-seed sweep, additive-Gaussian study, and the Weak-Pareto rows of the Yu comparison. The final aggregate outputs are already bundled under `reproduce/reference_results/branch_aware_campaign/`; rerunning is required only for independent reproduction or code changes.

# Running paper benchmarks

## Fast sanity run

```bash
FPDE_QUICK=1 bash scripts/run_publication_benchmarks.sh
```

This runs a small dataset/noise subset and checks that the full framework works.

## Paper run

```bash
bash scripts/run_publication_benchmarks.sh
```

The default paper run compares:

```text
weak_pareto          proposed: weak lib + best-subset Pareto-DE
vanilla_pareto       baseline: vanilla lib + best-subset Pareto-DE
weak_grid_stridge    baseline: weak lib + grid-search STRidge
weak_fixed_stability fixed-library stability ablation
```


## Notebook/script consistency

The teaching notebooks and benchmark scripts share the same configuration path.
Set the controls at the top of a notebook:

```python
dataset_name = "synthetic_time_space_fractional_RD"
noise_percent = 0.5
profile = "paper"
seed = 0
maxiter_override = None
popsize_override = None
```

This is equivalent to passing the same dataset/noise/profile/seed to
`scripts/run_all_methods.py`.  When `profile="paper"`, leave `maxiter` and
`popsize` unspecified unless you are intentionally running an ablation; the
dataset-specific paper budgets are chosen automatically.

## Manual method comparison

```bash
python scripts/run_all_methods.py \
  --profile notebook \
  --methods weak_pareto vanilla_pareto weak_grid_stridge weak_fixed_stability \
  --noise-levels 0 0.1 0.25 \
  --seeds 0 \
  --maxiter 0 \
  --popsize 2 \
  --weak-test-budget smoke \
  --quiet \
  --output-dir results/manual_check
```

## Noise tolerance sweep for the proposed method

```bash
python scripts/find_noise_tolerance.py \
  --noise-levels 0 0.05 0.1 0.2 0.25 0.5 1 2 \
  --seeds 0 \
  --maxiter 0 \
  --popsize 2 \
  --weak-test-budget smoke \
  --quiet \
  --output-dir results/noise_tolerance
```

The tolerance script reports the highest tested noise level at which `weak_pareto` recovers the true structure and passes the declared coefficient/order tolerances.


## Tracking proposed-method progress

Every `weak_pareto` run with an output directory writes:

```text
support_size_progress.csv
support_size_progress.json
```

These files list the best equation found at each support size `c`, including `alpha`, `beta_tuple`, coefficients, validation relative MSE, and the formatted equation. Use them to inspect whether the true structure appears at the expected support size before final Pareto selection.

## Interpreting recovery columns

`full_structure_recovered` is intentionally a symbolic-structure flag only. It checks whether the selected support/form matches the benchmark equation. It does not require the recovered alpha, beta, or coefficients to be close. For paper tables, report this flag together with `alpha_abs_error`, `max_matched_beta_abs_error`, and `max_coef_rel_error`.


## Candidate library scope

The paper configuration uses an overcomplete RHS class with terms `u^p D_x^beta u`, `p_values=(0,1,2)`, and canonical `cmax=4`. This is broader than the true bundled linear two-term equations, so the proposed method must reject competing nonlinear candidates. The manuscript reproduction drivers override this to `cmax=4`, matching the reported experiments. For a heavier stress test, pass `--cmax 5` or `FPDE_CMAX=5`; this is slower and may require larger DE budgets.

## Yu et al. neural fractional-discovery comparison

Install the PyTorch-enabled dependency set:

```bash
pip install -r requirements_yu2025.txt
```

Run the complete smoke workflow on CPU:

```bash
FPDE_QUICK=1 FPDE_DEVICE=cpu bash scripts/run_yu2025_comparison.sh
```

Run publication profiles:

```bash
FPDE_DEVICE=cpu bash scripts/run_yu2025_comparison.sh
```

Use CUDA for the neural reconstruction and automatic-differentiation phase:

```bash
FPDE_DEVICE=cuda bash scripts/run_yu2025_comparison.sh
```

The default contemporary neural fractional-discovery framework comparison runs `weak_pareto`, `yu2025_full`, and `yu2025_optimizer_only` on the FADE data. It uses noise levels `0 1 5` and seeds `0 1 2 3 4` in the full profile. Override these with space-separated variables, for example:

```bash
FPDE_NOISE="0 1 5 10 25" \
FPDE_SEEDS="0 1 2 3 4" \
FPDE_YU_EPOCHS=10000 \
FPDE_DEVICE=cuda \
bash scripts/run_yu2025_comparison.sh
```

The bash workflow first runs the complete test suite, then writes the shared noisy input and its SHA-256, runs each method, and exports CSV, JSON, per-run tracking, and a Markdown summary. The quick profile is only a software validation profile and must not be used for paper conclusions.

Yu's published operator realisation is one-sided and finite-terminal, whereas Weak-Pareto uses its periodic directional spectral convention on FADE. These are disclosed as different operator realisations sharing the field, nominal equation, and target orders; periodic signed-Riesz datasets lie outside the declared comparison scope and are marked `skipped_operator_scope`. The upstream Yu snapshot is not redistributed, so the packaged campaign records upstream byte identity as not verified. See [`YU2025_BASELINE_INTEGRATION.md`](YU2025_BASELINE_INTEGRATION.md) for the full adaptation and fairness audit.
