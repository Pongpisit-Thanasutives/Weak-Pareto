# Weak-Pareto: weak-form fractional differential equation discovery

**Companion article:** *Robust data-driven discovery of fractional differential equations via weak formulations and Pareto-based subset selection*  
**Target journal:** *Nonlinear Dynamics*  
**Authors:** Pongpisit Thanasutives and Yoshinobu Kawahara  
**Corresponding author:** Pongpisit Thanasutives (<pongpisit.thanasutives@riken.jp>)  
**Repository:** https://github.com/Pongpisit-Thanasutives/Weak-Pareto

> **Final verified release.** The temporal Caputo search treats
> $0<\alpha<1$, the exact integer mode $\alpha=1$, and $1<\alpha<2$ as
> separate operator modes, and interpolation never crosses an integer. The
> bundled reference outputs were regenerated after the normalized Pareto-metric
> and endpoint-excluded periodic-domain corrections and are the outputs used in
> the final manuscript.

This repository contains the audited branch-aware code for one central contribution:

> **Proposed method:** `weak_pareto` = weak-form fractional candidate library + Pareto-based best-subset discovery.

The baselines are deliberately chosen to isolate what matters:

| Method name | Role | Candidate library | Selector |
|---|---|---|---|
| `weak_pareto` | **Proposed method** | weak fractional adjoint library | Pareto-based best-subset search |
| `vanilla_pareto` | baseline | vanilla/pointwise derivative library | Pareto-based best-subset search |
| `weak_grid_stridge` | baseline | weak fractional adjoint library | alpha-grid STRidge |
| `weak_fixed_stability` | fixed-grid ablation | weak fractional adjoint library | repeated STRidge stability selection |

The contemporary Yu et al. neural fractional-discovery comparison adds two separately reported methods:

| Method name | Role | Reconstruction | Fractional/sparse discovery |
|---|---|---|---|
| `yu2025_full` | contemporary end-to-end neural fractional-discovery framework | coordinate neural network | Gauss--Jacobi + STRidge + differential evolution |
| `yu2025_optimizer_only` | component ablation | tensor-product quintic spline | Gauss--Jacobi + STRidge + differential evolution |

`weak_fixed_stability` is included only as a fixed-candidate-library ablation. It is **not** the proposed method because it does not perform continuous-order Pareto-based best-subset search. `yu2025_optimizer_only` is likewise not the complete Yu method; it isolates the sparse/global-optimization component from neural reconstruction.

## Data loading from notebooks

The benchmark loaders are robust to notebook working directories. You may run notebooks from `notebooks/` or the project root; calls such as

```python
from dataset_configs import benchmark_spec
spec = benchmark_spec("paper_FADE_tsfade_fft", profile="paper", noise_percent=0.0, seed=0)
```

will resolve the bundled `data/` directory automatically.

## Start here

- **Shortest tutorial:** [`examples/tutorial_weak_pareto.py`](examples/tutorial_weak_pareto.py)
- **Matching notebook:** [`notebooks/05_end_to_end_weak_pareto_tutorial.ipynb`](notebooks/05_end_to_end_weak_pareto_tutorial.ipynb)
- **Paper result-to-script map:** [`docs/PAPER_RESULTS_GUIDE.md`](docs/PAPER_RESULTS_GUIDE.md)
- **Public API and documentation policy:** [`docs/CODE_DOCUMENTATION.md`](docs/CODE_DOCUMENTATION.md)
- **Full reproduction commands:** [`reproduce/README.md`](reproduce/README.md)

Run the concise tutorial with:

```bash
python examples/tutorial_weak_pareto.py
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

To include the neural fractional-discovery framework, install:

```bash
pip install -r requirements_yu2025.txt
```

Run a fast end-to-end sanity check:

```bash
FPDE_QUICK=1 bash scripts/run_publication_benchmarks.sh
```

Run the paper-oriented benchmark:

```bash
bash scripts/run_publication_benchmarks.sh
```

The main result table is written to:

```text
results/publication_<timestamp>/COMBINED_SUMMARY.md
```

## Two-dimensional directional example

The supplementary package includes a self-contained linear 2D extension under [`two_dimensional/`](two_dimensional/). Each candidate term carries a coordinate label `x` or `y`; the implementation uses tensor-product weak rows and coordinate-specific spectral adjoints without forming a dense Kronecker matrix. Reproduce the two reported benchmarks with:

```bash
cd two_dimensional
./run_all.sh publication
```

The reported `90 x 80 x 80` grid produces 48,000 overlapping weak equations under the inherited paper rule. A `90 x 112 x 112` diagnostic is archived separately. The extension is deliberately limited to dense, periodic, uniform-grid, linear problems; sparse and nonlinear multidimensional discovery are not claimed.

## Yu et al. neural fractional-discovery comparison

Run the full CPU/GPU-aware comparison workflow with:

```bash
FPDE_QUICK=1 FPDE_DEVICE=cpu bash scripts/run_yu2025_comparison.sh
```

Remove `FPDE_QUICK=1` for the publication profiles, or set `FPDE_DEVICE=cuda` to run the PyTorch reconstruction/automatic-differentiation portion on CUDA. The default full run evaluates the FADE benchmark at noise levels `0 1 5` over seeds `0 1 2 3 4`.

The complete method audit, adaptations, operator-realisation differences, metrics, and environment overrides are documented in [`docs/YU2025_BASELINE_INTEGRATION.md`](docs/YU2025_BASELINE_INTEGRATION.md).

## Loading one dataset by name

For notebooks and small experiments, use the singular helper:

```python
from dataset_configs import benchmark_spec

spec = benchmark_spec("synthetic_space_fractional_RD", profile="notebook", noise_percent=0.1)
data = spec["data"]
config = spec["config"]
truth = spec["truth_spec"]
```

Use `benchmark_specs(...)` only when intentionally looping over many datasets in scripts.

## Learning notebooks

Start with `notebooks/00_notebook_map.ipynb`. The shortest end-to-end tutorial is `notebooks/05_end_to_end_weak_pareto_tutorial.ipynb`; Notebook 01 provides the more detailed candidate-library comparison. It manually compares vanilla library construction, weak library construction, and the proposed weak-library Pareto-DE method before calling the high-level scripts.

Every tutorial notebook has an editable first control cell, e.g.

```python
dataset_name = "synthetic_time_space_fractional_RD"
noise_percent = 0.5
profile = "paper"
seed = 0
maxiter_override = None
popsize_override = None
```

With the same controls, the notebook and `scripts/run_all_methods.py` use the same `benchmark_spec(...)` configuration and the same method dispatcher, so their selected equations and summary metrics should be consistent.

## Important scripts

| Script | Purpose |
|---|---|
| `scripts/run_all_methods.py` | run selected methods on selected datasets/noise levels |
| `scripts/run_publication_benchmarks.sh` | one-command paper benchmark workflow |
| `scripts/find_noise_tolerance.py` | sweep noise levels and find the highest tolerated level for the proposed method |
| `scripts/summarize_publication_results.py` | summarize benchmark CSV outputs |
| `scripts/export_canonical_configs.py` | export the one canonical config per dataset |
| `scripts/run_yu2025_comparison.py` | common evaluator for Weak-Pareto and both Yu variants |
| `scripts/run_yu2025_comparison.sh` | one-command CPU/GPU-aware neural-baseline workflow |

## Branch-aware temporal-order search

The Caputo operator changes definition when $n=\lceil\alpha\rceil$ changes.
Accordingly, the temporal search represents three modes explicitly:

- `fractional_subunit`: $0<\alpha<1$;
- `integer`: the exact candidate $\alpha=1$, evaluated without interpolation;
- `fractional_superunit`: $1<\alpha<2$.

For a declared interval that crosses one, the two fractional branches are
optimised separately on bounds that stop at `1 +/- alpha_branch_epsilon`
(default `1e-3`), and their minima are compared with the exact integer
candidate. The precomputed feature bank is never interpolated across one. The
selected `alpha_mode` is serialised in JSON/CSV outputs and is held fixed during
exact-order polishing. Whenever a declared interval extends above one, the superunit branch is
searched and its minimum competes with the subunit and exact-integer modes under
the same variance-normalised validation objective. A separate semi-analytic,
fixed-support experiment with true order $\alpha=1.65$ reports branch/order
recovery at 0%, 0.5%, and 1% noise; its final outputs are archived under
`reproduce/reference_results/superunit_final/`. The weak protocol uses the
transposed Caputo--L1 matrix on the temporal tests (`time_form="derivative"`).
For superunit orders this matrix is `L1(alpha-1) @ D1`; the transpose retains an
endpoint-dominated opposite-sign pair and therefore treats the initial rate
implicitly. This diagnostic does not claim unrestricted support discovery. Spatial spectral and Grünwald order families
remain continuous in their order and therefore do not use this temporal branch
machinery.

## Reproducibility policy

Each dataset has one canonical candidate search space in `dataset_configs.py` for the internal methods. Runtime profiles (`notebook` and `paper`) change compute budget, not the internal mathematical hypothesis class. The Yu framework variants retain Yu's published 12-column candidate library; this difference is recorded because the contemporary framework comparison is end-to-end rather than a selector-only ablation.

The benchmark outputs include `truth_spec.json`, `config_search_space.json`, `run_tracking.json`, and `method_comparison.csv` so every selected equation can be audited. The Yu comparison additionally stores clean-input hashes, the exact shared noisy field and its hash, adaptation metadata, checkpoints, and runtime phases.

Selection-stage fields retain fixed meanings throughout serialization: `train_rel_mse` and `val_rel_mse` are variance-normalised training and validation MSE, and `objective` is normally `log10(val_rel_mse)` (plus any declared search penalties). The final exact-order/full-data residual is stored separately as `full_data_rel_l2`; it does not overwrite validation, objective, AIC-type, or BIC-type fields. Pareto dominance, sparse-relaxed comparison, elbow selection, and stopping use `val_rel_mse`.

The binary operator-structure metric uses a predefined absolute order tolerance of `0.15`, together with exact temporal-mode and identity-term checks. This is an evaluation convention, not an optimizer input or a statistically fitted cutoff. Its sensitivity around the reported value is archived under `reproduce/reference_results/operator_tolerance_sensitivity/`.


## Candidate library scope

The manuscript reproduction configuration uses an overcomplete RHS class with terms `u^p D_x^beta u`, `p_values=(0,1,2)`, and `cmax=4`. This is broader than the true bundled linear two-term equations, so the proposed method must reject competing nonlinear candidates. For a heavier stress test, pass `--cmax 5` or `FPDE_CMAX=5`; this is slower and may require larger DE budgets.

### Post-selection inactive-term pruning

The manuscript candidate library intentionally includes competing nonlinear candidates through `p_values=(0,1,2)` and allows support sizes through `cmax=4`. In clean or low-noise data, a higher-cardinality model may occasionally lower validation error by a tiny numerical amount while giving the extra term negligible fitted RHS contribution. The reported selected equation therefore applies a non-oracle cleanup step: after Pareto-DE selection, term `j` is removed when its contribution ratio `||xi_j theta_j||_2 / ||Theta xi||_2` is below `selected_contribution_prune_tol` or when its coefficient is numerically zero. The remaining structure is refit by the same optimizer. This is applied equally to `weak_pareto` and `vanilla_pareto`; `support_size_progress.csv` still records the unpruned best equation at each support size.

## Blended additions and current mathematical status

An earlier independent weak-form derivation was compared against this package; the operator-level checks are documented in [`docs/WEAK_FORM_COMPARISON.md`](docs/WEAK_FORM_COMPARISON.md). Those checks support the weak adjoint formulas themselves.

The later presubmission review identified a distinct hypothesis-encoding problem for periodic signed-Riesz reaction--diffusion cases: the identity candidate at `beta=0` was interpolated in the same continuous bank as positive signed-Riesz orders, although their implemented zero-order limits are not the same. This has since been corrected by separating operator type from continuous derivative order (the discovery snaps any order below the first strictly-positive grid node to exact identity, and the recovery metric is mode-aware). The final campaign was regenerated after both this operator-mode correction and the normalized Pareto-metric and endpoint-excluded periodic-domain corrections. See [`RESULTS_STATUS.md`](RESULTS_STATUS.md) and [`docs/YU2025_BASELINE_INTEGRATION.md`](docs/YU2025_BASELINE_INTEGRATION.md).

Three additive, non-breaking extensions were merged into
`fractional_weak_form.py`:

- `compact_bump_test_matrix` — compactly supported test functions that zero the
  weak boundary functional exactly (for short records / non-periodic domains).
- `periodic_riesz_feller_adjoint_on_tests` — skewed Riesz–Feller operator for
  asymmetric stable/Lévy transport (append `theta` to the DE vector to search it).
- `refine_orders_local` — optional per-support-size local polish of the
  fractional orders on the smooth weak residual, after Pareto-DE.

Reproducibility scripts: `examples/compare_weak_temporal_targets.py`,
`examples/compare_test_functions.py`, `examples/demo_riesz_feller_skew.py`,
`examples/demo_order_polish.py`.


## Mathematical verification diagnostics

The supplement includes focused checks motivated by the final mathematical audit:

```bash
PYTHONPATH=. python reproduce/check_operator_convergence.py
PYTHONPATH=. python reproduce/check_nonlinear_bias.py
PYTHONPATH=. python reproduce/profile_temporal_order.py
```

The first checks the periodic Riesz spectral pair and Caputo--L1 convergence on an analytic polynomial. The second verifies both additive-Gaussian and multiplicative-uniform trace-bias formulas for a nonlinear data-weighted weak feature. The third is an oracle diagnostic of temporal-target bias on the two Riesz benchmarks; it profiles clean, fully noisy, clean-initial-slice, target-only-noise, and library-only-noise arms while respecting the temporal operator mode. It is not part of model selection. The final publication outputs are stored under `reproduce/reference_results/branch_aware_campaign/`; the controlled Yu-framework rows are under `reproduce/reference_results/yu_framework_comparison/`; and the reported superunit branch/order experiment is under `reproduce/reference_results/superunit_final/`. Diagnostic outputs and frozen-soil reference fits are stored alongside them.

The elbow selector uses a signed distance above the error--complexity chord, so a convex anti-elbow below the chord cannot be selected.

## Additive-Gaussian and reduced-sampling checks

The publication supplement includes a compact robustness runner for FADE and
fractional Burgers:

```bash
PYTHONPATH=. python scripts/run_alternative_robustness.py \
  --conditions additive_gaussian \
  --noise 10 \
  --seeds 0 1 2 3 4 \
  --restart \
  --outdir results/alternative_robustness
```

The same runner supports `half_time_sampling` (every second time snapshot, no
imputation) and `half_time_gaussian`. It writes each completed run immediately
to JSONL so a long local run can be resumed. The final five-seed additive-Gaussian results are archived under `reproduce/reference_results/branch_aware_campaign/additive_gaussian/`.

## Experimental frozen-soil creep

`real_data/frozen_soil_creep_weak.py` identifies the order and material
parameters of a fractional Kelvin model from published clay and silt creep
records using a smoothing fractional-integral equation rather than pointwise
fractional differentiation. See `external_data/frozen_soil/README.md` for data
provenance and acquisition instructions. The example is explicitly an
order-and-parameter fit within the physically motivated Kelvin support, not an
unrestricted structure-discovery benchmark.

## Optimized full branch-aware rerun

For the publication campaign, use the resumable orchestrator:

```bash
bash run_branch_aware_campaign.sh results/normalized_metric_final 5
```

The final argument is the number of independent worker processes. Choose a value appropriate for the available memory and CPU; the reported campaign used five. The reproduction cache is stored under
`results/normalized_metric_final/main/.discovery_cache`; do not delete it while the
campaign is incomplete. See `EFFICIENCY_NOTES.md` for the audit and safeguards.

## GitHub release checks

```bash
python scripts/check_documentation.py
python -m compileall -q .
pytest -q
python examples/tutorial_weak_pareto.py
```

