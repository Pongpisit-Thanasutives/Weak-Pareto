# API reference for the paper code

This document explains the main classes/functions used by the proposed method and baselines.


## Dataset helpers in `dataset_configs.py`

Use these functions to load benchmark data and the matching canonical config.

- `benchmark_spec(dataset_name, ...)`: **recommended notebook API**. Returns one dictionary for exactly one dataset name. The dictionary contains `data`, `config`, `truth_spec`, expected structure fields, and a config fingerprint.
- `load_benchmark(dataset_name, ...)`: tuple-oriented wrapper returning `(data, config, truth_spec)`.
- `benchmark_specs(dataset_names=[...], ...)`: plural helper for scripts that intentionally loop over many datasets. Omit `dataset_names` to run all loadable packaged benchmarks.
- `available_benchmark_dataset_names(...)`: returns the dataset names loadable in the current checkout.

Example:

```python
from dataset_configs import benchmark_spec

spec = benchmark_spec(
    "synthetic_space_fractional_RD",
    profile="notebook",
    noise_percent=0.1,
    seed=0,
)
data = spec["data"]
config = spec["config"]
truth = spec["truth_spec"]
```

This replaces the older pattern of calling `benchmark_specs(...)` and filtering a list by hand.

## `DiscoveryConfig` in `pareto_fde_discovery.py`

Holds the canonical search space and runtime budget for one discovery run.

Important fields:

- `alpha_grid`: declared temporal-order grid. With `branch_aware_time=True`, it is partitioned into mode-confined banks and never interpolated across `alpha=1`.
- `branch_aware_time`: enable separate subunit, exact-integer, and superunit Caputo modes (default `True`).
- `alpha_branch_epsilon`: numerical gap between a fractional branch and the exact integer candidate (default `1e-3`).
- `beta_grid`: spatial derivative orders used for interpolation/search.
- `cmax`: maximum number of RHS terms in the best-subset sweep.
- `p_values`: allowed powers in `u**p D_x^beta u`. The canonical paper config uses `(0,1,2)`; `p=0` gives a linear derivative term, `p=1` gives `u D_x^beta u`, and `p=2` gives `u^2 D_x^beta u`.
- `backend`: derivative backend used by the vanilla library.
- `maxiter`, `popsize`: differential-evolution budget.
- `selection`: Pareto-front model-selection rule.

Every method receives the same dataset-specific `DiscoveryConfig`.

## `PDEModel` in `pareto_fde_discovery.py`

Container for one discovered equation:

```text
D_t^alpha u = sum_j xi_j u**p_j D_x^beta_j u
```

Important methods:

- `canonicalized()`: sorts terms so equivalent equations have stable output order.
- `equation()`: returns a readable equation string.
- `to_dict()`: serializes the model for JSON/CSV tracking.

Metric fields keep distinct meanings:

- `train_rel_mse`: variance-normalised training MSE from model selection;
- `val_rel_mse`: variance-normalised held-out MSE used by the DE objective, Pareto dominance, elbow/sparse-relaxed selection, and stopping;
- `objective`: normally `log10(val_rel_mse)` plus declared search penalties;
- `aic`, `bic`: raw-MSE heuristic information-criterion fields, not conventional fixed-response likelihood comparisons across different temporal orders;
- `full_data_rel_l2`: the final exact-order/full-data relative L2 residual reported as `E_fit`.

Exact-order refitting updates the final coefficients and `full_data_rel_l2` but preserves the selection-stage fields. Direct evaluation removes interpolation error from the final conditional refit; the selected support, temporal mode, and optimisation basin can still depend on the order grid and search trajectory.

## `FractionalFeatureBank` in `pareto_fde_discovery.py`

The vanilla/strong-form candidate library.

- `target(alpha)`: pointwise estimate of `D_t^alpha u`.
- `spatial(beta)`: pointwise estimate of `D_x^beta u`.
- `library(p_tuple, beta_tuple)`: strong-form RHS matrix.

This is used by the `vanilla_pareto` baseline.

## `FractionalOperatorSpec` in `fractional_weak_form.py`

Describes a fractional operator used in a weak-form feature.

Important fields:

- `kind`: `caputo`, `riemann_liouville`, `grunwald_letnikov`, `riesz`, `integer`, or `identity`.
- `order`: fractional/integer derivative order.
- `axis`: `t` or `x`.
- `side`: `left`, `right`, or `symmetric` for one-sided operators.

## `SeparableWeakLibrary2D` in `fractional_weak_form.py`

Low-level tensor-product weak integral assembler. Given time tests `rho_k(t)` and space tests `psi_l(x)`, it builds rows of the form

```text
int int U(t,x) rho_k(t) psi_l(x) dx dt
```

and weak operator features such as

```text
int int U(t,x) rho_k(t) (D_x^beta)^* psi_l(x) dx dt.
```

## `WeakFractionalFeatureBank` in `weak_pareto_fde_discovery.py`

Weak replacement for the vanilla feature bank.

- `target(alpha)`: weak projection of the temporal fractional derivative.
- `spatial_feature(p, beta)`: weak feature for `u**p D_x^beta u`.
- `library(p_tuple, beta_tuple)`: weak RHS matrix for a proposed support.

This is the library used by the proposed method.

## Proposed method: `weak_pareto`

Location: `weak_pareto_fde_discovery.py`

The paper's proposed method is:

```text
weak candidate library + best-subset Pareto-DE
```

For scripts, call the one-line wrapper:

```python
summary = run_weak_pareto_discovery(data, config, output_dir=...)
```

The publication default is `time_form="derivative"`, which transfers the
branch-specific discrete Caputo--L1 matrix to the temporal tests by transpose.
For a superunit order, the matrix is `L1(alpha-1) @ D1`; its transpose retains
an endpoint-dominated opposite-sign pair, so the initial-rate dependence is
implicit rather than absent. The optional `time_form="caputo_integral"` is a
separate Volterra residual and is not used by the reported superunit experiment.

For teaching/debugging notebooks, the same method can be split into two explicit stages:

```python
bank = build_weak_candidate_library(data, config, test_budget="paper")
problem = build_best_subset_pareto_problem(bank, config)
summary = run_best_subset_pareto_de(problem, config, data=data, output_dir=...)
```

The split version and wrapper are equivalent.  The output summary contains:

- `support_size_progress`: best equation found for each support size `c`;
- `best_by_c`: machine-readable best `PDEModel` per support size;
- `pareto`: non-dominated support-size candidates;
- `selected`: final selected equation.

When `output_dir` is provided, the proposed method writes `support_size_progress.csv` and `support_size_progress.json` so you can track the progress of the best-subset sweep.

Command-line name:

```text
weak_pareto
```

## Baseline: `run_pareto_discovery(...)`

Location: `pareto_fde_discovery.py`

This is the vanilla-library Pareto baseline:

```text
vanilla candidate library + best-subset Pareto-DE
```

Command-line name:

```text
vanilla_pareto
```

## Baseline: `weak_grid_stridge_baseline(...)`

Location: `baselines.py`

This baseline keeps the weak candidate library but replaces Pareto-DE by alpha-grid search plus STRidge over the fixed beta grid.

Command-line name:

```text
weak_grid_stridge
```

## Baseline/ablation: `weak_fixed_library_stability_baseline(...)`

Location: `baselines.py`

This fixed-library ablation repeats weak Grid-STRidge over weak-test scales/splits and reports the most stable fixed-grid support.

Command-line name:

```text
weak_fixed_stability
```

It is only meaningful for fixed candidate libraries. It is not the proposed continuous-order Pareto-DE method.


## Notebook/script consistency helpers in `scripts/run_all_methods.py`

- `run_single_method(method, data, config, output_dir, ...)`: public dispatcher used by notebooks to run exactly one method through the same backend as the command-line script.
- `selected_model_dict(method, result)`: converts method-specific result objects into the same selected-model dictionary schema used in `method_comparison.csv`.
- `run_all_methods(args)`: batch runner used by `scripts/run_all_methods.py` and by notebook 02 for script-equivalent runs.

Notebook 01 uses these helpers so that manually executed notebook cases and command-line benchmark cases are consistent when the top-level controls match.

## Scripts

- `scripts/run_all_methods.py`: run any subset of methods on any subset of datasets/noise levels.
- `scripts/run_publication_benchmarks.sh`: one-command paper benchmark workflow.
- `scripts/find_noise_tolerance.py`: find the highest tested noise level tolerated by the proposed method.
- `scripts/summarize_publication_results.py`: summarize method-comparison CSVs.
- `scripts/export_canonical_configs.py`: export canonical dataset configurations.


CLI overrides: `scripts/run_all_methods.py` accepts `--cmax` and `--p-values` for ablations. The publication script accepts `FPDE_CMAX=5` and `FPDE_P_VALUES="0 1 2"` as environment overrides. Omit them for the canonical paper config.

### Post-selection inactive-term pruning

The paper candidate library intentionally includes competing nonlinear candidates through `p_values=(0,1,2)` and allows an extra support size through `cmax=4`. In clean or low-noise data, a higher-cardinality model may occasionally lower validation error by a tiny numerical amount while giving the extra term negligible fitted RHS contribution. The reported selected equation therefore applies a non-oracle cleanup step: after Pareto-DE selection, term `j` is removed when its contribution ratio `||xi_j theta_j||_2 / ||Theta xi||_2` is below `selected_contribution_prune_tol` or when its coefficient is numerically zero. The remaining structure is refit by the same optimizer. This is applied equally to `weak_pareto` and `vanilla_pareto`; `support_size_progress.csv` still records the unpruned best equation at each support size.

## Temporal mode metadata

`PDEModel.alpha_mode` is one of `fractional_subunit`, `integer`, or
`fractional_superunit`. It is serialised to JSON and CSV outputs and preserved
during exact-order polishing. A fractional value close to one is not equivalent
to the exact integer mode.

## Superunit Caputo diagnostic

`fpde_datasets.make_superunit_fractional_diffusion(...)` generates a
semi-analytic periodic field satisfying `D_t^1.65 u = 0.12 D_x^2 u` with zero
initial velocity.  The time dependence is evaluated with the Mittag--Leffler
series rather than the L1 scheme used by the discovery evaluator.

The diagnostic is registered as
`synthetic_superunit_fractional_diffusion`.  Its canonical candidate config is
broad, but `reproduce/run_superunit_experiment.py` deliberately fixes one
linear RHS term in order to isolate temporal-branch and continuous-order
recovery.  Use the resulting outputs only as a branch/order diagnostic, not as
an unrestricted support-discovery benchmark.
