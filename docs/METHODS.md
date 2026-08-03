# Methods used in the paper

## Branch-aware Caputo time orders

The Caputo family is separated into `0<alpha<1`, the exact `alpha=1`
integer operator, and `1<alpha<2`. Each admitted mode is optimised separately.
Feature interpolation and local polishing remain within the selected mode, with
`alpha_branch_epsilon=1e-3` separating fractional bounds from the exact integer
candidate. The manuscript's empirical validation is restricted to `0<alpha<=1`.
This branch mechanism is temporal and Caputo-specific; the spatial order
families used in the benchmarks vary continuously with order.

## Proposed method: `weak_pareto`

The proposed method is:

```text
weak fractional candidate library + best-subset Pareto-DE
```

The model class is

\[
D_t^\alpha u = \sum_{j=1}^c \xi_j u^{p_j}D_x^{\beta_j}u .
\]

The key change from the original vanilla method is the candidate library. Instead of computing noisy pointwise columns such as `D_x^beta u`, the weak library uses adjoint weak features:

\[
\langle D_x^\beta u,\phi\rangle = \langle u,(D_x^\beta)^*\phi\rangle .
\]

For nonlinear terms already present in the original model class,

\[
\langle u^pD_x^\beta u,\phi\rangle
=
\langle D_x^\beta u,u^p\phi\rangle
=
\langle u,(D_x^\beta)^*(u^p\phi)\rangle .
\]

For periodic FFT-generated directional benchmarks, the weak adjoint is matched to the same spectral operator used by the vanilla backend. If the model-side multiplier is \((ik)^\beta\), then the test-side adjoint uses \(\overline{(ik)^\beta}\). This avoids falsely mixing a periodic FFT dataset with a finite-domain one-sided GL/RL weak operator.

The selector is unchanged in spirit: for each support size `c`, differential evolution searches continuous `alpha` and `beta` values, coefficients are fitted by least squares, and the final structure is selected from the Pareto front.

The implementation exposes both a one-line wrapper and explicit teaching steps:

```python
bank = build_weak_candidate_library(data, config)
problem = build_best_subset_pareto_problem(bank, config)
summary = run_best_subset_pareto_de(problem, config, data=data)
```

The `support_size_progress` table records the best equation found at each support size. This is the preferred diagnostic for checking how the Pareto-DE search moved from one-term to two-term models.

The variance-normalised held-out score is the principal selection quantity throughout the DE objective, Pareto front, elbow/sparse-relaxed rule, and stopping logic. The final exact-order residual is serialized separately as `full_data_rel_l2`; it does not replace `train_rel_mse`, `val_rel_mse`, `objective`, or the heuristic AIC/BIC-type fields. Direct evaluation at polished orders removes interpolation error from the final conditional refit, while support, temporal mode, and optimisation basin can still depend on the grid and search trajectory.

## Baseline 1: `vanilla_pareto`

This uses the same best-subset Pareto-DE selector as the proposed method, but the library is the original vanilla/strong-form library. This baseline isolates the benefit of weak candidate construction only in combination with the same selector; the library-only effect is isolated by the Strong Pareto vs Weak Pareto (no-polishing) ablation, while the full weak-vs-strong comparison additionally includes the exact-order polishing and final coefficient refit.

## Baseline 2: `weak_grid_stridge`

This keeps the weak candidate library but replaces Pareto-DE with a fixed alpha/beta grid and STRidge thresholding. This baseline asks whether the improvement is due only to the weak library or also to the best-subset Pareto-DE search.

## Baseline 3: `weak_fixed_stability`

This is a fixed-candidate-library stability ablation. It repeats weak Grid-STRidge over weak-test scales and splits and reports the most stable fixed-grid structure. It is useful diagnostically, but it is not the proposed method because it does not perform continuous-order best-subset Pareto-DE.

## Result-scoring convention

The benchmark output separates symbolic recovery from numerical accuracy:

- `full_structure_recovered=True` means the selected equation has the correct symbolic support/form: the expected number of active RHS terms and the expected nonlinear powers `p`. It does not require small alpha, beta, or coefficient errors.
- `alpha_abs_error` reports the absolute error in the selected temporal fractional order.
- `max_matched_beta_abs_error` and `mean_matched_beta_abs_error` report errors in the matched RHS fractional orders.
- `max_coef_rel_error` reports the largest matched relative coefficient error when benchmark coefficient truth is available.
- `structure_and_orders_recovered` additionally requires the correct temporal mode, exact identity terms, and an absolute error no greater than `0.15` for every positive temporal or spatial derivative order. This tolerance is used only for reporting; it is not passed to the optimizer.

The value `0.15` is a predefined operational criterion rather than a statistically optimal cutoff. Across the positive true orders in the reported experiments (approximately `0.8` to `2.0`), it corresponds to relative deviations of `7.5%` to `18.75%`. Rescoring the 60 noisy matched runs gives Weak-Pareto recovery totals of `45/60`, `47/60`, and `47/60` at absolute tolerances `0.125`, `0.15`, and `0.175`, while Strong Pareto remains `0/60`. Continuous order errors are therefore reported alongside the binary counts. The supporting per-run and summary files are under `reproduce/reference_results/operator_tolerance_sensitivity/`.

### Post-selection inactive-term pruning

The paper candidate library intentionally includes competing nonlinear candidates through `p_values=(0,1,2)` and allows additional support sizes through `cmax=4`. In clean or low-noise data, a higher-cardinality model may occasionally lower validation error by a tiny numerical amount while giving the extra term negligible fitted RHS contribution. The reported selected equation therefore applies a non-oracle cleanup step: after Pareto-DE selection, term `j` is removed when its contribution ratio `||xi_j theta_j||_2 / ||Theta xi||_2` is below `selected_contribution_prune_tol` or when its coefficient is numerically zero. The remaining structure is refit by the same optimizer. This is applied equally to `weak_pareto` and `vanilla_pareto`; `support_size_progress.csv` still records the unpruned best equation at each support size.

## Elbow convention

After forming the nondominated error--complexity front, support size and `-log10(validation error)` are min--max normalised. The selected interior model maximises the signed distance above the endpoint chord; if no point lies above the chord, the smallest model is retained. This prevents selection of a convex anti-elbow.
