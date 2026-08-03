# Numerical-results status

The normalized-metric publication campaign and the controlled Yu-framework campaign have been completed and reconciled with the final manuscript. The bundled aggregate reference outputs are current submission evidence, not smoke-test output.

## Reference locations

- `reproduce/reference_results/branch_aware_campaign/`: final internal tables,   figures, additive-Gaussian results, Weak-Pareto Yu-framework-comparison rows, and   sanitized environment records.
- `reproduce/reference_results/yu_framework_comparison/`: final adapted neural fractional-discovery framework and optimiser-only per-run comparison tables, summary, manifest, and log.
- `reproduce/reference_results/diagnostics/`: operator, nonlinear-bias, temporal-target, and smoke diagnostics.
- `reproduce/reference_results/superunit_final/`: final five-seed fixed-support superunit branch/order experiment reported in the manuscript.
- `reproduce/reference_results/frozen_soil/`: derived Kelvin-model fit outputs.
- `two_dimensional/reference_results/`: independently reproduced 70-run 2D campaign and window sensitivity.
- `two_dimensional/resolution_112/results/`: independently reproduced 25-run spatial-grid diagnostic.

A rerun is not required to use the reported manuscript values. To reproduce the campaign independently, run from the package root:

```bash
bash run_branch_aware_campaign.sh results/normalized_metric_final 5
```

The final argument controls independent worker processes. The internal campaign alone also accepts `--jobs N`, `--jobs=N`, `-j N`, or `FPDE_REPRO_JOBS=N`.

## Superunit branch experiment

The completed five-seed fixed-support experiment is archived under
`reproduce/reference_results/superunit_final/` and is reported in the manuscript.
It uses semi-analytic data for `D_t^1.65 u = 0.12 D_x^2 u`, fixes one linear
support term, and discovers the temporal branch and the temporal/spatial orders.
Weak-Pareto gives 5/5 complete operator recovery at 0% and 0.5% noise; at 1%
noise it retains 5/5 superunit-branch recovery but only 1/5 complete operator
recovery. The matched strong-form comparator gives 5/5 clean recovery and 0/5
branch recovery at both noisy levels. This protocol is a branch/order diagnostic,
not unrestricted support discovery. The active weak superunit target uses the
composition `L1(alpha-1) @ D1`; after transposition, its temporal weights are
strongly concentrated on the first two samples. This is documented as an
implicit initial-rate treatment with endpoint-sensitive noise amplification.
The archived numerical rows are unchanged because this revision corrects the
interpretation and metadata rather than the implemented operator.

## Two-dimensional directional example

The final archive contains two linear periodic benchmarks on `90 x 80 x 80` data. Both Benchmark A and Benchmark B recover the correct support, coordinate directions, and derivative orders in all 25 noise--seed combinations through 20% multiplicative noise. The paper rule gives 48,000 overlapping weak equations. Benchmark A also retains 25/25 operator recovery on a `90 x 112 x 112` grid with 94,080 weak equations. The refined grid improves the 20% coefficient error but not every order error, so this is reported as a grid-stability diagnostic rather than a monotone convergence result. Broad spatial windows (`0.16L` and `0.24L`) fail the three-term Benchmark B support test, while the inherited paper rule and `0.10L` recover 5/5.
