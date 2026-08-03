# Two-dimensional Weak-Pareto example

This directory contains the example used for the paper's two-dimensional extension. It reuses `gaussian_test_matrix` and `caputo_l1_adjoint_tests` from the parent supplementary package and adds a coordinate-direction label to each linear candidate.

## Model class

```text
C_0 D_t^alpha u = sum_j xi_j X_beta_j^(d_j) u,   d_j in {x,y}.
```

The reported example is linear (`p_j=0`), periodic, and defined on a dense uniform rectangular grid. It demonstrates direction-aware continuous-order encoding; it does not claim sparse-data or nonlinear two-dimensional discovery.

## Reproduction

From this directory:

```bash
./run_all.sh publication       # five seed shards
./run_all.sh publication 1     # serial fallback
./run_all.sh smoke             # small integration check
```

Publication settings are a `90 x 80 x 80` field grid, five seeds, multiplicative-uniform noise levels `0/1/5/10/20%`, `c_max=4`, differential-evolution budget `7 x 24`, ridge parameter `1e-3`, 47 temporal and 59 spatial order nodes, and a 0.25 validation fraction. The one-dimensional paper rules are applied independently to both spatial axes. They give `30 x 40 x 40 = 48,000` overlapping tensor-product weak equations. These are integral measurements constructed from the same field, not independent observations.

Spatial widths supplied on the command line are fractions of the full periodic domain length and are converted once to the absolute width expected by the common Gaussian-test constructor. Temporal widths use the parent package's absolute paper rule. Boundary trimming is intentionally absent because it is a pointwise-feature control and would alter the weak Caputo target.

The separable contractions are evaluated without assembling a dense spatial Kronecker matrix. Repeated objective evaluations use exact Gram tables for the linearly interpolated order banks. The launcher defaults to one BLAS thread per seed shard to avoid oversubscription; set `WEAK_PARETO_BLAS_THREADS` only when deliberately changing that policy.

## Benchmarks and archived evidence

- **A:** two-term anisotropic directional diffusion.
- **B:** x-advection plus distinct x- and y-direction fractional diffusion terms.
- `reference_results/`: the independently reproduced 70-run archive used in the manuscript.
- `resolution_112/`: the `90 x 112 x 112` grid-stability diagnostic.
- `verification_refinement/`: independent L1 temporal-residual refinement.

The data generator propagates each active Fourier mode with a complex-capable Mittag--Leffler evaluator. Discovery instead evaluates the temporal target through the transposed L1 matrix, so generation and discovery do not share a temporal discretisation.

## Files

- `generate_2d_benchmarks.py`: semi-analytic data generation and independent L1 verification.
- `weak_pareto_2d.py`: tensor-product weak library, direction-aware search, elbow selection, and exact-order refit.
- `run_2d_experiments.py`: deterministic driver, optimal within-direction matching, and duplicate-safe summarisation.
- `run_all.sh`: reproduction launcher.
- `section_2d_extension.tex`: exact manuscript excerpt for the 2D main-text and appendix material.
