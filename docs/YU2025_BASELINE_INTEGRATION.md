# Yu et al. neural fractional-discovery framework integration and comparison scope

## Source status and provenance

The upstream Yu et al. source snapshot is **not included** in this supplementary archive because no clear redistribution licence was available. The package contains a newly written adapter, `external_baselines/yu2025/yu_baseline.py`, and a comparison runner, `scripts/run_yu2025_comparison.py`.

Because the upstream snapshot is absent, the packaged campaign cannot independently verify byte identity against an upstream file. The provenance manifest therefore records `yu_original_sha256: null`, `byte_identical: false`, `upstream_snapshot_included: false`, and `upstream_byte_identity_verified: false`. In this context, `byte_identical: false` means **not verified**, not a demonstrated mismatch. No undocumented prior independent check is presented as evidence.

The clean field actually consumed by every packaged method, every noisy realisation, and its SHA-256 digest are retained in the run directory. This verifies within-campaign input sharing, but not identity with an absent upstream snapshot.

## What is compared

The declared contemporary neural fractional-discovery framework comparison is limited to FADE. Weak-Pareto and the Yu adapter share:

- the packaged clean field;
- the nominal governing equation;
- the target temporal and spatial fractional orders;
- the same multiplicative-uniform noisy field for each seed;
- a frozen train/validation split for each run; and
- the same post hoc recovery definitions.

The numerical operator realisations are **different** and are not called mathematically compatible or identical:

- Weak-Pareto uses the repository's periodic directional spectral convention;
- the Yu adapter retains a one-sided finite-terminal fractional approximation.

The comparison is therefore an end-to-end, method-faithful comparison on shared data, not an operator-identical ablation. The periodic signed-Riesz reaction--diffusion benchmarks are outside this scope and are recorded as `skipped_operator_scope` rather than silently reinterpreted.

## Full framework and optimiser-only variant

The runner keeps two Yu variants distinct:

1. `yu2025_full`: neural field reconstruction, automatic differentiation, pointwise fractional features, STRidge, and differential-evolution order search;
2. `yu2025_optimizer_only`: a deterministic tensor-product spline replaces the neural reconstruction, while the Yu fractional-feature and selection stages remain.

The optimiser-only run is not described as the full Yu method. Runtime fields cover the stages executed by each variant. The final manuscript compares them with Weak-Pareto only after both campaigns were run sequentially on the same Apple M4 Pro CPU; the resulting wall-clock values are directly comparable for these tested implementations and configurations, but are not theoretical complexity measures.

## Adapter corrections and reproducibility controls

The adapter applies documented controls needed for a reproducible comparison:

- deterministic per-seed train/validation splits;
- a split frozen across optimiser evaluations;
- training-only coefficient and STRidge-penalty estimation;
- validation scoring only on held-out rows;
- recorded optimisation budgets and seeds; and
- explicit status and reason fields for skipped or failed runs.

These changes prevent validation-target leakage and split drift. They also mean the adapter is not claimed to be bit-for-bit identical to an upstream implementation.

## Recovery and coefficient-error definitions

Support/power recovery requires the selected number of terms and powers to match the truth after pruning. Operator-structure recovery additionally requires the declared temporal and spatial order tolerances and the correct operator modes.

Order and coefficient errors are conditioned on **support/power recovery**. Complete fractional-order recovery is not required before coefficient error is computed. This definition matches `reproduce/_repro_common.py::matched_errors`, the comparison runner, and the manuscript. Operator-structure recovery remains a separate reported column.

## Weak-Pareto selection and final-fit fields

Weak-Pareto uses the variance-normalised held-out score `val_rel_mse` for its differential-evolution objective, Pareto dominance, sparse-relaxed comparison, elbow selection, and stopping logic. Selection quantities retain their meanings after exact-order refitting:

- `train_rel_mse`: variance-normalised training MSE;
- `val_rel_mse`: variance-normalised validation MSE;
- `objective`: normally `log10(val_rel_mse)`, with any declared search penalties;
- `full_data_rel_l2`: final full-data relative L2 residual, reported as $\mathcal E_{\mathrm{fit}}$.

The exact-order step evaluates operators directly at polished orders, removing interpolation error from the final conditional refit. The selected support, temporal mode, and optimisation basin can nevertheless still depend on the order grid and search trajectory.

## Running the comparison

From the supplementary-code root:

```bash
FPDE_DEVICE=cpu bash scripts/run_yu2025_comparison.sh
```

For the complete branch-aware campaign, use `bash run_branch_aware_campaign.sh results/normalized_metric_final N`, where `N` is the desired worker count. The quick profile is a software check only and must not replace publication-profile runs.

## Final packaged outputs

The final Weak-Pareto rows are stored in `reproduce/reference_results/branch_aware_campaign/yu_weak/`. The adapted neural fractional-discovery framework and optimiser-only rows reproduced under the shared CPU environment are stored in `reproduce/reference_results/yu_framework_comparison/`. Both directories retain per-run noise-realisation hashes and the explicit unverified-upstream provenance status.

## Citation

X. Yu et al., “A data-driven framework for discovering fractional differential equations in complex systems,” *Nonlinear Dynamics*, 113, 24557–24577 (2025). DOI: 10.1007/s11071-025-11373-z.
