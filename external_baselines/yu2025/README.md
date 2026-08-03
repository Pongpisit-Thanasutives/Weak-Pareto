# Adapted neural fractional-discovery framework of Yu et al.

This directory contains the newly written adapter `yu_baseline.py`. The upstream Yu et al. source snapshot is **not redistributed** because no clear redistribution licence was available.

## Provenance limitation

An optional `original/` directory may be populated locally by a user who has obtained the upstream source. It is not part of the packaged archive and the adapter does not import it. When the directory is absent, the campaign cannot verify upstream byte identity; manifests record this as not verified rather than as a verified match or mismatch.

The campaign does retain hashes for the clean field it actually uses and for every shared noisy realisation. Those hashes establish that the packaged methods consume the same within-campaign inputs.

## Declared comparison scope

The packaged comparison is run on FADE. Both pipelines share the field, nominal equation, and target fractional orders, but their operator realisations differ:

- Weak-Pareto: periodic directional spectral convention;
- Yu adapter: one-sided finite-terminal approximation.

Thus this is a method-faithful comparison on shared data, not an operator-identical ablation. Periodic signed-Riesz reaction--diffusion requests are outside scope and receive status `skipped_operator_scope`.

## Variants

- `yu2025_full` runs neural reconstruction, automatic differentiation, pointwise fractional-feature construction, STRidge, and differential-evolution order search.
- `yu2025_optimizer_only` replaces the neural reconstruction with a deterministic tensor-product spline. It is an optimiser/feature-selection diagnostic, not the full neural fractional-discovery framework.

## Reproducibility controls

The adapter uses deterministic seeds, a frozen train/validation split, training-only coefficient and STRidge-penalty fitting, held-out validation scoring, and recorded phase runtimes. These documented corrections mean the adapter is not claimed to be bit-for-bit identical to the upstream implementation.

Coefficient errors are conditioned on correct support/power recovery; complete fractional-order recovery is reported separately as operator-structure recovery and is not a prerequisite for computing coefficient error.

See `../../docs/YU2025_BASELINE_INTEGRATION.md` and `NOTICE.md` for the full audit and source-status statement.
