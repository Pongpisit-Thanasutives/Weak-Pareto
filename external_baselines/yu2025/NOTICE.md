# Source and redistribution notice

The upstream Yu et al. code is **not redistributed** in this archive: the supplied snapshot did not contain a clear redistribution licence. Only the newly written adapter (`yu_baseline.py`), which reimplements the Yu et al. method and records all behavioural changes, is included.

To run the neural fractional-discovery framework, obtain the upstream code from the authors (see the paper below) and place it under `external_baselines/yu2025/original/`. The adapter is self-contained and does not import the upstream files; the comparison harness (`scripts/run_yu2025_comparison.py`) reads only the shared benchmark data in `data/`. When `original/` is absent, upstream byte identity is recorded as not verified. Hashes of the clean and noisy fields actually used by the packaged campaign are still retained for within-campaign provenance.

Adapter changes relative to the upstream method: deterministic per-seed train/validation splits, prevention of validation leakage in STRidge, a matched optimisation protocol and budget, and a bank-mode fractional-order search; see `yu_baseline.py` and `docs/YU2025_BASELINE_INTEGRATION.md`.

Paper:

X. Yu et al., “A data-driven framework for discovering fractional differential equations in complex systems,” *Nonlinear Dynamics*, 113, 24557–24577 (2025). DOI: 10.1007/s11071-025-11373-z.
