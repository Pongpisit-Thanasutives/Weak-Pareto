# Noise-tolerance selection for reporting clean/noisy cases

Use `scripts/find_noise_tolerance.py` to find, for each dataset, the highest tested noise level at which the proposed method still passes a declared recovery criterion.

The proposed method is:

```text
weak_pareto = weak library + best-subset Pareto-DE
```

A strict pass requires:

1. full structure recovery;
2. temporal alpha within the dataset tolerance;
3. RHS beta orders within the dataset tolerance;
4. maximum relative coefficient error below `--coef-rel-tol`.

Example:

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

The script writes:

```text
noise_tolerance_summary.csv
noise_tolerance_detailed.csv
NOISE_TOLERANCE_REPORT.md
```

The ADE control case is useful for structure recovery and order-error reporting, but its weak finite-domain diffusion coefficient can be biased under the current weak operator convention. Treat coefficient claims on that dataset cautiously.
