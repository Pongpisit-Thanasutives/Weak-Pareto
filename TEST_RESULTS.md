# Final verification results - 10 August 2026

The exact revised release tree was checked with:

```text
complete pytest suite                                  80/80 passed
Python AST syntax check                                68/68 files passed
bash -n on every shell script                          passed
LaTeX cross-reference/citation audit                   passed
PDF preflight and embedded-font audit                  passed
```

The existing regression suite continues to cover the benchmark smoke workflows,
operator/adjoint consistency, branch separation, superunit diagnostic,
two-dimensional extension, result/figure consistency, and Yu-baseline adapter.

## Revision-specific result check

Table 4 now contains clean reference rows for both Riesz reaction--diffusion
benchmarks. Five paper-profile clean seeds were run for each benchmark. Both
cases achieved 5/5 support/power recovery and 5/5 operator-structure recovery.
The ten per-seed records are archived at:

`reproduce/reference_results/branch_aware_campaign/main/rd_clean_per_seed.csv`

The Table-4 aggregate archive is:

`reproduce/reference_results/branch_aware_campaign/main/table_rd_noise.csv`

No core discovery/operator module was modified to obtain these rows.

## Figure regression status

The previously approved Figure 1, Figures 2--4, and their plotting data remain
unchanged in this revision. Existing tests bind their plotted values to archived
publication outputs and check the Figure-3 axis and six-point Figure-4 curve.
