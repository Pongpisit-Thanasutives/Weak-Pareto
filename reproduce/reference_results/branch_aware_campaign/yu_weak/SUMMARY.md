# Weak-Pareto versus Yu et al. (2025)

The comparison uses the same packaged field and identical multiplicative-uniform noise realizations per seed. Upstream byte identity was not verified in the packaged campaign because the upstream snapshot is absent.
Weak-Pareto uses a periodic directional spectral convention; the Yu adapter uses a one-sided finite-terminal approximation. The comparison is therefore shared-data and nominal-equation matched, not operator-identical.
Order and coefficient errors are shown only when the support/power pattern is correctly recovered; coefficient errors do not additionally require complete fractional-order recovery.

| Dataset | Noise | Seed | Method | Status | Operator recovery | Alpha error | Mean beta error | Mean coefficient error | Runtime (s) | Equation |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| paper_FADE_tsfade_fft | 0.0 | 0 | weak_pareto | ok | 1 | 0.0009733 | 0.01444 | 0.0104 | 6.373 | D_t^0.79903 u = (-0.99205)*D_x^0.99663 u + (0.48715)*D_x^1.72552 u |
| paper_FADE_tsfade_fft | 1.0 | 0 | weak_pareto | ok | 1 | 0.0008178 | 0.01857 | 0.01284 | 6.376 | D_t^0.79918 u = (-0.98783)*D_x^0.99584 u + (0.48649)*D_x^1.73298 u |
| paper_FADE_tsfade_fft | 5.0 | 0 | weak_pareto | ok | 1 | 0.0002605 | 0.026 | 0.01354 | 6.803 | D_t^0.79974 u = (-0.98171)*D_x^0.99533 u + (0.49121)*D_x^1.74733 u |
| paper_FADE_tsfade_fft | 0.0 | 1 | weak_pareto | ok | 1 | 0.0009736 | 0.01445 | 0.0104 | 6.533 | D_t^0.79903 u = (-0.99205)*D_x^0.99663 u + (0.48714)*D_x^1.72552 u |
| paper_FADE_tsfade_fft | 1.0 | 1 | weak_pareto | ok | 1 | 0.0004378 | 0.01214 | 0.005556 | 7.168 | D_t^0.79956 u = (-0.99483)*D_x^0.99793 u + (0.49405)*D_x^1.72221 u |
| paper_FADE_tsfade_fft | 5.0 | 1 | weak_pareto | ok | 1 | 0.001565 | 0.009079 | 0.02881 | 6.448 | D_t^0.80156 u = (-1.0234)*D_x^1.00720 u + (0.53424)*D_x^1.68904 u |
| paper_FADE_tsfade_fft | 0.0 | 2 | weak_pareto | ok | 1 | 0.0009736 | 0.01444 | 0.0104 | 7.058 | D_t^0.79903 u = (-0.99206)*D_x^0.99663 u + (0.48715)*D_x^1.72551 u |
| paper_FADE_tsfade_fft | 1.0 | 2 | weak_pareto | ok | 1 | 0.001001 | 0.01506 | 0.01069 | 6.535 | D_t^0.79900 u = (-0.99163)*D_x^0.99640 u + (0.48699)*D_x^1.72652 u |
| paper_FADE_tsfade_fft | 5.0 | 2 | weak_pareto | ok | 1 | 0.001208 | 0.005916 | 0.004015 | 7.053 | D_t^0.79879 u = (-1.0041)*D_x^0.99898 u + (0.49605)*D_x^1.71082 u |
| paper_FADE_tsfade_fft | 0.0 | 3 | weak_pareto | ok | 1 | 0.0009733 | 0.01444 | 0.0104 | 6.56 | D_t^0.79903 u = (-0.99205)*D_x^0.99663 u + (0.48715)*D_x^1.72552 u |
| paper_FADE_tsfade_fft | 1.0 | 3 | weak_pareto | ok | 1 | 0.0006042 | 0.0115 | 0.007564 | 6.723 | D_t^0.79940 u = (-0.99448)*D_x^0.99740 u + (0.49039)*D_x^1.72041 u |
| paper_FADE_tsfade_fft | 5.0 | 3 | weak_pareto | ok | 1 | 0.0007634 | 0.00994 | 0.01581 | 6.615 | D_t^0.80076 u = (-1.018)*D_x^1.00369 u + (0.5136)*D_x^1.68381 u |
| paper_FADE_tsfade_fft | 0.0 | 4 | weak_pareto | ok | 1 | 0.0009734 | 0.01444 | 0.0104 | 6.515 | D_t^0.79903 u = (-0.99206)*D_x^0.99663 u + (0.48715)*D_x^1.72551 u |
| paper_FADE_tsfade_fft | 1.0 | 4 | weak_pareto | ok | 1 | 0.0008283 | 0.01302 | 0.0102 | 6.724 | D_t^0.79917 u = (-0.99313)*D_x^0.99655 u + (0.48647)*D_x^1.72260 u |
| paper_FADE_tsfade_fft | 5.0 | 4 | weak_pareto | ok | 1 | 0.0003443 | 0.007034 | 0.009295 | 6.527 | D_t^0.79966 u = (-1.0159)*D_x^1.00068 u + (0.49728)*D_x^1.68661 u |

## Protocol

- Weak-Pareto profile: `paper`; exact-order polishing and exact coefficient refit are enabled.
- Yu profile: `paper`; protocol `matched`; order mode `bank`.
- `yu2025_full` uses the neural reconstruction + Gauss–Jacobi + STRidge + DE framework.
- `yu2025_optimizer_only` replaces only the neural reconstruction by a tensor-product quintic spline, retaining Gauss–Jacobi + STRidge + DE.
- The adapted neural fractional-discovery framework and optimizer-only comparison remain separate rows.
- Yu runs are intentionally skipped for periodic signed-Riesz datasets because those operators fall outside the adapter's declared comparison scope.
