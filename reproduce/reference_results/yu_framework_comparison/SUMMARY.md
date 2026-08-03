# Weak-Pareto versus Yu et al. (2025)

The comparison uses the same packaged field and identical multiplicative-uniform noise realizations per seed. Upstream byte identity was not verified in the packaged campaign because the upstream snapshot is absent.
Weak-Pareto uses a periodic directional spectral convention; the Yu adapter uses a one-sided finite-terminal approximation. The comparison is therefore shared-data and nominal-equation matched, not operator-identical.
Order and coefficient errors are shown only when the support/power pattern is correctly recovered; coefficient errors do not additionally require complete fractional-order recovery.

| Dataset | Noise | Seed | Method | Status | Operator recovery | Alpha error | Mean beta error | Mean coefficient error | Runtime (s) | Equation |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| paper_FADE_tsfade_fft | 0.0 | 0 | yu2025_full | ok | 0 | 0.01348 | 0.07691 | 0.06943 | 230 | D_t^0.786517 u = - 1.09733 u_x + 0.458474 D_x^1.54618 u |
| paper_FADE_tsfade_fft | 0.0 | 0 | yu2025_optimizer_only | ok | 0 | 0.01742 | 0.1001 | 0.09008 | 18.11 | D_t^0.782578 u = - 1.12861 u_x + 0.44845 D_x^1.49973 u |
| paper_FADE_tsfade_fft | 1.0 | 0 | yu2025_full | ok | 1 | 0.01047 | 0.06365 | 0.05584 | 313.6 | D_t^0.789528 u = - 1.07984 u_x + 0.468164 D_x^1.57269 u |
| paper_FADE_tsfade_fft | 1.0 | 0 | yu2025_optimizer_only | ok | 0 | — | — | — | 10.83 | D_t^0.997639 u = - 0.465927 u_x |
| paper_FADE_tsfade_fft | 5.0 | 0 | yu2025_full | ok | 1 | 0.003731 | 0.03456 | 0.02751 | 333.3 | D_t^0.796269 u = - 1.04259 u_x + 0.487563 D_x^1.63088 u |
| paper_FADE_tsfade_fft | 5.0 | 0 | yu2025_optimizer_only | ok | 0 | — | — | — | 15.57 | D_t^0.999 u = - 0.225651 u_x |
| paper_FADE_tsfade_fft | 0.0 | 1 | yu2025_full | ok | 0 | 0.01264 | 0.08412 | 0.06761 | 350.2 | D_t^0.78736 u = - 1.09743 u_x + 0.462213 D_x^1.53176 u |
| paper_FADE_tsfade_fft | 0.0 | 1 | yu2025_optimizer_only | ok | 0 | 0.0172 | 0.102 | 0.09088 | 25.91 | D_t^0.782801 u = - 1.12984 u_x + 0.448074 D_x^1.49592 u |
| paper_FADE_tsfade_fft | 1.0 | 1 | yu2025_full | ok | 0 | 0.01191 | 0.08443 | 0.0663 | 335.5 | D_t^0.788092 u = - 1.09652 u_x + 0.463923 D_x^1.53114 u |
| paper_FADE_tsfade_fft | 1.0 | 1 | yu2025_optimizer_only | ok | 0 | — | — | — | 11.79 | D_t^0.992717 u = - 0.477551 u_x |
| paper_FADE_tsfade_fft | 5.0 | 1 | yu2025_full | ok | 0 | 0.009473 | 0.07579 | 0.0555 | 326.5 | D_t^0.790527 u = - 1.08282 u_x + 0.471825 D_x^1.54841 u |
| paper_FADE_tsfade_fft | 5.0 | 1 | yu2025_optimizer_only | ok | 0 | — | — | — | 9.229 | D_t^0.996136 u = - 0.262449 u_x |
| paper_FADE_tsfade_fft | 0.0 | 2 | yu2025_full | ok | 0 | 0.01114 | 0.08952 | 0.07316 | 346.2 | D_t^0.788862 u = - 1.10237 u_x + 0.456038 D_x^1.52096 u |
| paper_FADE_tsfade_fft | 0.0 | 2 | yu2025_optimizer_only | ok | 0 | 0.01657 | 0.103 | 0.09024 | 27.9 | D_t^0.783433 u = - 1.12992 u_x + 0.449448 D_x^1.49398 u |
| paper_FADE_tsfade_fft | 1.0 | 2 | yu2025_full | ok | 0 | 0.01088 | 0.08069 | 0.06891 | 339.4 | D_t^0.789122 u = - 1.0934 u_x + 0.455581 D_x^1.53863 u |
| paper_FADE_tsfade_fft | 1.0 | 2 | yu2025_optimizer_only | ok | 0 | — | — | — | 10.84 | D_t^0.81064 u = - 0.856261 u_x |
| paper_FADE_tsfade_fft | 5.0 | 2 | yu2025_full | ok | 1 | 0.003307 | 0.006073 | 0.008123 | 232.6 | D_t^0.796693 u = - 1.01127 u_x + 0.495023 D_x^1.68785 u |
| paper_FADE_tsfade_fft | 5.0 | 2 | yu2025_optimizer_only | ok | 0 | — | — | — | 13.98 | D_t^0.999 u = - 0.206441 u_x |
| paper_FADE_tsfade_fft | 0.0 | 3 | yu2025_full | ok | 0 | 0.01462 | 0.08687 | 0.07732 | 349.3 | D_t^0.785381 u = - 1.10836 u_x + 0.45372 D_x^1.52625 u |
| paper_FADE_tsfade_fft | 0.0 | 3 | yu2025_optimizer_only | ok | 0 | 0.01645 | 0.1042 | 0.09073 | 29.23 | D_t^0.783546 u = - 1.13046 u_x + 0.449 D_x^1.49153 u |
| paper_FADE_tsfade_fft | 1.0 | 3 | yu2025_full | ok | 0 | 0.01298 | 0.08217 | 0.07072 | 336.6 | D_t^0.787018 u = - 1.09957 u_x + 0.45812 D_x^1.53566 u |
| paper_FADE_tsfade_fft | 1.0 | 3 | yu2025_optimizer_only | ok | 0 | — | — | — | 10.3 | D_t^0.991503 u = - 0.475461 u_x |
| paper_FADE_tsfade_fft | 5.0 | 3 | yu2025_full | ok | 1 | 0.0006663 | 0.03325 | 0.0139 | 194.3 | D_t^0.799334 u = - 1.0272 u_x + 0.499404 D_x^1.63351 u |
| paper_FADE_tsfade_fft | 5.0 | 3 | yu2025_optimizer_only | ok | 0 | — | — | — | 9.019 | D_t^0.993989 u = - 0.221412 u_x |
| paper_FADE_tsfade_fft | 0.0 | 4 | yu2025_full | ok | 1 | 0.01027 | 0.06897 | 0.06122 | 338.1 | D_t^0.789731 u = - 1.08168 u_x + 0.459233 D_x^1.56206 u |
| paper_FADE_tsfade_fft | 0.0 | 4 | yu2025_optimizer_only | ok | 0 | 0.01656 | 0.1014 | 0.08958 | 25.89 | D_t^0.783437 u = - 1.12811 u_x + 0.448949 D_x^1.49711 u |
| paper_FADE_tsfade_fft | 1.0 | 4 | yu2025_full | ok | 1 | 0.01053 | 0.06919 | 0.06036 | 335.7 | D_t^0.789467 u = - 1.08209 u_x + 0.461362 D_x^1.56161 u |
| paper_FADE_tsfade_fft | 1.0 | 4 | yu2025_optimizer_only | ok | 0 | — | — | — | 10.28 | D_t^0.994342 u = - 0.470599 u_x |
| paper_FADE_tsfade_fft | 5.0 | 4 | yu2025_full | ok | 1 | 0.004681 | 0.05321 | 0.03502 | 306.6 | D_t^0.795319 u = - 1.05661 u_x + 0.48658 D_x^1.59358 u |
| paper_FADE_tsfade_fft | 5.0 | 4 | yu2025_optimizer_only | ok | 0 | — | — | — | 9.963 | D_t^0.997842 u = - 0.234329 u_x |

## Protocol

- Weak-Pareto profile: `notebook`; exact-order polishing and exact coefficient refit are enabled.
- Yu profile: `paper`; protocol `matched`; order mode `bank`.
- `yu2025_full` uses the neural reconstruction + Gauss–Jacobi + STRidge + DE framework.
- `yu2025_optimizer_only` replaces only the neural reconstruction by a tensor-product quintic spline, retaining Gauss–Jacobi + STRidge + DE.
- The adapted neural fractional-discovery framework and optimizer-only comparison remain separate rows.
- Yu runs are intentionally skipped for periodic signed-Riesz datasets because those operators fall outside the adapter's declared comparison scope.
