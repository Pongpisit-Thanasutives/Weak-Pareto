# Weak-form comparison and blended improvements

This note records an independent review of the weak fractional-derivative
formulation in this package, a direct comparison against a second, independently
derived weak formulation, and three additive improvements that have been merged
into `fractional_weak_form.py`. All numerical claims below were reproduced on the
two bundled benchmarks (`data/tsfade_fft.dat`, `data/Convection_diffusion.dat`)
with `numpy` 2.4 / `scipy` 1.17.

The supporting scripts live in `examples/`:
`compare_weak_temporal_targets.py`, `compare_test_functions.py`,
`demo_riesz_feller_skew.py`, `demo_order_polish.py`.

---

## 1. Summary verdicts

* **Is the package's weak formulation mathematically correct?** Yes. The adjoint
  identities used for the spatial operators are reproduced to machine precision;
  the discrete Caputo–L1 adjoint matches the strong L1 derivative to machine
  precision; the Caputo initial-polynomial correction is algebraically the
  correct boundary term; and the nonlinear advective identity used for the
  `u^p D_x^beta u` encoding is verified to ~1e-15. The current bundled test suite (64
  tests) passes.

* **Is it essentially the same as the independent formulation, or more/less
  noise-robust?** Essentially the **same**. On the benchmark terms the two weak
  formulations are not merely similar — the spatial fractional adjoints are
  *bit-identical*, and the independent temporal target (Grünwald–Letnikov
  transpose with an initial-value correction) coincides with this package's
  default Caputo mode. Empirically, on the real FADE benchmark the temporal
  variants are statistically indistinguishable, and all of them dominate the
  strong (pointwise-derivative) baseline by a large margin. Neither weak form is
  meaningfully more robust than the other; the decisive effect is *weak vs
  strong*, which both share.

* **Net recommendation.** Keep this package as the production framework (its
  Pareto-DE framework, exact-discretisation-matched adjoints, train/validation
  scoring and metrics are more complete). Adopt the three additions below, of
  which the Riesz–Feller operator is the only one that closes a genuine
  capability gap on these benchmarks; the other two are principled levers for
  regimes outside the two bundled datasets.

---

## 2. Mathematical equivalence (Q1, Q2)

### 2.1 Spatial fractional operators are identical

The weak spatial column is `<X_beta u, phi> = <u, X_beta^* phi>`. Both
formulations implement `X_beta^*` as a Fourier multiplier on the test
functions: `-|k|^beta` for the self-adjoint Riesz operator, and the conjugate
symbol `conj((i k)^beta)` (followed by the real part) for the periodic
directional derivative used by `tsfade_fft`. Applied to identical test
functions on identical grids, the two implementations agree to floating-point
rounding:

| order `beta` | `|Riesz_A - Riesz_B|` | `|directional_A - directional_B|` |
|---:|---:|---:|
| 0.5 | 0.0 | 2.0e-31 |
| 1.0 | 0.0 | 0.0 |
| 1.7 | 0.0 | 0.0 |
| 2.0 | 0.0 | 0.0 |
| 2.5 | 0.0 | 2.0e-28 |

Because both benchmark equations are linear (all true terms have power `p=0`),
this establishes that on the terms that matter the two weak formulations apply
*the same operator*. They are mathematically the same, not approximately so.

### 2.2 Caputo temporal target is equivalent

For `0 < alpha <= 1`, this package writes the Caputo weak target as
`<u - u(0,.), (_t D_T^alpha) phi>` — the right Riemann–Liouville derivative on
the test function, with the initial field subtracted from the data. The
independent derivation instead subtracts an explicit boundary functional,
`-u(a) * (1/Gamma(1-alpha)) * \int (tau-a)^{-alpha} phi(tau) d tau`. These are
algebraically identical: `<u - u(0), (_t D_T^alpha) phi> = <u, (_t D_T^alpha)
phi> - u(0) <1, (_t D_T^alpha) phi>`, and the second term equals the explicit
kernel functional because `\int (_t D_T^alpha phi) = (1/Gamma(1-alpha)) \int
tau^{-alpha} phi`. The "subtract the initial field" packaging is the cleaner
discrete realisation (it reuses the same GL transpose and avoids a singular
quadrature), and it cross-validates the explicit-kernel derivation.

For the reported `spectral_l1` experiments, the active temporal target uses the
**exact L1 transpose** (`time_form="derivative"`), which transfers the discrete
Caputo matrix to the temporal tests. For a superunit order the matrix is
`L1(alpha-1) @ D1`; its transpose retains an endpoint-dominated opposite-sign
weight pair inherited from the one-sided first row of `D1`. The reported
superunit target therefore treats the initial rate implicitly and can be much
more noise-sensitive than the subunit target. The package also retains an
optional **Volterra / fractional-integral form** (`time_form="caputo_integral"`).
For orders above one, that optional path subtracts an initial slope estimated
explicitly from the first two samples; it is useful for diagnostics but is not
used for the reported superunit results.

---

## 3. Noise-robustness experiments (Q1, Q2)

All experiments fix the true support and orders and report the relative
coefficient error `||xi_hat - xi_star|| / ||xi_star||` (mean over four noise
seeds) against multiplicative uniform noise, on the real FADE benchmark
`D_t^0.8 u = -1.0 D_x^1 u + 0.5 D_x^1.7 u`.

### 3.1 Temporal target: independent (GL) vs package (L1, Volterra) vs strong

| noise % | GL-transpose (independent) | L1-transpose (package) | Volterra (package) | strong baseline |
|---:|---:|---:|---:|---:|
| 0  | 0.0116 | 0.0074 | 0.0084 | 0.0138 |
| 1  | 0.0110 | 0.0074 | 0.0099 | 0.4191 |
| 5  | 0.0094 | 0.0087 | 0.0164 | 0.8062 |
| 10 | 0.0104 | 0.0116 | 0.0247 | 1.1184 |
| 20 | 0.0159 | 0.0177 | 0.0414 | 1.2914 |

Recovered temporal order (truth `alpha = 0.8`), by minimum-residual scan at the
true spatial support:

| noise % | GL (independent) | L1 (package) | Volterra (package) |
|---:|---:|---:|---:|
| 0  | 0.800 | 0.800 | 0.820 |
| 5  | 0.800 | 0.800 | 0.815 |
| 10 | 0.800 | 0.800 | 0.815 |
| 20 | 0.800 | 0.795 | 0.815 |

**Reading.** (i) The strong baseline is destroyed by noise — already 42% error
at 1% noise, exceeding 100% by 20% — whereas every weak variant stays near 1–2%.
This ~80× separation at 20% noise is the real robustness result, and both
formulations share it. (ii) The GL transpose (the independent approach) and the
L1 transpose (this package) are statistically tied for coefficients, and the GL
transpose is in fact marginally better for `alpha` recovery. The L1 transpose's
exact-discretisation match buys only a small advantage at zero noise. (iii) The
Volterra form is slightly worse here, both in coefficient error at high noise and
in a small positive `alpha` bias (~0.015). It remains useful when the time record
is short or when one wishes to avoid any temporal derivative, but it is not the
best default on these datasets.

### 3.2 Test functions: Gaussian vs compact-support

Varying only the time-test family (GL-transpose target):

| noise % | coeff, Gaussian | coeff, compact | alpha, Gaussian | alpha, compact |
|---:|---:|---:|---:|---:|
| 0  | 0.0116 | 0.0116 | 0.800 | 0.800 |
| 5  | 0.0094 | 0.0099 | 0.800 | 0.800 |
| 10 | 0.0104 | 0.0111 | 0.800 | 0.800 |
| 20 | 0.0159 | 0.0158 | 0.800 | 0.785 |

The two families are tied. Gaussian tests kept in the interior already drive the
omitted temporal boundary term to a negligible level on this record length, so
compact support yields no measurable gain *here*. Its value is principled rather
than empirical on these benchmarks (see §5.1).

---

## 4. How the two formulations differ (Q1, Q3)

1. **Nonlinear candidate form.** This package implements the **advective** class
   `u^p D_x^beta u` via the correct identity `<u^p X_beta u, phi> = <u,
   X_beta^*(u^p phi)>` (verified to ~1e-15 for Riesz and directional operators,
   `p in {1,2}`). This matches the stated encoding. The independent module used
   a **conservation/flux** form `D_x^beta(u^p)`; for `p=0` the two coincide, so
   they agree on every true benchmark term and differ only on nonlinear
   competing-candidate columns. For the `u^p D_x^beta u` search, *this package's form is
   the correct one*. The note's caveat is also correct: for `p>0` the adjoint
   acts on the data-weighted test `u^p phi`, so nonlinear columns retain genuine
   data dependence and are less noise-robust than linear columns.

2. **Temporal target options.** The independent derivation offers exactly one
   temporal target (GL transpose + IC), which equals this package's default.
   This package adds the exact L1 transpose and the Volterra form (§2.2, §3.1).

3. **Test functions.** This package uses Gaussian and Fourier tests (Fourier
   makes a periodic Riesz column diagonal and therefore exact). The independent
   module used compactly supported Messenger–Bortz bumps. Now available here as
   an option (§5.1).

4. **Selection.** This package uses best-subset Pareto-DE per support size,
   contribution-norm pruning and a train/validation rule. The independent route
   sketched a scale-free thresholding score plus a robust-selection programme
   (information criteria for heavy-tailed residuals, knockoff/e-BH false-
   discovery control). The latter is complementary and is noted as a research
   direction in §6, not merged.

---

## 5. Blended additions (merged into `fractional_weak_form.py`)

All three are additive; the existing API and the 16-test suite are unchanged.

### 5.1 `compact_bump_test_matrix` — compactly supported test functions

`(1 - s^2)^power` bumps with half-support `3*width`. Unlike Gaussians, they and
their derivatives vanish *exactly* outside their support, so the omitted boundary
functional `B_L(f, phi)` for a derivative-form temporal target is exactly zero
rather than merely small. Verified: the bumps are numerically zero at the record
endpoints. On the bundled benchmarks this ties the Gaussian tests (§3.2); its
intended use is short time horizons and non-periodic spatial domains, where the
boundary term does not otherwise vanish.

### 5.2 `periodic_riesz_feller_adjoint_on_tests` — skewed (Feller) operator

Adjoint of the Riesz–Feller operator with multiplier `-|k|^order *
exp(i sgn(k) theta pi/2)` (conjugated for the test side, real part taken). At
`theta = 0` it reproduces the symmetric Riesz operator bit-for-bit; the skewed
adjoint identity holds to ~1e-12; the admissibility guard `|theta| <= min(order,
2-order)` is enforced. This closes a real gap: the package previously had only
symmetric Riesz and directional operators, neither of which can represent the
*asymmetric* transport of skewed alpha-stable / Lévy flights.

Demonstration on synthetic skewed data `u_t = 0.2 * D^1.6_(theta=0.4) u`:

| operator | recovered `D` | residual |
|---|---:|---:|
| symmetric Riesz (only prior option) | 0.162 | 0.588 |
| Riesz–Feller, theta = 0.2 | 0.190 | 0.309 |
| Riesz–Feller, theta = 0.3 | 0.198 | 0.156 |
| Riesz–Feller, theta = 0.4 (truth) | **0.200** | **0.000** |

The symmetric operator cannot fit skewed data (59% residual, wrong coefficient);
the Riesz–Feller operator recovers the coefficient exactly with a clean,
monotone residual descent in `theta`, so the skew is identifiable. To search over
skew, add `theta` to the differential-evolution vector alongside `(alpha, beta)`.

### 5.3 `refine_orders_local` — local order polish

A derivative-free simplex polish of the fractional orders on the (smooth) weak
residual, intended as an optional per-support-size step *after* the existing
Pareto-DE search. Because the weak residual is smooth in the continuous orders —
unlike the strong-form thresholded-regression loss, which jumps as the support
changes — a few steps refine a coarse optimum cheaply. On FADE, a deliberately
coarse start `(alpha, beta_2) = (0.90, 1.50)` (residual 0.114) polishes to
`(0.799, 1.711)` (residual 1.4e-3) against the truth `(0.800, 1.700)`.

---

## 6. Honest limitations and a research direction

* The two weak formulations are equivalent on the linear benchmark terms, so the
  additions here do not, and cannot, make the *weak form itself* more robust than
  it already is on these datasets. The genuine robustness margin is weak vs
  strong, which the package already exploits.
* Fractional-order identification is not made universally well-conditioned: at
  20% noise the spatial order can still drift when two candidate orders give
  similar residuals (the package documents this honestly, and §3.1 shows it).
* The most promising un-merged idea for *harder* structure recovery (heavy noise,
  Lévy-type residuals) is to replace the plain least-squares refit inside the
  Pareto-DE objective with a robust / false-discovery-controlled selector
  (robust information criteria; knockoff filters with e-BH control). This targets
  the support-selection step rather than the weak operators and is left as a
  follow-up.

---

## 7. Pre-publication fixes (addressed before generating paper results)

Two changes were made on top of the comparison work above, to remove a
methodological hazard and to close a capability gap before the numbers are
generated for the paper.

### 7.1 Exact-order refit (item 1) — final conditional refit without interpolation

**Problem.** Differential evolution proposes *continuous* fractional orders, but
the search-time weak features are obtained by interpolation across a precomputed
order grid. Earlier canonical convenience grids could also contain inserted
benchmark orders. The publication campaign now replaces those grids with dense,
truth-agnostic uniform grids. Interpolation can nevertheless bias the final
coefficient refit when a proposed order lies between precomputed nodes.

**Fix.**

* `WeakFractionalFeatureBank.target_exact(alpha)` and
  `library_exact(alpha, p_tuple, beta_tuple)` evaluate the weak operators
  *exactly* at the requested continuous orders (no grid, no interpolation), via
  the existing exact constructors `_target_direct` / `_space_feature_direct`.
* `refit_selected_exact(bank, model)` recomputes the final selected model's
  coefficients and full-data residual at its continuous orders. The selection-stage
  training/validation scores, objective, and AIC/BIC-type fields retain their
  original meanings; the full-data residual is stored separately as
  `full_data_rel_l2`.
* `_polish_selected_orders_exact(...)` optionally polishes `(alpha, beta)` on the
  directly evaluated weak residual through a bounded local search around the DE
  optimum. Direct evaluation removes interpolation error from this final
  conditional refit; the selected support, temporal mode, and optimisation basin
  can still depend on the precompute grid and search trajectory.
* Config flags (`pareto_fde_discovery.DiscoveryConfig`):
  `exact_order_refit: bool = True` is on by default, while
  `exact_order_polish: bool = False` is the lightweight API default and is
  enabled explicitly by the publication drivers.
  `fit_coefficients_for_structure` now uses exact features when available, so the
  truth-structure diagnostic is exact too. The discovery summary carries
  `selected_exact_refit` with `full_data_rel_l2` and the coefficient/order shifts
  versus the interpolated fit.
* `dataset_configs.with_uniform_order_grids(config, n_alpha=47, n_beta=59)`
  rebuilds the search grid as a dense **uniform** grid over the same ranges with
  *no* true orders inserted (prime node counts so round true orders are not
  nodes). Recommended for publication, together with `exact_order_refit=True`.

**Verification** (FADE, true support `{(0,1.0),(0,1.7)}`):

| check | interpolated vs exact |
| --- | --- |
| on-grid (true orders are grid nodes) | identical (`|interp − exact| = 0`) — this *is* the "grid encodes the answer" optic |
| off-grid `(alpha,beta_2)=(0.83,1.63)` | interpolation bias `|interp − exact| ≈ 4.7e-4`, removed by the exact refit |
| within-framework (5% noise) | exact refit shifts a coefficient by `≈ 3.1e-2` vs the interpolated fit; orders polished to the exact-residual minimum, all logged in `selected_exact_refit` |

### 7.2 A genuinely nonlinear benchmark (item 2) — `synthetic_fractional_burgers`

**Problem.** Every bundled benchmark (paper and synthetic) is *linear*: every true
term has power `p = 0`. The headline model class is `u^p D_x^beta u` with
`p in {0,1,2}`, and the data-weighted advective weak form
`<u, (D_x^beta)^*(u^p phi)>` is the main generalisation over plain weak-SINDy,
yet it was only verified as an algebraic identity, never as recovery on data with
a true nonlinear term.

**Fix.** Added a nonlinear fractional Burgers benchmark
(`fpde_datasets.make_fractional_burgers`, registered as
`synthetic_fractional_burgers` everywhere: loader, truth spec, candidate config,
benchmark lists, coefficient truth):

```
u_t = -1.0 * u u_x + 0.25 * D_x^1.7 u        (directional spectral operator, periodic, integer time)
   => encoding {(p=1, beta=1.0): -1.0,  (p=0, beta=1.7): 0.25}.
```

The PDE is integrated pseudo-spectrally (explicit RK4, 2/3 dealiasing) on a fine
grid, then subsampled to the paper grid `(150, 120)`. Parameters were tuned so the
solution stays smooth and band-limited *within the coarse-grid Nyquist* (coarse
strong-form residual `≈ 1.6e-4`, spectral tail energy `≈ 6e-8`) while the
nonlinear term remains comparable in magnitude to the diffusion term
(`||u u_x|| ≈ 1.4 ||0.25 D^1.7 u||`), so it is genuinely identifiable rather than a
perturbation.

**Recovery demonstration** (`examples/demo_fractional_burgers.py`). Best-subset
over a 7-term pool (2 true terms + 5 competing candidates: `u_x`, `u^2 u_x`, `u_xx`,
`u D^1.7 u`, `D^0.5 u`), exact weak features:

| noise | winner | gap to closest competing structure (residual) | true-structure coef | rel-coef-err |
| --- | --- | --- | --- | --- |
| 0%  | `{u u_x, D^1.7 u}` ✓ | `9.2e-5` vs `3.3e-1` (≈3650×) | `[-1.0000, +0.2500]` | `0.0000` |
| 10% | `{u u_x, D^1.7 u}` ✓ | `4.5e-2` vs `3.4e-1` (≈7.6×)  | `[-1.0023, +0.2487]` | `0.0026` |
| 25% | `{u u_x, D^1.7 u}` ✓ | `1.1e-1` vs `3.6e-1` (≈3.2×)  | `[-1.0058, +0.2456]` | `0.0071` |

The closest competing structure in every case is integer diffusion `u_xx` (`beta = 2`):
the weak form distinguishes the **fractional** order `beta = 1.7` from the integer
order, and the **nonlinear** advective `u u_x` from the linear `u_x` and from
`u^2 u_x`, up to 25% noise. The full weak Pareto-DE framework (with the item-1
exact refit) recovers the structure automatically at 0% noise:
selected `{(p=1,beta=1.0): -1.0, (p=0,beta=1.7): 0.25}`, `alpha = 1.0`, exact
residual `9.2e-5`.

### 7.3 Recommended settings for the publication runs

* Keep `exact_order_refit=True`; set `exact_order_polish=True`, and rebuild grids
  with `with_uniform_order_grids(config)` so reported orders/coefficients are
  evaluated without interpolation error at the polished orders. The selected support, temporal mode, and optimisation basin can still depend on the order grid and search trajectory.
* Include `synthetic_fractional_burgers` so the nonlinear capability is
  demonstrated, not asserted; report the integer-vs-fractional (`u_xx` vs
  `D^1.7 u`) margin explicitly.
* Independent of these two items, two recommendations from §3/§6 still stand for
  the paper: report the noise-degradation curve to the breakdown point (push above
  the 2% default and add an additive-Gaussian noise model alongside the
  multiplicative one), and add an external published baseline rather than only the
  internal ablations. Do not report the Volterra temporal target (small `alpha`
  bias, §3.1).
