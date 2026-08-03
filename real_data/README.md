# Applying weak-Pareto to real spatiotemporal data

This directory contains **standalone** code for running the discovery method on
**real** measured fields, kept separate from the synthetic-benchmark
reproduction in `../reproduce/`. It provides a data loader, a finite-domain
discovery preset, and a runnable end-to-end demonstration.

```
real_data/
  real_fade.py              # loaders + bounded-domain config + discover()
  run_real.py               # command-line driver: run on your own data file
  demo_synthetic_column.py   # runnable template on a synthesised tracer field
  frozen_soil_creep_weak.py # integral fractional-Kelvin fit to external creep data
  README.md                  # this file
```

> **Experimental example.** The paper includes an integral-form fractional
> Kelvin fit to irregular, naturally noisy frozen-soil creep records. The two
> external MAT files are not redistributed; acquisition instructions and
> expected hashes are in `../external_data/frozen_soil/README.md`. Once those
> files are supplied, reproduce the result with
> `PYTHONPATH=. python real_data/frozen_soil_creep_weak.py`.

## The file to run on your own data

`run_real.py` is the command-line driver. Obtain a dataset (see pointers below),
get it onto a uniform `(t, x)` grid as CSV or NPZ, then:

```bash
# matrix CSV (first row = x, first column = t, interior = u), advective column,
# time-fractional, publication-quality search, saving a summary + Pareto front:
PYTHONPATH=. python real_data/run_real.py --csv column.csv --layout matrix \
    --space-side left --beta-min 0.7 --beta-max 2.0 --powers 0,1 \
    --budget paper --out results_real/column

# tidy CSV with columns t,x,u:
PYTHONPATH=. python real_data/run_real.py --csv column_long.csv --layout long --budget paper

# NPZ with arrays t, x, U; integer time order, two-sided (Riesz-like) dispersion:
PYTHONPATH=. python real_data/run_real.py --npz field.npz --integer-time \
    --space-side symmetric --budget paper
```

`--budget paper` is the publication-quality setting; `--budget smoke` is a fast
wiring check. The driver prints the discovered equation and the full
support-size Pareto front, and with `--out PREFIX` writes `PREFIX_summary.json`
and `PREFIX_pareto.csv`. Run `python real_data/run_real.py -h` for all options.

To sanity-check the framework before you have data, run the synthetic template:

```bash
PYTHONPATH=. python real_data/demo_synthetic_column.py            # standard budget
PYTHONPATH=. python real_data/demo_synthetic_column.py --budget smoke
```

(The synthetic demo is a wiring template, not a benchmark; see the closing note.)

## The one thing to get right: bounded domains are not periodic

Several of the paper's synthetic benchmarks live on a periodic box and use the
periodic spectral operators (Riesz `-|k|^β`, directional `(ik)^β`). **Real
transport experiments do not.** A tracer column, a contaminant plume, or a
heat-conduction bar occupies a finite interval `x ∈ [0, L]` with a source/inflow
boundary at one end and a free/outflow boundary at the other. Using a periodic
operator there is physically wrong.

This module therefore configures the **one-sided Grünwald–Letnikov / Riemann–
Liouville** spatial operators (`backend="regularized"`, `spectral_riesz=False`,
`regularized_space_side="left"`), exactly as the bounded integer-order
advection–diffusion control in the paper does. The weak formulation moves those
one-sided operators onto compactly supported test functions whose support stays
away from the inflow/outflow boundaries, so the boundary terms of the fractional
integration by parts vanish (Appendix A). For a column flowing in `+x` with
inflow at `x = 0`, the **left** (causal) side is the physical choice; use
`"symmetric"` for a Riesz-like two-sided dispersion on the bounded interval.

The time operator is the Caputo–L1 derivative (`time_fractional=True`), which is
appropriate whether the time order is fractional (anomalous, memory-laden
transport) or integer (set `time_fractional=False`).

## Loading your data

`u` must end up on a **uniform** `(t, x)` grid as an `(n_t, n_x)` array, because
the weak quadrature assumes uniform spacing.

```python
from real_data.real_fade import load_field_csv, load_field_npz, regrid_scattered, discover, bounded_transport_config

# (a) a matrix CSV: first row = x coordinates, first column = t coordinates, interior = u
data = load_field_csv("column.csv", layout="matrix")

# (b) a tidy CSV with columns t,x,u already on a lattice
data = load_field_csv("column_long.csv", layout="long")

# (c) an .npz holding arrays t (n_t,), x (n_x,), U (n_t,n_x)
data = load_field_npz("column.npz")

# (d) irregular samples -> resample once onto a regular grid (no finer than the real resolution)
data = regrid_scattered(t_samples, x_samples, u_samples, n_t=80, n_x=80)

config = bounded_transport_config(time_fractional=True, space_side="left",
                                  beta_range=(0.7, 2.0), p_values=(0, 1))
summary = discover(data, config)          # prints the discovered equation + held-out residual
```

No ground truth is needed or used. The output is the discovered equation and its
held-out weak residual; physical interpretation is up to you.

## Candidate public datasets

Fully curated, ready-to-load spatiotemporal fractional fields are rare: much
"fractional" real data are one-dimensional time series (mean-squared
displacement, relaxation), which are fractional **ODEs**, not the PDE fields
this method targets. The most suitable sources of a genuine `u(x, t)` field are
anomalous solute/gas transport in porous and fractured media and non-Fourier
heat conduction. Concrete starting points:

- **Soil-column miscible-displacement breakthrough curves.** The classic
  finite-domain fADE datasets are collections of column breakthrough curves
  (e.g. the seven experiment sets / 53 BTCs used by Zhang, Benson and
  collaborators for the spatial-fADE on a finite column). These are modelled
  with left/right Grünwald derivatives on `[0, L]` — directly the configuration
  here. Many are tabulated in the transport-modelling literature and the MADE /
  Cape Cod tracer-test reports; a single column with concentration sectioned
  along `x` over time gives an `(x, t)` field, while outlet-only BTCs are 1-D in
  space (treat as a time series, or several outlet depths stacked).

- **X-ray-monitored column experiments.** Setups where an X-ray source/detector
  traverses the column record tracer concentration *averaged on each
  cross-section* as a function of `(x, t)` (see e.g. arXiv:1608.08363, a
  finite-column tracer test with constant Darcy velocity and an inlet flux).
  This is the cleanest kind of spatiotemporal field for the method.

- **Gas breakthrough in geological media.** Sub-diffusive gas transport with
  heavy late-time tailing is captured by a *time*-fractional convection–
  diffusion equation; set `time_fractional=True` with an integer-order space
  term (`beta_range` bracketing 1–2, `p_values=(0,)`).

- **Non-Fourier / anomalous heat conduction.** Temperature fields `T(x, t)` in
  heterogeneous or low-dimensional conductors are described by time-fractional
  heat equations; the same bounded-domain preset applies with the temperature as
  `u`.

For any of these: match `space_side` to the flow orientation (or `"symmetric"`
for two-sided dispersion), set `beta_range` to bracket the expected dispersion
order (often 1.5–2 for fADE), choose `time_fractional` by whether memory/tailing
is present, and keep `p_values=(0, 1)` unless a specific nonlinearity is
expected. Regrid no finer than the genuine measurement resolution so you do not
manufacture high-wavenumber content for the weak features to fit.

## A realistic expectation

Discovery on real anomalous-transport data is harder than on the synthetic
benchmarks: fields are often advection-dominated, fronts can be sharp, the
domain is bounded, and the dispersion order and a nearby advection order can
trade off. Treat the recovered orders as estimates, inspect the full
support-size Pareto front (returned in the summary) rather than only the
selected model, and tune the search ranges, optimisation budget, and
regularisation (`lam_t`, `lam_x`) to the data. The order-identifiability
behaviour and its dependence on noise are discussed in Section 5 of the paper.
