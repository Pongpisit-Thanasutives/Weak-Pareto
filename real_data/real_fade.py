"""Apply weak-Pareto FDE discovery to a real spatiotemporal field on a *bounded* domain.

This module is deliberately separate from the synthetic-benchmark reproduction
code.  Real transport experiments (tracer columns, contaminant breakthrough,
non-Fourier heat conduction) live on a finite interval ``x in [0, L]`` with an
inflow/source boundary and a free outflow boundary; they are *not* periodic.
The periodic spectral operators used for several synthetic benchmarks therefore
do not apply.  Instead we use the package's one-sided Grunwald--Letnikov / Caputo
operators (``backend="regularized"``, ``spectral_riesz=False``), exactly as the
bounded integer-order advection--diffusion control does, and let the weak
formulation move those one-sided operators onto compactly supported test
functions whose support stays away from the inflow/outflow boundaries.

The public entry points are:

* :func:`load_field_csv` / :func:`load_field_npz` -- read ``u`` on a regular
  ``(t, x)`` grid into the :class:`GridDataset` the discovery code expects;
* :func:`regrid_scattered` -- resample irregular (t, x, u) samples onto a
  regular grid by linear interpolation (column data are often irregular);
* :func:`bounded_transport_config` -- a :class:`DiscoveryConfig` preset for
  finite-domain transport (one-sided space operator, Caputo or integer time);
* :func:`discover` -- run weak-Pareto on a loaded field and return the summary.

No ground-truth is required or used.  Because a real governing equation is
unknown, the output is the discovered equation plus its held-out weak residual;
the caller is responsible for physical interpretation.

See ``README.md`` in this directory for concrete public datasets, how to obtain
them, and the operator/boundary configuration each one needs.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from fpde_datasets import GridDataset
from pareto_fde_discovery import DiscoveryConfig
from weak_pareto_fde_discovery import run_weak_pareto_discovery


# --------------------------------------------------------------------------- #
#  Data ingestion                                                             #
# --------------------------------------------------------------------------- #
def _validate_grid(t: NDArray[np.float64], x: NDArray[np.float64], U: NDArray[np.float64]) -> None:
    if U.ndim != 2:
        raise ValueError(f"U must be 2-D with shape (n_t, n_x); got {U.shape}")
    if U.shape != (t.size, x.size):
        raise ValueError(f"U has shape {U.shape} but (n_t, n_x)=({t.size}, {x.size})")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(x))):
        raise ValueError("t and x must be finite")
    for g, nm in ((t, "t"), (x, "x")):
        d = np.diff(g)
        if np.any(d <= 0):
            raise ValueError(f"{nm} must be strictly increasing")
        if np.max(np.abs(d - d[0])) > 1e-6 * abs(d[0]):
            raise ValueError(
                f"{nm} must be uniformly spaced for the weak quadrature; "
                f"resample with regrid_scattered() first"
            )
    if not np.all(np.isfinite(U)):
        raise ValueError("U contains non-finite values; clean or interpolate first")


def make_dataset(
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    U: NDArray[np.float64],
    name: str = "real_field",
) -> GridDataset:
    """Wrap a regular ``(t, x)`` field ``U`` of shape ``(n_t, n_x)`` as a GridDataset.

    ``truth`` is ``None`` (unknown for real data) and the backend recommendation
    is left unset; choose it via :func:`bounded_transport_config`.
    """
    t = np.asarray(t, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    U = np.asarray(U, dtype=float)
    _validate_grid(t, x, U)
    return GridDataset(U=U, t=t, x=x, name=name, truth=None, recommended_backend="regularized")


def load_field_npz(path: str, name: str | None = None) -> GridDataset:
    """Load a field from a ``.npz`` with arrays ``t`` (n_t,), ``x`` (n_x,), ``U`` (n_t, n_x)."""
    with np.load(path) as d:
        missing = {"t", "x", "U"} - set(d.files)
        if missing:
            raise KeyError(f"{path} is missing arrays {sorted(missing)} (need t, x, U)")
        return make_dataset(d["t"], d["x"], d["U"], name=name or path)


def load_field_csv(
    path: str,
    name: str | None = None,
    layout: Literal["matrix", "long"] = "matrix",
) -> GridDataset:
    """Load a field from CSV.

    ``layout="matrix"``: first row holds the ``x`` coordinates (its leading cell
    is ignored), the first column holds the ``t`` coordinates, and the interior
    is ``U``.  ``layout="long"``: three columns ``t, x, u`` (a tidy table), which
    is regridded onto the unique sorted ``t`` and ``x`` values; if the long table
    is scattered rather than on a lattice, use :func:`regrid_scattered` instead.
    """
    if layout == "matrix":
        raw = np.genfromtxt(path, delimiter=",", dtype=float)
        x = raw[0, 1:]
        t = raw[1:, 0]
        U = raw[1:, 1:]
        return make_dataset(t, x, U, name=name or path)
    if layout == "long":
        raw = np.genfromtxt(path, delimiter=",", dtype=float, names=True)
        cols = raw.dtype.names
        if cols is None or len(cols) < 3:
            raise ValueError("long layout needs a header with at least t, x, u columns")
        tc, xc, uc = cols[:3]
        t_all, x_all, u_all = raw[tc], raw[xc], raw[uc]
        t = np.unique(t_all)
        x = np.unique(x_all)
        U = np.full((t.size, x.size), np.nan)
        ti = {v: i for i, v in enumerate(t)}
        xi = {v: i for i, v in enumerate(x)}
        for tt, xx, uu in zip(t_all, x_all, u_all):
            U[ti[tt], xi[xx]] = uu
        return make_dataset(t, x, U, name=name or path)
    raise ValueError("layout must be 'matrix' or 'long'")


def regrid_scattered(
    t_samples: NDArray[np.float64],
    x_samples: NDArray[np.float64],
    u_samples: NDArray[np.float64],
    n_t: int,
    n_x: int,
    name: str = "real_field_regridded",
) -> GridDataset:
    """Resample scattered ``(t, x, u)`` observations onto a regular ``n_t x n_x`` grid.

    Uses SciPy's ``griddata`` (linear, with nearest-neighbour fill at the convex
    hull edge).  Real column/heat data are frequently irregular in time or space;
    the weak quadrature needs a uniform grid, so regrid once up front.  Choose
    ``n_t, n_x`` no finer than the genuine resolution of the measurements to
    avoid manufacturing spurious high-wavenumber content.
    """
    from scipy.interpolate import griddata

    t_samples = np.asarray(t_samples, float).ravel()
    x_samples = np.asarray(x_samples, float).ravel()
    u_samples = np.asarray(u_samples, float).ravel()
    t = np.linspace(float(t_samples.min()), float(t_samples.max()), int(n_t))
    x = np.linspace(float(x_samples.min()), float(x_samples.max()), int(n_x))
    TT, XX = np.meshgrid(t, x, indexing="ij")
    pts = np.column_stack([t_samples, x_samples])
    U = griddata(pts, u_samples, (TT, XX), method="linear")
    holes = ~np.isfinite(U)
    if holes.any():
        U[holes] = griddata(pts, u_samples, (TT[holes], XX[holes]), method="nearest")
    return make_dataset(t, x, U, name=name)


# --------------------------------------------------------------------------- #
#  Discovery configuration for finite-domain transport                         #
# --------------------------------------------------------------------------- #
def bounded_transport_config(
    *,
    time_fractional: bool = True,
    space_side: Literal["left", "right", "symmetric"] = "left",
    alpha_range: tuple[float, float] = (0.50, 1.20),
    beta_range: tuple[float, float] = (0.70, 2.00),
    cmax: int = 3,
    p_values: tuple[int, ...] = (0, 1),
    seed: int = 0,
    **overrides: Any,
) -> DiscoveryConfig:
    """A finite-domain transport preset mirroring the bounded ADE control.

    Parameters
    ----------
    time_fractional:
        ``True`` searches a Caputo time order in ``alpha_range`` (anomalous /
        memory transport); ``False`` fixes the integer time derivative
        (``alpha_range`` is then ignored and the grid is pinned to 1).
    space_side:
        Direction of the one-sided spatial operator.  ``"left"`` is the causal
        (down-gradient, inflow-at-``x=0``) Grunwald stencil appropriate for an
        advective column flowing in ``+x``; use ``"right"`` for the reverse
        orientation, or ``"symmetric"`` for a Riesz-like two-sided dispersion on
        the bounded interval.
    alpha_range, beta_range:
        Continuous search ranges for the time and space orders.  ``beta_range``
        should bracket the dispersion order (often near 1.5--2 for fADE).
    p_values:
        Candidate integer powers; ``(0, 1)`` covers linear transport plus a
        Burgers-type advective nonlinearity.  Add ``2`` only with reason.

    Any further ``DiscoveryConfig`` field can be passed through ``overrides``.
    """
    af = (1.00,) if not time_fractional else (1.00,)
    alpha_grid = (
        _grid_with_required(1.00, 1.00, 1, (1.00,))
        if not time_fractional
        else _grid_with_required(alpha_range[0], alpha_range[1], 9, af)
    )
    base = dict(
        backend="regularized",
        alpha_grid=alpha_grid,
        beta_grid=_grid_with_required(beta_range[0], beta_range[1], 27, (1.00, 2.00)),
        cmax=cmax,
        p_values=tuple(p_values),
        max_patterns_per_c=None,
        maxiter=20,
        popsize=6,
        seed=seed,
        val_fraction=0.25,
        trim_t=3,
        trim_x=6,
        lam_t=1e-8,
        lam_x=1e-6,
        power_mode="positive",
        spectral_riesz=False,                 # finite-domain: NOT periodic Riesz
        regularized_space_side=space_side,    # one-sided GL/RL adjoint on the test side
        selection="elbow",
        exact_order_refit=True,
        auto_stop=True,
        auto_stop_min_c=2,
        auto_stop_patience=1,
        auto_stop_rel_improvement=0.03,
        auto_stop_log10_improvement=0.02,
        auto_stop_use_selection_stability=True,
        auto_stop_selection_patience=1,
    )
    base.update(overrides)
    return DiscoveryConfig(**base)


def _grid_with_required(lo: float, hi: float, n: int, required: tuple[float, ...]) -> NDArray[np.float64]:
    """Uniform grid on [lo, hi] with ``required`` values snapped in (mirror of dataset_configs)."""
    g = list(np.linspace(float(lo), float(hi), int(n)))
    for r in required:
        if not any(abs(r - v) < 1e-9 for v in g):
            g.append(float(r))
    return np.array(sorted(set(round(v, 10) for v in g)), dtype=float)


# --------------------------------------------------------------------------- #
#  Run discovery                                                               #
# --------------------------------------------------------------------------- #
def discover(
    data: GridDataset,
    config: DiscoveryConfig | None = None,
    *,
    test_budget: Literal["smoke", "standard", "paper"] = "standard",
    verbose: bool = True,
) -> dict[str, Any]:
    """Run weak-Pareto discovery on a loaded real field; return the summary dict.

    With no ``config`` a default finite-domain time-fractional transport preset
    is used.  The returned summary contains the selected model under
    ``summary["selected"]`` (orders, powers, coefficients) and the held-out weak
    validation error; no ground-truth comparison is performed.
    """
    if config is None:
        config = bounded_transport_config()
    summary = run_weak_pareto_discovery(data, config, test_budget=test_budget, verbose=verbose)
    if verbose:
        sel = summary["selected"]
        print("\nDiscovered finite-domain transport model:")
        print(f"  time order alpha = {sel.get('alpha'):.4f}")
        print(f"  spatial orders   = {[round(float(b), 4) for b in sel['beta_tuple']]}")
        print(f"  powers           = {[int(p) for p in sel['p_tuple']]}")
        print(f"  coefficients     = {[round(float(c), 5) for c in sel['coefficients']]}")
        print(f"  held-out weak residual E_val = {sel.get('val_rel_mse', float('nan')):.4e}")
    return summary


__all__ = [
    "make_dataset",
    "load_field_npz",
    "load_field_csv",
    "regrid_scattered",
    "bounded_transport_config",
    "discover",
]
