#!/usr/bin/env python3
"""Integral-form identification of a fractional Kelvin model from creep data.

This example uses the irregular, naturally noisy frozen-soil strain records
reported by Yu et al. (Nonlinear Dynamics, 2025).  It is deliberately framed as
an order-and-parameter identification problem within the physically motivated
fractional Kelvin support,

    D_t^alpha epsilon = c0 + c1 epsilon,
    c0 = sigma / eta,      c1 = -E / eta.

Applying the Riemann--Liouville integral I_t^alpha to the Caputo equation gives

    epsilon(t) - epsilon(0)
        = c0 t^alpha / Gamma(alpha + 1) + c1 I_t^alpha epsilon(t).

The right-hand side contains smoothing integrals rather than pointwise
fractional derivatives.  For each alpha, c0 and c1 are obtained by linear least
squares; alpha is selected by a bounded one-dimensional optimisation.

The external data files are not redistributed.  See
``external_data/frozen_soil/README.md`` for acquisition instructions.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.io import loadmat
from scipy.optimize import minimize_scalar
from scipy.special import gamma


@dataclass(frozen=True)
class SoilSpec:
    """Store one frozen-soil dataset and its published reference parameters."""
    name: str
    filename: str
    stress_mpa: float
    reference_alpha: float
    reference_eta: float
    reference_E: float


SOILS = (
    SoilSpec("clay", "creep.mat", 1.11, 0.371, 0.1030, 0.320),
    SoilSpec("silt", "silt.mat", 1.14, 0.562, 0.0778, 0.442),
)


def _deduplicate_by_mean(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort observations and average repeated timestamps."""
    order = np.argsort(t)
    t = np.asarray(t, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    unique_t = np.unique(t)
    unique_y = np.array([np.mean(y[t == ti]) for ti in unique_t], dtype=float)
    return unique_t, unique_y


def load_creep(path: Path, n_grid: int = 400) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load irregular observations and construct a shape-preserving quadrature grid.

    The fractional Kelvin step-response condition epsilon(0)=0 is prepended.
    PCHIP is used only to evaluate the fractional integral on a uniform grid;
    no pointwise derivative is taken from the interpolant.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. See external_data/frozen_soil/README.md for acquisition instructions."
        )
    mat = loadmat(path)
    t_obs = np.asarray(mat["t"], dtype=float).ravel()
    eps_obs = np.asarray(mat["ex"], dtype=float).ravel()
    n_raw = int(len(t_obs))
    t_obs, eps_obs = _deduplicate_by_mean(t_obs, eps_obs)

    if t_obs[0] > 0.0:
        t_aug = np.concatenate(([0.0], t_obs))
        eps_aug = np.concatenate(([0.0], eps_obs))
    else:
        t_aug, eps_aug = t_obs, eps_obs

    t = np.linspace(0.0, float(t_aug[-1]), int(n_grid))
    eps = np.asarray(PchipInterpolator(t_aug, eps_aug)(t), dtype=float)
    return n_raw, t_obs, eps_obs, t, eps


def fractional_integral_piecewise_constant(values: np.ndarray, h: float, alpha: float) -> np.ndarray:
    """Left-endpoint product integration for I_t^alpha values on a uniform grid."""
    values = np.asarray(values, dtype=float)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    n = len(values)
    out = np.zeros(n, dtype=float)
    k = np.arange(n - 1, dtype=float)
    weights = (h**alpha / gamma(alpha + 1.0)) * ((k + 1.0) ** alpha - k**alpha)
    for i in range(1, n):
        # Interval j contributes with lag i-j; reverse the first i weights.
        out[i] = np.dot(weights[:i][::-1], values[:i])
    return out


def fit_at_order(t: np.ndarray, eps: np.ndarray, alpha: float) -> dict:
    """Fit the two Kelvin coefficients at a fixed fractional order."""
    h = float(t[1] - t[0])
    integral = fractional_integral_piecewise_constant(eps, h, alpha)
    design = np.column_stack((t**alpha / gamma(alpha + 1.0), integral))
    rows = np.arange(1, len(t))  # the t=0 equation is identically zero
    coeff, *_ = np.linalg.lstsq(design[rows], eps[rows], rcond=None)
    fitted = design @ coeff
    residual = eps - fitted
    rel = float(np.linalg.norm(residual[rows]) / (np.linalg.norm(eps[rows]) + 1e-14))
    return {
        "alpha": float(alpha),
        "c0": float(coeff[0]),
        "c1": float(coeff[1]),
        "relative_integral_residual": rel,
        "fitted": fitted,
    }


def identify(path: Path, spec: SoilSpec, n_grid: int = 400) -> dict:
    """Estimate the fractional Kelvin order and physical parameters for one soil."""
    n_raw, t_obs, eps_obs, t, eps = load_creep(path, n_grid=n_grid)

    def objective(alpha: float) -> float:
        return fit_at_order(t, eps, float(alpha))["relative_integral_residual"]

    opt = minimize_scalar(
        objective,
        bounds=(0.10, 0.90),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 250},
    )
    fit = fit_at_order(t, eps, float(opt.x))
    c0, c1 = fit["c0"], fit["c1"]
    eta = float(spec.stress_mpa / c0)
    elasticity = float(-c1 * eta)
    tau_r = float((eta / elasticity) ** (1.0 / fit["alpha"]))
    reference_tau_r = float((spec.reference_eta / spec.reference_E) ** (1.0 / spec.reference_alpha))

    return {
        "soil": spec.name,
        "n_raw_observations": n_raw,
        "n_unique_observations": int(len(t_obs)),
        "n_quadrature_grid": int(len(t)),
        "stress_mpa": spec.stress_mpa,
        "alpha": fit["alpha"],
        "c0": c0,
        "c1": c1,
        "eta": eta,
        "E": elasticity,
        "tau_r": tau_r,
        "relative_integral_residual": fit["relative_integral_residual"],
        "reference_alpha": spec.reference_alpha,
        "reference_eta": spec.reference_eta,
        "reference_E": spec.reference_E,
        "reference_tau_r": reference_tau_r,
        "alpha_abs_difference": abs(fit["alpha"] - spec.reference_alpha),
        "eta_rel_difference": abs(eta - spec.reference_eta) / spec.reference_eta,
        "E_rel_difference": abs(elasticity - spec.reference_E) / spec.reference_E,
        "tau_r_rel_difference": abs(tau_r - reference_tau_r) / reference_tau_r,
        "data_time_min": float(t_obs.min()),
        "data_time_max": float(t_obs.max()),
        "model": "D_t^alpha epsilon = c0 + c1 epsilon",
        "eta_units": "MPa * time^alpha",
        "tau_r_units": "source-data time unit",
        "method_note": (
            "Order and parameters are identified within the physically motivated "
            "fractional Kelvin support; eta values at different alpha have different "
            "dimensions, so tau_r=(eta/E)^(1/alpha) is the comparable time scale."
        ),
    }


def main() -> None:
    """Parse command-line options and write the frozen-soil identification outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="external_data/frozen_soil")
    parser.add_argument("--outdir", default="results/frozen_soil_creep")
    parser.add_argument("--n-grid", type=int, default=400)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in SOILS:
        result = identify(data_dir / spec.filename, spec, n_grid=args.n_grid)
        rows.append(result)
        (outdir / f"{spec.name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))

    fields = [
        "soil", "n_raw_observations", "n_unique_observations", "stress_mpa", "alpha", "eta", "E", "tau_r",
        "reference_alpha", "reference_eta", "reference_E", "reference_tau_r",
        "relative_integral_residual", "c0", "c1",
    ]
    with (outdir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (outdir / "manifest.json").write_text(json.dumps({
        "model": "D_t^alpha epsilon = sigma/eta - (E/eta) epsilon",
        "integral_form": "epsilon-epsilon(0) = c0*t^alpha/Gamma(alpha+1) + c1*I_t^alpha epsilon",
        "quadrature": "left-endpoint product integration on a 400-point PCHIP grid",
        "alpha_bounds": [0.10, 0.90],
        "data_files_redistributed": False,
        "soil_specs": [asdict(spec) for spec in SOILS],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
