#!/usr/bin/env python3
"""Monte Carlo checks for additive and multiplicative nonlinear-feature bias.

For the raw p=1 weak feature

    theta_h(u) = h u^T A^{*,h} D_phi u,

independent additive noise with variance sigma^2 gives the trace bias

    sigma^2 h tr(A^{*,h} D_phi).

For multiplicative noise ``u = u* (1 + rho zeta)`` with
``Var(zeta)=sigma_zeta^2``, the corresponding conditional bias is

    rho^2 sigma_zeta^2 h tr(D_{u*} A^{*,h} D_phi D_{u*}).

The script checks both formulas for the periodic first derivative and a
noninteger directional derivative.  It applies to the raw nonlinear feature;
positivity clipping changes the calculation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def directional_matrix(n: int, beta: float, length: float = 2.0 * np.pi) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.arange(n, dtype=float) * (length / n)
    h = length / n
    kappa = 2.0 * np.pi * np.fft.fftfreq(n, d=h)
    multiplier = (1j * kappa).astype(complex) ** float(beta)
    multiplier[np.isclose(kappa, 0.0)] = 0.0
    identity = np.eye(n)
    matrix = np.fft.ifft(np.fft.fft(identity, axis=0) * multiplier[:, None], axis=0).real
    return x, matrix, h


def quadratic_value(u: np.ndarray, qmat: np.ndarray, h: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        return np.asarray(h * u @ qmat @ u)
    return h * np.einsum("bi,ij,bj->b", u, qmat, u, optimize=True)


def run(samples: int = 10000, sigma: float = 0.2, rho: float = 0.2, seed: int = 0) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for n in (64, 128, 256):
        for beta in (1.0, 1.7):
            x, operator, h = directional_matrix(n, beta)
            distance = (x - np.pi + np.pi) % (2.0 * np.pi) - np.pi
            phi = np.exp(-0.5 * (distance / 0.7) ** 2)
            qmat = operator.T @ np.diag(phi)
            ustar = 1.0 + 0.25 * np.cos(x) + 0.15 * np.sin(2.0 * x)
            clean = float(quadratic_value(ustar, qmat, h))
            n_samples = samples if n < 256 else max(4000, samples // 2)

            additive = rng.normal(0.0, sigma, size=(n_samples, n))
            additive_values = quadratic_value(ustar[None, :] + additive, qmat, h) - clean
            predicted_add = float(sigma**2 * h * np.trace(qmat))
            rows.append({
                "noise_model": "additive_gaussian",
                "n": n,
                "beta": beta,
                "predicted_bias": predicted_add,
                "observed_bias": float(np.mean(additive_values)),
                "monte_carlo_standard_error": float(np.std(additive_values, ddof=1) / np.sqrt(n_samples)),
            })

            zeta = rng.uniform(-1.0, 1.0, size=(n_samples, n))
            noisy_mult = ustar[None, :] * (1.0 + float(rho) * zeta)
            mult_values = quadratic_value(noisy_mult, qmat, h) - clean
            Du = np.diag(ustar)
            predicted_mult = float(rho**2 * (1.0 / 3.0) * h * np.trace(Du @ qmat @ Du))
            rows.append({
                "noise_model": "multiplicative_uniform",
                "n": n,
                "beta": beta,
                "predicted_bias": predicted_mult,
                "observed_bias": float(np.mean(mult_values)),
                "monte_carlo_standard_error": float(np.std(mult_values, ddof=1) / np.sqrt(n_samples)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/nonlinear_bias.csv")
    args = parser.parse_args()
    rows = run(args.samples, args.sigma, args.rho, args.seed)
    for row in rows:
        print(row)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
