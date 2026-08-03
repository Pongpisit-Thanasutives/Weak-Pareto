"""Monte Carlo check of Proposition 1's grid-refinement variance rates.

The experiment holds a smooth periodic separable test function fixed on the
physical unit square, refines n_t=n_x=n, and compares

* the weak feature <eta, X_beta^* phi>_h, whose variance is predicted to scale
  as n^-2 when both grid dimensions are refined; and
* one pointwise spectral derivative of eta, whose variance is predicted to
  scale as n^(2 beta).

Usage:
    python reproduce/check_variance_scaling.py
    python reproduce/check_variance_scaling.py --trials 2000 --out results/variance_scaling.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def run_check(
    *,
    beta: float = 1.5,
    grids: tuple[int, ...] = (24, 32, 48, 64, 96),
    trials: int = 2000,
    seed: int = 0,
) -> tuple[list[dict[str, float]], float, float]:
    if beta <= 0:
        raise ValueError("beta must be positive")
    if trials < 2:
        raise ValueError("trials must be at least 2")
    if any(n < 8 for n in grids):
        raise ValueError("all grid sizes must be at least 8")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    batch_size = 100

    for n in grids:
        h = 1.0 / float(n)
        t = np.arange(n, dtype=float) * h
        x = np.arange(n, dtype=float) * h

        # Fixed smooth physical test function phi(t,x)=rho(t) psi(x).
        rho = np.sin(2.0 * np.pi * t) + 0.5 * np.cos(4.0 * np.pi * t)
        psi = np.cos(2.0 * np.pi * x) + 0.3 * np.sin(6.0 * np.pi * x)

        kappa = 2.0 * np.pi * np.fft.fftfreq(n, d=h)
        multiplier = np.abs(kappa) ** beta
        multiplier[0] = 0.0
        adjoint_psi = np.fft.ifft(multiplier * np.fft.fft(psi)).real
        omega = rho[:, None] * adjoint_psi[None, :]

        weak_samples: list[np.ndarray] = []
        strong_samples: list[np.ndarray] = []
        for start in range(0, trials, batch_size):
            current = min(batch_size, trials - start)
            eta = rng.standard_normal((current, n, n))

            weak_samples.append((h * h) * np.sum(eta * omega[None, :, :], axis=(1, 2)))

            eta_hat = np.fft.fft(eta[:, 0, :], axis=-1)
            derivative = np.fft.ifft(multiplier[None, :] * eta_hat, axis=-1).real
            strong_samples.append(derivative[:, 0])

        weak = np.concatenate(weak_samples)
        strong = np.concatenate(strong_samples)
        rows.append(
            {
                "n": float(n),
                "h": h,
                "weak_variance": float(np.var(weak, ddof=1)),
                "strong_variance": float(np.var(strong, ddof=1)),
            }
        )

    log_n = np.log(np.array([row["n"] for row in rows], dtype=float))
    weak_slope = float(
        np.polyfit(log_n, np.log([row["weak_variance"] for row in rows]), 1)[0]
    )
    strong_slope = float(
        np.polyfit(log_n, np.log([row["strong_variance"] for row in rows]), 1)[0]
    )
    return rows, weak_slope, strong_slope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grids", type=int, nargs="+", default=[24, 32, 48, 64, 96])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows, weak_slope, strong_slope = run_check(
        beta=args.beta,
        grids=tuple(args.grids),
        trials=args.trials,
        seed=args.seed,
    )

    print("n, weak variance, strong variance")
    for row in rows:
        print(f"{int(row['n']):3d}, {row['weak_variance']:.6e}, {row['strong_variance']:.6e}")
    print(f"weak log-log slope:   {weak_slope:.3f} (theory: -2.000)")
    print(f"strong log-log slope: {strong_slope:.3f} (theory: {2.0 * args.beta:.3f})")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
