#!/usr/bin/env python3
"""Verify consistency of the released fractional operators against analytic pairs.

This complements the exact discrete-adjoint identity, which checks transpose
bookkeeping but not continuum accuracy. Two checks are reported:

1. Periodic Riesz: R_beta sin(m x) = -|m|^beta sin(m x). The FFT operator is
   spectrally exact up to floating-point roundoff for a resolved Fourier mode.
2. Subunit Caputo-L1: D_t^alpha t^q = Gamma(q+1)/Gamma(q-alpha+1)
   t^(q-alpha). The observed refinement rate should approach 2-alpha for
   smooth data.
3. Superunit Caputo composition: the separately implemented 1 < alpha < 2
   branch is checked against the same analytic power-law identity. This is a
   consistency/convergence check of the active code path, not a benchmark-level
   recovery claim.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.special import gamma

from fractional_weak_form import caputo_l1_matrix, periodic_riesz_on_tests


def run() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []

    beta = 1.7
    mode = 3
    for n in (32, 64, 128, 256):
        x = np.arange(n, dtype=float) * (2.0 * np.pi / n)
        f = np.sin(mode * x)
        numerical = periodic_riesz_on_tests(f[None, :], x, beta)[0]
        exact = -(abs(mode) ** beta) * f
        rel = float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))
        rows.append({"check": "periodic_riesz", "n": n, "error": rel, "rate": float("nan")})

    q = 3.0
    for alpha, label, valid_start in (
        (0.7, "caputo_l1_subunit", 1),
        (1.3, "caputo_l1_superunit", 2),
    ):
        previous_error: float | None = None
        previous_h: float | None = None
        for n in (65, 129, 257, 513):
            t = np.linspace(0.0, 1.0, n)
            h = float(t[1] - t[0])
            f = t**q
            numerical = np.asarray(caputo_l1_matrix(n, alpha, h) @ f, dtype=float)
            exact = gamma(q + 1.0) / gamma(q - alpha + 1.0) * t ** (q - alpha)
            rel = float(
                np.linalg.norm(numerical[valid_start:] - exact[valid_start:])
                / np.linalg.norm(exact[valid_start:])
            )
            rate = float("nan")
            if previous_error is not None and previous_h is not None:
                rate = float(np.log(previous_error / rel) / np.log(previous_h / h))
            rows.append({"check": label, "n": n, "error": rel, "rate": rate})
            previous_error, previous_h = rel, h
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/operator_convergence.csv")
    args = parser.parse_args()
    rows = run()
    for row in rows:
        print(row)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "n", "error", "rate"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
