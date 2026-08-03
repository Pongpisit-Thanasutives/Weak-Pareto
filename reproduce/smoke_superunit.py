"""Focused clean/noisy smoke test for the superunit Caputo branch.

This diagnostic fixes the one-term support ``D_x^beta u`` and tests whether the
branch-aware search recovers a temporal order in ``(1, 2)`` and the expected
second-order spatial diffusion from semi-analytic data.  It is a software and
branch-recovery check, not a publication result.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
import time

import numpy as np

from fpde_datasets import make_superunit_fractional_diffusion
from pareto_fde_discovery import DiscoveryConfig, run_pareto_discovery
from weak_pareto_fde_discovery import run_weak_pareto_discovery

TRUE_ALPHA = 1.65
TRUE_BETA = 2.0
TRUE_COEFFICIENT = 0.12


def _config(seed: int, noise_percent: float) -> DiscoveryConfig:
    return DiscoveryConfig(
        backend="spectral_l1",
        alpha_grid=tuple(np.linspace(0.65, 1.85, 35)),
        beta_grid=tuple(np.linspace(1.50, 2.50, 25)),
        cmax=1,
        p_values=(0,),
        max_patterns_per_c=None,
        maxiter=8,
        popsize=5,
        polish=False,
        seed=int(seed),
        noise_percent=float(noise_percent),
        val_fraction=0.25,
        trim_t=2,
        trim_x=0,
        spectral_riesz=False,
        selection="elbow",
        exact_order_refit=True,
        exact_order_polish=True,
        auto_stop=False,
        progress=False,
        progress_de=False,
        ridge=1e-4,
    )


def _row(method: str, noise: float, seed: int, summary: dict, runtime: float) -> dict[str, object]:
    selected = summary["selected"]
    beta = float(selected["beta_tuple"][0])
    coefficient = float(selected["coefficients"][0])
    alpha = float(selected["alpha"])
    mode = str(selected["alpha_mode"])
    return {
        "method": method,
        "noise_percent": float(noise),
        "seed": int(seed),
        "alpha_mode": mode,
        "alpha": alpha,
        "beta": beta,
        "coefficient": coefficient,
        "e_alpha": abs(alpha - TRUE_ALPHA),
        "e_beta": abs(beta - TRUE_BETA),
        "e_coefficient": abs(coefficient - TRUE_COEFFICIENT) / TRUE_COEFFICIENT,
        "branch_recovered": mode == "fractional_superunit",
        "selected_equation": str(selected["equation"]),
        "runtime_seconds": float(runtime),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.5])
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = make_superunit_fractional_diffusion(alpha=TRUE_ALPHA, diffusivity=TRUE_COEFFICIENT)
    rows: list[dict[str, object]] = []
    for noise in args.noise:
        for seed in args.seeds:
            cfg = _config(seed, noise)
            t0 = time.perf_counter()
            weak = run_weak_pareto_discovery(
                data,
                cfg,
                output_dir=args.outdir / f"weak_noise{noise:g}_seed{seed}",
                verbose=False,
                test_budget="standard",
                test_counts=(16, 20),
            )
            rows.append(_row("Weak-Pareto", noise, seed, weak, time.perf_counter() - t0))

            strong_cfg = dataclasses.replace(cfg, exact_order_polish=False)
            t0 = time.perf_counter()
            strong = run_pareto_discovery(
                data,
                strong_cfg,
                output_dir=args.outdir / f"strong_noise{noise:g}_seed{seed}",
                verbose=False,
            )
            rows.append(_row("Strong Pareto", noise, seed, strong, time.perf_counter() - t0))

    csv_path = args.outdir / "superunit_smoke_per_run.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    weak_rows = [row for row in rows if row["method"] == "Weak-Pareto"]
    failures = [
        row for row in weak_rows
        if not bool(row["branch_recovered"])
        or float(row["e_alpha"]) > 0.15
        or float(row["e_beta"]) > 0.15
    ]
    payload = {
        "publication_quality": False,
        "purpose": "focused superunit branch-recovery smoke test",
        "support_fixed": "one linear spatial term with continuous beta",
        "truth": {"alpha": TRUE_ALPHA, "beta": TRUE_BETA, "coefficient": TRUE_COEFFICIENT},
        "rows": rows,
        "weak_acceptance_failures": failures,
    }
    json_path = args.outdir / "superunit_smoke_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        print(
            f"[superunit-smoke] {row['method']} noise={row['noise_percent']:g}% "
            f"seed={row['seed']} mode={row['alpha_mode']} alpha={row['alpha']:.4f} "
            f"beta={row['beta']:.4f} coeff={row['coefficient']:.4f}"
        )
    if failures:
        raise RuntimeError(f"Weak-Pareto superunit smoke acceptance failed for {len(failures)} run(s)")
    print(f"[superunit-smoke] passed; summary: {json_path}")


if __name__ == "__main__":
    main()
