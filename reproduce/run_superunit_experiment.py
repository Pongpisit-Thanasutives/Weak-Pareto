#!/usr/bin/env python3
"""Run the dedicated clean/noisy superunit Caputo branch experiment.

The experiment uses semi-analytic data for
``D_t^1.65 u = 0.12 D_x^2 u`` with zero initial velocity.  Its primary protocol
fixes the one-term linear support while searching the temporal branch and the
continuous temporal/spatial orders.  This isolates the previously unreported
superunit regime; it should not be described as unrestricted support discovery.

Results are appended to JSONL and can be resumed.  Weak-Pareto and the matched
strong-form Pareto baseline receive the same noisy realisation for every seed.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import dataclasses
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from dataset_configs import load_benchmark, with_uniform_order_grids
from pareto_fde_discovery import run_pareto_discovery
from weak_pareto_fde_discovery import run_weak_pareto_discovery
from reproduce._repro_common import _attach_full_data_rel_l2

TRUE_ALPHA = 1.65
TRUE_BETA = 2.0
TRUE_COEFFICIENT = 0.12
METHODS = ("Weak-Pareto", "Strong Pareto")
WEAK_TIME_FORM = "derivative"
WEAK_TIME_DISCRETIZATION = "caputo_l1_adjoint"


def _config(seed: int, noise: float, fast: bool, weak: bool):
    _, cfg, _ = load_benchmark(
        "synthetic_superunit_fractional_diffusion",
        profile="paper",
        seed=seed,
    )
    cfg = with_uniform_order_grids(cfg)
    return dataclasses.replace(
        cfg,
        cmax=1,
        p_values=(0,),
        noise_percent=float(noise),
        seed=int(seed),
        maxiter=5 if fast else int(cfg.maxiter),
        popsize=4 if fast else int(cfg.popsize),
        exact_order_refit=True,
        exact_order_polish=bool(weak),
        auto_stop=False,
        progress=False,
        progress_de=False,
        ridge=1e-4,
    )


def _run_one(task: tuple[str, float, int, bool]) -> dict[str, Any]:
    method, noise, seed, fast = task
    data, _, _ = load_benchmark(
        "synthetic_superunit_fractional_diffusion",
        profile="paper",
        seed=seed,
    )
    weak = method == "Weak-Pareto"
    cfg = _config(seed, noise, fast, weak)
    started = time.perf_counter()
    if weak:
        summary = run_weak_pareto_discovery(
            data,
            cfg,
            verbose=False,
            test_budget="standard" if fast else "paper",
            test_counts=(16, 20) if fast else (24, 32),
            time_form=WEAK_TIME_FORM,
        )
    else:
        summary = run_pareto_discovery(data, cfg, verbose=False)
        _attach_full_data_rel_l2(data, cfg, summary)
    runtime_seconds = time.perf_counter() - started
    selected = summary["selected"]
    alpha = float(selected["alpha"])
    beta = float(selected["beta_tuple"][0])
    coefficient = float(selected["coefficients"][0])
    mode = str(selected["alpha_mode"])
    return {
        "method": method,
        "noise_percent": float(noise),
        "seed": int(seed),
        "alpha_mode": mode,
        "alpha": alpha,
        "beta": beta,
        "coefficient": coefficient,
        "branch_recovered": mode == "fractional_superunit",
        "operator_recovered": bool(
            mode == "fractional_superunit"
            and abs(alpha - TRUE_ALPHA) <= 0.15
            and abs(beta - TRUE_BETA) <= 0.15
        ),
        "e_alpha": abs(alpha - TRUE_ALPHA),
        "e_beta": abs(beta - TRUE_BETA),
        "e_coefficient_relative": abs(coefficient - TRUE_COEFFICIENT) / TRUE_COEFFICIENT,
        "val_rel_mse": float(selected["val_rel_mse"]),
        "full_data_rel_l2": float(selected.get("full_data_rel_l2", float("nan"))),
        "runtime_seconds": float(runtime_seconds),
        "selected_equation": str(selected["equation"]),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in METHODS:
        for noise in sorted({float(row["noise_percent"]) for row in rows}):
            group = [row for row in rows if row["method"] == method and float(row["noise_percent"]) == noise]
            if not group:
                continue
            def stats(key: str) -> tuple[float, float]:
                vals = np.asarray([float(row[key]) for row in group], dtype=float)
                return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ea, ea_sd = stats("e_alpha")
            eb, eb_sd = stats("e_beta")
            ec, ec_sd = stats("e_coefficient_relative")
            out.append({
                "method": method,
                "noise_percent": noise,
                "n_seeds": len(group),
                "branch_recovery": sum(bool(row["branch_recovered"]) for row in group),
                "operator_recovery": sum(bool(row["operator_recovered"]) for row in group),
                "e_alpha_mean": ea,
                "e_alpha_sd": ea_sd,
                "e_beta_mean": eb,
                "e_beta_sd": eb_sd,
                "e_coefficient_relative_mean": ec,
                "e_coefficient_relative_sd": ec_sd,
            })
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--jobs", "-j", type=int, default=2)
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    args.outdir.mkdir(parents=True, exist_ok=True)

    jsonl_path = args.outdir / "per_seed.jsonl"
    rows = _read_jsonl(jsonl_path)
    done = {(row["method"], float(row["noise_percent"]), int(row["seed"])) for row in rows}
    tasks = [
        (method, float(noise), int(seed), bool(args.fast))
        for method in args.methods
        for noise in args.noise
        for seed in args.seeds
        if (method, float(noise), int(seed)) not in done
    ]

    if tasks:
        with cf.ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
            for row in executor.map(_run_one, tasks):
                rows.append(row)
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=True) + "\n")
                print(
                    f"[superunit] {row['method']} noise={row['noise_percent']:g}% seed={row['seed']} "
                    f"mode={row['alpha_mode']} alpha={row['alpha']:.4f} beta={row['beta']:.4f}"
                )

    rows = sorted(rows, key=lambda row: (row["method"], float(row["noise_percent"]), int(row["seed"])))
    per_seed_csv = args.outdir / "per_seed.csv"
    if rows:
        with per_seed_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = _aggregate(rows)
    summary_csv = args.outdir / "summary.csv"
    if summary:
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    manifest = {
        "publication_quality": not bool(args.fast),
        "purpose": "focused superunit temporal-branch recovery experiment",
        "support_fixed": "one linear term; beta and temporal branch/order are discovered",
        "truth": {"alpha": TRUE_ALPHA, "beta": TRUE_BETA, "coefficient": TRUE_COEFFICIENT},
        "noise_percent": [float(value) for value in args.noise],
        "seeds": [int(value) for value in args.seeds],
        "methods": list(args.methods),
        "weak_time_form": WEAK_TIME_FORM,
        "weak_time_discretization": WEAK_TIME_DISCRETIZATION,
        "weak_initial_rate_treatment": "implicit_D1_in_composed_L1",
        "weak_endpoint_noise_scaling": "O(sigma^2 h_x h_t^-2) for a fixed continuum separable test before scalar row rescaling under iid additive noise",
        "jobs": int(args.jobs),
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[superunit] wrote {per_seed_csv}")
    print(f"[superunit] wrote {summary_csv}")


if __name__ == "__main__":
    main()
