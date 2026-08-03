#!/usr/bin/env python3
"""Alternative-noise and reduced-sampling robustness checks.

This experiment complements the manuscript's multiplicative-uniform noise
study. It evaluates FADE and fractional Burgers under:

1. additive Gaussian noise with standard deviation ``level`` percent of the
   clean field's standard deviation;
2. regular 50% temporal sampling (every second snapshot), with no imputation;
3. the combination of 1 and 2 (optional).

The exact same perturbed or subsampled field is passed to Weak-Pareto and the
strong-form Pareto baseline. Publication mode uses five seeds and the paper DE
budget. Results are appended to JSONL immediately, so long local runs can be
resumed safely after interruption.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from dataset_configs import load_benchmark, with_uniform_order_grids
from fpde_datasets import GridDataset
from pareto_fde_discovery import run_pareto_discovery
from weak_pareto_fde_discovery import run_weak_pareto_discovery
from reproduce._repro_common import (
    NICE_NAME,
    de_budget,
    matched_errors,
    _attach_full_data_rel_l2,
)

BENCHMARKS = ("paper_FADE_tsfade_fft", "synthetic_fractional_burgers")
METHODS = ("Weak-Pareto", "Strong Pareto")


def add_additive_gaussian_noise(U: np.ndarray, percent: float, seed: int) -> np.ndarray:
    """Add zero-mean Gaussian noise scaled by the clean-field standard deviation."""
    U = np.asarray(U, dtype=float)
    if percent <= 0:
        return U.copy()
    rng = np.random.default_rng(seed)
    sigma = 0.01 * float(percent) * float(np.std(U))
    return U + sigma * rng.standard_normal(U.shape)


def replace_field(data: GridDataset, U: np.ndarray, suffix: str) -> GridDataset:
    return GridDataset(
        U=np.asarray(U, dtype=float),
        t=np.asarray(data.t, dtype=float),
        x=np.asarray(data.x, dtype=float),
        name=f"{data.name}_{suffix}",
        truth=data.truth,
        recommended_backend=data.recommended_backend,
    )


def half_time_sampling(data: GridDataset) -> GridDataset:
    """Retain every second time snapshot (approximately half the observations)."""
    idx = np.arange(0, len(data.t), 2, dtype=int)
    return GridDataset(
        U=np.asarray(data.U[idx, :], dtype=float),
        t=np.asarray(data.t[idx], dtype=float),
        x=np.asarray(data.x, dtype=float),
        name=f"{data.name}_half_time",
        truth=data.truth,
        recommended_backend=data.recommended_backend,
    )


def make_config(name: str, seed: int, fast: bool, *, weak: bool):
    _, cfg, truth = load_benchmark(name, profile="paper", seed=seed)
    cfg = dataclasses.replace(
        cfg,
        noise_percent=0.0,  # perturbations are applied explicitly above
        seed=int(seed),
        progress=False,
        exact_order_refit=True,
        exact_order_polish=weak,
        cmax=4,
        **de_budget(fast),
    )
    return with_uniform_order_grids(cfg), truth


def make_data(name: str, condition: str, seed: int, noise_percent: float) -> tuple[GridDataset, Any]:
    clean, _, truth = load_benchmark(name, profile="paper", seed=seed)
    if condition == "additive_gaussian":
        data = replace_field(
            clean,
            add_additive_gaussian_noise(clean.U, noise_percent, seed),
            f"gaussian{noise_percent:g}",
        )
    elif condition == "half_time_sampling":
        data = half_time_sampling(clean)
    elif condition == "half_time_gaussian":
        sparse = half_time_sampling(clean)
        data = replace_field(
            sparse,
            add_additive_gaussian_noise(sparse.U, noise_percent, seed),
            f"half_time_gaussian{noise_percent:g}",
        )
    else:
        raise ValueError(condition)
    return data, truth


def run_method(
    name: str,
    condition: str,
    seed: int,
    fast: bool,
    noise_percent: float,
    method: str,
) -> dict[str, Any]:
    data, truth = make_data(name, condition, seed, noise_percent)
    weak = method == "Weak-Pareto"
    cfg, _ = make_config(name, seed, fast, weak=weak)
    if weak:
        summary = run_weak_pareto_discovery(data, cfg, test_budget="paper", verbose=False)
    elif method == "Strong Pareto":
        summary = run_pareto_discovery(data, cfg, verbose=False)
        _attach_full_data_rel_l2(data, cfg, summary)
    else:
        raise ValueError(method)
    metrics = matched_errors(name, summary["selected"], truth)
    return {
        "benchmark": NICE_NAME[name],
        "dataset_name": name,
        "condition": condition,
        "noise_percent": float(noise_percent if "gaussian" in condition else 0.0),
        "sampling_fraction": 0.5 if "half_time" in condition else 1.0,
        "seed": int(seed),
        "method": method,
        "support_power_ok": bool(metrics["support_power_ok"]),
        "operator_structure_ok": bool(metrics["operator_structure_ok"]),
        "e_alpha": float(metrics["e_alpha"]),
        "e_beta_max": float(metrics["e_beta_max"]),
        "e_xi_max": float(metrics["e_xi_max"]),
        "e_xi_l2": float(metrics["e_xi_l2"]),
        "fit_residual": float(metrics["full_data_rel_l2"]),
        "selected_equation": summary["selected"].get("equation", ""),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["benchmark"], row["condition"], row["method"]), []).append(row)
    out = []
    for (benchmark, condition, method), rs in groups.items():
        correct = [r for r in rs if r["support_power_ok"]]

        def stats_or_nan(key: str) -> tuple[float, float]:
            vals = np.asarray([r[key] for r in correct if np.isfinite(r[key])], dtype=float)
            if not len(vals):
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        ea_m, ea_s = stats_or_nan("e_alpha")
        eb_m, eb_s = stats_or_nan("e_beta_max")
        ex_m, ex_s = stats_or_nan("e_xi_max")
        ex2_m, ex2_s = stats_or_nan("e_xi_l2")
        fit = np.asarray([r["fit_residual"] for r in rs], dtype=float)
        out.append({
            "benchmark": benchmark,
            "condition": condition,
            "method": method,
            "n_seeds": len(rs),
            "support_power_recovery": sum(r["support_power_ok"] for r in rs),
            "operator_structure_recovery": sum(r["operator_structure_ok"] for r in rs),
            "e_alpha_mean": ea_m,
            "e_alpha_sd": ea_s,
            "e_beta_max_mean": eb_m,
            "e_beta_max_sd": eb_s,
            "e_xi_max_mean": ex_m,
            "e_xi_max_sd": ex_s,
            "e_xi_l2_mean": ex2_m,
            "e_xi_l2_sd": ex2_s,
            "fit_residual_mean": float(np.mean(fit)),
            "fit_residual_sd": float(np.std(fit, ddof=1)) if len(fit) > 1 else 0.0,
        })
    return sorted(out, key=lambda r: (r["benchmark"], r["condition"], r["method"]))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_key(row: dict[str, Any]) -> tuple:
    return (
        row["dataset_name"], row["condition"], float(row["noise_percent"]),
        int(row["seed"]), row["method"],
    )


def _run_task(task):
    name, condition, seed, fast, noise, method = task
    return run_method(name, condition, seed, fast, noise, method)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--noise", type=float, default=10.0,
                        help="Gaussian standard deviation as percent of clean-field standard deviation")
    parser.add_argument("--conditions", nargs="+", default=["additive_gaussian", "half_time_sampling"],
                        choices=["additive_gaussian", "half_time_sampling", "half_time_gaussian"])
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="explicit seeds; defaults to 0 1 in --fast mode and 0 ... 4 otherwise")
    parser.add_argument("--outdir", default="results/alternative_robustness")
    parser.add_argument("--restart", action="store_true", help="discard an existing incremental JSONL file")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("FPDE_REPRO_JOBS", "1")),
                        help="parallel independent runs; completed rows remain resumable")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else ([0, 1] if args.fast else [0, 1, 2, 3, 4])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl = outdir / "per_seed.jsonl"
    if args.restart and jsonl.exists():
        jsonl.unlink()
    rows = read_jsonl(jsonl)
    completed = {row_key(row) for row in rows}

    tasks = []
    for condition in args.conditions:
        for name in args.benchmarks:
            for seed in seeds:
                for method in args.methods:
                    expected = (name, condition, float(args.noise if "gaussian" in condition else 0.0), seed, method)
                    if expected in completed:
                        print(f"[skip] {condition} {name} seed={seed} {method}", flush=True)
                    else:
                        tasks.append((name, condition, seed, args.fast, args.noise, method))

    jobs = max(1, int(args.jobs))
    print(f"[alternative robustness] {len(tasks)} missing runs; jobs={jobs}", flush=True)
    with jsonl.open("a") as stream:
        if jobs == 1:
            iterator = map(_run_task, tasks)
            for i, row in enumerate(iterator, 1):
                stream.write(json.dumps(row) + "\n"); stream.flush()
                rows.append(row); completed.add(row_key(row))
                print(f"[done {i}/{len(tasks)}] {row}", flush=True)
        else:
            with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = {ex.submit(_run_task, task): task for task in tasks}
                done_n = 0
                for fut in cf.as_completed(futs):
                    row = fut.result(); done_n += 1
                    stream.write(json.dumps(row) + "\n"); stream.flush()
                    rows.append(row); completed.add(row_key(row))
                    print(f"[done {done_n}/{len(tasks)}] {row}", flush=True)

    # De-duplicate in case separate invocations targeted overlapping subsets.
    unique = {row_key(row): row for row in rows}
    rows = sorted(unique.values(), key=lambda r: row_key(r))
    with (outdir / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    agg = aggregate(rows)
    with (outdir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(agg[0]))
        writer.writeheader()
        writer.writerows(agg)
    (outdir / "manifest.json").write_text(json.dumps({
        "noise_definition": "additive Gaussian sd = noise_percent/100 * std(clean field)",
        "sampling_definition": "retain every second time snapshot; no interpolation or imputation",
        "seeds_requested": seeds,
        "conditions_requested": args.conditions,
        "benchmarks_requested": args.benchmarks,
        "methods_requested": args.methods,
        "noise_percent": args.noise,
        "fast": args.fast,
        "incremental_output": str(jsonl),
    }, indent=2) + "\n")
    for row in agg:
        print(row)


if __name__ == "__main__":
    main()
