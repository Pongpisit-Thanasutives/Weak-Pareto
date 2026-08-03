"""Precompute shared deterministic discovery runs in parallel.

This script only fills the persistent reproduction cache used by the table and
figure generators.  It does not change any numerical budget, seed, objective,
or discovery code.  Re-running it is safe: completed configurations are loaded
from cache and skipped quickly.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import time
from typing import Any

from _repro_common import (
    APPENDIX_ADJUSTMENTS,
    APPENDIX_BENCHMARKS,
    MAIN_BENCHMARKS,
    ROBUSTNESS_BENCHMARKS,
    default_noise,
    default_seeds,
    run_strong,
    run_weak,
)

NOISY_LEVEL = {
    "paper_FADE_tsfade_fft": 10.0,
    "synthetic_fractional_burgers": 10.0,
    "synthetic_space_fractional_RD": 5.0,
    "synthetic_time_space_fractional_RD": 5.0,
}
RD_BENCHMARKS = ("synthetic_space_fractional_RD", "synthetic_time_space_fractional_RD")


def _key(task: dict[str, Any]) -> tuple:
    overrides = tuple(sorted((task.get("overrides") or {}).items()))
    return (
        task["method"], task["name"], float(task["noise"]), int(task["seed"]),
        bool(task["fast"]), overrides,
    )


def build_tasks(fast: bool, smoke: bool = False) -> list[dict[str, Any]]:
    seeds = default_seeds(fast)
    noises = default_noise(fast)
    tasks: list[dict[str, Any]] = []
    if smoke:
        return [
            dict(method="weak", name="synthetic_fractional_burgers", noise=10.0, seed=0, fast=True, overrides=None),
            dict(method="strong", name="synthetic_fractional_burgers", noise=10.0, seed=0, fast=True, overrides=None),
        ]

    def add(method: str, name: str, noise: float, seed: int, overrides=None):
        tasks.append(dict(method=method, name=name, noise=float(noise), seed=int(seed),
                          fast=bool(fast), overrides=overrides))

    # Main table.
    for name in MAIN_BENCHMARKS:
        for seed in seeds:
            add("weak", name, 10.0, seed)

    # Weak-vs-strong robustness table and figure.
    for name in ROBUSTNESS_BENCHMARKS:
        for noise in noises:
            for seed in seeds:
                add("weak", name, noise, seed)
                add("strong", name, noise, seed)

    # Riesz order-sensitivity table.
    rd_noises = (2.0, 5.0, 10.0)
    for name in RD_BENCHMARKS:
        for noise in rd_noises:
            for seed in seeds:
                add("weak", name, noise, seed)

    # Appendix default and adjusted cases.
    for name in APPENDIX_BENCHMARKS:
        for seed in seeds:
            add("weak", name, 10.0, seed)
            add("weak", name, 10.0, seed, APPENDIX_ADJUSTMENTS[name])

    # Representative equations and forward validation (main benchmarks).
    for name in MAIN_BENCHMARKS:
        for noise in (0.0, NOISY_LEVEL[name]):
            for seed in seeds:
                add("weak", name, noise, seed)

    unique: dict[tuple, dict[str, Any]] = {}
    for task in tasks:
        unique[_key(task)] = task
    return list(unique.values())


def _run(task: dict[str, Any]) -> tuple[str, str, float, int, float]:
    t0 = time.perf_counter()
    kwargs = dict(
        name=task["name"], noise=task["noise"], seed=task["seed"],
        fast=task["fast"], overrides=task.get("overrides"), use_cache=True,
    )
    if task["method"] == "weak":
        run_weak(**kwargs)
    else:
        run_strong(**kwargs)
    return task["method"], task["name"], task["noise"], task["seed"], time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("FPDE_REPRO_JOBS", "2")))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    jobs = max(1, int(args.jobs))
    tasks = build_tasks(args.fast, smoke=args.smoke)
    print(f"[precompute] {len(tasks)} unique deterministic runs; jobs={jobs}", flush=True)
    t0 = time.perf_counter()
    if jobs == 1:
        for i, task in enumerate(tasks, 1):
            method, name, noise, seed, dt = _run(task)
            print(f"[precompute {i:>3d}/{len(tasks)}] {method:6s} {name:38s} "
                  f"noise={noise:>5g}% seed={seed}  {dt:7.1f}s", flush=True)
    else:
        with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
            future_to_task = {ex.submit(_run, task): task for task in tasks}
            done = 0
            for fut in cf.as_completed(future_to_task):
                task = future_to_task[fut]
                done += 1
                try:
                    method, name, noise, seed, dt = fut.result()
                    print(f"[precompute {done:>3d}/{len(tasks)}] {method:6s} {name:38s} "
                          f"noise={noise:>5g}% seed={seed}  {dt:7.1f}s", flush=True)
                except Exception as exc:
                    print(f"[precompute FAILED] {task}: {exc!r}", flush=True)
                    raise
    print(f"[precompute] complete in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
