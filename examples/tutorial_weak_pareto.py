#!/usr/bin/env python3
"""End-to-end tutorial for weak-form FPDE discovery with Weak-Pareto.

The tutorial uses the bundled FADE benchmark and deliberately small optimizer
budgets so it runs quickly. It separates the two conceptual stages:

1. construct an adjoint-consistent weak candidate library; and
2. run Pareto-based best-subset selection over support size and orders.

The publication experiments use broader power sets, support sizes, five seeds,
and the full optimizer budget; see ``docs/PAPER_RESULTS_GUIDE.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_configs import benchmark_spec, with_uniform_order_grids  # noqa: E402
from fractional_weak_form import fit_least_squares  # noqa: E402
from weak_pareto_fde_discovery import (  # noqa: E402
    build_best_subset_pareto_problem,
    build_weak_candidate_library,
    run_best_subset_pareto_de,
)


def run_tutorial(output_dir: Path, noise_percent: float, seed: int) -> dict:
    """Build the weak library, inspect one candidate, and run Pareto selection."""
    spec = benchmark_spec(
        "paper_FADE_tsfade_fft",
        data_dir=ROOT / "data",
        profile="paper",
        noise_percent=noise_percent,
        seed=seed,
        maxiter=2,
        popsize=4,
        cmax=2,
        p_values=(0,),
    )
    data = spec["data"]
    config = with_uniform_order_grids(spec["config"])

    print(f"Dataset: {data.name}")
    print(f"Grid: {data.U.shape}; truth: {data.truth}")

    # Stage 1: construct weak integral measurements and order-indexed features.
    bank = build_weak_candidate_library(
        data,
        config,
        test_budget="smoke",
        verbose=True,
    )
    print(
        "Weak rows:", bank.n_points,
        "=", bank.time_tests.shape[0], "time tests x",
        bank.space_tests.shape[0], "space tests",
    )
    print("Temporal modes:", bank.available_alpha_modes())

    # Inspect the known FADE support before asking the optimizer to discover it.
    target = bank.target(0.80, alpha_mode="fractional_subunit")
    theta = bank.library((0, 0), (1.0, 1.7))
    coefficients, diagnostics = fit_least_squares(theta, target, ridge=config.ridge)
    print("\nManual weak fit on the FADE support")
    print("  coefficients:", np.array2string(coefficients, precision=4))
    print("  relative residual:", f"{diagnostics['rel_rmse']:.3e}")

    # Stage 2: optimize the support/order candidates and select the Pareto elbow.
    problem = build_best_subset_pareto_problem(bank, config)
    summary = run_best_subset_pareto_de(
        problem,
        config,
        data=data,
        output_dir=output_dir,
        verbose=False,
    )
    selected = summary["selected"]
    print("\nSelected equation")
    print(" ", selected["equation"])
    print("Support-size progress:")
    for row in summary["support_size_progress"]:
        print(f"  c={row['c']}: val_rel_mse={row['val_rel_mse']:.3e}  {row['equation']}")

    (output_dir / "tutorial_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    """Parse tutorial options and execute the end-to-end example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-percent", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "tutorial_weak_pareto",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_tutorial(args.output_dir, args.noise_percent, args.seed)


if __name__ == "__main__":
    main()
