#!/usr/bin/env python3
"""Instrumented temporal-order profiles for the two Riesz benchmarks.

The diagnostic fixes the correct two-term support (identity plus one positive
Riesz derivative), profiles the held-out objective over alpha and minimises over
the positive spatial order.  Five data-path arms separate the source of an
order shift:

* clean target / clean library;
* noisy target / noisy library;
* noisy field with the clean initial slice restored;
* noisy target / clean library;
* clean target / noisy library.

It is an oracle diagnostic, not part of model selection.  Temporal modes are
reported explicitly, and the fractional grid never interpolates across alpha=1.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from fpde_datasets import GridDataset, add_multiplicative_uniform_noise
from fractional_weak_form import caputo_l1_adjoint_tests
from pareto_fde_discovery import ParetoFDEOptimizer, train_val_split
from reproduce._repro_common import _config_for, NICE_NAME
from temporal_modes import infer_alpha_mode
from weak_pareto_fde_discovery import WeakFractionalFeatureBank

BENCHMARKS = ("synthetic_space_fractional_RD", "synthetic_time_space_fractional_RD")


class CompositeBank:
    """Use the target from one weak-row set and the RHS library from another."""

    def __init__(self, target_bank: WeakFractionalFeatureBank, library_bank: WeakFractionalFeatureBank):
        self.target_bank = target_bank
        self.library_bank = library_bank
        self.alpha_grid = target_bank.alpha_grid
        self.beta_grid = library_bank.beta_grid
        self.n_points = target_bank.n_points

    def available_alpha_modes(self):
        return self.target_bank.available_alpha_modes()

    def alpha_grid_for_mode(self, mode):
        return self.target_bank.alpha_grid_for_mode(mode)

    def alpha_bounds(self, mode):
        return self.target_bank.alpha_bounds(mode)

    def target(self, alpha, alpha_mode=None):
        # Keep the RHS bank's active alpha synchronized for the optional
        # Caputo-integral form, although the principal diagnostic uses the
        # derivative form.
        self.library_bank._active_alpha = 1.0 if alpha_mode == "integer" else float(alpha)
        return self.target_bank.target(alpha, alpha_mode=alpha_mode)

    def library(self, p_tuple, beta_tuple):
        return self.library_bank.library(p_tuple, beta_tuple)


def _with_field(data: GridDataset, U: np.ndarray, suffix: str) -> GridDataset:
    return GridDataset(
        U=np.asarray(U, dtype=float),
        t=np.asarray(data.t, dtype=float),
        x=np.asarray(data.x, dtype=float),
        name=f"{data.name}_{suffix}",
        truth=data.truth,
        recommended_backend=data.recommended_backend,
    )


def _build_banks(name: str, noise: float, seed: int):
    data, config, truth = _config_for(name, 0.0, seed, False, weak=True)
    cfg = replace(config, noise_percent=0.0, progress=False, progress_de=False)
    U_clean = np.asarray(data.U, dtype=float)
    U_noisy = add_multiplicative_uniform_noise(U_clean, float(noise), seed=int(seed))
    U_clean_initial = U_noisy.copy()
    U_clean_initial[0, :] = U_clean[0, :]

    clean = WeakFractionalFeatureBank(_with_field(data, U_clean, "clean"), cfg, test_budget="paper")
    clean.precompute(verbose=False)
    kwargs = dict(
        config=cfg,
        time_tests=clean.time_tests,
        space_tests=clean.space_tests,
        test_budget="paper",
        time_kind=clean.time_kind,
        space_kind=clean.space_kind,
        space_side=clean.space_side,
        time_form=clean.time_form,
    )
    noisy = WeakFractionalFeatureBank(_with_field(data, U_noisy, "noisy"), **kwargs)
    clean_initial = WeakFractionalFeatureBank(_with_field(data, U_clean_initial, "clean_initial"), **kwargs)
    noisy.precompute(verbose=False)
    clean_initial.precompute(verbose=False)
    return data, cfg, truth, clean, noisy, clean_initial


def profile(name: str, noise: float, seed: int, n_alpha: int) -> list[dict[str, float | int | str]]:
    data, config, truth, clean, noisy, clean_initial = _build_banks(name, noise, seed)
    arms = {
        "clean": CompositeBank(clean, clean),
        "fully_noisy": CompositeBank(noisy, noisy),
        "clean_initial_slice": CompositeBank(clean_initial, clean_initial),
        "target_only_noise": CompositeBank(noisy, clean),
        "library_only_noise": CompositeBank(clean, noisy),
    }
    positive_beta_min = min(float(b) for b in config.beta_grid if float(b) > 1e-10)
    beta_max = max(map(float, config.beta_grid))
    lo, hi = float(config.alpha_grid[0]), float(config.alpha_grid[-1])
    eps = float(getattr(config, "alpha_branch_epsilon", 1e-3))
    alpha_values = list(np.linspace(lo, min(hi, 1.0 - eps), int(n_alpha))) if lo < 1.0 - eps else []
    if lo <= 1.0 <= hi:
        alpha_values.append(1.0)
    if hi > 1.0 + eps:
        alpha_values.extend(np.linspace(max(lo, 1.0 + eps), hi, max(5, int(n_alpha // 4))))
    alpha_values.extend([float(truth.expected_alpha)])
    alpha_values = sorted(set(float(a) for a in alpha_values if lo <= a <= hi))

    train_idx, val_idx = train_val_split(clean.n_points, config.val_fraction, config.seed)
    optimizers = {arm: ParetoFDEOptimizer(bank, train_idx, val_idx, config) for arm, bank in arms.items()}
    rows: list[dict[str, float | int | str]] = []
    for alpha in alpha_values:
        alpha_mode = infer_alpha_mode(alpha)
        b_clean = clean.target(alpha, alpha_mode=alpha_mode)
        b_noisy = noisy.target(alpha, alpha_mode=alpha_mode)
        b_clean_initial = clean_initial.target(alpha, alpha_mode=alpha_mode)
        temporal_weights = caputo_l1_adjoint_tests(clean.time_tests, data.t, alpha)
        for arm, optimizer in optimizers.items():
            result = minimize_scalar(
                lambda beta: optimizer.evaluate(
                    float(alpha), (0, 0), (0.0, float(beta)), alpha_mode=alpha_mode
                ).val_rel_mse,
                bounds=(positive_beta_min, beta_max),
                method="bounded",
                options={"xatol": 1e-5, "maxiter": 120},
            )
            rows.append({
                "benchmark": NICE_NAME[name],
                "noise_percent": float(noise),
                "seed": int(seed),
                "arm": arm,
                "alpha": float(alpha),
                "alpha_mode": alpha_mode,
                "best_beta": float(result.x),
                "validation_error": float(result.fun),
                "true_alpha": float(truth.expected_alpha),
                "target_noise_l2": float(np.linalg.norm(b_noisy - b_clean)),
                "initial_slice_effect_l2": float(np.linalg.norm(b_noisy - b_clean_initial)),
                "target_variance_clean": float(np.var(b_clean)),
                "target_variance_noisy": float(np.var(b_noisy)),
                "temporal_adjoint_weight_norm": float(np.linalg.norm(temporal_weights)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 2.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-alpha", type=int, default=81)
    parser.add_argument("--out", default="results/temporal_order_profile.csv")
    args = parser.parse_args()
    rows: list[dict[str, float | int | str]] = []
    for name in BENCHMARKS:
        for noise in args.noise:
            current = profile(name, noise, args.seed, args.n_alpha)
            rows.extend(current)
            for arm in sorted({str(r["arm"]) for r in current}):
                subset = [r for r in current if r["arm"] == arm]
                best = min(subset, key=lambda row: float(row["validation_error"]))
                print(
                    f"{best['benchmark']} noise={noise:g}% arm={arm}: "
                    f"minimum alpha={best['alpha']:.6f} ({best['alpha_mode']}), "
                    f"true={best['true_alpha']:.6f}, beta={best['best_beta']:.6f}, "
                    f"E_val={best['validation_error']:.3e}"
                )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
