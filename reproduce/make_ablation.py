"""Component ablation on the FADE benchmark (presubmission review item 3.3).

Isolates the three ingredients of Weak-Pareto by toggling one at a time:

  Method                        Weak library  Continuous orders  Exact polishing
  Strong Pareto                 no            yes                no
  Weak Grid-STRidge             yes           no                 no
  Weak Pareto (no polishing)    yes           yes                no
  Full Weak-Pareto              yes           yes                yes

For each method it reports support/power recovery, operator-structure recovery,
the structure-conditioned order and coefficient errors, and the full-data
residual, averaged over seeds.  Emits ``table_ablation.{csv,tex}``.

    PYTHONPATH=. python3 reproduce/make_ablation.py            # full budget
    PYTHONPATH=. python3 reproduce/make_ablation.py --fast     # quick smoke
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Any

import numpy as np

from _repro_common import (
    _config_for, run_weak, run_strong, matched_errors, default_seeds, NICE_NAME,
)
from make_tables import _agg, _cell, _pm
from baselines import weak_grid_stridge_baseline
from weak_pareto_fde_discovery import WeakFractionalFeatureBank, run_weak_pareto_discovery

ABLATION_BENCHMARK = "paper_FADE_tsfade_fft"


def _residual_cell(mean: float, sd: float) -> str:
    """Readable mean +/- sd formatting for residuals spanning many decades."""
    if not np.isfinite(mean):
        return "--"
    if abs(mean) >= 5e-3:
        return _pm(mean, sd)
    def part(x: float) -> str:
        if x == 0:
            return "0"
        e = int(np.floor(np.log10(abs(x))))
        m = x / (10.0 ** e)
        return f"{m:.1f}{{\\times}}10^{{{e}}}"
    return f"${part(mean)}\\pm {part(sd)}$"


def _baseline_selected(result: Any, *, full_data_rel_l2: float = float("nan")) -> dict[str, Any]:
    """Adapt a BaselineResult to the dict consumed by matched_errors."""
    terms = list(result.terms)
    return {
        "p_tuple": [int(p) for p, _ in terms],
        "beta_tuple": [float(b) for _, b in terms],
        "coefficients": [float(c) for c in result.coefficients],
        "alpha": float(result.alpha),
        "val_rel_mse": float(result.val_rel_mse),
        "full_data_rel_l2": float(full_data_rel_l2),
    }


def _weak_grid_selected(name: str, noise: float, seed: int, fast: bool) -> dict[str, Any]:
    data, cfg, _ = _config_for(name, noise, seed, fast, weak=True)
    res = weak_grid_stridge_baseline(data, cfg, test_budget="paper", verbose=False)
    # Re-evaluate the selected fixed-grid equation on all weak rows so E_fit has
    # the same explicit full-data relative-L2 definition used by the Pareto rows.
    bank = WeakFractionalFeatureBank(data, cfg, test_budget="paper")
    bank.precompute(verbose=False)
    y = np.asarray(bank.target(float(res.alpha)), dtype=float)
    X = np.column_stack([bank.spatial_feature(int(p), float(b)) for p, b in res.terms])
    coef = np.asarray(res.coefficients, dtype=float)
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    rel = float(np.linalg.norm(y[finite] - X[finite] @ coef) / (np.linalg.norm(y[finite]) + 1e-12))
    return _baseline_selected(res, full_data_rel_l2=rel)


# (label, weak_library, continuous_orders, exact_polishing, selector)
METHODS = [
    ("Strong Pareto", "no", "yes", "no",
     lambda name, noise, seed, fast: run_strong(name, noise, seed, fast)[0]["selected"]),
    ("Weak Grid-STRidge", "yes", "no", "no",
     lambda name, noise, seed, fast: _weak_grid_selected(name, noise, seed, fast)),
    ("Weak Pareto (no polish)", "yes", "yes", "no",
     lambda name, noise, seed, fast: run_weak(name, noise, seed, fast,
                                              overrides={"exact_order_polish": False})[0]["selected"]),
    ("Full Weak-Pareto", "yes", "yes", "yes",
     lambda name, noise, seed, fast: run_weak(name, noise, seed, fast)[0]["selected"]),
]


def ablation_table(outdir: str, fast: bool, noise: float = 10.0) -> str:
    seeds = default_seeds(fast)
    _, _, truth = _config_for(ABLATION_BENCHMARK, noise, seeds[0], fast, weak=True)
    rows = []
    for label, wl, co, ep, selector in METHODS:
        ms = []
        for seed in seeds:
            try:
                selected = selector(ABLATION_BENCHMARK, noise, seed, fast)
                ms.append(matched_errors(ABLATION_BENCHMARK, selected, truth))
            except Exception as exc:  # a baseline may fail on a given seed
                print(f"[ablation] {label} seed {seed} failed: {exc}")
        if not ms:
            continue
        a = _agg(ms)
        row = {
            "method": label, "weak_library": wl, "continuous_orders": co,
            "exact_polishing": ep, "noise_percent": noise, "n_seeds": a["n"],
            "support_power_recovery": a["n_sp"], "operator_structure_recovery": a["n_os"],
        }
        for key in ("e_beta_max", "e_xi_max", "e_xi_l2", "full_data_rel_l2"):
            row[key], row[key + "_sd"] = a[key]
        rows.append(row)
        print(f"[ablation] {label:26s}  sp {a['n_sp']}/{a['n']} os {a['n_os']}/{a['n']}  "
              f"e_beta={row['e_beta_max']:.3f} e_xi={row['e_xi_max']:.3f} E_fit={row['full_data_rel_l2']:.2e}")
    path = os.path.join(outdir, "table_ablation.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    _E = chr(92) * 2  # two backslashes = LaTeX row terminator
    _NL = chr(10)
    with open(os.path.join(outdir, "table_ablation.tex"), "w") as f:
        for r in rows:
            row = (f"{r['method']} & {r['weak_library']} & {r['continuous_orders']} & "
                   f"{r['exact_polishing']} & {r['support_power_recovery']}/{r['n_seeds']} & "
                   f"{r['operator_structure_recovery']}/{r['n_seeds']} & "
                   f"{_cell(r['e_beta_max'], r['e_beta_max_sd'])} & "
                   f"{_cell(r['e_xi_max'], r['e_xi_max_sd'])} & "
                   f"{_residual_cell(r['full_data_rel_l2'], r['full_data_rel_l2_sd'])}")
            f.write(row + " " + _E + _NL)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Component ablation on FADE.")
    ap.add_argument("--fast", action="store_true", help="quick smoke budget")
    ap.add_argument("--outdir", default="results", help="output directory")
    ap.add_argument("--noise", type=float, default=10.0, help="noise level (percent)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    path = ablation_table(args.outdir, args.fast, noise=args.noise)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
