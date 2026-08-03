#!/usr/bin/env python3
"""Find the largest noise level tolerated by Weak Pareto (weak library + best-subset Pareto-DE).

This script answers a practical paper-writing question:

    For each benchmark dataset, what is the highest tested noise level at which
    the main proposed method still recovers the true structure and has small
    order/coefficient errors?

The default method is ``weak_pareto`` because this is the main proposed
method: weak candidate library + best-subset Pareto-DE.  The script can either run a fresh sweep through ``run_all_methods.py``
or analyze an existing ``method_comparison.csv`` file.

Definitions
-----------
The result table separates three questions:

1. ``full_structure_recovered``: the selected symbolic structure/support matches
   the benchmark form.  This does **not** require alpha, beta, or coefficients to
   be within any tolerance.
2. ``order_accuracy_pass``: the selected alpha and matched beta orders are within
   the visible benchmark tolerances.
3. ``strict_pass``: structure is recovered, order accuracy passes, and maximum
   relative coefficient error is below ``--coef-rel-tol``.

The order tolerances and coefficient truths are used only after discovery for
evaluation.  They are not passed to the optimizer.

Important caveat
----------------
The ``paper_ADE_Convection_diffusion`` benchmark has a known weak finite-domain
coefficient-scaling mismatch for the diffusion coefficient in the current weak
operator convention.  Therefore the script reports two tolerance columns:

* ``highest_structure_noise``: requires only symbolic structure recovery.
* ``highest_order_accuracy_noise``: requires structure plus alpha/beta accuracy.
* ``highest_strict_noise``: requires structure, order accuracy, and coefficients.

Use the strict column for coefficient claims.  Use the structure and order columns
when discussing symbolic recovery and fractional-order accuracy separately.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ground-truth coefficients for the packaged benchmarks.  Evaluation only.
TRUE_COEFFICIENTS: dict[str, dict[tuple[int, float], float]] = {
    "paper_ADE_Convection_diffusion": {(0, 1.0): -1.0, (0, 2.0): 0.25},
    "paper_FADE_tsfade_fft": {(0, 1.0): -1.0, (0, 1.7): 0.5},
    "synthetic_space_fractional_RD": {(0, 0.0): 0.04, (0, 1.65): 0.18},
    "synthetic_time_space_fractional_RD": {(0, 0.0): 0.03, (0, 1.55): 0.12},
    "synthetic_two_fractional_rhs": {(0, 0.55): 0.05, (0, 2.8): 0.005},
}


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def coefficient_errors_for_row(row: pd.Series) -> tuple[float, float, str]:
    """Return mean/max relative coefficient error for matched terms.

    Terms are matched by nearest beta among terms with the same polynomial power
    and only counted if the beta error is within the row's beta tolerance.
    """
    dataset = str(row["dataset"])
    truth = TRUE_COEFFICIENTS.get(dataset, {})
    terms = _parse_jsonish(row.get("selected_terms", "[]"))
    coeffs = _parse_jsonish(row.get("selected_coefficients", "[]"))
    beta_tol = float(row.get("beta_tol", 0.10))
    used: set[tuple[int, float]] = set()
    rel_errors: list[float] = []
    matched_payload: list[dict[str, Any]] = []
    for term, c_hat in zip(terms, coeffs):
        p_hat, beta_hat = int(term[0]), float(term[1])
        candidates = []
        for (p_true, beta_true), c_true in truth.items():
            if (p_true, beta_true) in used or p_true != p_hat:
                continue
            candidates.append((abs(beta_hat - beta_true), p_true, beta_true, c_true))
        if not candidates:
            continue
        beta_err, p_true, beta_true, c_true = min(candidates, key=lambda z: z[0])
        if beta_err <= beta_tol:
            used.add((p_true, beta_true))
            rel = abs(float(c_hat) - float(c_true)) / max(abs(float(c_true)), 1e-12)
            rel_errors.append(float(rel))
            matched_payload.append({
                "expected": [int(p_true), float(beta_true)],
                "selected": [int(p_hat), float(beta_hat)],
                "c_true": float(c_true),
                "c_hat": float(c_hat),
                "relative_error": float(rel),
            })
    if not rel_errors:
        return float("inf"), float("inf"), json.dumps(matched_payload)
    return float(np.mean(rel_errors)), float(np.max(rel_errors)), json.dumps(matched_payload)


def analyze_results(method_csv: Path, output_dir: Path, coef_rel_tol: float) -> pd.DataFrame:
    """Analyze a method_comparison.csv and write tolerance summaries."""
    df = pd.read_csv(method_csv)
    if "method" in df.columns:
        df = df[df["method"] == "weak_pareto"].copy()
    if df.empty:
        raise ValueError(f"No weak_pareto rows found in {method_csv}")

    coeff_stats = df.apply(coefficient_errors_for_row, axis=1, result_type="expand")
    df["mean_coeff_rel_error"] = coeff_stats[0]
    df["max_coeff_rel_error"] = coeff_stats[1]
    df["matched_coefficients_json"] = coeff_stats[2]
    df["structure_pass"] = df["full_structure_recovered"].astype(str).str.lower().isin(["true", "1"])
    df["order_accuracy_pass"] = (
        df["structure_pass"]
        & (pd.to_numeric(df["alpha_abs_error"], errors="coerce") <= pd.to_numeric(df["alpha_tol"], errors="coerce"))
        & (pd.to_numeric(df["max_matched_beta_abs_error"], errors="coerce") <= pd.to_numeric(df["beta_tol"], errors="coerce"))
    )
    df["strict_pass"] = df["order_accuracy_pass"] & (df["max_coeff_rel_error"] <= float(coef_rel_tol))

    rows: list[dict[str, Any]] = []
    for dataset, g in df.groupby("dataset", sort=True):
        g = g.sort_values("noise_percent")
        strict = g[g["strict_pass"]]
        structure = g[g["structure_pass"]]
        order = g[g["order_accuracy_pass"]]
        best_strict = None if strict.empty else strict.iloc[-1]
        best_structure = None if structure.empty else structure.iloc[-1]
        best_order = None if order.empty else order.iloc[-1]
        rows.append({
            "dataset": dataset,
            "highest_structure_noise": np.nan if best_structure is None else float(best_structure["noise_percent"]),
            "highest_order_accuracy_noise": np.nan if best_order is None else float(best_order["noise_percent"]),
            "highest_strict_noise": np.nan if best_strict is None else float(best_strict["noise_percent"]),
            "selected_noise_for_coefficient_claim": np.nan if best_strict is None else float(best_strict["noise_percent"]),
            "selected_equation_strict": "" if best_strict is None else str(best_strict.get("selected_equation", "")),
            "alpha_abs_error_strict": np.nan if best_strict is None else float(best_strict["alpha_abs_error"]),
            "max_beta_abs_error_strict": np.nan if best_strict is None else float(best_strict["max_matched_beta_abs_error"]),
            "max_coeff_rel_error_strict": np.nan if best_strict is None else float(best_strict["max_coeff_rel_error"]),
            "selected_equation_structure": "" if best_structure is None else str(best_structure.get("selected_equation", "")),
            "selected_equation_order_accuracy": "" if best_order is None else str(best_order.get("selected_equation", "")),
            "note": (
                "Strict coefficient tolerance not passed at any tested noise level. "
                "Use structure/order columns separately or fix coefficient normalization."
                if best_strict is None else "Strict pass."
            ),
        })
    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "noise_tolerance_detailed.csv", index=False)
    summary.to_csv(output_dir / "noise_tolerance_summary.csv", index=False)

    lines = [
        "# Noise tolerance of proposed Weak Pareto-DE",
        "",
        f"Analyzed: `{method_csv}`",
        f"Coefficient relative-error tolerance: `{coef_rel_tol:g}`",
        "",
        "A strict pass requires structure recovery, alpha/beta order accuracy, and coefficient error below the tolerance.",
        "Structure recovery is reported separately from order and coefficient errors.",
        "",
        "| Dataset | Highest structure noise | Highest order-accuracy noise | Highest strict noise | Max coeff. rel. err. at strict noise | Equation at strict noise |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, r in summary.iterrows():
        struct_noise = "—" if pd.isna(r["highest_structure_noise"]) else f"{r['highest_structure_noise']:.3g}%"
        order_noise = "—" if pd.isna(r["highest_order_accuracy_noise"]) else f"{r['highest_order_accuracy_noise']:.3g}%"
        strict_noise = "—" if pd.isna(r["highest_strict_noise"]) else f"{r['highest_strict_noise']:.3g}%"
        coef_err = "—" if pd.isna(r["max_coeff_rel_error_strict"]) else f"{100*r['max_coeff_rel_error_strict']:.2f}%"
        eq = str(r["selected_equation_strict"]) if str(r["selected_equation_strict"]) else "—"
        lines.append(f"| `{r['dataset']}` | {struct_noise} | {order_noise} | {strict_noise} | {coef_err} | {eq} |")
    lines += [
        "",
        "## Interpretation guidance",
        "",
        "Use `highest_structure_noise` for symbolic-structure claims.",
        "Use `highest_order_accuracy_noise` for structure plus alpha/beta order-accuracy claims.",
        "Use `highest_strict_noise` for clean/noisy equation tables that include coefficient accuracy.",
        "If a dataset has no strict pass, the weak formulation may still recover the structure but the current coefficient normalization should not be used for a coefficient-accuracy claim.",
    ]
    (output_dir / "NOISE_TOLERANCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run_sweep(args: argparse.Namespace) -> Path:
    """Run run_all_methods.py and return the produced method_comparison.csv path."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_all_methods.py"),
        "--methods", "weak_pareto",
        "--profile", args.profile,
        "--noise-levels", *[str(v) for v in args.noise_levels],
        "--seeds", *[str(v) for v in args.seeds],
        "--weak-test-budget", args.weak_test_budget,
        "--stability-splits", str(args.stability_splits),
        "--stability-width-scales", *[str(v) for v in args.stability_width_scales],
        "--output-dir", str(args.output_dir / "raw_sweep"),
    ]
    if args.datasets:
        cmd += ["--datasets", *args.datasets]
    if args.maxiter is not None:
        cmd += ["--maxiter", str(args.maxiter)]
    if args.popsize is not None:
        cmd += ["--popsize", str(args.popsize)]
    if args.quiet:
        cmd += ["--quiet"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    return args.output_dir / "raw_sweep" / "method_comparison.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--existing-results", type=Path, default=None, help="Existing method_comparison.csv to analyze instead of running a fresh sweep.")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results" / "noise_tolerance")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--noise-levels", nargs="+", type=float, default=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--profile", choices=["notebook", "paper"], default="notebook")
    ap.add_argument("--maxiter", type=int, default=0)
    ap.add_argument("--popsize", type=int, default=2)
    ap.add_argument("--weak-test-budget", choices=["smoke", "standard", "paper"], default="smoke")
    ap.add_argument("--stability-splits", type=int, default=1)
    ap.add_argument("--stability-width-scales", nargs="+", type=float, default=[1.0])
    ap.add_argument("--coef-rel-tol", type=float, default=0.25)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    method_csv = args.existing_results if args.existing_results is not None else run_sweep(args)
    summary = analyze_results(method_csv, args.output_dir, args.coef_rel_tol)
    print(summary.to_string(index=False))
    print(f"\nWrote {args.output_dir / 'NOISE_TOLERANCE_REPORT.md'}")


if __name__ == "__main__":
    main()
