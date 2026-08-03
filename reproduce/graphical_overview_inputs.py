"""Validated data inputs for manuscript Figure 1.

The graphical overview is intentionally assembled from archived publication
outputs rather than manually copied constants.  This module centralises those
lookups so the manuscript renderer and regression tests use the same source of
truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN_ARCHIVE = ROOT / "reproduce" / "reference_results" / "branch_aware_campaign" / "main"
GAUSSIAN_ARCHIVE = (
    ROOT / "reproduce" / "reference_results" / "branch_aware_campaign" / "additive_gaussian"
)
TWO_D_RESULTS = ROOT / "two_dimensional" / "reference_results" / "per_run_B.jsonl"


@dataclass(frozen=True)
class GraphicalOverviewInputs:
    """Numerical values displayed in the graphical overview."""

    support_sizes: tuple[int, ...]
    validation_errors: tuple[float, ...]
    selected_support: int
    weak_support_recovery: int
    strong_support_recovery: int
    recovery_denominator: int
    fade_alpha: float
    fade_coefficients: tuple[float, float]
    fade_orders: tuple[float, float]
    twod_alpha: float
    twod_coefficients: tuple[float, float, float]
    twod_orders: tuple[float, float, float]
    twod_seed: int


def _fade_representative() -> tuple[float, tuple[float, float], tuple[float, float]]:
    """Read the representative 10% multiplicative-noise FADE equation."""
    equations = pd.read_csv(MAIN_ARCHIVE / "table_equations.csv")
    row = equations[(equations["benchmark"] == "FADE") & (equations["noise"] == 10)]
    if len(row) != 1:
        raise RuntimeError("Expected exactly one 10% FADE representative equation.")
    values = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?", row.iloc[0]["discovered_eq"])]
    if len(values) != 5:
        raise RuntimeError(f"Could not parse the archived FADE equation: {row.iloc[0]['discovered_eq']}")
    alpha, c1, beta1, c2, beta2 = values
    return alpha, (c1, c2), (beta1, beta2)


def _additive_gaussian_counts() -> tuple[int, int, int]:
    """Read the common support-recovery counts for both Gaussian benchmarks."""
    summary = pd.read_csv(GAUSSIAN_ARCHIVE / "summary.csv")
    subset = summary[summary["benchmark"].isin(["FADE", "Fractional Burgers"])]
    if len(subset) != 4:
        raise RuntimeError("Incomplete additive-Gaussian summary for Figure 1.")
    denominators = set(int(v) for v in subset["n_seeds"])
    if len(denominators) != 1:
        raise RuntimeError("Inconsistent seed counts in the additive-Gaussian summary.")
    weak = set(int(v) for v in subset[subset["method"] == "Weak-Pareto"]["support_power_recovery"])
    strong = set(int(v) for v in subset[subset["method"] == "Strong Pareto"]["support_power_recovery"])
    if len(weak) != 1 or len(strong) != 1:
        raise RuntimeError("The two Gaussian benchmarks do not share the displayed recovery counts.")
    return weak.pop(), strong.pop(), denominators.pop()


def _twod_example(seed: int = 1) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    """Read one fixed 10% multiplicative-noise anisotropic 2-D discovery."""
    selected = None
    for line in TWO_D_RESULTS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("benchmark") == "B" and int(row.get("seed", -1)) == seed and abs(float(row.get("noise", -1.0)) - 0.10) < 1e-12:
            selected = row
            break
    if selected is None:
        raise RuntimeError(f"Missing 10% Benchmark B result for seed {seed}.")
    terms = list(zip(selected["terms"], selected["xi"]))
    adv = next((term, coef) for term, coef in terms if term[0] == "x" and float(term[2]) < 1.3)
    xdiff = next((term, coef) for term, coef in terms if term[0] == "x" and float(term[2]) >= 1.3)
    ydiff = next((term, coef) for term, coef in terms if term[0] == "y")
    coefficients = (float(adv[1]), float(xdiff[1]), float(ydiff[1]))
    orders = (float(adv[0][2]), float(xdiff[0][2]), float(ydiff[0][2]))
    return float(selected["alpha"]), coefficients, orders


def load_graphical_overview_inputs(*, twod_seed: int = 1) -> GraphicalOverviewInputs:
    """Load and validate every numerical value displayed in Figure 1."""
    progress = pd.read_csv(MAIN_ARCHIVE / "table_progress.csv").sort_values("c")
    selected_rows = progress[progress["selected"].astype(str).str.lower().isin(["true", "1"])]
    if len(selected_rows) != 1:
        raise RuntimeError("Expected one selected Pareto support size.")
    weak, strong, denominator = _additive_gaussian_counts()
    fade_alpha, fade_coefficients, fade_orders = _fade_representative()
    twod_alpha, twod_coefficients, twod_orders = _twod_example(twod_seed)
    return GraphicalOverviewInputs(
        support_sizes=tuple(int(v) for v in progress["c"]),
        validation_errors=tuple(float(v) for v in progress["val_rel_mse"]),
        selected_support=int(selected_rows.iloc[0]["c"]),
        weak_support_recovery=weak,
        strong_support_recovery=strong,
        recovery_denominator=denominator,
        fade_alpha=fade_alpha,
        fade_coefficients=fade_coefficients,
        fade_orders=fade_orders,
        twod_alpha=twod_alpha,
        twod_coefficients=twod_coefficients,
        twod_orders=twod_orders,
        twod_seed=twod_seed,
    )
