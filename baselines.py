"""Baseline methods for the paper benchmarks.

The proposed method in this repository is **weak library + best-subset Pareto-DE**
(``weak_pareto`` in ``scripts/run_all_methods.py``).  This module contains
non-proposed comparators:

1. ``vanilla_pareto`` is implemented in :mod:`pareto_fde_discovery`; it uses the
   same best-subset Pareto-DE optimizer but with the original pointwise/strong
   candidate library.
2. ``weak_grid_stridge_baseline`` uses the weak candidate library but replaces
   best-subset Pareto-DE by a grid-search STRidge selector.
3. ``weak_fixed_library_stability_baseline`` uses the same fixed weak grid
   library and repeats STRidge over weak-test scales/splits.  It is an
   ablation/diagnostic baseline for fixed candidate libraries, not the proposed
   continuous-order Pareto-DE method.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence
import json
import math

import numpy as np
from numpy.typing import NDArray

from pareto_fde_discovery import (
    DiscoveryConfig,
    FractionalFeatureBank,
    ParetoFDEOptimizer,
    PDEModel,
    generate_p_patterns,
    train_val_split,
    write_json,
)
from fpde_datasets import GridDataset
from weak_pareto_fde_discovery import WeakFractionalFeatureBank, weak_grouped_train_val_split, _base_weak_widths

EPS = 1e-14


@dataclass
class BaselineResult:
    """Container for one baseline-discovery result.
    
    This mirrors the fields reported by the proposed method: equation string,
    temporal order, selected RHS terms, coefficients, support size, validation
    errors, BIC, and any method-specific metadata.
    """
    name: str
    equation: str
    alpha: float
    terms: list[tuple[int, float]]
    coefficients: list[float]
    support_size: int
    val_mse: float
    val_rel_mse: float
    bic: float
    extra: dict

    def to_dict(self) -> dict:
        """Convert the baseline result to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "equation": self.equation,
            "alpha": float(self.alpha),
            "terms": [[int(p), float(b)] for p, b in self.terms],
            "coefficients": [float(v) for v in self.coefficients],
            "support_size": int(self.support_size),
            "val_mse": float(self.val_mse),
            "val_rel_mse": float(self.val_rel_mse),
            "bic": float(self.bic),
            "extra": self.extra,
        }


def _normalize_train(Xtr: NDArray[np.float64], Xv: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Normalize columns using training-set norms and apply the same scaling to validation rows."""
    scale = np.linalg.norm(Xtr, axis=0)
    scale[scale < EPS] = 1.0
    return Xtr / scale[None, :], Xv / scale[None, :], scale


def _ridge_fit(X: NDArray[np.float64], y: NDArray[np.float64], ridge: float = 1e-10) -> NDArray[np.float64]:
    """Solve ridge or ordinary least-squares regression on finite rows."""
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xf, yf = X[finite], y[finite]
    if Xf.size == 0:
        return np.zeros(X.shape[1])
    if ridge <= 0:
        coef, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
        return coef
    return np.linalg.solve(Xf.T @ Xf + ridge * np.eye(X.shape[1]), Xf.T @ yf)


def stridge_threshold(
    Xtr: NDArray[np.float64],
    ytr: NDArray[np.float64],
    *,
    threshold: float,
    ridge: float = 1e-8,
    maxit: int = 20,
) -> NDArray[np.float64]:
    """Sequential threshold ridge regression on already-normalized columns."""
    active = np.ones(Xtr.shape[1], dtype=bool)
    coef = np.zeros(Xtr.shape[1], dtype=float)
    for _ in range(maxit):
        if not np.any(active):
            break
        small_coef = _ridge_fit(Xtr[:, active], ytr, ridge=ridge)
        new_active = active.copy()
        idx = np.flatnonzero(active)
        new_active[idx[np.abs(small_coef) < threshold]] = False
        if np.array_equal(new_active, active):
            coef[active] = small_coef
            break
        active = new_active
    if np.any(active):
        coef[active] = _ridge_fit(Xtr[:, active], ytr, ridge=ridge)
    return coef


def _equation(alpha: float, terms: Sequence[tuple[int, float]], coef: Sequence[float], digits: int = 5) -> str:
    """Format a baseline model as a readable FPDE string."""
    pieces = []
    for (p, beta), xi in zip(terms, coef):
        if abs(float(xi)) < 1e-12:
            continue
        if p == 0:
            term = f"D_x^{beta:.{digits}f} u"
        elif p == 1:
            term = f"u D_x^{beta:.{digits}f} u"
        else:
            term = f"u^{p} D_x^{beta:.{digits}f} u"
        pieces.append(f"({xi:.{digits}g})*{term}")
    return f"D_t^{alpha:.{digits}f} u = " + (" + ".join(pieces) if pieces else "0")


def grid_stridge_baseline(
    data: GridDataset,
    config: DiscoveryConfig,
    *,
    thresholds: Sequence[float] = tuple(np.logspace(-5, -1, 17)),
    ridge: float = 1e-8,
    max_terms: int | None = None,
    verbose: bool = False,
) -> BaselineResult:
    """Overcomplete-library baseline using finite-order grid + STRidge.

    It searches alpha over ``config.alpha_grid``.  For each alpha, RHS columns are
    all combinations ``u^p D_x^beta u`` for p in ``p_values`` and beta in
    ``beta_grid``.  This baseline can discover terms but only at pre-discretized
    orders, and thresholding can keep correlated nuisance terms.
    """
    bank = FractionalFeatureBank(data, config)
    bank.precompute(verbose=verbose)
    train_idx, val_idx = train_val_split(bank.n_points, config.val_fraction, config.seed)
    terms = [(int(p), float(b)) for p in config.p_values for b in config.beta_grid]
    X_full = np.column_stack([bank.u_power(p) * bank.spatial(beta) for p, beta in terms])
    Xtr_raw, Xv_raw = X_full[train_idx], X_full[val_idx]
    best: BaselineResult | None = None
    for alpha in config.alpha_grid:
        y = bank.target(float(alpha))
        ytr, yv = y[train_idx], y[val_idx]
        Xtr, Xv, scale = _normalize_train(Xtr_raw, Xv_raw)
        for thr in thresholds:
            coef_scaled = stridge_threshold(Xtr, ytr, threshold=float(thr), ridge=ridge)
            active = np.abs(coef_scaled) > 0

            # STRidge thresholding can keep many nearly-collinear fractional
            # derivative columns.  For a fair comparison with the Pareto methods
            # under the same canonical benchmark class, enforce ``max_terms`` by
            # retaining the largest normalized coefficients and refitting on that
            # support.  This makes Grid-STRidge a working baseline instead of a
            # frequent "no admissible model" failure when cmax=2.
            if max_terms is not None and int(active.sum()) > max_terms:
                keep = np.argsort(np.abs(coef_scaled))[-int(max_terms):]
                active = np.zeros_like(active, dtype=bool)
                active[keep] = True
                refit = np.zeros_like(coef_scaled)
                refit[active] = _ridge_fit(Xtr[:, active], ytr, ridge=ridge)
                coef_scaled = refit

            coef = coef_scaled / scale
            active = np.abs(coef_scaled) > 0
            finite = np.isfinite(yv) & np.all(np.isfinite(Xv_raw), axis=1)
            if not np.any(finite):
                continue
            resid = yv[finite] - Xv_raw[finite] @ coef
            val_mse = float(np.mean(resid**2))
            val_rel = float(val_mse / (np.var(yv[finite]) + EPS))
            k_eff = int(active.sum()) + 1  # active coefficients + alpha grid choice
            bic = float(len(yv[finite]) * math.log(max(val_mse, np.finfo(float).tiny)) + math.log(max(len(yv[finite]), 2)) * k_eff)
            selected_terms = [terms[i] for i in np.flatnonzero(active)]
            selected_coef = [float(coef[i]) for i in np.flatnonzero(active)]
            res = BaselineResult(
                name="grid_STRidge",
                equation=_equation(float(alpha), selected_terms, selected_coef),
                alpha=float(alpha),
                terms=selected_terms,
                coefficients=selected_coef,
                support_size=len(selected_terms),
                val_mse=val_mse,
                val_rel_mse=val_rel,
                bic=bic,
                extra={"threshold": float(thr), "candidate_columns": len(terms)},
            )
            if best is None or res.bic < best.bic:
                best = res
    if best is None:
        raise RuntimeError("Grid-STRidge failed to produce any model")
    return best


def fixed_cardinality_de_baseline(
    data: GridDataset,
    config: DiscoveryConfig,
    *,
    c: int = 2,
    verbose: bool = False,
) -> BaselineResult:
    """DE at a user-specified support size; no Pareto complexity selection."""
    bank = FractionalFeatureBank(data, config)
    bank.precompute(verbose=verbose)
    train_idx, val_idx = train_val_split(bank.n_points, config.val_fraction, config.seed)
    opt = ParetoFDEOptimizer(bank, train_idx, val_idx, config)
    include = {1: [(0,)], 2: [(0, 0), (0, 1)], 3: [(0, 0, 0), (0, 0, 1)]}.get(c, [])
    patterns = generate_p_patterns(c, config.p_values, include=include, max_patterns=config.max_patterns_per_c, seed=config.seed + 909)
    best_model: PDEModel | None = None
    for j, pat in enumerate(patterns):
        m = opt.optimize_fixed_pattern(pat, seed=config.seed + 1000 + j)
        if best_model is None or m.val_rel_mse < best_model.val_rel_mse:
            best_model = m
    if best_model is None:
        raise RuntimeError("fixed-cardinality DE failed")
    d = best_model.to_dict()
    return BaselineResult(
        name=f"fixed_c_DE_c{c}",
        equation=d["equation"],
        alpha=float(d["alpha"]),
        terms=[(int(p), float(b)) for p, b in d["terms"]],
        coefficients=[float(v) for v in d["coefficients"]],
        support_size=int(d["c"]),
        val_mse=float(d["val_mse"]),
        val_rel_mse=float(d["val_rel_mse"]),
        bic=float(d["bic"]),
        extra={"note": "DE without complexity selection; c was supplied by the user"},
    )


def save_baselines(path: str | Path, results: Sequence[BaselineResult]) -> None:
    """Save baseline results as a JSON list."""
    write_json(path, [r.to_dict() for r in results])


def weak_grid_stridge_baseline(
    data: GridDataset,
    config: DiscoveryConfig,
    *,
    thresholds: Sequence[float] = tuple(np.logspace(-5, -1, 17)),
    ridge: float = 1e-8,
    max_terms: int | None = None,
    test_budget: str = "smoke",
    width_scale: float = 1.0,
    split_seed_offset: int = 0,
    verbose: bool = False,
) -> BaselineResult:
    """Fixed-grid STRidge baseline using the weak candidate library.

    This is the non-Pareto weak-library baseline used in the paper.  It keeps the
    weak feature construction identical to the proposed method, but it removes
    the best-subset Pareto-DE search.  Instead, it builds one overcomplete weak
    matrix containing every pair ``(p, beta)`` from the canonical grid, searches
    the temporal order ``alpha`` on ``config.alpha_grid``, and applies sequential
    threshold ridge regression.

    Parameters
    ----------
    data, config:
        Benchmark data and the shared canonical search space.  Noise is injected
        inside the weak feature bank from ``config.noise_percent`` and
        ``config.seed``.
    thresholds:
        STRidge threshold values to scan.
    ridge:
        Ridge penalty used inside each STRidge refit.
    max_terms:
        Optional cap on active terms.  For the packaged benchmarks this is set
        to ``config.cmax`` so all methods respect the same maximum complexity.
    test_budget, width_scale:
        Weak-test resolution and window scale.  These affect numerical
        integration, not the symbolic candidate library.
    split_seed_offset:
        Extra offset used by the fixed-library stability ablation to generate
        different train/validation splits without changing the data noise.
    """
    tw, xw = _base_weak_widths(data, test_budget)  # type: ignore[arg-type]
    bank = WeakFractionalFeatureBank(
        data,
        config,
        test_budget=test_budget,  # type: ignore[arg-type]
        time_width=tw * float(width_scale),
        space_width=xw * float(width_scale),
    )
    bank.precompute(verbose=verbose)
    train_idx, val_idx = train_val_split(bank.n_points, config.val_fraction, config.seed + int(split_seed_offset))
    terms = [(int(p), float(b)) for p in config.p_values for b in config.beta_grid]

    # For the main derivative-form weak library, RHS columns do not depend on
    # alpha.  Build the overcomplete fixed library once.
    X_full = np.column_stack([bank.spatial_feature(p, beta) for p, beta in terms])
    Xtr_raw, Xv_raw = X_full[train_idx], X_full[val_idx]

    best: BaselineResult | None = None
    for alpha in config.alpha_grid:
        y = bank.target(float(alpha))
        ytr, yv = y[train_idx], y[val_idx]
        Xtr, Xv, scale = _normalize_train(Xtr_raw, Xv_raw)
        for thr in thresholds:
            coef_scaled = stridge_threshold(Xtr, ytr, threshold=float(thr), ridge=ridge)
            active = np.abs(coef_scaled) > 0
            if max_terms is not None and int(active.sum()) > int(max_terms):
                keep = np.argsort(np.abs(coef_scaled))[-int(max_terms):]
                active = np.zeros_like(active, dtype=bool)
                active[keep] = True
                refit = np.zeros_like(coef_scaled)
                refit[active] = _ridge_fit(Xtr[:, active], ytr, ridge=ridge)
                coef_scaled = refit
            coef = coef_scaled / scale
            active = np.abs(coef_scaled) > 0
            finite = np.isfinite(yv) & np.all(np.isfinite(Xv_raw), axis=1)
            if not np.any(finite):
                continue
            resid = yv[finite] - Xv_raw[finite] @ coef
            val_mse = float(np.mean(resid**2))
            val_rel = float(val_mse / (np.var(yv[finite]) + EPS))
            k_eff = int(active.sum()) + 1
            bic = float(len(yv[finite]) * math.log(max(val_mse, np.finfo(float).tiny)) + math.log(max(len(yv[finite]), 2)) * k_eff)
            selected_terms = [terms[i] for i in np.flatnonzero(active)]
            selected_coef = [float(coef[i]) for i in np.flatnonzero(active)]
            res = BaselineResult(
                name="weak_grid_STRidge",
                equation=_equation(float(alpha), selected_terms, selected_coef),
                alpha=float(alpha),
                terms=selected_terms,
                coefficients=selected_coef,
                support_size=len(selected_terms),
                val_mse=val_mse,
                val_rel_mse=val_rel,
                bic=bic,
                extra={
                    "threshold": float(thr),
                    "candidate_columns": len(terms),
                    "test_budget": str(test_budget),
                    "width_scale": float(width_scale),
                },
            )
            if best is None or res.bic < best.bic:
                best = res
    if best is None:
        raise RuntimeError("weak Grid-STRidge failed to produce any model")
    return best


def _structure_signature(result: BaselineResult, beta_round: int = 2) -> tuple:
    """Canonical structure signature used by the fixed-library stability ablation."""
    return (
        round(float(result.alpha), beta_round),
        tuple(sorted((int(p), round(float(b), beta_round)) for p, b in result.terms)),
    )


def weak_fixed_library_stability_baseline(
    data: GridDataset,
    config: DiscoveryConfig,
    *,
    thresholds: Sequence[float] = tuple(np.logspace(-5, -1, 17)),
    ridge: float = 1e-8,
    max_terms: int | None = None,
    test_budget: str = "smoke",
    width_scales: Sequence[float] = (0.85, 1.0, 1.2),
    n_splits: int = 2,
    verbose: bool = False,
) -> BaselineResult:
    """Stability ablation for a fixed weak grid library.

    This is **not** the proposed method.  It is included only to answer the
    question: if the symbolic candidate library is fixed to the alpha/beta grid,
    can repeated weak STRidge selections identify a stable support?  Because the
    candidate orders are fixed to the grid, this baseline should not be compared
    as a continuous-order Pareto-DE method.
    """
    runs: list[BaselineResult] = []
    for i, ws in enumerate(width_scales):
        for split in range(int(n_splits)):
            cfg = replace(config, seed=int(config.seed) + 1009 * split)
            res = weak_grid_stridge_baseline(
                data,
                cfg,
                thresholds=thresholds,
                ridge=ridge,
                max_terms=max_terms,
                test_budget=test_budget,
                width_scale=float(ws),
                split_seed_offset=37 * split,
                verbose=verbose,
            )
            runs.append(res)
    if not runs:
        raise RuntimeError("weak fixed-library stability baseline had no runs")
    counts: dict[tuple, list[BaselineResult]] = {}
    for r in runs:
        counts.setdefault(_structure_signature(r), []).append(r)
    # Choose the most frequent structure; break ties by median BIC.
    best_sig, best_group = min(
        counts.items(),
        key=lambda kv: (-len(kv[1]), float(np.median([r.bic for r in kv[1]]))),
    )
    # Report the best validation member within the winning stable group.
    best = min(best_group, key=lambda r: r.val_rel_mse)
    best.extra = dict(best.extra)
    best.extra.update({
        "stability_runs": len(runs),
        "winning_frequency": len(best_group) / max(1, len(runs)),
        "winning_signature": str(best_sig),
        "method_note": "fixed weak grid library + repeated STRidge; not continuous-order Pareto-DE",
    })
    best.name = "weak_fixed_library_stability"
    return best
