"""Shared helpers for reproducing the paper's tables and figures.

These scripts drive the released package (``dataset_configs``,
``weak_pareto_fde_discovery``, ``pareto_fde_discovery``) to regenerate every
non-placeholder number and figure in the manuscript.

Run from the repository root, e.g.::

    PYTHONPATH=. python3 reproduce/make_tables.py --fast
    PYTHONPATH=. python3 reproduce/make_figures.py --fast --figdir /path/to/paper/figures

Omit ``--fast`` for the full publication settings (more seeds, larger DE budget);
this is slower but reproduces the paper-scale results.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# Make the package importable whether or not it is on PYTHONPATH.
try:  # pragma: no cover - import plumbing
    import dataset_configs  # noqa: F401
except ImportError:  # pragma: no cover
    _pkg = os.environ.get("FRACTIONAL_PARETO_DIR")
    if _pkg:
        sys.path.insert(0, _pkg)
    else:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_configs import load_benchmark, with_uniform_order_grids  # noqa: E402
from temporal_modes import infer_alpha_mode
from weak_pareto_fde_discovery import (  # noqa: E402
    build_weak_candidate_library,
    coefficient_truth,
    fit_coefficients_for_structure,
    run_weak_pareto_discovery,
)
from pareto_fde_discovery import run_pareto_discovery, FractionalFeatureBank  # noqa: E402

# Benchmarks used in the paper.
# Main-text benchmarks (Table 2), run with the non-restrictive default settings.
MAIN_BENCHMARKS = (
    "paper_FADE_tsfade_fft",
    "synthetic_space_fractional_RD",
    "synthetic_time_space_fractional_RD",
    "synthetic_fractional_burgers",
)
# Appendix "challenging cases": no fractional derivative (ADE) and more than one
# fractional spatial derivative (two-term Riesz). The non-restrictive defaults
# mishandle these; case-specific hyperparameters recover them (Appendix).
APPENDIX_BENCHMARKS = (
    "paper_ADE_Convection_diffusion",
    "synthetic_two_fractional_rhs",
)
# Hyperparameter adjustments found effective for the appendix cases.
APPENDIX_ADJUSTMENTS = {
    "paper_ADE_Convection_diffusion": {"p_values": (0,)},
    "synthetic_two_fractional_rhs": {"p_values": (0,), "selection": "aic"},
}
# Weak-vs-strong robustness comparison uses well-behaved main benchmarks.
ROBUSTNESS_BENCHMARKS = ("paper_FADE_tsfade_fft", "synthetic_fractional_burgers")
ALL_BENCHMARKS = MAIN_BENCHMARKS + APPENDIX_BENCHMARKS
NICE_NAME = {
    "paper_FADE_tsfade_fft": "FADE",
    "paper_ADE_Convection_diffusion": "ADE",
    "synthetic_space_fractional_RD": "Frac. RD (space)",
    "synthetic_time_space_fractional_RD": "Frac. RD (time-space)",
    "synthetic_two_fractional_rhs": "Two-term Riesz",
    "synthetic_fractional_burgers": "Fractional Burgers",
}


def de_budget(fast: bool) -> dict[str, int]:
    """Differential-evolution budget; small for --fast, paper-scale otherwise."""
    return {"maxiter": 8, "popsize": 5} if fast else {"maxiter": 24, "popsize": 7}


def default_noise(fast: bool) -> list[float]:
    return [0.0, 5.0, 10.0] if fast else [0.0, 1.0, 5.0, 10.0, 20.0]


def default_seeds(fast: bool) -> list[int]:
    return [0, 1] if fast else [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
#  Running the proposed (weak) and strong-form (pointwise) pipelines           #
# --------------------------------------------------------------------------- #
def _config_for(name: str, noise: float, seed: int, fast: bool, *, weak: bool,
                overrides: dict[str, Any] | None = None):
    data, cfg, truth = load_benchmark(name, profile="paper", seed=seed)
    cfg = dataclasses.replace(
        cfg, noise_percent=float(noise), seed=int(seed), progress=False,
        exact_order_refit=True, exact_order_polish=weak, **de_budget(fast),
    )
    # Publication grid: dense, uniform, truth-agnostic (Sec. 4.5 / Appendix C).
    cfg = with_uniform_order_grids(cfg)
    # Non-restrictive main-text setting: auto_stop on, cmax = 4, powers {0,1,2}
    # (auto_stop and p_values come from the "paper" profile). Per-case overrides
    # for the appendix challenging cases are applied last.
    cfg = dataclasses.replace(cfg, cmax=4, **(overrides or {}))
    return data, cfg, truth


# Persistent cache used only by the reproduction drivers.  It does not alter the
# discovery algorithm; it prevents the same deterministic seed/configuration from
# being recomputed by several table and figure scripts.  Set FPDE_REPRO_CACHE to
# enable it (run_all.sh does this automatically).
_CACHE_SCHEMA = "branch-aware-repro-cache-v2"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    names = (
        "pareto_fde_discovery.py", "weak_pareto_fde_discovery.py",
        "fractional_weak_form.py", "fpde_derivatives.py", "fpde_datasets.py",
        "dataset_configs.py", "temporal_modes.py", "reproduce/_repro_common.py",
    )
    h = hashlib.sha256(_CACHE_SCHEMA.encode("utf-8"))
    for name in names:
        path = root / name
        h.update(name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()[:20]


_SOURCE_FINGERPRINT = _source_fingerprint()


def _cache_path(method: str, name: str, noise: float, seed: int, fast: bool,
                overrides: dict[str, Any] | None) -> Path | None:
    root = os.environ.get("FPDE_REPRO_CACHE", "").strip()
    if not root:
        return None
    payload = {
        "schema": _CACHE_SCHEMA, "source": _SOURCE_FINGERPRINT,
        "method": method, "name": name, "noise": float(noise),
        "seed": int(seed), "fast": bool(fast),
        "overrides": _jsonable(overrides or {}),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    path = Path(root) / method / f"{digest}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _cache_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        try:
            import fcntl  # POSIX/macOS; the campaign is run on those platforms.
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        handle.close()


def _load_cache(path: Path | None):
    if path is None or not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
        if os.environ.get("FPDE_REPRO_CACHE_VERBOSE") == "1":
            print(f"[cache hit] {path.name}", flush=True)
        return value
    except Exception:
        # A killed writer cannot poison a resumed campaign.
        try:
            path.unlink()
        except OSError:
            pass
        return None


def _store_cache(path: Path | None, value: Any) -> None:
    if path is None:
        return
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def run_weak(name: str, noise: float, seed: int, fast: bool,
             overrides: dict[str, Any] | None = None, *,
             use_cache: bool = True) -> tuple[dict[str, Any], Any]:
    path = _cache_path("weak", name, noise, seed, fast, overrides) if use_cache else None
    if path is not None:
        with _cache_lock(path):
            cached = _load_cache(path)
            if cached is not None:
                return cached
            data, cfg, truth = _config_for(name, noise, seed, fast, weak=True, overrides=overrides)
            t0 = time.perf_counter()
            summary = run_weak_pareto_discovery(data, cfg, test_budget="paper", verbose=False)
            summary["_repro_wall_time_s"] = float(time.perf_counter() - t0)
            value = (summary, truth)
            _store_cache(path, value)
            return value
    data, cfg, truth = _config_for(name, noise, seed, fast, weak=True, overrides=overrides)
    t0 = time.perf_counter()
    summary = run_weak_pareto_discovery(data, cfg, test_budget="paper", verbose=False)
    summary["_repro_wall_time_s"] = float(time.perf_counter() - t0)
    return summary, truth


def _attach_full_data_rel_l2(data, cfg, summary) -> None:
    """Attach the strong model's full-data relative L2 residual explicitly.

    The weak framework reports the relative L2 residual of its final exact-order
    conditional refit over all rows.  Here we recompute the corresponding strong
    model quantity and store it as ``full_data_rel_l2``.  Selection-stage
    ``train_rel_mse``, ``val_rel_mse``, ``objective``, and heuristic AIC/BIC-type
    fields are not overwritten.  The evaluation rows are each framework's own
    (weak-form versus pointwise) rows, which differ by construction.
    """
    sel = summary.get("selected") if isinstance(summary, dict) else None
    if not sel or int(sel.get("c", 0)) < 1:
        return
    try:
        alpha = float(sel["alpha"])
        p_tuple = tuple(int(v) for v in sel["p_tuple"])
        beta_tuple = tuple(float(v) for v in sel["beta_tuple"])
        coef = np.asarray(sel["coefficients"], dtype=float)
        bank = FractionalFeatureBank(data, cfg)
        bank.precompute(verbose=False)
        alpha_mode = str(sel.get("alpha_mode", infer_alpha_mode(alpha)))
        y = np.asarray(bank.target(alpha, alpha_mode=alpha_mode), dtype=float)
        X = np.asarray(bank.library(p_tuple, beta_tuple), dtype=float)
        finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        yf, Xf = y[finite], X[finite]
        if yf.size == 0 or Xf.shape[1] != coef.shape[0]:
            return
        rel = float(np.linalg.norm(yf - Xf @ coef) / (np.linalg.norm(yf) + 1e-12))
        sel["full_data_rel_l2"] = rel
    except Exception:
        return  # leave the original value if recomputation fails


def run_strong(name: str, noise: float, seed: int, fast: bool,
               overrides: dict[str, Any] | None = None, *,
               use_cache: bool = True) -> tuple[dict[str, Any], Any]:
    # Same selector and search ranges, pointwise (strong-form) candidate library.
    path = _cache_path("strong", name, noise, seed, fast, overrides) if use_cache else None
    if path is not None:
        with _cache_lock(path):
            cached = _load_cache(path)
            if cached is not None:
                return cached
            data, cfg, truth = _config_for(name, noise, seed, fast, weak=False, overrides=overrides)
            t0 = time.perf_counter()
            summary = run_pareto_discovery(data, cfg, verbose=False)
            _attach_full_data_rel_l2(data, cfg, summary)
            summary["_repro_wall_time_s"] = float(time.perf_counter() - t0)
            value = (summary, truth)
            _store_cache(path, value)
            return value
    data, cfg, truth = _config_for(name, noise, seed, fast, weak=False, overrides=overrides)
    t0 = time.perf_counter()
    summary = run_pareto_discovery(data, cfg, verbose=False)
    _attach_full_data_rel_l2(data, cfg, summary)
    summary["_repro_wall_time_s"] = float(time.perf_counter() - t0)
    return summary, truth


# --------------------------------------------------------------------------- #
#  Matching a selected model to the truth and scoring errors                   #
# --------------------------------------------------------------------------- #
DECLARED_BETA_TOL = 0.15   # operator-structure tolerance on fractional orders
DECLARED_ALPHA_TOL = 0.15  # operator-structure tolerance on the temporal order
# Identity/derivative operator mode is distinguished by the fractional order: the
# discovery snaps any order below the first strictly-positive grid node to exactly
# zero (identity), so a term is in the identity mode iff its order is (numerically)
# zero, and in the derivative mode otherwise.  Recovery must respect this discrete
# mode: an identity truth is recovered only by an identity selection, and a
# derivative truth only by a (positive-order) derivative within tolerance -- a
# low-order Riesz term does not recover an identity, and vice versa.
IDENTITY_ORDER_TOL = 1e-8  # |beta| <= this counts as the identity operator mode


def matched_errors(name: str, selected: dict[str, Any], truth_spec: Any) -> dict[str, Any]:
    """Recovery flags and alpha/beta/coefficient/validation errors.

    Two recovery flags are returned (per the presubmission review):
    ``support_power_ok`` requires the correct number of terms and matching
    integer powers (the cardinality/power pattern), while
    ``operator_structure_ok`` additionally requires every matched fractional
    order to fall within ``DECLARED_BETA_TOL`` of its truth -- so an identity
    term ($\\beta=0$) reported as a low-order Riesz term does *not* count as an
    operator-structure recovery.  Terms are matched greedily: true terms, in
    order, consume the nearest same-power selected term.  Order/coefficient
    errors cover the matched pairs; callers condition them on recovery (they are
    scientifically meaningful only when the support/powers are correct).
    See Sec. 5.1.
    """
    truth_terms = list(truth_spec.expected_terms)          # [(p, beta), ...]
    coef_true = coefficient_truth(name)                    # [xi, ...] in truth order
    alpha_true = float(truth_spec.expected_alpha)

    sel_p = [int(p) for p in selected["p_tuple"]]
    sel_b = [float(b) for b in selected["beta_tuple"]]
    sel_x = [float(x) for x in selected["coefficients"]]

    support_size_match = len(sel_p) == len(truth_terms)
    symbolic_ok = support_size_match
    e_beta: list[float] = []
    e_xi: list[float] = []
    e_xi_abs: list[float] = []
    matched_coef_true: list[float] = []
    power_hits = 0
    operator_hits = 0
    used = [False] * len(sel_p)
    for (pt, bt), xt in zip(truth_terms, coef_true):
        cand = [i for i in range(len(sel_p)) if not used[i] and sel_p[i] == pt]
        if not cand:
            symbolic_ok = False
            continue
        i = min(cand, key=lambda i: abs(sel_b[i] - bt))
        used[i] = True
        power_hits += 1
        b_err = abs(sel_b[i] - bt)
        # Mode-aware operator hit (see IDENTITY_ORDER_TOL above): identity truth
        # (beta*=0) requires an identity selection; derivative truth (beta*>0)
        # requires a positive-order derivative within DECLARED_BETA_TOL.
        truth_is_identity = abs(bt) <= IDENTITY_ORDER_TOL
        sel_is_identity = abs(sel_b[i]) <= IDENTITY_ORDER_TOL
        if truth_is_identity:
            op_hit = sel_is_identity
        else:
            op_hit = (not sel_is_identity) and (b_err <= DECLARED_BETA_TOL)
        operator_hits += int(op_hit)
        e_beta.append(b_err)
        e_xi.append(abs(sel_x[i] - xt) / (abs(xt) + 1e-12))
        e_xi_abs.append(abs(sel_x[i] - xt))
        matched_coef_true.append(float(xt))

    support_power_ok = bool(support_size_match and power_hits == len(truth_terms))
    selected_alpha = float(selected["alpha"])
    selected_alpha_mode = str(selected.get("alpha_mode", infer_alpha_mode(selected_alpha)))
    truth_alpha_mode = infer_alpha_mode(alpha_true)
    alpha_mode_ok = selected_alpha_mode == truth_alpha_mode
    e_alpha = abs(selected_alpha - alpha_true)
    operator_structure_ok = bool(
        support_size_match
        and operator_hits == len(truth_terms)
        and alpha_mode_ok
        and e_alpha <= DECLARED_ALPHA_TOL
    )
    # L2 relative coefficient error over the matched terms (stable metric, 2.5)
    if e_xi_abs and support_power_ok:
        e_xi_l2 = float(np.linalg.norm(e_xi_abs) / (np.linalg.norm(matched_coef_true) + 1e-12))
    else:
        e_xi_l2 = float("nan")
    return {
        "symbolic_form_ok": bool(symbolic_ok),      # retained alias (== support_power_ok)
        "support_power_ok": support_power_ok,
        "operator_structure_ok": operator_structure_ok,
        "e_alpha": e_alpha,
        "alpha_mode_ok": bool(alpha_mode_ok),
        "selected_alpha_mode": selected_alpha_mode,
        "truth_alpha_mode": truth_alpha_mode,
        "e_beta_max": (max(e_beta) if e_beta else float("nan")),
        "e_xi_max": (max(e_xi) if e_xi else float("nan")),
        "e_xi_l2": e_xi_l2,
        "e_xi_abs_max": (max(e_xi_abs) if e_xi_abs else float("nan")),
        "val_rel_mse": float(selected.get("val_rel_mse", float("nan"))),
        "full_data_rel_l2": float(selected.get("full_data_rel_l2", float("nan"))),
    }


def support_progress_curve(summary: dict[str, Any]) -> tuple[list[int], list[float]]:
    """Extract (support size, validation error) from a weak-framework summary."""
    rows = summary.get("support_size_progress", [])
    cs, errs = [], []
    for r in rows:
        c = int(r.get("c"))
        err = r.get("val_rel_mse", r.get("train_rel_mse", r.get("objective")))
        if err is None:
            continue
        cs.append(c)
        errs.append(float(err))
    order = np.argsort(cs)
    return [cs[i] for i in order], [errs[i] for i in order]


def support_progress_full(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-support-size rows with train/validation error and the selected flag."""
    rows = summary.get("support_size_progress", [])
    selected_c = int(summary["selected"]["c"])
    out = []
    for r in sorted(rows, key=lambda r: int(r.get("c"))):
        c = int(r.get("c"))
        out.append({
            "c": c,
            "train_rel_mse": float(r.get("train_rel_mse", float("nan"))),
            "val_rel_mse": float(r.get("val_rel_mse", r.get("objective", float("nan")))),
            "selected": (c == selected_c),
        })
    return out


def weak_rows(summary: dict[str, Any]) -> int:
    """Number of weak regression rows K used by a weak-framework run."""
    return int(summary.get("weak_feature_bank", {}).get("n_weak_rows", 0))
