#!/usr/bin/env python3
"""Run the clean benchmark workflow for fractional PDE/FDE discovery.

Methods
-------
weak_pareto
    **Proposed method.** Best-subset Pareto-DE using the weak-adjoint fractional
    candidate library.
vanilla_pareto
    Baseline. The same best-subset Pareto-DE optimizer, but with the original
    pointwise/strong candidate library.
weak_grid_stridge
    Baseline. The weak candidate library is kept, but best-subset Pareto-DE is
    replaced by alpha-grid search plus STRidge over a fixed overcomplete library.
weak_fixed_stability
    Baseline/ablation. Fixed weak grid library plus repeated STRidge selections
    over weak-test scales/splits. This stability baseline is meaningful only
    because the candidate alpha/beta library is fixed to the grid.

Examples
--------
Smoke test on all packaged datasets::

    python scripts/run_all_methods.py \
        --methods weak_pareto vanilla_pareto weak_grid_stridge weak_fixed_stability \
        --profile notebook --noise-levels 0 5 --maxiter 1 --popsize 2

"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import weak_grid_stridge_baseline, weak_fixed_library_stability_baseline
from dataset_configs import benchmark_specs, config_search_space_fingerprint, dataset_config_philosophy
from pareto_fde_discovery import run_pareto_discovery, write_json
from weak_pareto_fde_discovery import (
    WeakStabilityConfig,
    model_order_metrics,
    run_stability_selected_weak_pareto_discovery,
    run_weak_pareto_discovery,
    coefficient_truth,
)

METHODS = (
    "weak_pareto",
    "vanilla_pareto",
    "weak_grid_stridge",
    "weak_fixed_stability",
)


def _as_tuple_float(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _selected_from_summary(method: str, result: Any) -> dict[str, Any]:
    """Normalize method-specific result objects to one selected-model dict."""
    if method in {"weak_grid_stridge", "weak_fixed_stability"}:
        d = result.to_dict()
        return {
            "equation": d["equation"],
            "alpha": float(d["alpha"]),
            "terms": d["terms"],
            "coefficients": d["coefficients"],
            "c": int(d["support_size"]),
            "val_mse": float(d["val_mse"]),
            "val_rel_mse": float(d["val_rel_mse"]),
            "full_data_rel_l2": float("nan"),
            "bic": float(d["bic"]),
        }
    return dict(result["selected"])


def selected_model_dict(method: str, result: Any) -> dict[str, Any]:
    """Public wrapper used by notebooks to normalize one method result.

    The benchmark script writes every selected model to a common schema before
    scoring.  Teaching notebooks should use this helper instead of duplicating
    method-specific logic, so notebook outputs match script outputs.
    """
    return _selected_from_summary(method, result)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(obj: Any) -> Any:
    """Convert NumPy/scalar/path objects to JSON-friendly values for tracking."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def _coefficient_error_metrics(
    selected: dict[str, Any],
    expected_terms: list[tuple[int, float]],
    expected_coefficients: list[float] | None,
) -> dict[str, Any]:
    """Match selected terms to truth terms and report coefficient errors.

    Coefficients are scored only after structural matching by p and nearest beta.
    These metrics are post-hoc diagnostics; they are never used during fitting.
    """
    if not expected_coefficients or len(expected_coefficients) != len(expected_terms):
        return {
            "coefficient_scored": False,
            "mean_coef_abs_error": np.nan,
            "max_coef_abs_error": np.nan,
            "mean_coef_rel_error": np.nan,
            "max_coef_rel_error": np.nan,
            "coef_errors_json": "[]",
        }
    terms = [(int(p), float(b)) for p, b in selected.get("terms", [])]
    coefs = [float(c) for c in selected.get("coefficients", [])]
    unmatched = list(zip(terms, coefs))
    records: list[dict[str, Any]] = []
    abs_errs: list[float] = []
    rel_errs: list[float] = []
    for (p_true, b_true), c_true in zip(expected_terms, expected_coefficients):
        candidates = [(i, abs(tb[0][1] - b_true)) for i, tb in enumerate(unmatched) if tb[0][0] == p_true]
        if not candidates:
            records.append({
                "expected_term": [int(p_true), float(b_true)],
                "expected_coefficient": float(c_true),
                "matched": False,
            })
            abs_errs.append(float("inf"))
            rel_errs.append(float("inf"))
            continue
        i, beta_err = min(candidates, key=lambda z: z[1])
        (p_sel, b_sel), c_sel = unmatched.pop(i)
        ae = abs(float(c_sel) - float(c_true))
        re = ae / max(abs(float(c_true)), 1e-14)
        abs_errs.append(float(ae))
        rel_errs.append(float(re))
        records.append({
            "expected_term": [int(p_true), float(b_true)],
            "selected_term": [int(p_sel), float(b_sel)],
            "beta_abs_error": float(beta_err),
            "expected_coefficient": float(c_true),
            "selected_coefficient": float(c_sel),
            "coefficient_abs_error": float(ae),
            "coefficient_rel_error": float(re),
            "matched": True,
        })
    finite_abs = [v for v in abs_errs if np.isfinite(v)]
    finite_rel = [v for v in rel_errs if np.isfinite(v)]
    return {
        "coefficient_scored": True,
        "mean_coef_abs_error": float(np.mean(finite_abs)) if finite_abs else float("inf"),
        "max_coef_abs_error": float(np.max(finite_abs)) if finite_abs else float("inf"),
        "mean_coef_rel_error": float(np.mean(finite_rel)) if finite_rel else float("inf"),
        "max_coef_rel_error": float(np.max(finite_rel)) if finite_rel else float("inf"),
        "coef_errors_json": json.dumps(records),
    }


def _experiment_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Return the top-level tracking manifest for a benchmark run."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "profile": args.profile,
        "methods": list(args.methods),
        "datasets": list(args.datasets) if args.datasets else "all packaged datasets",
        "noise_levels": [float(v) for v in args.noise_levels],
        "seeds": [int(v) for v in args.seeds],
        "weak_test_budget": args.weak_test_budget,
        "stability_splits": int(args.stability_splits),
        "stability_width_scales": [float(v) for v in args.stability_width_scales],
        "maxiter_override": args.maxiter,
        "popsize_override": args.popsize,
        "cmax_override": args.cmax,
        "p_values_override": args.p_values,
        "config_philosophy": dataset_config_philosophy(),
        "output_dir": str(args.output_dir),
    }


def _run_tracking_payload(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    method: str,
    seed: int,
    noise: float,
    data: Any,
    config: Any,
    spec: dict[str, Any],
    selected: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    return _json_ready({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "truth_equation": getattr(data, "truth", ""),
        "method": method,
        "profile": args.profile,
        "seed": int(seed),
        "noise_percent": float(noise),
        "status": status,
        "error": error,
        "data_shape": list(getattr(data, "shape", ())),
        "dt": getattr(data, "dt", None),
        "dx": getattr(data, "dx", None),
        "truth_spec": spec.get("truth_spec_dict", {}),
        "config_search_space": config_search_space_fingerprint(config),
        "runtime_config": {
            "maxiter": int(config.maxiter),
            "popsize": int(config.popsize),
            "cmax": int(config.cmax),
            "selection": config.selection,
            "auto_stop": bool(config.auto_stop),
            "val_fraction": float(config.val_fraction),
            "noise_percent": float(config.noise_percent),
            "seed": int(config.seed),
        },
        "weak_tracking": {
            "weak_test_budget": args.weak_test_budget,
            "stability_splits": int(args.stability_splits),
            "stability_width_scales": [float(v) for v in args.stability_width_scales],
        },
        "selected": selected or {},
        "structure_metrics": metrics or {},
    })


def _run_one_method(
    method: str,
    data: Any,
    config: Any,
    out_dir: Path,
    *,
    weak_test_budget: str,
    stability_splits: int,
    stability_width_scales: tuple[float, ...],
    verbose: bool,
) -> Any:
    out_dir.mkdir(parents=True, exist_ok=True)
    if method == "vanilla_pareto":
        return run_pareto_discovery(data, config, output_dir=out_dir, verbose=verbose, export_selected_fde=True)
    if method == "weak_pareto":
        return run_weak_pareto_discovery(
            data,
            config,
            output_dir=out_dir,
            verbose=verbose,
            export_selected_fde=True,
            test_budget=weak_test_budget,  # type: ignore[arg-type]
        )
    if method == "weak_grid_stridge":
        result = weak_grid_stridge_baseline(
            data, config, verbose=verbose, max_terms=config.cmax, test_budget=weak_test_budget
        )
        write_json(out_dir / "summary.json", result.to_dict())
        return result
    if method == "weak_fixed_stability":
        result = weak_fixed_library_stability_baseline(
            data,
            config,
            verbose=verbose,
            max_terms=config.cmax,
            test_budget=weak_test_budget,
            width_scales=stability_width_scales,
            n_splits=stability_splits,
        )
        write_json(out_dir / "summary.json", result.to_dict())
        return result
    raise ValueError(f"unknown method={method!r}; valid methods are {METHODS}")


def run_single_method(
    method: str,
    data: Any,
    config: Any,
    output_dir: Path,
    *,
    weak_test_budget: str = "smoke",
    stability_splits: int = 2,
    stability_width_scales: tuple[float, ...] = (0.85, 1.0, 1.2),
    verbose: bool = False,
) -> Any:
    """Run exactly one benchmark method using the same backend as the script.

    This function is intentionally public for notebooks.  It delegates to the
    same method dispatcher used by :func:`run_all_methods`, so an interactive
    notebook case and a command-line benchmark case are consistent when they use
    the same dataset, profile, noise level, seed, and runtime settings.
    """
    return _run_one_method(
        method,
        data,
        config,
        output_dir,
        weak_test_budget=weak_test_budget,
        stability_splits=stability_splits,
        stability_width_scales=stability_width_scales,
        verbose=verbose,
    )


def run_all_methods(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "experiment_manifest.json", _experiment_manifest(args))
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for noise in args.noise_levels:
            specs = benchmark_specs(
                data_dir=args.data_dir,
                profile=args.profile,
                maxiter=args.maxiter,
                popsize=args.popsize,
                cmax=args.cmax,
                noise_percent=float(noise),
                seed=int(seed),
                dataset_names=args.datasets,
            )
            for spec in specs:
                dataset_name = spec["dataset_name"]
                data = spec["data"]
                config = spec["config"]
                config.progress = bool(args.progress)
                config.progress_de = bool(args.progress_de)
                if args.quiet:
                    config.progress = False
                    config.progress_de = False
                truth_terms = [(int(p), float(b)) for p, b in spec["expected_terms"]]
                truth_spec = spec["truth_spec"]
                for method in args.methods:
                    print(f"\n[{dataset_name}] seed={seed} noise={noise:g}% method={method}")
                    method_dir = output_dir / f"seed_{int(seed)}" / dataset_name / f"noise_{noise:g}" / method
                    selected: dict[str, Any] | None = None
                    metrics: dict[str, Any] | None = None
                    try:
                        write_json(method_dir / "truth_spec.json", spec.get("truth_spec_dict", {}))
                        write_json(method_dir / "config_search_space.json", config_search_space_fingerprint(config))
                        result = run_single_method(
                            method,
                            data,
                            config,
                            method_dir,
                            weak_test_budget=args.weak_test_budget,
                            stability_splits=args.stability_splits,
                            stability_width_scales=_as_tuple_float(args.stability_width_scales),
                            verbose=not args.quiet,
                        )
                        selected = _selected_from_summary(method, result)
                        metrics = model_order_metrics(
                            selected,
                            spec["expected_alpha"],
                            truth_terms,
                            alpha_tol=float(truth_spec.alpha_tol),
                            beta_tol=float(truth_spec.beta_tol),
                        )
                        metrics.update(_coefficient_error_metrics(
                            selected, truth_terms, coefficient_truth(dataset_name)
                        ))
                        row = {
                            "dataset": dataset_name,
                            "truth": data.truth,
                            "seed": int(seed),
                            "noise_percent": float(noise),
                            "method": method,
                            "selected_equation": selected.get("equation", ""),
                            "selected_alpha": float(selected.get("alpha", np.nan)),
                            "selected_terms": json.dumps(selected.get("terms", [])),
                            "selected_coefficients": json.dumps(selected.get("coefficients", [])),
                            "selected_c": int(selected.get("c", len(selected.get("terms", [])))),
                            "val_rel_mse": float(selected.get("val_rel_mse", np.nan)),
                            "full_data_rel_l2": float(selected.get("full_data_rel_l2", np.nan)),
                            "expected_alpha": float(spec["expected_alpha"]),
                            "expected_terms": json.dumps([[int(p), float(b)] for p, b in truth_terms]),
                            **metrics,
                            "status": "ok",
                        }
                        write_json(method_dir / "run_tracking.json", _run_tracking_payload(
                            args=args, dataset_name=dataset_name, method=method, seed=int(seed),
                            noise=float(noise), data=data, config=config, spec=spec, selected=selected,
                            metrics=metrics, status="ok"
                        ))
                    except Exception as exc:  # keep large batch runs from dying at one dataset.
                        row = {
                            "dataset": dataset_name,
                            "truth": getattr(data, "truth", ""),
                            "seed": int(seed),
                            "noise_percent": float(noise),
                            "method": method,
                            "status": "failed",
                            "error": repr(exc),
                        }
                        write_json(method_dir / "run_tracking.json", _run_tracking_payload(
                            args=args, dataset_name=dataset_name, method=method, seed=int(seed),
                            noise=float(noise), data=data, config=config, spec=spec, selected=selected,
                            metrics=metrics, status="failed", error=repr(exc)
                        ))
                        print(f"FAILED: {exc!r}")
                    rows.append(row)
                    _write_csv(output_dir / "method_comparison.csv", rows)
                    write_json(output_dir / "method_comparison.json", rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results" / "clean_method_comparison")
    ap.add_argument("--datasets", nargs="*", default=None, help="Subset of dataset names. Omit for all packaged datasets.")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=["weak_pareto", "vanilla_pareto", "weak_grid_stridge", "weak_fixed_stability"])
    ap.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 5.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0], help="Random seeds for synthetic initial conditions and shared noise realizations.")
    ap.add_argument("--profile", choices=["notebook", "paper"], default="notebook")
    ap.add_argument("--maxiter", type=int, default=None, help="Optional DE generations override. Omit to use the selected profile.")
    ap.add_argument("--popsize", type=int, default=None, help="Optional DE population multiplier override. Omit to use the selected profile.")
    ap.add_argument("--cmax", type=int, default=None)
    ap.add_argument("--p-values", nargs="+", type=int, default=None, help="Optional candidate powers override, e.g. --p-values 0 1 2. Omit for canonical paper config.")
    ap.add_argument("--weak-test-budget", choices=["smoke", "standard", "paper"], default="smoke")
    ap.add_argument("--stability-splits", type=int, default=2)
    ap.add_argument("--stability-width-scales", nargs="+", type=float, default=[0.85, 1.0, 1.2])
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--progress-de", action="store_true")
    args = ap.parse_args()
    rows = run_all_methods(args)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\nFinished {len(rows)} runs; {ok} succeeded. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
