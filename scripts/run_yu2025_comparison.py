#!/usr/bin/env python3
"""Compare Weak-Pareto against the adapted neural fractional-discovery framework of Yu et al. (2025).

The default comparison uses the packaged ``paper_FADE_tsfade_fft`` field and its
nominal equation/target orders.  The upstream Yu source snapshot is not
redistributed, so the packaged campaign does not verify byte identity against an
absent upstream file.  The operator realizations are also distinct: Weak-Pareto
uses its periodic directional spectral convention, whereas the Yu adapter uses a
one-sided finite-terminal approximation.  Periodic signed-Riesz benchmarks fall
outside the adapter's declared comparison scope and are recorded as skipped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_configs import benchmark_spec
from dataset_configs import config_search_space_fingerprint, with_uniform_order_grids
from external_baselines.yu2025 import YuBaselineConfig, run_yu_baseline
from fpde_datasets import add_multiplicative_uniform_noise
from weak_pareto_fde_discovery import coefficient_truth, run_weak_pareto_discovery

METHODS = ("weak_pareto", "yu2025_full", "yu2025_optimizer_only")
YU_SHARED_FIELD_DATASETS = {"paper_FADE_tsfade_fft", "paper_ADE_Convection_diffusion"}



def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _input_hash_manifest(datasets: Sequence[str], data_dir: Path) -> dict[str, Any]:
    filenames = {
        "paper_FADE_tsfade_fft": "tsfade_fft.dat",
        "paper_ADE_Convection_diffusion": "Convection_diffusion.dat",
    }
    original_dir = ROOT / "external_baselines" / "yu2025" / "original" / "dataset"
    out: dict[str, Any] = {}
    for dataset_name in datasets:
        filename = filenames.get(dataset_name)
        if filename is None:
            continue
        project_path = data_dir / filename
        original_path = original_dir / filename
        project_hash = _sha256_file(project_path) if project_path.exists() else None
        original_hash = _sha256_file(original_path) if original_path.exists() else None
        out[dataset_name] = {
            "project_path": str(project_path),
            "project_sha256": project_hash,
            "yu_original_path": str(original_path),
            "yu_original_sha256": original_hash,
            "byte_identical": bool(project_hash is not None and project_hash == original_hash),
            "upstream_snapshot_included": bool(original_path.exists()),
            "upstream_byte_identity_verified": bool(
                project_hash is not None and original_hash is not None and project_hash == original_hash
            ),
            "verification_status": (
                "verified_equal"
                if project_hash is not None and original_hash is not None and project_hash == original_hash
                else "not_verified_upstream_snapshot_absent"
                if not original_path.exists()
                else "verified_different"
            ),
        }
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _active_terms(selected: dict[str, Any]) -> list[tuple[int, float, float]]:
    terms = [(int(p), float(b)) for p, b in selected.get("terms", [])]
    coefs = [float(v) for v in selected.get("coefficients", [])]
    if len(terms) != len(coefs):
        coefs = [1.0] * len(terms)
    if not coefs:
        return []
    scale = max(abs(v) for v in coefs)
    tol = max(1e-10, 1e-4 * scale)
    return [(p, b, c) for (p, b), c in zip(terms, coefs) if abs(c) > tol]


def _score_selected(
    selected: dict[str, Any],
    *,
    expected_alpha: float,
    expected_terms: Sequence[tuple[int, float]],
    expected_coefficients: Sequence[float] | None,
    alpha_tol: float,
    beta_tol: float,
) -> dict[str, Any]:
    """Strict, structure-conditioned scoring requested by the prior review."""
    active = _active_terms(selected)
    unmatched = active.copy()
    matches: list[dict[str, Any]] = []
    power_hits = 0
    operator_hits = 0
    beta_errors: list[float] = []
    coef_abs: list[float] = []
    coef_rel: list[float] = []

    for j, (p_true, b_true) in enumerate(expected_terms):
        candidates = [(i, abs(item[1] - b_true)) for i, item in enumerate(unmatched) if item[0] == p_true]
        if not candidates:
            matches.append({"expected": [p_true, b_true], "matched": False})
            continue
        i, b_err = min(candidates, key=lambda z: z[1])
        p_sel, b_sel, c_sel = unmatched.pop(i)
        power_hits += 1
        beta_ok = bool(b_err <= beta_tol)
        operator_hits += int(beta_ok)
        beta_errors.append(float(b_err))
        record: dict[str, Any] = {
            "expected": [int(p_true), float(b_true)],
            "selected": [int(p_sel), float(b_sel)],
            "selected_coefficient": float(c_sel),
            "beta_abs_error": float(b_err),
            "matched": True,
            "operator_hit": beta_ok,
        }
        if expected_coefficients is not None and j < len(expected_coefficients):
            c_true = float(expected_coefficients[j])
            ae = abs(c_sel - c_true)
            re = ae / max(abs(c_true), 1e-14)
            record.update({"expected_coefficient": c_true, "coef_abs_error": ae, "coef_rel_error": re})
            coef_abs.append(ae)
            coef_rel.append(re)
        matches.append(record)

    selected_c = len(active)
    expected_c = len(expected_terms)
    support_size_match = selected_c == expected_c
    support_power_recovered = bool(support_size_match and power_hits == expected_c)
    rhs_operator_structure_recovered = bool(support_size_match and operator_hits == expected_c)
    alpha_error = abs(float(selected.get("alpha", np.nan)) - float(expected_alpha))
    alpha_hit = bool(np.isfinite(alpha_error) and alpha_error <= alpha_tol)
    full_operator_structure_recovered = bool(rhs_operator_structure_recovered and alpha_hit)

    # Parameter errors are reported only when the support/power pattern is correct,
    # consistently with the main experiments (matched_errors in reproduce/_repro_common.py).
    conditioned = support_power_recovered
    coef_conditioned = support_power_recovered and len(coef_rel) == expected_c
    return {
        "selected_c_active": int(selected_c),
        "support_size_match": bool(support_size_match),
        "support_power_recovered": support_power_recovered,
        "rhs_operator_structure_recovered": rhs_operator_structure_recovered,
        "full_operator_structure_recovered": full_operator_structure_recovered,
        "alpha_abs_error_conditioned": float(alpha_error) if conditioned else np.nan,
        "mean_beta_abs_error_conditioned": float(np.mean(beta_errors)) if conditioned and beta_errors else np.nan,
        "max_beta_abs_error_conditioned": float(np.max(beta_errors)) if conditioned and beta_errors else np.nan,
        "mean_coef_abs_error_conditioned": float(np.mean(coef_abs)) if coef_conditioned else np.nan,
        "max_coef_abs_error_conditioned": float(np.max(coef_abs)) if coef_conditioned else np.nan,
        "max_coef_rel_error_conditioned": float(np.max(coef_rel)) if coef_conditioned else np.nan,
        "coef_l2_rel_error_conditioned": (
            float(np.linalg.norm(coef_abs) / (np.linalg.norm(expected_coefficients) + 1e-14))
            if coef_conditioned and expected_coefficients is not None
            else np.nan
        ),
        "matching_json": json.dumps(matches),
    }


def _selected_from_weak(summary: dict[str, Any]) -> dict[str, Any]:
    return dict(summary["selected"])


def _yu_cfg_for_dataset(args: argparse.Namespace, dataset_name: str, seed: int, noise: float, surrogate: str) -> YuBaselineConfig:
    cfg = YuBaselineConfig.for_profile(
        args.yu_profile,
        surrogate=surrogate,
        protocol=args.yu_protocol,
        device=args.device,
        dtype=args.yu_dtype,
        seed=int(seed),
        noise_percent=float(noise),
        epochs=args.yu_epochs,
        de_maxiter=args.yu_de_maxiter,
        de_popsize=args.yu_de_popsize,
        order_mode=args.yu_order_mode,
        quadrature_nodes=args.yu_quadrature_nodes,
        verbose=not args.quiet,
    )
    if dataset_name == "paper_ADE_Convection_diffusion":
        cfg.alpha_bounds = (0.999, 1.0)
        cfg.activation = "tanh"
    return cfg


def _summary_markdown(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    lines = [
        "# Weak-Pareto versus Yu et al. (2025)",
        "",
        "The comparison uses the same packaged field and identical multiplicative-uniform noise realizations per seed. Upstream byte identity was not verified in the packaged campaign because the upstream snapshot is absent.",
        "Weak-Pareto uses a periodic directional spectral convention; the Yu adapter uses a one-sided finite-terminal approximation. The comparison is therefore shared-data and nominal-equation matched, not operator-identical.",
        "Order and coefficient errors are shown only when the support/power pattern is correctly recovered; coefficient errors do not additionally require complete fractional-order recovery.",
        "",
        "| Dataset | Noise | Seed | Method | Status | Operator recovery | Alpha error | Mean beta error | Mean coefficient error | Runtime (s) | Equation |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        def fmt(value: Any) -> str:
            try:
                x = float(value)
                return "—" if not np.isfinite(x) else f"{x:.4g}"
            except Exception:
                return "—"
        eq = str(row.get("selected_equation", "")).replace("|", "\\|")
        lines.append(
            f"| {row.get('dataset','')} | {row.get('noise_percent','')} | {row.get('seed','')} | "
            f"{row.get('method','')} | {row.get('status','')} | "
            f"{int(bool(row.get('full_operator_structure_recovered', False)))} | "
            f"{fmt(row.get('alpha_abs_error_conditioned'))} | {fmt(row.get('mean_beta_abs_error_conditioned'))} | "
            f"{fmt(row.get('mean_coef_abs_error_conditioned'))} | {fmt(row.get('runtime_seconds'))} | {eq} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Weak-Pareto profile: `{manifest['weak_profile']}`; exact-order polishing and exact coefficient refit are enabled.",
            f"- Yu profile: `{manifest['yu_profile']}`; protocol `{manifest['yu_protocol']}`; order mode `{manifest['yu_order_mode']}`.",
            "- `yu2025_full` uses the neural reconstruction + Gauss–Jacobi + STRidge + DE framework.",
            "- `yu2025_optimizer_only` replaces only the neural reconstruction by a tensor-product quintic spline, retaining Gauss–Jacobi + STRidge + DE.",
            "- The adapted neural fractional-discovery framework and optimizer-only comparison remain separate rows.",
            "- Yu runs are intentionally skipped for periodic signed-Riesz datasets because those operators fall outside the adapter's declared comparison scope.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "methods": list(args.methods),
        "datasets": list(args.datasets),
        "noise_levels": list(args.noise_levels),
        "seeds": list(args.seeds),
        "weak_profile": args.weak_profile,
        "yu_profile": args.yu_profile,
        "yu_protocol": args.yu_protocol,
        "yu_order_mode": args.yu_order_mode,
        "device": args.device,
        "input_file_hashes": _input_hash_manifest(args.datasets, args.data_dir),
        "fairness_rules": [
            "same clean data file and same multiplicative-uniform noise generator/seed",
            "same post-hoc truth metadata and strict operator-recovery tolerances",
            "runtime includes each method's complete fitting framework",
            "Yu neural and optimizer-only variants are reported separately",
            "periodic signed-Riesz cases outside the Yu adapter's declared scope are skipped rather than silently redefined",
        ],
    }
    _write_json(args.output_dir / "experiment_manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for noise in args.noise_levels:
            for dataset_name in args.datasets:
                spec = benchmark_spec(
                    dataset_name,
                    data_dir=args.data_dir,
                    profile=args.weak_profile,
                    maxiter=args.weak_maxiter,
                    popsize=args.weak_popsize,
                    noise_percent=float(noise),
                    seed=int(seed),
                )
                data, cfg, truth = spec["data"], spec["config"], spec["truth_spec"]
                if args.truth_agnostic_grids:
                    n_alpha, n_beta = ((9, 13) if args.weak_profile == "notebook" else (47, 59))
                    cfg = with_uniform_order_grids(cfg, n_alpha=n_alpha, n_beta=n_beta)
                cfg.exact_order_refit = True
                cfg.exact_order_polish = True
                cfg.progress = bool(args.progress and not args.quiet)
                cfg.progress_de = False
                expected_terms = [(int(p), float(b)) for p, b in spec["expected_terms"]]
                expected_coefs = coefficient_truth(dataset_name)
                shared_noisy = add_multiplicative_uniform_noise(
                    data.U, float(noise), seed=int(seed)
                )
                shared_input_dir = (
                    args.output_dir / "shared_inputs" / f"seed_{seed}" / dataset_name / f"noise_{noise:g}"
                )
                shared_input_dir.mkdir(parents=True, exist_ok=True)
                np.save(shared_input_dir / "noisy_field.npy", shared_noisy)
                noise_sha256 = _sha256_array(shared_noisy)
                _write_json(
                    shared_input_dir / "input_metadata.json",
                    {
                        "dataset": dataset_name,
                        "noise_percent": float(noise),
                        "seed": int(seed),
                        "noise_model": "multiplicative uniform: U*(1 + percent/100 * Uniform[-1,1])",
                        "array_sha256": noise_sha256,
                    },
                )

                for method in args.methods:
                    method_dir = args.output_dir / f"seed_{seed}" / dataset_name / f"noise_{noise:g}" / method
                    method_dir.mkdir(parents=True, exist_ok=True)
                    tracking_path = method_dir / "comparison_tracking.json"
                    if tracking_path.exists() and not args.restart:
                        try:
                            prior = json.loads(tracking_path.read_text(encoding="utf-8"))
                            prior_row = prior.get("row", {})
                            if prior_row.get("status") == "ok":
                                rows.append(prior_row)
                                print(f"[skip completed] {dataset_name} noise={noise:g}% seed={seed} method={method}", flush=True)
                                _write_csv(args.output_dir / "yu_method_comparison.csv", rows)
                                _write_json(args.output_dir / "yu_method_comparison.json", rows)
                                continue
                        except Exception:
                            pass
                    _write_json(method_dir / "truth_spec.json", spec["truth_spec_dict"])
                    status = "ok"
                    error = ""
                    selected: dict[str, Any] = {}
                    runtime = np.nan
                    metadata: dict[str, Any] = {}
                    print(f"\n[{dataset_name}] noise={noise:g}% seed={seed} method={method}")
                    try:
                        if method == "weak_pareto":
                            _write_json(method_dir / "config_search_space.json", config_search_space_fingerprint(cfg))
                            start = time.perf_counter()
                            result = run_weak_pareto_discovery(
                                data,
                                cfg,
                                output_dir=method_dir,
                                verbose=not args.quiet,
                                export_selected_fde=True,
                                test_budget=args.weak_test_budget,
                            )
                            runtime = time.perf_counter() - start
                            selected = _selected_from_weak(result)
                            metadata = {"exact_refit": True, "exact_order_polish": True}
                        else:
                            if dataset_name not in YU_SHARED_FIELD_DATASETS:
                                status = "skipped_operator_scope"
                                error = (
                                    "This periodic signed-Riesz benchmark is outside the declared Yu adapter "
                                    "comparison scope; it is not silently reinterpreted with a one-sided operator."
                                )
                            else:
                                surrogate = "neural" if method == "yu2025_full" else "spline"
                                yu_cfg = _yu_cfg_for_dataset(args, dataset_name, int(seed), float(noise), surrogate)
                                yu_result = run_yu_baseline(data, yu_cfg, output_dir=method_dir)
                                runtime = yu_result.runtime_seconds
                                selected = yu_result.selected_model_dict()
                                metadata = dict(yu_result.metadata)
                                saved_noisy = np.load(method_dir / "shared_noisy_field.npy")
                                saved_hash = _sha256_array(saved_noisy)
                                if saved_hash != noise_sha256:
                                    raise RuntimeError(
                                        "Yu framework did not use the predefined shared noise realization"
                                    )
                                metadata["shared_noise_verified"] = True
                                metadata["shared_noise_sha256"] = saved_hash
                    except Exception as exc:
                        status = "failed"
                        error = repr(exc)
                        print(f"FAILED: {error}")

                    score = (
                        _score_selected(
                            selected,
                            expected_alpha=float(spec["expected_alpha"]),
                            expected_terms=expected_terms,
                            expected_coefficients=expected_coefs,
                            alpha_tol=float(truth.alpha_tol),
                            beta_tol=float(truth.beta_tol),
                        )
                        if status == "ok"
                        else {}
                    )
                    row = {
                        "dataset": dataset_name,
                        "truth": data.truth,
                        "noise_percent": float(noise),
                        "seed": int(seed),
                        "method": method,
                        "noise_realization_sha256": noise_sha256,
                        "status": status,
                        "error": error,
                        "selected_equation": selected.get("equation", ""),
                        "selected_alpha": selected.get("alpha", np.nan),
                        "selected_terms": json.dumps(selected.get("terms", [])),
                        "selected_coefficients": json.dumps(selected.get("coefficients", [])),
                        "val_rel_mse": selected.get("val_rel_mse", np.nan),
                        "full_data_rel_l2": selected.get("full_data_rel_l2", np.nan),
                        "runtime_seconds": runtime,
                        **score,
                    }
                    rows.append(row)
                    _write_json(
                        method_dir / "comparison_tracking.json",
                        {
                            "row": row,
                            "selected": selected,
                            "method_metadata": metadata,
                            "truth_spec": spec["truth_spec_dict"],
                        },
                    )
                    _write_csv(args.output_dir / "yu_method_comparison.csv", rows)
                    _write_json(args.output_dir / "yu_method_comparison.json", rows)

    (args.output_dir / "SUMMARY.md").write_text(_summary_markdown(rows, manifest), encoding="utf-8")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results" / "yu2025_comparison")
    ap.add_argument("--datasets", nargs="+", default=["paper_FADE_tsfade_fft"])
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 1.0, 5.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--weak-profile", choices=["notebook", "paper"], default="notebook")
    ap.add_argument("--weak-test-budget", choices=["smoke", "standard", "paper"], default="smoke")
    ap.add_argument("--weak-maxiter", type=int, default=None)
    ap.add_argument("--weak-popsize", type=int, default=None)
    ap.add_argument("--truth-agnostic-grids", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--yu-profile", choices=["smoke", "standard", "paper"], default="smoke")
    ap.add_argument("--yu-protocol", choices=["matched", "faithful"], default="matched")
    ap.add_argument("--yu-order-mode", choices=["bank", "exact"], default="bank")
    ap.add_argument("--yu-epochs", type=int, default=None)
    ap.add_argument("--yu-de-maxiter", type=int, default=None)
    ap.add_argument("--yu-de-popsize", type=int, default=None)
    ap.add_argument("--yu-quadrature-nodes", type=int, default=5)
    ap.add_argument("--yu-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--restart", action="store_true", help="ignore completed per-run tracking files")
    args = ap.parse_args()
    rows = run(args)
    ok = sum(r.get("status") == "ok" for r in rows)
    print(f"\nFinished {len(rows)} runs; {ok} succeeded. Summary: {args.output_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
