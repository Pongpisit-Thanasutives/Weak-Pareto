#!/usr/bin/env python3
"""Summarize and sanity-check benchmark CSV files.

This script reads the ``method_comparison.csv`` files produced by
``scripts/run_all_methods.py`` and writes compact Markdown/JSON summaries for
paper tables.  The proposed method is ``weak_pareto``: weak library +
best-subset Pareto-DE.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def _to_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.is_dir():
            path = path / "method_comparison.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(dict(r) for r in csv.DictReader(f))
    return rows


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get("method", "")), _to_float(r.get("noise_percent"), 0.0))].append(r)

    table: list[dict[str, Any]] = []
    for (method, noise), rs in sorted(groups.items(), key=lambda z: (z[0][1], z[0][0])):
        ok = [r for r in rs if r.get("status") == "ok"]
        full = [_to_bool(r.get("full_structure_recovered")) for r in ok]
        rhs_f1 = [_to_float(r.get("rhs_f1")) for r in ok if math.isfinite(_to_float(r.get("rhs_f1")))]
        alpha_err = [_to_float(r.get("alpha_abs_error")) for r in ok if math.isfinite(_to_float(r.get("alpha_abs_error")))]
        beta_err = [_to_float(r.get("max_matched_beta_abs_error")) for r in ok if math.isfinite(_to_float(r.get("max_matched_beta_abs_error")))]
        val_rel = [_to_float(r.get("val_rel_mse")) for r in ok if math.isfinite(_to_float(r.get("val_rel_mse")))]
        coef_rel = [_to_float(r.get("max_coef_rel_error")) for r in ok if math.isfinite(_to_float(r.get("max_coef_rel_error")))]
        table.append({
            "method": method,
            "noise_percent": noise,
            "runs": len(rs),
            "ok_runs": len(ok),
            "symbolic_structure_recovery_rate": mean(full) if full else 0.0,
            "full_recovery_rate": mean(full) if full else 0.0,  # backward-compatible alias
            "mean_rhs_f1": mean(rhs_f1) if rhs_f1 else math.nan,
            "mean_alpha_abs_error": mean(alpha_err) if alpha_err else math.nan,
            "mean_max_beta_abs_error": mean(beta_err) if beta_err else math.nan,
            "median_val_rel_mse": median(val_rel) if val_rel else math.nan,
            "mean_max_coef_rel_error": mean(coef_rel) if coef_rel else math.nan,
        })
    meta = {
        "n_rows": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "datasets": sorted({str(r.get("dataset", "")) for r in rows}),
        "methods": sorted({str(r.get("method", "")) for r in rows}),
        "noise_levels": sorted({_to_float(r.get("noise_percent"), 0.0) for r in rows}),
    }
    return table, meta


def write_markdown(path: Path, table: list[dict[str, Any]], meta: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Publication benchmark summary",
        "",
        f"Rows: {meta['n_rows']}; successful runs: {meta['n_ok']}",
        "",
        f"Datasets: `{', '.join(meta['datasets'])}`",
        "",
        f"Methods: `{', '.join(meta['methods'])}`",
        "",
        "## Aggregate recovery table",
        "",
        "| Noise (%) | Method | Runs | OK | Symbolic structure recovery | Mean RHS F1 | Mean alpha error | Mean max beta error | Mean max coef rel error | Median val rel MSE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['noise_percent']:g} | `{r['method']}` | {r['runs']} | {r['ok_runs']} | "
            f"{r['symbolic_structure_recovery_rate']:.3f} | {r['mean_rhs_f1']:.3f} | "
            f"{r['mean_alpha_abs_error']:.4g} | {r['mean_max_beta_abs_error']:.4g} | {r['mean_max_coef_rel_error']:.4g} | "
            f"{r['median_val_rel_mse']:.4g} |"
        )
    lines += ["", "## Failed runs", ""]
    failed = [r for r in rows if r.get("status") != "ok"]
    if not failed:
        lines.append("No failed runs.")
    else:
        lines += ["| Dataset | Noise | Method | Error |", "|---|---:|---|---|"]
        for r in failed:
            lines.append(f"| `{r.get('dataset','')}` | {r.get('noise_percent','')} | `{r.get('method','')}` | `{r.get('error','')}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True, help="CSV files or result directories containing method_comparison.csv")
    ap.add_argument("--output-md", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--require-proposed-zero-full", action="store_true", help="Fail unless weak_pareto recovers the symbolic structure of every 0-noise run.")
    args = ap.parse_args()

    rows = read_rows(args.inputs)
    table, meta = summarize(rows)
    write_markdown(args.output_md, table, meta, rows)
    json_path = args.output_json or args.output_md.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"meta": meta, "table": table}, indent=2), encoding="utf-8")

    if args.require_proposed_zero_full:
        proposed_zero = [
            r for r in rows
            if r.get("method") == "weak_pareto" and abs(_to_float(r.get("noise_percent"), 0.0)) < 1e-12
        ]
        bad = [r for r in proposed_zero if r.get("status") != "ok" or not _to_bool(r.get("full_structure_recovered"))]
        if not proposed_zero:
            raise SystemExit("No zero-noise weak_pareto rows found; cannot verify sanity condition.")
        if bad:
            examples = "; ".join(f"{r.get('dataset')}:{r.get('recovery_label')}" for r in bad[:5])
            raise SystemExit(f"Zero-noise proposed weak_pareto sanity check failed for {len(bad)} run(s): {examples}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
