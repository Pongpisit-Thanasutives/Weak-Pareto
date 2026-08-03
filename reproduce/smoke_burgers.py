"""Tiny deterministic Burgers smoke workflow.

This is a software-validation workflow, not a publication experiment.  It uses a
small endpoint-excluded Burgers field and minimal weak/strong Pareto-DE budgets,
then writes a compact CSV/JSON summary and one diagnostic field image.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from fpde_datasets import make_fractional_burgers
from reproduce._repro_common import _attach_full_data_rel_l2
from pareto_fde_discovery import DiscoveryConfig, run_pareto_discovery
from weak_pareto_fde_discovery import run_weak_pareto_discovery


def _config() -> DiscoveryConfig:
    return DiscoveryConfig(
        backend="spectral_l1",
        alpha_grid=tuple(np.linspace(0.9, 1.1, 5)),
        beta_grid=tuple(np.linspace(0.8, 1.9, 7)),
        cmax=2,
        p_values=(0, 1),
        max_patterns_per_c=2,
        maxiter=1,
        popsize=2,
        polish=False,
        seed=0,
        val_fraction=0.25,
        trim_t=2,
        trim_x=2,
        spectral_riesz=False,
        selection="elbow",
        exact_order_refit=True,
        exact_order_polish=False,
        auto_stop=False,
        progress=False,
        progress_de=False,
    )


def _row(method: str, summary: dict, runtime: float) -> dict[str, object]:
    selected = summary["selected"]
    val_rel = float(selected["val_rel_mse"])
    full_rel = float(selected.get("full_data_rel_l2", math.nan))
    if not math.isfinite(val_rel):
        raise RuntimeError(f"{method} produced a non-finite normalized validation score")
    return {
        "method": method,
        "selected_equation": selected["equation"],
        "selected_c": int(selected["c"]),
        "val_rel_mse": val_rel,
        "full_data_rel_l2": full_rel,
        "runtime_seconds": float(runtime),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--figdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    data = make_fractional_burgers(
        nt=20, nx=20, nx_fine=40, t_end=0.4, Lx=10.0, dt_fine=0.004
    )
    cfg = _config()

    rows: list[dict[str, object]] = []
    t0 = time.perf_counter()
    weak = run_weak_pareto_discovery(
        data,
        cfg,
        output_dir=args.outdir / "weak",
        verbose=False,
        test_budget="smoke",
        test_counts=(3, 4),
    )
    rows.append(_row("Weak-Pareto (tiny smoke)", weak, time.perf_counter() - t0))

    t0 = time.perf_counter()
    strong = run_pareto_discovery(
        data,
        cfg,
        output_dir=args.outdir / "strong",
        verbose=False,
    )
    _attach_full_data_rel_l2(data, cfg, strong)
    rows.append(_row("Strong Pareto (tiny smoke)", strong, time.perf_counter() - t0))

    csv_path = args.outdir / "burgers_smoke_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.outdir / "burgers_smoke_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "publication_quality": False,
                "dataset": data.name,
                "shape": list(data.U.shape),
                "time_grid_endpoint_excluded": bool(data.t[-1] < 0.4),
                "rows": rows,
            },
            indent=2,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    image = ax.imshow(
        data.U,
        aspect="auto",
        origin="lower",
        extent=[float(data.x[0]), float(data.x[-1]), float(data.t[0]), float(data.t[-1])],
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Tiny fractional-Burgers smoke field")
    fig.colorbar(image, ax=ax, label="u")
    fig.tight_layout()
    fig_path = args.figdir / "burgers_smoke_field.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    for required in (csv_path, json_path, fig_path):
        if not required.is_file() or required.stat().st_size == 0:
            raise RuntimeError(f"smoke output was not written: {required}")

    print(f"[smoke] dataset={data.name} shape={data.U.shape}")
    for row in rows:
        print(
            f"[smoke] {row['method']}: c={row['selected_c']} "
            f"val_rel_mse={row['val_rel_mse']:.3e} "
            f"runtime={row['runtime_seconds']:.2f}s"
        )
    print(f"[smoke] summary: {json_path}")
    print(f"[smoke] figure:  {fig_path}")


if __name__ == "__main__":
    main()
