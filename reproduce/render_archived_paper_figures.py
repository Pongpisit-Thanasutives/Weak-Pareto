#!/usr/bin/env python3
"""Render manuscript Figures 2--4 from archived publication outputs.

This script is the fast, deterministic route for rebuilding the exact graphical
assets shipped with the submission.  It reads the archived CSV summaries under
``reproduce/reference_results/branch_aware_campaign/main`` and the bundled
benchmark data, then writes both vector PDF and high-resolution PNG files.

Use ``reproduce/make_figures.py`` to recompute the underlying results from the
full discovery pipeline before rendering them again.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from dataset_configs import load_benchmark

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reproduce" / "reference_results" / "branch_aware_campaign" / "main"

BLUE = "#0C5DA5"
RED = "#C83232"
ORANGE = "#E68613"
GREEN = "#1B9E77"
PURPLE = "#6A51A3"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", format="pdf", dpi=600, bbox_inches="tight",
                pad_inches=0.04, metadata={"Creator": "Matplotlib from archived publication outputs"})
    fig.savefig(outdir / f"{stem}.png", format="png", dpi=600,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def robustness(outdir: Path) -> None:
    df = pd.read_csv(ARCHIVE / "table_robustness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.65))
    for ax, benchmark, title in zip(
        axes,
        ["FADE", "Fractional Burgers"],
        ["FADE", "Fractional Burgers"],
    ):
        d = df[df["benchmark"] == benchmark].sort_values("noise_percent")
        ax.plot(d["noise_percent"], d["weak_e_xi_max"], "o-", color=BLUE)
        strong = d[np.isfinite(d["strong_e_xi_max"])]
        for _, row in strong.iterrows():
            colour = ORANGE if benchmark == "FADE" and np.isclose(row["noise_percent"], 5.0) else RED
            ax.axhline(float(row["strong_e_xi_max"]), color=colour, linestyle=":", linewidth=2.3)
        ax.set_xlabel("noise level (%)")
        ax.set_ylabel(r"rel. coefficient error $e_{\xi}^{\max}$")
        ax.set_title(title)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].text(-0.14, 1.08, "(a)", transform=axes[0].transAxes,
                 fontsize=12, fontweight="bold")
    axes[1].text(-0.14, 1.08, "(b)", transform=axes[1].transAxes,
                 fontsize=12, fontweight="bold")
    handles = [
        Line2D([0], [0], color=BLUE, marker="o", label="weak (proposed)"),
        Line2D([0], [0], color=RED, linestyle=":", label="strong-form (0% noise)"),
        Line2D([0], [0], color=ORANGE, linestyle=":", label="strong-form (FADE, 5% noise)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, outdir, "Fig2")


def pareto(outdir: Path) -> None:
    df = pd.read_csv(ARCHIVE / "table_progress.csv")
    fig, ax = plt.subplots(figsize=(5.3, 3.7))
    ax.plot(df["c"], df["val_rel_mse"], "o-", color=BLUE)
    sel = df[df["selected"].astype(str).str.lower().isin(["true", "1"])]
    if not sel.empty:
        row = sel.iloc[0]
        ax.scatter([row["c"]], [row["val_rel_mse"]], s=160,
                   facecolors="none", edgecolors=ORANGE, linewidths=2.3,
                   label="selected (elbow)", zorder=5)
    ax.set_xlabel("support size $c$")
    ax.set_ylabel(r"validation error $\mathcal{E}_{\mathrm{val}}$")
    ax.set_xticks(df["c"].astype(int).tolist())
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, outdir, "Fig3")


def burgers(outdir: Path) -> None:
    df = pd.read_csv(ARCHIVE / "figure_burgers_curve.csv").sort_values("noise_percent")
    data, _, _ = load_benchmark("synthetic_fractional_burgers", profile="paper", seed=0)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.65))
    im = axes[0].pcolormesh(data.x, data.t, data.U, shading="auto", cmap="RdBu_r", rasterized=True)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title("solution $u(t,x)$")
    fig.colorbar(im, ax=axes[0], shrink=0.84)
    axes[1].semilogy(df["noise_percent"], df["true_residual"], "o-",
                    color=GREEN, label="true structure")
    axes[1].semilogy(df["noise_percent"], df["closest_competing_residual"], "s--",
                    color=PURPLE, label="closest competing structure")
    axes[1].set_xlabel("noise level (%)")
    axes[1].set_ylabel("weak residual")
    axes[1].set_title("recovery margin")
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    axes[0].text(-0.15, 1.08, "(a)", transform=axes[0].transAxes,
                 fontsize=12, fontweight="bold")
    axes[1].text(-0.15, 1.08, "(b)", transform=axes[1].transAxes,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, "Fig4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figdir", default=str(ROOT.parent / "Manuscript" / "figures"))
    parser.add_argument("--only", choices=["robustness", "pareto", "burgers"])
    args = parser.parse_args()
    outdir = Path(args.figdir).resolve()
    if args.only in (None, "robustness"):
        robustness(outdir)
    if args.only in (None, "pareto"):
        pareto(outdir)
    if args.only in (None, "burgers"):
        burgers(outdir)
    print(f"wrote publication figures to {outdir}")


if __name__ == "__main__":
    main()
