"""Regenerate the paper's figures.

Figures produced (PDF and PNG):
  * Fig2.{pdf,png}     -> Fig. 2 (coefficient error vs noise, weak vs strong)
  * Fig3.{pdf,png}     -> Fig. 3 (validation error vs support size, elbow + plateau)
  * Fig4.{pdf,png}     -> Fig. 4 (nonlinear Burgers solution + recovery margin)

Usage (from the repository root):
    PYTHONPATH=. python3 reproduce/make_figures.py --fast --figdir /path/to/paper/figures
    PYTHONPATH=. python3 reproduce/make_figures.py --figdir /path/to/paper/figures   # paper-scale

If --figdir is omitted, figures are written to ./results/figures.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from _repro_common import (
    ROBUSTNESS_BENCHMARKS,
    NICE_NAME,
    coefficient_truth,
    default_noise,
    default_seeds,
    fit_coefficients_for_structure,
    load_benchmark,
    matched_errors,
    run_strong,
    run_weak,
    support_progress_curve,
)
from fpde_datasets import make_fractional_burgers
from weak_pareto_fde_discovery import build_weak_candidate_library

from pathlib import Path
import shutil

# All paper figures use the LaTeX-enabled SciencePlots ``science`` style.
# The official style file is bundled for reproducibility when the optional
# ``scienceplots`` Python package is unavailable. Deliberately do not use
# ``no-latex``: Figures 2--4 must share the same TeX-rendered typography.
if shutil.which("latex") is None:
    raise RuntimeError(
        "LaTeX is required to reproduce the paper figures. "
        "Install a TeX distribution or use the pre-generated figures."
    )
try:
    import scienceplots  # noqa: F401  (registers the SciencePlots styles)
    plt.style.use("science")
except ImportError:  # pragma: no cover - exercised in minimal environments
    style_path = Path(__file__).resolve().parent / "styles" / "science.mplstyle"
    plt.style.use(str(style_path))

# Fonts are enlarged for legibility in print.
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})

# Consistent, colour-blind-safe palette
_C_WEAK = "#0C5DA5"    # blue   -> weak (proposed)
_C_STRONG = "#FF2C00"  # red    -> strong-form at 0%
_C_STRONG_FADE5 = "#FF9500"  # orange -> strong-form FADE at 5%
_C_SEL = "#FF8C00"     # orange -> selected-model highlight


def _curve(name: str, noises: list[float], seeds: list[int], fast: bool, runner):
    mean, lo, hi = [], [], []
    for noise in noises:
        vals = []
        for seed in seeds:
            summary, truth = runner(name, noise, seed, fast)
            m = matched_errors(name, summary["selected"], truth)
            # condition the coefficient error on support/power recovery: an
            # unrecovered structure has no meaningful coefficient error
            if m["symbolic_form_ok"]:
                vals.append(m["e_xi_max"])
        vals = np.array([v for v in vals if np.isfinite(v)]) if any(np.isfinite(v) for v in vals) else np.array([np.nan])
        mean.append(np.nanmean(vals))
        lo.append(np.nanmin(vals))
        hi.append(np.nanmax(vals))
    return np.array(mean), np.array(lo), np.array(hi)


def robustness_figure(figdir: str, fast: bool) -> str:
    noises = default_noise(fast)
    seeds = default_seeds(fast)
    fig, axes = plt.subplots(
        1, len(ROBUSTNESS_BENCHMARKS),
        figsize=(4.3 * len(ROBUSTNESS_BENCHMARKS), 3.7),
    )
    if len(ROBUSTNESS_BENCHMARKS) == 1:
        axes = [axes]
    x = np.array(noises)
    for ax, name in zip(axes, ROBUSTNESS_BENCHMARKS):
        wm, wl, wh = _curve(name, noises, seeds, fast, run_weak)
        sm, _, _ = _curve(name, noises, seeds, fast, run_strong)
        ax.plot(x, wm, "o-", color=_C_WEAK)

        # Parameter error is defined only at noise levels where the strong-form
        # method recovers the support/power pattern. Show those isolated values
        # as horizontal dotted references instead of visually connecting them.
        finite_idx = np.flatnonzero(np.isfinite(sm))
        for idx in finite_idx:
            noise = float(x[idx])
            colour = _C_STRONG_FADE5 if (name == "paper_FADE_tsfade_fft" and np.isclose(noise, 5.0)) else _C_STRONG
            ax.axhline(float(sm[idx]), color=colour, linestyle=":", linewidth=2.4)

        ax.set_xlabel(r"noise level (\%)")
        ax.set_ylabel("rel. coefficient error $e_\\xi^{\\max}$")
        ax.set_title(NICE_NAME[name])
        ax.grid(alpha=0.3)

    axes[0].text(-0.12, 1.12, "(a)", transform=axes[0].transAxes,
                 fontsize=16, fontweight="bold", va="top")
    if len(axes) > 1:
        axes[1].text(-0.12, 1.12, "(b)", transform=axes[1].transAxes,
                     fontsize=16, fontweight="bold", va="top")

    legend_handles = [
        Line2D([0], [0], color=_C_WEAK, marker="o", linestyle="-",
               linewidth=2.0, markersize=7, label="weak (proposed)"),
        Line2D([0], [0], color=_C_STRONG, linestyle=":", linewidth=2.4,
               label=r"strong-form (0\% noise)"),
        Line2D([0], [0], color=_C_STRONG_FADE5, linestyle=":", linewidth=2.4,
               label=r"strong-form (FADE, 5\% noise)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.035),
               handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    path = os.path.join(figdir, "Fig2.pdf")
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(figdir, "Fig2.png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def pareto_figure(figdir: str, fast: bool) -> str:
    # One representative weak run; show the support-size Pareto front.
    summary, _ = run_weak("paper_FADE_tsfade_fft", 5.0, 0, fast)
    cs, errs = support_progress_curve(summary)
    selected_c = int(summary["selected"]["c"])
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(cs, errs, "o-", color=_C_WEAK)
    if selected_c in cs:
        i = cs.index(selected_c)
        ax.scatter([cs[i]], [errs[i]], s=170, facecolors="none",
                   edgecolors=_C_SEL, linewidths=2.5, label="selected (elbow)")
    ax.set_xlabel("support size $c$")
    ax.set_ylabel("validation error $\\mathcal{E}_{\\mathrm{val}}$")
    ax.set_xticks(cs)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figdir, "Fig3.pdf")
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(figdir, "Fig3.png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


_BURGERS_POOL = [(0, 1.0), (1, 1.0), (2, 1.0), (0, 1.7), (0, 2.0), (1, 1.7), (0, 0.5)]
_BURGERS_TRUE = frozenset([(1, 1.0), (0, 1.7)])


def burgers_figure(figdir: str, fast: bool) -> str:
    name = "synthetic_fractional_burgers"
    data, cfg, _ = load_benchmark(name, profile="paper", seed=0)
    noises = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0] if not fast else [0.0, 10.0, 25.0]
    true_res, competing_res = [], []
    for noise in noises:
        c = dataclasses.replace(cfg, noise_percent=float(noise), seed=0)
        bank = build_weak_candidate_library(data, c, test_budget="paper", verbose=False)
        bank.precompute(verbose=False)
        scored = []
        for pair in itertools.combinations(_BURGERS_POOL, 2):
            _, rel = fit_coefficients_for_structure(bank, 1.0, list(pair))
            scored.append((rel, frozenset(pair)))
        scored.sort(key=lambda r: r[0])
        true_rel = next(r for r, s in scored if s == _BURGERS_TRUE)
        competing_rel = next(r for r, s in scored if s != _BURGERS_TRUE)
        true_res.append(true_rel)
        competing_res.append(competing_rel)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    im = axes[0].pcolormesh(data.x, data.t, data.U, shading="auto", cmap="RdBu_r")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title("solution $u(t,x)$")
    fig.colorbar(im, ax=axes[0], shrink=0.85)
    axes[1].semilogy(noises, true_res, "o-", color="#1b9e77", label="true structure")
    axes[1].semilogy(noises, competing_res, "s--", color="#7570b3", label="closest competing structure")
    axes[1].set_xlabel(r"noise level (\%)")
    axes[1].set_ylabel("weak residual")
    axes[1].set_title("recovery margin")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[0].text(-0.16, 1.10, "(a)", transform=axes[0].transAxes,
                 fontsize=16, fontweight="bold", va="top")
    axes[1].text(-0.16, 1.10, "(b)", transform=axes[1].transAxes,
                 fontsize=16, fontweight="bold", va="top")
    fig.tight_layout()
    path = os.path.join(figdir, "Fig4.pdf")
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(figdir, "Fig4.png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="small budget/seeds for a quick run")
    ap.add_argument("--figdir", default=os.path.join("results", "figures"),
                    help="output directory for figures (point at the paper's figures/ to overwrite placeholders)")
    ap.add_argument("--only", choices=["robustness", "pareto", "burgers"], default=None)
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    if args.only in (None, "burgers"):
        print("wrote", burgers_figure(args.figdir, args.fast))
    if args.only in (None, "pareto"):
        print("wrote", pareto_figure(args.figdir, args.fast))
    if args.only in (None, "robustness"):
        print("wrote", robustness_figure(args.figdir, args.fast))


if __name__ == "__main__":
    main()
