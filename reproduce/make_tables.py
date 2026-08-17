"""Regenerate the paper's result tables (CSV + LaTeX rows).

Tables produced include:
  * main       -> proposed Weak-Pareto on the four main benchmarks
  * rd_noise   -> Riesz reaction--diffusion order identifiability versus noise
  * robustness -> weak versus strong-form under the same selector
  * progress   -> support-size progress for one benchmark
  * runtime    -> runtime and search budget
  * burgers    -> nonlinear fractional Burgers recovery and competing-structure margin

Usage (from the repository root):
    PYTHONPATH=. python3 reproduce/make_tables.py --fast
    PYTHONPATH=. python3 reproduce/make_tables.py            # paper-scale (slower)
    PYTHONPATH=. python3 reproduce/make_tables.py --only main
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import os
import time

import numpy as np

from _repro_common import (
    APPENDIX_ADJUSTMENTS,
    APPENDIX_BENCHMARKS,
    MAIN_BENCHMARKS,
    NICE_NAME,
    ROBUSTNESS_BENCHMARKS,
    coefficient_truth,
    default_noise,
    default_seeds,
    fit_coefficients_for_structure,
    load_benchmark,
    matched_errors,
    run_strong,
    run_weak,
    support_progress_full,
    weak_rows,
)
from weak_pareto_fde_discovery import build_weak_candidate_library


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _fmt_sci(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / (10.0 ** e)
    if round(abs(m), 1) >= 10.0:   # rounding bumped the mantissa into the next decade
        m /= 10.0
        e += 1
    return f"${m:.1f}{{\\times}}10^{{{e}}}$"

def _mean_std(xs) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    xs = list(xs)
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


def _pm(m: float, sd: float) -> str:
    """Compact LaTeX 'mean +/- std' to two decimals."""
    return f"${m:.2f}\\pm {sd:.2f}$"


def main_metrics_table(outdir: str, fast: bool, noise: float = 10.0) -> str:
    """Table 2: proposed weak_pareto across the four main benchmarks at one noise level."""
    seeds = default_seeds(fast)
    rows = []
    for name in MAIN_BENCHMARKS:
        ms = []
        for seed in seeds:
            summary, truth = run_weak(name, noise, seed, fast)
            ms.append(matched_errors(name, summary["selected"], truth))
        a = _agg(ms)
        row = {
            "benchmark": NICE_NAME[name], "noise_percent": noise, "n_seeds": a["n"],
            "support_power_recovery": a["n_sp"], "operator_structure_recovery": a["n_os"],
        }
        for key in ("e_alpha", "e_beta_max", "e_xi_max", "e_xi_l2", "full_data_rel_l2"):
            row[key], row[key + "_sd"] = a[key]
        rows.append(row)
        print(f"[main] {row['benchmark']:>22s}  sp {a['n_sp']}/{a['n']}  os {a['n_os']}/{a['n']}  "
              f"e_alpha={row['e_alpha']:.3f} e_beta={row['e_beta_max']:.3f} "
              f"e_xi={row['e_xi_max']:.3f} e_xi2={row['e_xi_l2']:.3f} E_fit={row['full_data_rel_l2']:.2e}")
    path = os.path.join(outdir, "table_main.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(outdir, "table_main.tex"), "w") as f:
        for r in rows:
            row = (f"{r['benchmark']} & {r['support_power_recovery']}/{r['n_seeds']} & "
                   f"{r['operator_structure_recovery']}/{r['n_seeds']} & "
                   f"{_cell(r['e_alpha'], r['e_alpha_sd'])} & {_cell(r['e_beta_max'], r['e_beta_max_sd'])} & "
                   f"{_cell(r['e_xi_max'], r['e_xi_max_sd'])} & {_cell(r['e_xi_l2'], r['e_xi_l2_sd'])} & "
                   f"{_pm(r['full_data_rel_l2'], r['full_data_rel_l2_sd'])}")
            f.write(row + " " + chr(92) * 2 + "\n")
    return path


def robustness_table(outdir: str, fast: bool) -> str:
    """Table 3: weak vs strong-form library under the same selector, vs noise."""
    noises = default_noise(fast)
    seeds = default_seeds(fast)
    rows = []
    for name in ROBUSTNESS_BENCHMARKS:
        for noise in noises:
            agg = {"weak": [], "strong": []}
            for seed in seeds:
                for tag, runner in (("weak", run_weak), ("strong", run_strong)):
                    summary, truth = runner(name, noise, seed, fast)
                    agg[tag].append(matched_errors(name, summary["selected"], truth))
            # Parameter errors (order/coefficient) are conditioned on support/power
            # recovery, consistently with the aggregate tables; an em dash (nan) is
            # reported when no seed recovers the structure. The fit residual
            # E_fit (full_data_rel_l2) is a within-framework diagnostic and is reported
            # over all seeds.
            def _cond(ms, key):
                return _mean([m[key] for m in ms if m["symbolic_form_ok"]])
            row = {
                "benchmark": NICE_NAME[name], "noise_percent": noise, "n_seeds": len(seeds),
                "weak_symbolic_recovery": sum(m["symbolic_form_ok"] for m in agg["weak"]),
                "weak_e_alpha": _cond(agg["weak"], "e_alpha"),
                "weak_e_beta_max": _cond(agg["weak"], "e_beta_max"),
                "weak_e_xi_max": _cond(agg["weak"], "e_xi_max"),
                "weak_full_data_rel_l2": _mean([m["full_data_rel_l2"] for m in agg["weak"]]),
                "strong_symbolic_recovery": sum(m["symbolic_form_ok"] for m in agg["strong"]),
                "strong_e_alpha": _cond(agg["strong"], "e_alpha"),
                "strong_e_beta_max": _cond(agg["strong"], "e_beta_max"),
                "strong_e_xi_max": _cond(agg["strong"], "e_xi_max"),
                "strong_full_data_rel_l2": _mean([m["full_data_rel_l2"] for m in agg["strong"]]),
            }
            rows.append(row)
            print(f"[robustness] {row['benchmark']:>5s} noise={noise:>5.1f}%  "
                  f"weak {row['weak_symbolic_recovery']}/{row['n_seeds']} (e_xi={row['weak_e_xi_max']:.3f}) | "
                  f"strong {row['strong_symbolic_recovery']}/{row['n_seeds']} (e_xi={row['strong_e_xi_max']:.3f})")
    path = os.path.join(outdir, "table_robustness.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # LaTeX rows for Table 3 (FADE): recovery, e_xi, E_val for weak and strong.
    with open(os.path.join(outdir, "table_robustness_FADE.tex"), "w") as f:
        for r in rows:
            if r["benchmark"] != "FADE":
                continue
            _x = lambda v: ("--" if not np.isfinite(v) else f"{v:.2f}")
            f.write(f"{r['noise_percent']:.0f} & {r['weak_symbolic_recovery']}/{r['n_seeds']} & "
                    f"{_x(r['weak_e_xi_max'])} & {_fmt_sci(r['weak_full_data_rel_l2'])} & "
                    f"{r['strong_symbolic_recovery']}/{r['n_seeds']} & {_x(r['strong_e_xi_max'])} & "
                    f"{_fmt_sci(r['strong_full_data_rel_l2'])} \\\\\n")
    return path


def progress_table(outdir: str, fast: bool, name: str = "paper_FADE_tsfade_fft", noise: float = 5.0) -> str:
    """Table 4: support-size progress for one benchmark."""
    summary, _ = run_weak(name, noise, 0, fast)
    rows = support_progress_full(summary)
    path = os.path.join(outdir, "table_progress.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["c", "train_rel_mse", "val_rel_mse", "selected"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(outdir, "table_progress.tex"), "w") as f:
        for r in rows:
            mark = "$\\checkmark$" if r["selected"] else ""
            f.write(f"{r['c']} & {_fmt_sci(r['train_rel_mse'])} & {_fmt_sci(r['val_rel_mse'])} & {mark} \\\\\n")
    print(f"[progress] {NICE_NAME[name]} @ {noise}%: " +
          ", ".join(f"c={r['c']}:val={r['val_rel_mse']:.2e}{'*' if r['selected'] else ''}" for r in rows))
    return path


def runtime_table(outdir: str, fast: bool, names=("paper_FADE_tsfade_fft",
                                                  "synthetic_space_fractional_RD",
                                                  "synthetic_fractional_burgers")) -> str:
    """Table 5: runtime and search budget (single seed)."""
    rows = []
    for name in names:
        t0 = time.time(); sw, _ = run_weak(name, 10.0, 0, fast, use_cache=False); tw = time.time() - t0
        t0 = time.time(); ss, _ = run_strong(name, 10.0, 0, fast, use_cache=False); ts = time.time() - t0
        cfg = sw.get("config", {})
        budget = f"{int(cfg.get('popsize', 0))}x{int(cfg.get('maxiter', 0))}"
        row = {"benchmark": NICE_NAME[name], "weak_time_s": round(tw, 1),
               "strong_time_s": round(ts, 1), "weak_rows": weak_rows(sw), "de_budget": budget}
        rows.append(row)
        print(f"[runtime] {row['benchmark']:>22s}  weak {row['weak_time_s']}s  strong {row['strong_time_s']}s  "
              f"K={row['weak_rows']}  DE={budget}")
    path = os.path.join(outdir, "table_runtime.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(outdir, "table_runtime.tex"), "w") as f:
        for r in rows:
            budget_tex = r["de_budget"].replace("x", r"\times ")
            f.write(f"{r['benchmark']} & {r['weak_time_s']:.0f} & {r['strong_time_s']:.0f} & "
                    f"$\\sim${r['weak_rows']} & ${budget_tex}$ \\\\\n")
    return path


_BURGERS_POOL = [(0, 1.0), (1, 1.0), (2, 1.0), (0, 1.7), (0, 2.0), (1, 1.7), (0, 0.5)]
_BURGERS_TRUE = frozenset([(1, 1.0), (0, 1.7)])


def burgers_table(outdir: str, fast: bool) -> str:
    """Nonlinear fractional Burgers: selection, competing-structure margin, coefficient error."""
    name = "synthetic_fractional_burgers"
    data, cfg, truth = load_benchmark(name, profile="paper", seed=0)
    coef_true = np.array(coefficient_truth(name))
    noises = default_noise(fast) if fast else [0.0, 10.0, 25.0]
    rows = []
    for noise in noises:
        c = dataclasses.replace(cfg, noise_percent=float(noise), seed=0)
        bank = build_weak_candidate_library(data, c, test_budget="paper", verbose=False)
        bank.precompute(verbose=False)
        scored = []
        for pair in itertools.combinations(_BURGERS_POOL, 2):
            _, rel = fit_coefficients_for_structure(bank, 1.0, list(pair))
            scored.append((rel, frozenset(pair)))
        scored.sort(key=lambda r: r[0])
        best_rel, best_supp = scored[0]
        closest_competing_rel = next(r for r, s in scored if s != _BURGERS_TRUE)
        coef_fit, true_rel = fit_coefficients_for_structure(bank, 1.0, [(1, 1.0), (0, 1.7)])
        e_xi = float(np.max(np.abs(np.array(coef_fit) - coef_true) / (np.abs(coef_true) + 1e-12)))
        row = {
            "noise_percent": noise, "true_selected": bool(best_supp == _BURGERS_TRUE),
            "true_residual": true_rel, "closest_competing_residual": closest_competing_rel,
            "margin_ratio": closest_competing_rel / (true_rel + 1e-18), "e_xi_max": e_xi,
        }
        rows.append(row)
        print(f"[burgers] noise={noise:>5.1f}%  selected={'TRUE' if row['true_selected'] else 'WRONG'}  "
              f"margin={row['margin_ratio']:.1f}x  e_xi={e_xi:.3f}")
    path = os.path.join(outdir, "table_burgers.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(outdir, "table_burgers.tex"), "w") as f:
        for r in rows:
            sel = "yes" if r["true_selected"] else "no"
            f.write(f"{r['noise_percent']:.0f} & {sel} & "
                    f"$\\sim${r['margin_ratio']:.0f}$\\times$ & {r['e_xi_max']:.2f} \\\\\n")
    return path


_ADJ_LABEL = {
    "paper_ADE_Convection_diffusion": r"$p\in\{0\}$",
    "synthetic_two_fractional_rhs": r"$p\in\{0\}$, AIC-type",
}


RD_BENCHMARKS = ("synthetic_space_fractional_RD", "synthetic_time_space_fractional_RD")


def conditioning_table(outdir: str, fast: bool, name: str = "paper_FADE_tsfade_fft",
                       spacings=(0.5, 0.25, 0.1, 0.05)) -> str:
    """Conditioning of a FIXED dense fractional dictionary (by-design comparison).

    A fixed-dictionary alternative to the continuous-order encoding must
    enumerate fractional orders on a grid of spacing dbeta. This table reports,
    for the noiseless weak library of one benchmark, the number of columns, the
    mutual coherence (max |cosine| between distinct normalised columns), and the
    condition number of the column-normalised dictionary, as dbeta shrinks.
    Near-unit coherence and exploding condition numbers quantify why dense
    enumeration destabilises selection, motivating the continuous-order search.
    """
    import dataclasses
    import numpy as np
    from weak_pareto_fde_discovery import build_weak_candidate_library

    data, cfg, _ = load_benchmark(name, profile="paper", seed=0)
    cfg = dataclasses.replace(cfg, noise_percent=0.0, seed=0, progress=False)
    bank = build_weak_candidate_library(data, cfg, test_budget="paper", verbose=False)
    bank.precompute(verbose=False)
    alpha = 1.0
    rows = []
    for db in spacings:
        orders = np.round(np.arange(0.1, 3.0 + 1e-9, db), 10)
        terms_p = tuple(0 for _ in orders)
        theta = np.asarray(bank.library_exact(alpha, terms_p, tuple(float(b) for b in orders)))
        finite = np.all(np.isfinite(theta), axis=1)
        theta = theta[finite]
        norms = np.linalg.norm(theta, axis=0)
        theta_n = theta / np.maximum(norms, 1e-300)
        gram = theta_n.T @ theta_n
        off = gram - np.eye(gram.shape[0])
        coherence = float(np.max(np.abs(off)))
        cond = float(np.linalg.cond(theta_n))
        row = {"delta_beta": db, "n_columns": int(theta.shape[1]),
               "coherence": coherence, "condition_number": cond}
        rows.append(row)
        print(f"[conditioning] dbeta={db:<5} cols={row['n_columns']:>3d} "
              f"coherence={coherence:.6f} cond={cond:.3e}")
    path = os.path.join(outdir, "table_conditioning.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(outdir, "table_conditioning.tex"), "w") as f:
        for r in rows:
            f.write(f"{r['delta_beta']} & {r['n_columns']} & {r['coherence']:.4f} & "
                    f"{_fmt_sci(r['condition_number'])} \\\\\n")
    return path


def _fmt_exi(v: float) -> str:
    """Format a relative coefficient error, using scientific notation when large."""
    if v >= 100:
        return _fmt_sci(v)
    if v >= 10:
        return f"{v:.0f}"
    return f"{v:.2f}"


def _exi_cell(m: float, sd: float) -> str:
    """Coefficient-error cell: mean-only scientific when inflated, else mean+/-sd."""
    return _fmt_sci(m) if abs(m) >= 10.0 else _pm(m, sd)


def _cell(m: float, sd: float) -> str:
    """Conditioned mean+/-sd cell; em-dash when no seed satisfies the condition."""
    if m is None or not np.isfinite(m):
        return "--"
    return f"${m:.2f}\\pm {sd:.2f}$"


def _agg(ms) -> dict:
    """Aggregate per-seed matched_errors into recovery counts and conditioned errors.

    Two recovery counts are reported: ``n_sp`` (support/power pattern) and
    ``n_os`` (operator structure, i.e. orders within tolerance).  Order and
    coefficient errors are averaged only over the seeds whose support and powers
    are recovered (returning NaN -> em-dash when none are), while the full-data
    residual is averaged over all seeds.  Implements the structure-conditioned
    reporting of the presubmission review (items 1.6, 2.5, 3.4).
    """
    n = len(ms)
    keep = [m for m in ms if m["support_power_ok"]]

    def cond(key):
        xs = [m[key] for m in keep if np.isfinite(m[key])]
        return _mean_std(xs) if xs else (float("nan"), float("nan"))

    def allseeds(key):
        xs = [m[key] for m in ms if np.isfinite(m[key])]
        return _mean_std(xs) if xs else (float("nan"), float("nan"))

    return {
        "n": n,
        "n_sp": sum(m["support_power_ok"] for m in ms),
        "n_os": sum(m["operator_structure_ok"] for m in ms),
        "e_alpha": cond("e_alpha"),
        "e_beta_max": cond("e_beta_max"),
        "e_xi_max": cond("e_xi_max"),
        "e_xi_l2": cond("e_xi_l2"),
        "full_data_rel_l2": allseeds("full_data_rel_l2"),
    }


def _rd_cell(m: float, sd: float) -> str:
    """Table-4 cell with extra precision for clean, sub-percent errors."""
    if m is None or not np.isfinite(m):
        return "--"
    if abs(float(m)) < 0.01 and abs(float(sd)) < 0.01:
        return f"${m:.4f}\\pm {sd:.4f}$"
    return _cell(m, sd)


def rd_noise_table(outdir: str, fast: bool, noises=(0.0, 2.0, 5.0, 10.0)) -> str:
    """Order identifiability of the two Riesz reaction--diffusion benchmarks vs noise, including clean references."""
    seeds = default_seeds(fast)
    rows = []
    for name in RD_BENCHMARKS:
        for noise in noises:
            ms = []
            for seed in seeds:
                summary, truth = run_weak(name, noise, seed, fast)
                ms.append(matched_errors(name, summary["selected"], truth))
            a = _agg(ms)
            row = {
                "benchmark": NICE_NAME[name], "noise_percent": noise, "n_seeds": a["n"],
                "support_power_recovery": a["n_sp"], "operator_structure_recovery": a["n_os"],
            }
            for key in ("e_alpha", "e_beta_max", "e_xi_max", "e_xi_l2", "full_data_rel_l2"):
                row[key], row[key + "_sd"] = a[key]
            rows.append(row)
            print(f"[rd_noise] {row['benchmark']:>22s} @ {noise:>4.1f}%  sp {a['n_sp']}/{a['n']} os {a['n_os']}/{a['n']}  "
                  f"e_beta={row['e_beta_max']:.3f}  e_xi={row['e_xi_max']:.3f}")
    path = os.path.join(outdir, "table_rd_noise.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(outdir, "table_rd_noise.tex"), "w") as f:
        for r in rows:
            row = (f"{r['benchmark']} & {r['noise_percent']:.0f} & {r['support_power_recovery']}/{r['n_seeds']} & "
                   f"{r['operator_structure_recovery']}/{r['n_seeds']} & "
                   f"{_rd_cell(r['e_alpha'], r['e_alpha_sd'])} & {_rd_cell(r['e_beta_max'], r['e_beta_max_sd'])} & "
                   f"{_rd_cell(r['e_xi_max'], r['e_xi_max_sd'])} & {_rd_cell(r['full_data_rel_l2'], r['full_data_rel_l2_sd'])}")
            f.write(row + " " + chr(92) * 2 + "\n")
    return path


def appendix_table(outdir: str, fast: bool, noise: float = 10.0) -> str:
    """Appendix challenging cases under default vs adjusted hyperparameters (structure-conditioned)."""
    seeds = default_seeds(fast)
    rows = []
    for name in APPENDIX_BENCHMARKS:
        for regime, overrides, label in (
            ("default", None, "default"),
            ("adjusted", APPENDIX_ADJUSTMENTS[name], _ADJ_LABEL[name]),
        ):
            ms = []
            for seed in seeds:
                summary, truth = run_weak(name, noise, seed, fast, overrides=overrides)
                ms.append(matched_errors(name, summary["selected"], truth))
            a = _agg(ms)
            row = {
                "benchmark": NICE_NAME[name], "regime": regime, "setting": label,
                "noise_percent": noise, "n_seeds": a["n"],
                "support_power_recovery": a["n_sp"], "operator_structure_recovery": a["n_os"],
            }
            for key in ("e_alpha", "e_beta_max", "e_xi_max", "e_xi_l2", "full_data_rel_l2"):
                row[key], row[key + "_sd"] = a[key]
            rows.append(row)
            print(f"[appendix] {row['benchmark']:>14s} [{regime:>8s}: {label}]  "
                  f"sp {a['n_sp']}/{a['n']} os {a['n_os']}/{a['n']}  e_beta={row['e_beta_max']:.3f}  e_xi={row['e_xi_max']:.3f}")
    path = os.path.join(outdir, "table_appendix.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(outdir, "table_appendix.tex"), "w") as f:
        for r in rows:
            row = (f"{r['benchmark']} & {r['setting']} & {r['support_power_recovery']}/{r['n_seeds']} & "
                   f"{r['operator_structure_recovery']}/{r['n_seeds']} & "
                   f"{_cell(r['e_beta_max'], r['e_beta_max_sd'])} & "
                   f"{_cell(r['e_xi_max'], r['e_xi_max_sd'])} & {_pm(r['full_data_rel_l2'], r['full_data_rel_l2_sd'])}")
            f.write(row + " " + chr(92) * 2 + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="small budget/seeds for a quick run")
    ap.add_argument("--outdir", default="results", help="directory for CSV/LaTeX output")
    ap.add_argument("--only", choices=["main", "robustness", "progress", "runtime", "burgers", "appendix", "rd_noise", "conditioning"], default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    todo = [args.only] if args.only else ["main", "robustness", "progress", "runtime", "burgers", "appendix", "rd_noise", "conditioning"]
    fns = {
        "main": main_metrics_table, "robustness": robustness_table, "progress": progress_table,
        "runtime": runtime_table, "burgers": burgers_table, "appendix": appendix_table,
        "rd_noise": rd_noise_table, "conditioning": conditioning_table,
    }
    for key in todo:
        print(f"== {key} table ==")
        print("wrote", fns[key](args.outdir, args.fast))


if __name__ == "__main__":
    main()
