"""Representative discovered equations vs. ground truth.

For each principal benchmark, at clean data and at the main noisy operating
point, this prints the complete discovered equation
    D_t^{alpha} u = sum_j xi_j u^{p_j} D_x^{beta_j} u
next to the true equation.  The representative seed is chosen by a declared
rule: the seed whose worst spatial-order error e_beta^max is the median across
the reported seeds (ties broken by the smaller coefficient error), so the shown
equation is typical rather than best-case.  A fixed seed can be forced with
--seed.

Usage (from the package root):
    PYTHONPATH=. python3 reproduce/make_equations.py            # full budget, 5 seeds
    PYTHONPATH=. python3 reproduce/make_equations.py --fast     # quick demo, 2 seeds
    PYTHONPATH=. python3 reproduce/make_equations.py --seed 0   # force a fixed seed

Writes <outdir>/table_equations.tex and .csv.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import warnings

import numpy as np

try:
    from ._repro_common import (
        _config_for, matched_errors, MAIN_BENCHMARKS, APPENDIX_BENCHMARKS,
        APPENDIX_ADJUSTMENTS, NICE_NAME, coefficient_truth, run_weak,
    )
except ImportError:  # direct script execution from the package root
    from _repro_common import (
        _config_for, matched_errors, MAIN_BENCHMARKS, APPENDIX_BENCHMARKS,
        APPENDIX_ADJUSTMENTS, NICE_NAME, coefficient_truth, run_weak,
    )

warnings.filterwarnings("ignore")

RIESZ_BENCHMARKS = {
    "synthetic_space_fractional_RD",
    "synthetic_time_space_fractional_RD",
    "synthetic_two_fractional_rhs",
}

# Main noisy operating point for the representative equations (per-benchmark).
# Well-behaved benchmarks are shown at 10%; the harder Riesz reaction-diffusion
# cases are shown at 5% as illustrative degradation cases; this is not claimed
# to be a shared moderate-noise operating point for complete operator recovery.
NOISY_LEVEL = {
    "paper_FADE_tsfade_fft": 10.0,
    "synthetic_fractional_burgers": 10.0,
    "synthetic_space_fractional_RD": 5.0,
    "synthetic_time_space_fractional_RD": 5.0,
    "paper_ADE_Convection_diffusion": 10.0,
    "synthetic_two_fractional_rhs": 5.0,
}


def _term_latex(name: str, p: int, beta: float, xi: float, *, lead: bool) -> str:
    """One signed model term in readable LaTeX, with the correct operator symbol."""
    mag = abs(xi)
    sign = "-" if xi < 0 else "+"
    coeff = f"{mag:.2f}"
    # operator factor
    if abs(beta) < 5e-3:                      # identity -> reaction u^{p+1}
        op = "u" if p == 0 else (f"u^{{{p+1}}}")
    else:
        if name not in RIESZ_BENCHMARKS and abs(beta - 1.0) < 1e-10:
            deriv = "u_x"
        elif abs(beta - 2.0) < 1e-10:
            deriv = "u_{xx}"
        else:
            if name in RIESZ_BENCHMARKS:
                deriv = f"\\mathcal{{R}}_{{{beta:.2f}}}u"
            else:
                deriv = f"D_x^{{{beta:.2f}}}u"
        if p == 0:
            op = deriv
        elif p == 1:
            op = f"u\\,{deriv}"
        else:
            op = f"u^{{{p}}}\\,{deriv}"
    body = f"{coeff}\\,{op}"
    if lead:
        return (f"-{body}" if xi < 0 else body)
    return f" {sign} {body}"


def equation_latex(name: str, alpha: float, terms: list[tuple[int, float, float]], alpha_mode: str | None = None) -> str:
    """Assemble the equation using explicit temporal and spatial operator modes."""
    if alpha_mode == "integer" or (alpha_mode is None and abs(alpha - 1.0) < 1e-11):
        lhs = r"\partial_t u = "
    else:
        lhs = f"D_t^{{{alpha:.4f}}}u = "
    if not terms:
        return lhs + "0"
    # order terms by |beta| for stable presentation
    terms = sorted(terms, key=lambda t: (t[0], t[1]))
    rhs = "".join(_term_latex(name, p, b, x, lead=(i == 0)) for i, (p, b, x) in enumerate(terms))
    return lhs + rhs


def true_equation_latex(name: str, truth) -> str:
    terms = [(int(p), float(b), float(x))
             for (p, b), x in zip(truth.expected_terms, coefficient_truth(name))]
    true_mode = "integer" if abs(float(truth.expected_alpha) - 1.0) < 1e-11 else ("fractional_subunit" if float(truth.expected_alpha) < 1.0 else "fractional_superunit")
    return equation_latex(name, float(truth.expected_alpha), terms, true_mode)


def discovered_equation(name: str, noise: float, seeds: list[int], fast: bool, force_seed: int | None):
    """Run discovery for the given seeds; return (latex, seed, recovered, pruned, e_beta, e_xi)."""
    adj = APPENDIX_ADJUSTMENTS.get(name, {})
    runs = []
    use_seeds = [force_seed] if force_seed is not None else seeds
    for seed in use_seeds:
        s, truth = run_weak(name, noise, seed, fast, overrides=(adj or None))
        m = matched_errors(name, s["selected"], truth)
        runs.append((seed, s["selected"], m, truth))
    # declared representative rule: median e_beta_max seed (tie -> smaller e_xi_max)
    runs_sorted = sorted(runs, key=lambda r: (float(np.nan_to_num(r[2]["e_beta_max"], nan=1e9)),
                                              float(np.nan_to_num(r[2]["e_xi_max"], nan=1e9))))
    seed, sel, m, truth = runs_sorted[len(runs_sorted) // 2]
    terms = [(int(p), float(b), float(x))
             for p, b, x in zip(sel["p_tuple"], sel["beta_tuple"], sel["coefficients"])]
    latex = equation_latex(name, float(sel["alpha"]), terms, sel.get("alpha_mode"))
    n_pruned = int(sel.get("n_pruned", 0)) if isinstance(sel, dict) else 0
    return latex, int(seed), bool(m["operator_structure_ok"]), n_pruned, float(m["e_beta_max"]), float(m["e_xi_max"]), truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="2 seeds, reduced DE budget")
    ap.add_argument("--seed", type=int, default=None, help="force a fixed representative seed")
    ap.add_argument("--benchmarks", default="main", choices=["main", "all"])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    seeds = [0, 1] if args.fast else [0, 1, 2, 3, 4]
    names = list(MAIN_BENCHMARKS) + (list(APPENDIX_BENCHMARKS) if args.benchmarks == "all" else [])
    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for name in names:
        for noise in (0.0, NOISY_LEVEL[name]):
            latex, seed, rec, pruned, eb, ex, truth = discovered_equation(
                name, noise, seeds, args.fast, args.seed)
            true_ltx = true_equation_latex(name, truth)
            rows.append(dict(benchmark=NICE_NAME[name], noise=f"{noise:.0f}", seed=seed,
                             recovered="yes" if rec else "no", pruned=pruned,
                             true_eq=true_ltx, discovered_eq=latex))
            print(f"[eq] {NICE_NAME[name]:20s} {noise:4.0f}%  seed {seed}  "
                  f"rec={'Y' if rec else 'N'}  e_beta={eb:.3f} e_xi={ex:.3f}")
            print(f"        true: ${true_ltx}$")
            print(f"        disc: ${latex}$")

    with open(os.path.join(args.outdir, "table_equations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    nl = " \\\\\n"
    with open(os.path.join(args.outdir, "table_equations.tex"), "w") as f:
        f.write("% Representative discovered equations -- reproduce/make_equations.py\n")
        f.write("Benchmark & Noise & Discovered equation (representative seed)" + nl)
        cur = None
        for r in rows:
            if r["benchmark"] != cur:
                cur = r["benchmark"]
                f.write("\\midrule\n")
                f.write(f"\\multicolumn{{3}}{{l}}{{\\emph{{{r['benchmark']}}} --- true: ${r['true_eq']}$}}" + nl)
            tag = "" if r["recovered"] == "yes" else "$^{\\dagger}$"
            f.write(f"{r['benchmark']} & {r['noise']}\\% & ${r['discovered_eq']}${tag}" + nl)
    print(f"[eq] wrote {args.outdir}/table_equations.{{csv,tex}}")


if __name__ == "__main__":
    main()
