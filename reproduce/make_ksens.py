"""Sensitivity of Weak-Pareto to the number of weak rows K.

Varies the number of test functions (n_t^test, n_x^test) -- and hence the
number of weak rows K = n_t^test * n_x^test -- around the paper preset,
holding everything else at the main-text configuration (uniform order grids,
cmax=4, DE budget, elbow + automatic stopping).  Reports symbolic recovery
and the error metrics of Sec. 5.1 versus K.

Usage (from the package root):
    PYTHONPATH=. python3 reproduce/make_ksens.py            # full budget (5 seeds)
    PYTHONPATH=. python3 reproduce/make_ksens.py --fast     # quick demo (2 seeds)

Writes <outdir>/table_ksens.csv and <outdir>/table_ksens.tex.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import warnings

import numpy as np

from _repro_common import _config_for, matched_errors  # noqa: E402
from weak_pareto_fde_discovery import (  # noqa: E402
    run_weak_pareto_discovery,
    build_weak_candidate_library,
    _default_test_counts,
)

warnings.filterwarnings("ignore")

# Scale factors applied to the paper-preset test counts.  1.0 is the setting
# used for every table and figure in the paper.
FULL_SCALES = (0.35, 0.5, 0.71, 1.0, 1.41)
FAST_SCALES = (0.35, 0.5, 1.0, 1.41)


def _pm(mean: float, sd: float, digits: int = 2) -> str:
    return f"${mean:.{digits}f}\\pm {sd:.{digits}f}$"


def _sci(v: float) -> str:
    if not np.isfinite(v):
        return "--"
    m, e = f"{v:.1e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def _one_discovery(task):
    name, noise, seed, fast, c_t, c_x = task
    data, cfg, truth = _config_for(name, noise, seed, fast, weak=True)
    summary = run_weak_pareto_discovery(
        data, cfg, test_budget="paper", test_counts=(c_t, c_x), verbose=False,
        export_selected_fde=False,
    )
    m = matched_errors(name, summary["selected"], truth)
    return {
        "name": name, "noise": float(noise), "seed": int(seed), "fast": bool(fast),
        "c_t": int(c_t), "c_x": int(c_x),
        "recovered": bool(m["symbolic_form_ok"]),
        "operator_recovered": bool(m["operator_structure_ok"]),
        "e_alpha": float(m["e_alpha"]), "e_beta": float(m["e_beta_max"]),
        "e_xi": float(m["e_xi_max"]), "efit": float(m["full_data_rel_l2"]),
    }


def _row_key(row):
    return (row["name"], float(row["noise"]), int(row["seed"]), bool(row["fast"]),
            int(row["c_t"]), int(row["c_x"]))


def run_study(name: str, noise: float, seeds: list[int], scales: tuple[float, ...],
              fast: bool, outdir: str, jobs: int = 1) -> str:
    os.makedirs(outdir, exist_ok=True)
    data0, _, _ = _config_for(name, noise, seeds[0], fast, weak=True)
    nt, nx = np.asarray(data0.U).shape
    base_t, base_x = _default_test_counts(nt, nx, budget="paper")
    counts = [(max(4, round(base_t * s)), max(6, round(base_x * s))) for s in scales]

    incremental = os.path.join(outdir, "ksens_per_run.jsonl")
    existing = []
    if os.path.exists(incremental):
        with open(incremental) as f:
            existing = [json.loads(line) for line in f if line.strip()]
    done = {_row_key(r): r for r in existing}
    tasks = []
    for c_t, c_x in counts:
        for seed in seeds:
            task = (name, float(noise), int(seed), bool(fast), int(c_t), int(c_x))
            key = (name, float(noise), int(seed), bool(fast), int(c_t), int(c_x))
            if key not in done:
                tasks.append(task)

    if tasks:
        print(f"[ksens] {len(tasks)} missing runs; jobs={max(1, jobs)}", flush=True)
        with open(incremental, "a") as stream:
            if max(1, jobs) == 1:
                iterator = map(_one_discovery, tasks)
                for i, row in enumerate(iterator, 1):
                    done[_row_key(row)] = row
                    stream.write(json.dumps(row) + "\n"); stream.flush()
                    print(f"[ksens run {i}/{len(tasks)}] {row['c_t']}x{row['c_x']} seed={row['seed']}", flush=True)
            else:
                with cf.ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
                    futs = {ex.submit(_one_discovery, task): task for task in tasks}
                    completed = 0
                    for fut in cf.as_completed(futs):
                        row = fut.result(); completed += 1
                        done[_row_key(row)] = row
                        stream.write(json.dumps(row) + "\n"); stream.flush()
                        print(f"[ksens run {completed}/{len(tasks)}] {row['c_t']}x{row['c_x']} seed={row['seed']}", flush=True)
    else:
        print("[ksens] all per-seed runs already present; aggregating", flush=True)

    rows = []
    for c_t, c_x in counts:
        recs = [done[(name, float(noise), int(seed), bool(fast), c_t, c_x)] for seed in seeds]
        K = c_t * c_x
        rec = sum(int(r["recovered"]) for r in recs)
        op_rec = sum(int(r.get("operator_recovered", bool(r["recovered"] and r["e_alpha"] <= 0.15 and r["e_beta"] <= 0.15))) for r in recs)
        ea = [r["e_alpha"] for r in recs]; eb = [r["e_beta"] for r in recs]
        ex = [r["e_xi"] for r in recs]; ev = [r["efit"] for r in recs]
        kappa = float("nan")
        try:
            d0, c0, tr0 = _config_for(name, noise, seeds[0], fast, weak=True)
            bank = build_weak_candidate_library(d0, c0, test_budget="paper",
                                                test_counts=(c_t, c_x), verbose=False)
            p_t = [int(p) for (p, _) in tr0.expected_terms]
            b_t = [float(b) for (_, b) in tr0.expected_terms]
            X = np.asarray(bank.library_exact(float(tr0.expected_alpha), p_t, b_t), dtype=float)
            good = np.all(np.isfinite(X), axis=0)
            Xn = X[:, good]
            norms = np.linalg.norm(Xn, axis=0); norms[norms == 0] = 1.0
            sv = np.linalg.svd(Xn / norms, compute_uv=False)
            kappa = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
        except Exception:
            pass
        row = {
            "tests": f"{c_t}x{c_x}", "K": K,
            "support_power_recovery": f"{rec}/{len(seeds)}",
            "operator_structure_recovery": f"{op_rec}/{len(seeds)}",
            "e_alpha_mean": float(np.nanmean(ea)), "e_alpha_sd": float(np.nanstd(ea, ddof=1)),
            "e_beta_mean": float(np.nanmean(eb)), "e_beta_sd": float(np.nanstd(eb, ddof=1)),
            "e_xi_mean": float(np.nanmean(ex)), "efit_mean": float(np.nanmean(ev)),
            "kappa": kappa,
        }
        rows.append(row)
        print(f"[ksens] {c_t}x{c_x} (K={K:>5d})  supp={row['support_power_recovery']} op={row['operator_structure_recovery']}  "
              f"e_a={row['e_alpha_mean']:.2f}  e_b={row['e_beta_mean']:.2f}  "
              f"e_xi={row['e_xi_mean']:.2e}  Efit={row['efit_mean']:.2e}  kappa={kappa:.1e}", flush=True)

    csv_path = os.path.join(outdir, "table_ksens.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    tex_path = os.path.join(outdir, "table_ksens.tex")
    nl = " \\\\\n"
    with open(tex_path, "w") as f:
        f.write("% K-sensitivity (weak rows) -- produced by reproduce/make_ksens.py\n")
        f.write("Tests $n_t^{\\mathrm{test}}{\\times}n_x^{\\mathrm{test}}$ & $K$ & support/power & operator structure & "
                "$e_\\alpha$ & $e_\\beta^{\\max}$ & $e_\\xi^{\\max}$ & $\\mathcal E_{\\mathrm{fit}}$ & $\\kappa$" + nl)
        for r in rows:
            tests = r["tests"].replace("x", "{\\times}")
            cells = [
                f"${tests}$", str(r["K"]), r["support_power_recovery"], r["operator_structure_recovery"],
                _pm(r["e_alpha_mean"], r["e_alpha_sd"], digits=3),
                _pm(r["e_beta_mean"], r["e_beta_sd"], digits=2),
                _sci(r["e_xi_mean"]), _sci(r["efit_mean"]), _sci(r.get("kappa", float("nan"))),
            ]
            f.write(" & ".join(cells) + nl)
    return csv_path

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="paper_FADE_tsfade_fft")
    ap.add_argument("--noise", type=float, default=10.0)
    ap.add_argument("--fast", action="store_true", help="2 seeds, reduced DE budget")
    ap.add_argument("--scales", default=None, help="comma-separated scale factors")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("FPDE_REPRO_JOBS", "1")))
    args = ap.parse_args()

    seeds = [0, 1] if args.fast else [0, 1, 2, 3, 4]
    scales = tuple(float(x) for x in args.scales.split(",")) if args.scales \
        else (FAST_SCALES if args.fast else FULL_SCALES)
    path = run_study(args.benchmark, args.noise, seeds, scales, args.fast, args.outdir, jobs=args.jobs)
    print(f"[ksens] wrote {path} (and .tex)")


if __name__ == "__main__":
    main()
