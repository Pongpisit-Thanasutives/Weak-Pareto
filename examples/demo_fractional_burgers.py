"""Demonstration: weak-form recovery of a genuinely NONLINEAR fractional PDE.

All other bundled benchmarks are linear (every true term has power ``p=0``).  This
one has a true ``p=1`` advective term, so it exercises the data-weighted weak form

    <u, (D_x^beta)^*(u^p phi)>

rather than only the linear ``p=0`` columns.  True equation (directional spectral
operator, periodic, integer time):

    u_t = -1.0 * u u_x + 0.25 * D_x^1.7 u
        => encoding terms {(p=1, beta=1.0): -1.0, (p=0, beta=1.7): 0.25}.

Two demonstrations:
  (A) a controlled best-subset over a 7-term pool (2 true + 5 competing candidates), using
      exact weak features, at 0 / 10 / 25 % noise -- shows the true pair wins and the
      coefficients are recovered, and in particular that fractional D^1.7 is
      distinguished from integer u_xx (beta=2);
  (B) one end-to-end run of the full weak Pareto-DE framework (with the exact-order
      refit of item 1) confirming the structure is recovered automatically.

Run from the repository root:  PYTHONPATH=. python3 examples/demo_fractional_burgers.py
"""
from __future__ import annotations

import dataclasses
import itertools
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from dataset_configs import load_benchmark
from weak_pareto_fde_discovery import (
    build_weak_candidate_library,
    fit_coefficients_for_structure,
    run_weak_pareto_discovery,
)

TRUE = {(1, 1.0): -1.0, (0, 1.7): 0.25}
TRUE_SUPPORT = frozenset(TRUE)
POOL = [(0, 1.0), (1, 1.0), (2, 1.0), (0, 1.7), (0, 2.0), (1, 1.7), (0, 0.5)]
LABEL = {
    (0, 1.0): "u_x", (1, 1.0): "u*u_x [TRUE]", (2, 1.0): "u^2*u_x",
    (0, 1.7): "D^1.7 u [TRUE]", (0, 2.0): "u_xx", (1, 1.7): "u*D^1.7 u", (0, 0.5): "D^0.5 u",
}


def demo_controlled_best_subset() -> None:
    data, cfg, _ = load_benchmark("synthetic_fractional_burgers", profile="paper", seed=0)
    print("(A) Controlled best-subset over a 7-term pool (size 2), exact weak features.")
    print("    True equation:", data.truth, "\n")
    truth_vec = np.array([TRUE[(1, 1.0)], TRUE[(0, 1.7)]])
    for noise in (0.0, 10.0, 25.0):
        c = dataclasses.replace(cfg, noise_percent=noise, seed=0)
        bank = build_weak_candidate_library(data, c, test_budget="paper", verbose=False)
        bank.precompute(verbose=False)
        scored = []
        for pair in itertools.combinations(POOL, 2):
            coef, rel = fit_coefficients_for_structure(bank, 1.0, list(pair))
            scored.append((rel, frozenset(pair), list(pair), coef))
        scored.sort(key=lambda r: r[0])
        best = scored[0]
        ok = "== TRUE structure" if best[1] == TRUE_SUPPORT else "!! WRONG"
        print(f"--- noise {noise:>4.1f}% ---  winner {ok}")
        for rel, supp, terms, coef in scored[:3]:
            names = " , ".join(f"{coef[i]:+.3f}*{LABEL[terms[i]]}" for i in range(2))
            tag = "  <-- selected" if supp == best[1] else ""
            print(f"      rel-resid={rel:.4e}   {names}{tag}")
        ct, _ = fit_coefficients_for_structure(bank, 1.0, [(1, 1.0), (0, 1.7)])
        err = np.linalg.norm(ct - truth_vec) / np.linalg.norm(truth_vec)
        print(f"      true-structure coef = [{ct[0]:+.4f}, {ct[1]:+.4f}]  (truth [-1.0000, +0.2500])"
              f"  rel-coef-err={err:.4f}\n")


def demo_end_to_end() -> None:
    data, cfg, _ = load_benchmark("synthetic_fractional_burgers", profile="paper", seed=0)
    cfg = dataclasses.replace(cfg, noise_percent=0.0, maxiter=10, popsize=6,
                              exact_order_refit=True, exact_order_polish=True, progress=False)
    print("(B) Full weak Pareto-DE framework (0% noise), with exact-order refit.")
    t0 = time.time()
    summary = run_weak_pareto_discovery(data, cfg, test_budget="paper", verbose=False)
    sel = summary["selected"]
    terms = list(zip(sel["p_tuple"], [round(b, 3) for b in sel["beta_tuple"]],
                     [round(float(x), 4) for x in sel["coefficients"]]))

    def matches(got: list[tuple[int, float]]) -> bool:
        if len(got) != 2:
            return False
        need, used, ok = [(1, 1.0), (0, 1.7)], [False, False], 0
        for p, b in got:
            for j, (pt, bt) in enumerate(need):
                if not used[j] and p == pt and abs(b - bt) < 0.07:
                    used[j] = True
                    ok += 1
                    break
        return ok == 2

    print(f"    finished in {time.time() - t0:.0f}s; selected c = {sel['c']}, alpha = {round(sel['alpha'], 4)}")
    print("    selected terms (p, beta, coef):", terms)
    print("    structure matches truth {u*u_x, D^1.7 u}:",
          matches(list(zip(sel["p_tuple"], sel["beta_tuple"]))))
    print("    exact_order_refit:", summary.get("selected_exact_refit"))


if __name__ == "__main__":
    demo_controlled_best_subset()
    demo_end_to_end()
