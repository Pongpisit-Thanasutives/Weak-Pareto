"""Command-line driver: discover a finite-domain FDE from a real data file.

This is the file to run on your own measured field.  It loads ``u(t, x)`` from a
CSV or NPZ, runs the weak-Pareto discovery with the finite-domain (one-sided,
non-periodic) operator configuration, prints the discovered equation and the
full support-size Pareto front, and optionally writes a JSON summary and a CSV
of the front.

No ground truth is used; the output is the discovered equation plus its held-out
weak residual.  See ``README.md`` for dataset pointers and the
operator/boundary choices each dataset needs.

Examples
--------
Matrix CSV (first row = x, first column = t, interior = u), advective column
flowing in +x, time-fractional, publication-quality search:

    PYTHONPATH=. python real_data/run_real.py --csv column.csv --layout matrix \
        --space-side left --beta-min 0.7 --beta-max 2.0 --powers 0,1 \
        --budget paper --out results_real/column

Tidy CSV with columns t,x,u:

    PYTHONPATH=. python real_data/run_real.py --csv column_long.csv --layout long --budget paper

NPZ holding arrays t, x, U; integer time order, two-sided (Riesz-like) dispersion:

    PYTHONPATH=. python real_data/run_real.py --npz field.npz --integer-time \
        --space-side symmetric --budget paper
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

from real_data.real_fade import (
    bounded_transport_config,
    discover,
    load_field_csv,
    load_field_npz,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Discover a finite-domain FDE from a real (t, x) field.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=str, help="path to a CSV field")
    src.add_argument("--npz", type=str, help="path to an .npz with arrays t, x, U")
    ap.add_argument("--layout", choices=["matrix", "long"], default="matrix",
                    help="CSV layout: 'matrix' (x in first row, t in first column) or 'long' (columns t,x,u)")
    ap.add_argument("--space-side", choices=["left", "right", "symmetric"], default="left",
                    help="one-sided operator orientation; 'left' for inflow at x=0 flowing +x, 'symmetric' for two-sided dispersion")
    ap.add_argument("--integer-time", action="store_true", help="fix the time order to 1 (default: search a Caputo fractional order)")
    ap.add_argument("--alpha-min", type=float, default=0.5)
    ap.add_argument("--alpha-max", type=float, default=1.2)
    ap.add_argument("--beta-min", type=float, default=0.7)
    ap.add_argument("--beta-max", type=float, default=2.0)
    ap.add_argument("--cmax", type=int, default=3, help="maximum support size to search")
    ap.add_argument("--powers", type=str, default="0,1", help="comma-separated candidate integer powers, e.g. 0,1")
    ap.add_argument("--budget", choices=["smoke", "standard", "paper"], default="paper",
                    help="search/quadrature budget; 'paper' is publication quality, 'smoke' is a fast wiring check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="output path prefix; writes <prefix>_summary.json and <prefix>_pareto.csv")
    return ap.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = _parse_args()

    if args.csv is not None:
        data = load_field_csv(args.csv, layout=args.layout)
    else:
        data = load_field_npz(args.npz)
    print(f"Loaded field '{data.name}': shape (n_t, n_x) = {data.U.shape}.")

    powers = tuple(int(p) for p in args.powers.split(",") if p.strip() != "")
    config = bounded_transport_config(
        time_fractional=not args.integer_time,
        space_side=args.space_side,
        alpha_range=(args.alpha_min, args.alpha_max),
        beta_range=(args.beta_min, args.beta_max),
        cmax=args.cmax,
        p_values=powers,
        seed=args.seed,
    )

    summary = discover(data, config, test_budget=args.budget, verbose=True)

    # full support-size Pareto front (estimates at each complexity)
    print("\nSupport-size Pareto front (held-out weak residual at each c):")
    pareto = summary.get("pareto", {})
    for c in sorted(pareto, key=lambda k: int(k)):
        m = pareto[c]
        print(f"  c={c}: E_val={m['val_rel_mse']:.4e}   {m['equation']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        sel = summary["selected"]
        with open(f"{args.out}_summary.json", "w") as f:
            json.dump(
                {
                    "dataset": data.name,
                    "grid": list(data.U.shape),
                    "config": {
                        "time_fractional": not args.integer_time,
                        "space_side": args.space_side,
                        "alpha_range": [args.alpha_min, args.alpha_max],
                        "beta_range": [args.beta_min, args.beta_max],
                        "cmax": args.cmax,
                        "powers": list(powers),
                        "budget": args.budget,
                        "seed": args.seed,
                    },
                    "selected": {
                        "equation": sel["equation"],
                        "alpha": float(sel["alpha"]),
                        "beta_tuple": [float(b) for b in sel["beta_tuple"]],
                        "p_tuple": [int(p) for p in sel["p_tuple"]],
                        "coefficients": [float(c) for c in sel["coefficients"]],
                        "val_rel_mse": float(sel["val_rel_mse"]),
                        "bic": float(sel["bic"]),
                        "aic": float(sel["aic"]),
                    },
                },
                f,
                indent=2,
            )
        import csv

        with open(f"{args.out}_pareto.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["support_size", "val_rel_mse", "alpha", "beta_tuple", "p_tuple", "equation"])
            for c in sorted(pareto, key=lambda k: int(k)):
                m = pareto[c]
                w.writerow([
                    c, m["val_rel_mse"], m["alpha"],
                    ";".join(f"{float(b):.5f}" for b in m["beta_tuple"]),
                    ";".join(str(int(p)) for p in m["p_tuple"]),
                    m["equation"],
                ])
        print(f"\nWrote {args.out}_summary.json and {args.out}_pareto.csv")


if __name__ == "__main__":
    main()
