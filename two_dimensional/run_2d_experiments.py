"""Reproduce and summarize the two-dimensional Weak-Pareto experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from . import weak_pareto_2d as wp2
except ImportError:  # direct script execution
    import weak_pareto_2d as wp2

TRUTH = {
    "A": {"alpha": 0.85, "terms": [("x", 1.70, 0.30), ("y", 1.40, 0.20)]},
    "B": {
        "alpha": 0.85,
        "terms": [("x", 1.00, -0.60), ("x", 1.70, 0.30), ("y", 1.40, 0.20)],
    },
}

SETTINGS = {
    "A": dict(noise=(0.0, 0.01, 0.05, 0.10, 0.20), seeds=tuple(range(5)), cmax=4, width=None),
    "B": dict(noise=(0.0, 0.01, 0.05, 0.10, 0.20), seeds=tuple(range(5)), cmax=4, width=None),
    "ablation": dict(noise=(0.05,), seeds=tuple(range(5)), cmax=4, width="sweep"),
}
ABLATION_WIDTHS = (None, 0.10, 0.16, 0.24)
ORDER_TOLERANCE = 0.15


def match_and_score(model: dict, truth: dict):
    """Match selected terms to truth optimally within each direction."""
    got: dict[str, list[tuple[float, float]]] = {}
    for (direction, power, beta), xi in zip(model["terms"], model["xi"]):
        if int(power) != 0:
            return None
        got.setdefault(direction, []).append((float(beta), float(xi)))

    wanted: dict[str, list[tuple[float, float]]] = {}
    for direction, beta, xi in truth["terms"]:
        wanted.setdefault(direction, []).append((float(beta), float(xi)))

    if {key: len(value) for key, value in got.items()} != {
        key: len(value) for key, value in wanted.items()
    }:
        return None

    beta_errors: list[float] = []
    coefficient_errors: list[float] = []
    for direction, targets in wanted.items():
        estimates = got[direction]
        cost = np.array(
            [[abs(est[0] - target[0]) for est in estimates] for target in targets],
            dtype=float,
        )
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            beta_true, xi_true = targets[i]
            beta_hat, xi_hat = estimates[j]
            beta_errors.append(abs(beta_hat - beta_true))
            coefficient_errors.append(abs(xi_hat - xi_true) / (abs(xi_true) + 1e-12))

    return {
        "e_alpha": abs(float(model["alpha"]) - float(truth["alpha"])),
        "e_beta_max": float(max(beta_errors)),
        "e_xi_max": float(max(coefficient_errors)),
    }


def one_run(U, t, x, y, truth, rho, seed, cmax, width, maxiter=24, popsize=7):
    rng = np.random.default_rng(700 + int(seed))
    noisy = U if rho == 0.0 else U * (1.0 + rho * rng.uniform(-1.0, 1.0, U.shape))
    start = time.perf_counter()
    best, models, errors = wp2.discover(
        noisy,
        t,
        x,
        y,
        cmax=cmax,
        seed=seed,
        spatial_width=width,
        early_stop=False,
        maxiter=maxiter,
        popsize=popsize,
    )
    score = match_and_score(best, truth)
    selected_c = len(best["terms"])
    Kt, Kx = wp2.paper_test_counts(t.size, x.size)
    _, Ky = wp2.paper_test_counts(t.size, y.size)
    return {
        "noise": float(rho),
        "seed": int(seed),
        "width": width,
        "cmax": int(cmax),
        "selected_support": selected_c,
        "weak_test_grid": [Kt, Kx, Ky],
        "weak_rows": int(Kt * Kx * Ky),
        "alpha": float(best["alpha"]),
        "terms": [[d, int(p), float(beta)] for d, p, beta in best["terms"]],
        "xi": [float(value) for value in best["xi"]],
        "validation_curve": [float(value) for value in errors],
        "support_recovered": score is not None,
        "operator_recovered": bool(
            score is not None
            and score["e_alpha"] <= ORDER_TOLERANCE
            and score["e_beta_max"] <= ORDER_TOLERANCE
        ),
        "errors": score,
        "seconds": float(time.perf_counter() - start),
    }


def _record_key(record: dict):
    width = "paper-rule" if record["width"] is None else f"{float(record['width']):.8g}"
    return (
        record["experiment"],
        record["benchmark"],
        width,
        float(record["noise"]),
        int(record["seed"]),
    )


def load_records(out_dir: Path) -> list[dict]:
    """Load result shards and reject conflicting duplicate run keys."""
    records: OrderedDict[tuple, dict] = OrderedDict()
    for path in sorted(out_dir.glob("per_run*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = _record_key(record)
            if key in records and record != records[key]:
                raise ValueError(f"conflicting duplicate record {key} in {path}")
            records[key] = record
    return list(records.values())


def _mean_std(records: list[dict], key: str):
    values = np.array([row["errors"][key] for row in records], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def write_summary(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    rows = load_records(out_dir)
    groups: dict[tuple, list[dict]] = {}
    for record in rows:
        width = "paper-rule" if record["width"] is None else f"{float(record['width']):g}"
        key = (record["experiment"], record["benchmark"], width, float(record["noise"]))
        groups.setdefault(key, []).append(record)

    fields = (
        "experiment", "benchmark", "width", "noise", "n_seeds",
        "support_recovered", "operator_recovered", "weak_rows",
        "e_alpha_mean", "e_alpha_std", "e_beta_max_mean", "e_beta_max_std",
        "e_xi_max_mean", "e_xi_max_std", "seconds_mean",
    )
    summary_path = out_dir / "summary_2d.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (experiment, benchmark, width, noise), group in sorted(groups.items()):
            recovered = [row for row in group if row["support_recovered"]]
            ea_m, ea_s = _mean_std(recovered, "e_alpha")
            eb_m, eb_s = _mean_std(recovered, "e_beta_max")
            ex_m, ex_s = _mean_std(recovered, "e_xi_max")
            weak_rows = {row["weak_rows"] for row in group}
            if len(weak_rows) != 1:
                raise ValueError(f"inconsistent weak-row counts in group {(experiment, noise)}")
            writer.writerow({
                "experiment": experiment,
                "benchmark": benchmark,
                "width": width,
                "noise": noise,
                "n_seeds": len(group),
                "support_recovered": sum(row["support_recovered"] for row in group),
                "operator_recovered": sum(row["operator_recovered"] for row in group),
                "weak_rows": weak_rows.pop(),
                "e_alpha_mean": ea_m,
                "e_alpha_std": ea_s,
                "e_beta_max_mean": eb_m,
                "e_beta_max_std": eb_s,
                "e_xi_max_mean": ex_m,
                "e_xi_max_std": ex_s,
                "seconds_mean": float(np.mean([row["seconds"] for row in group])),
            })
    print(f"wrote {summary_path} ({len(rows)} unique runs, {len(groups)} cells)")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--experiment", default="A", choices=("A", "B", "ablation"))
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--seed-list")
    parser.add_argument("--tag", default="")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--noise", help="comma-separated fractions, e.g. 0,0.01,0.05")
    parser.add_argument("--cmax", type=int)
    parser.add_argument("--width", help="spatial width as a domain fraction, or paper")
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--popsize", type=int, default=7)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        write_summary(out_dir)
        return

    benchmark = "B" if args.experiment == "ablation" else args.experiment
    config = dict(SETTINGS[args.experiment])
    if args.seeds is not None:
        config["seeds"] = tuple(range(args.seeds))
    if args.seed_list:
        config["seeds"] = tuple(int(value) for value in args.seed_list.split(",") if value)
    if args.noise:
        config["noise"] = tuple(float(value) for value in args.noise.split(","))
    if args.cmax is not None:
        config["cmax"] = args.cmax
    if args.width is not None:
        value = str(args.width).strip().lower()
        config["width"] = None if value in {"paper", "paper-rule", "default"} else float(value)

    data_path = Path(args.data) / f"benchmark_{benchmark}.npz"
    with np.load(data_path) as archive:
        t, x, y, U = (archive[key] for key in ("t", "x", "y", "U"))

    widths = ABLATION_WIDTHS if config["width"] == "sweep" else (config["width"],)
    suffix = f"_{args.tag}" if args.tag else ""
    output_path = out_dir / f"per_run{suffix}.jsonl"
    mode = "a" if args.append else "w"
    written = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        for width in widths:
            for noise in config["noise"]:
                for seed in config["seeds"]:
                    record = one_run(
                        U, t, x, y, TRUTH[benchmark], noise, seed,
                        config["cmax"], width, args.maxiter, args.popsize,
                    )
                    record.update({
                        "experiment": args.experiment,
                        "benchmark": benchmark,
                        "maxiter": args.maxiter,
                        "popsize": args.popsize,
                    })
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    written += 1
                    score = record["errors"]
                    label = "paper-rule" if width is None else f"width={width:.2f}"
                    status = (
                        f"e_a={score['e_alpha']:.4g}, e_b={score['e_beta_max']:.4g}, "
                        f"e_xi={score['e_xi_max']:.4g}"
                        if score else "support not recovered"
                    )
                    print(
                        f"{benchmark} {label} noise={100*noise:g}% seed={seed}: "
                        f"{status} ({record['seconds']:.2f}s)", flush=True,
                    )

    import scipy

    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "experiment": args.experiment,
        "benchmark": benchmark,
        "seeds": list(config["seeds"]),
        "noise": list(config["noise"]),
        "cmax": config["cmax"],
        "widths": list(widths),
        "maxiter": args.maxiter,
        "popsize": args.popsize,
        "data_shape": list(U.shape),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    environment_path = out_dir / f"environment{suffix or '_' + args.experiment}.json"
    environment_path.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {written} runs to {output_path}")


if __name__ == "__main__":
    main()
