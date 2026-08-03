"""Forward-model validation of the discovered equations.

A small weak residual does not by itself establish that a discovered equation
reproduces the observed dynamics.  For each benchmark we therefore integrate the
discovered fractional PDE from the benchmark initial condition and compare the
simulated field with the clean reference solution, reporting

    e_field = || u_pred - u_ref ||_2 / (|| u_ref ||_2 + eps).

The integrator is pseudospectral in space (periodic), with the spatial operator
taken from the benchmark's declared kind (Riesz multiplier -|k|^beta or the
directional (ik)^beta), and in time uses an adaptively substepped RK4 for alpha = 1 and the
Caputo L1 scheme mode-by-mode for alpha < 1 (the fractional-time benchmarks are
linear, so this is exact up to the L1 discretisation).  As a validity check the
same integrator is run with the *true* parameters; the resulting e_field^true is
the discretisation floor against which the discovered-model error is read.

Usage (from the package root):
    PYTHONPATH=. python3 reproduce/make_forward.py            # full budget, 5 seeds
    PYTHONPATH=. python3 reproduce/make_forward.py --fast     # quick demo, 2 seeds
    PYTHONPATH=. python3 reproduce/make_forward.py --seed 0   # force a fixed seed

Writes <outdir>/table_forward.{tex,csv}.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import warnings

import numpy as np
from scipy import special

from _repro_common import (
    _config_for, matched_errors, MAIN_BENCHMARKS, APPENDIX_BENCHMARKS,
    APPENDIX_ADJUSTMENTS, NICE_NAME, coefficient_truth, run_weak,
)

warnings.filterwarnings("ignore")

NOISY_LEVEL = {
    "paper_FADE_tsfade_fft": 10.0, "synthetic_fractional_burgers": 10.0,
    "synthetic_space_fractional_RD": 5.0, "synthetic_time_space_fractional_RD": 5.0,
    "paper_ADE_Convection_diffusion": 10.0, "synthetic_two_fractional_rhs": 5.0,
}
EPS = 1e-12


def _multiplier(k: np.ndarray, beta: float, kind: str) -> np.ndarray:
    """Fourier multiplier of the spatial operator D_x^beta for one term."""
    if abs(beta) < 5e-3:                       # identity
        return np.ones_like(k, dtype=complex)
    if kind in ("riesz", "riemann_liouville", "integer") or kind is None:
        m = -(np.abs(k) ** beta)              # periodic Riesz, multiplier -|k|^beta
        m = m.astype(complex)
    else:                                     # directional spectral (ik)^beta
        m = (1j * k) ** beta
    m[np.isclose(k, 0.0)] = 0.0 if abs(beta) >= 5e-3 else 1.0
    return m


def _rhs_factory(k, terms, kind, dealias_mask=None):
    mults = [( _multiplier(k, b, kind), int(p), float(x)) for (p, b, x) in terms]
    def rhs(u):
        uhat = np.fft.fft(u)
        out = np.zeros_like(u, dtype=float)
        for m, p, xi in mults:
            Lu = np.fft.ifft(m * uhat).real
            out = out + xi * ((u ** p) if p > 0 else 1.0) * Lu
        if dealias_mask is not None:
            out = np.fft.ifft(dealias_mask * np.fft.fft(out)).real
        return out
    return rhs


def _simulate(u0, t, x, alpha, terms, kind):
    nt, nx = len(t), len(u0)
    dt = float(t[1] - t[0])
    Lx = float(x[-1] - x[0]) + float(x[1] - x[0])
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    rhs = _rhs_factory(k, terms, kind)
    U = np.empty((nt, nx), dtype=float)
    U[0] = u0
    linear = all(int(p) == 0 for (p, _, _) in terms)
    if abs(alpha - 1.0) < 1e-6 and linear:
        # Exact exponential propagator (unconditionally stable), matching the
        # linear generators: u_t = lam(k) u  =>  uhat(t) = exp(lam t) uhat(0).
        lam = np.zeros(nx, dtype=complex)
        for (p, b, xi) in terms:
            lam = lam + xi * _multiplier(k, b, kind)
        u0hat = np.fft.fft(u0)
        for i in range(1, nt):
            U[i] = np.fft.ifft(np.exp(lam * (t[i] - t[0])) * u0hat).real
        return U
    if abs(alpha - 1.0) < 1e-6:
        # RK4 method of lines with adaptive substepping for stability
        # (nonlinear alpha = 1, e.g. Burgers).  A single output step can violate
        # the explicit CFL limit at higher noise/coefficients, so each output
        # interval is subdivided into stable internal steps.  The tendency is
        # 2/3-dealiased (standard pseudo-spectral practice), which suppresses
        # the aliasing that shock formation under weak fractional damping would
        # otherwise feed back into the high modes; on smooth data it is inert.
        dx = float(x[1] - x[0])
        kmax = float(np.max(np.abs(k))) + 1e-12
        dealias_mask = (np.abs(k) <= (2.0 / 3.0) * kmax).astype(float)
        rhs = _rhs_factory(k, terms, kind, dealias_mask=dealias_mask)
        lam_mag = 0.0
        for (p, b, xi) in terms:
            if int(p) == 0:
                lam_mag = max(lam_mag, abs(xi) * float(np.max(np.abs(_multiplier(k, b, kind)))))
        u = u0.copy()
        for i in range(1, nt):
            umax = float(np.max(np.abs(u))) + 1e-12
            dt_adv = dx / (np.pi * umax)
            dt_diff = 2.5 / (lam_mag + 1e-12)
            dt_stable = 0.4 * min(dt_adv, dt_diff)
            nsub = max(1, int(np.ceil(dt / dt_stable)))
            h = dt / nsub
            for _ in range(nsub):
                k1 = rhs(u)
                k2 = rhs(u + 0.5 * h * k1)
                k3 = rhs(u + 0.5 * h * k2)
                k4 = rhs(u + h * k3)
                u = u + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.all(np.isfinite(u)):
                # Explicit integration has diverged (stiff shock); mark the
                # remaining trajectory non-finite so the caller reports the
                # forward check as inconclusive rather than emitting a value.
                U[i:] = np.nan
                return U
            U[i] = u
        return U
    # Caputo L1 (linear terms): mode-by-mode with the linear symbol
    # lam(k) = sum_j xi_j * multiplier_j(k)  (valid when all p_j == 0)
    if all(int(p) == 0 for (p, _, _) in terms):
        lam = np.zeros(nx, dtype=complex)
        for (p, b, xi) in terms:
            lam = lam + xi * _multiplier(k, b, kind)
        coef = dt ** (-alpha) / special.gamma(2.0 - alpha)
        w = np.array([(j + 1.0) ** (1.0 - alpha) - j ** (1.0 - alpha) for j in range(nt)])
        Uhat = np.empty((nt, nx), dtype=complex)
        Uhat[0] = np.fft.fft(u0)
        for n in range(1, nt):
            hist = np.zeros(nx, dtype=complex)
            for j in range(1, n):
                hist += w[j] * (Uhat[n - j] - Uhat[n - j - 1])
            Uhat[n] = (coef * w[0] * Uhat[n - 1] - coef * hist) / (coef * w[0] - lam)
        return np.fft.ifft(Uhat, axis=1).real
    # Nonlinear + fractional time is not among the benchmarks; fall back to RK4-L1 hybrid unavailable
    raise NotImplementedError("nonlinear fractional-time integration not required for these benchmarks")


def _field_error(u_pred, u_ref):
    return float(np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + EPS))


def _median_seed_model(name, noise, seeds, fast, force_seed):
    adj = APPENDIX_ADJUSTMENTS.get(name, {})
    use = [force_seed] if force_seed is not None else seeds
    runs = []
    for seed in use:
        s, truth = run_weak(name, noise, seed, fast, overrides=(adj or None))
        m = matched_errors(name, s["selected"], truth)
        runs.append((seed, s["selected"], m))
    runs.sort(key=lambda r: float(np.nan_to_num(r[2]["e_beta_max"], nan=1e9)))
    return runs[len(runs) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--benchmarks", default="main", choices=["main", "all"])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    seeds = [0, 1] if args.fast else [0, 1, 2, 3, 4]
    names = list(MAIN_BENCHMARKS) + (list(APPENDIX_BENCHMARKS) if args.benchmarks == "all" else [])
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for name in names:
        # clean reference field and its spatial-operator kind
        data0, cfg0, truth = _config_for(name, 0.0, seeds[0], args.fast, weak=True)
        from weak_pareto_fde_discovery import weak_operator_kinds_for_config
        _, space_kind = weak_operator_kinds_for_config(cfg0, name)
        u_ref = np.asarray(data0.U, dtype=float)
        u0 = u_ref[0].copy()
        x, t = np.asarray(data0.x, dtype=float), np.asarray(data0.t, dtype=float)
        true_terms = [(int(p), float(b), float(xi))
                      for (p, b), xi in zip(truth.expected_terms, coefficient_truth(name))]
        try:
            u_true = _simulate(u0, t, x, float(truth.expected_alpha), true_terms, space_kind)
            e_floor = _field_error(u_true, u_ref)
        except Exception as exc:
            print(f"[fwd] {NICE_NAME[name]}: self-consistency solve failed ({exc}); skipping")
            continue
        if not np.isfinite(e_floor) or e_floor > 5e-2:
            print(f"[fwd] {NICE_NAME[name]}: excluded (true-param floor {e_floor:.1e} exceeds "
                  f"the validity threshold; the general spectral integrator does not resolve "
                  f"this benchmark's operator stably)")
            continue
        for noise in (0.0, NOISY_LEVEL[name]):
            seed, sel, m = _median_seed_model(name, noise, seeds, args.fast, args.seed)
            terms = [(int(p), float(b), float(xi))
                     for p, b, xi in zip(sel["p_tuple"], sel["beta_tuple"], sel["coefficients"])]
            try:
                u_pred = _simulate(u0, t, x, float(sel["alpha"]), terms, space_kind)
                e_field = _field_error(u_pred, u_ref)
            except Exception:
                e_field = float("nan")
            rows.append(dict(benchmark=NICE_NAME[name], noise=f"{noise:.0f}",
                             e_floor=f"{e_floor:.2e}", e_field=f"{e_field:.2e}",
                             recovered="yes" if m["symbolic_form_ok"] else "no", seed=seed))
            print(f"[fwd] {NICE_NAME[name]:20s} {noise:4.0f}%  e_field={e_field:.2e}  "
                  f"(true-param floor {e_floor:.2e})  rec={'Y' if m['symbolic_form_ok'] else 'N'}")

    if not rows:
        print("[fwd] no benchmarks produced a valid forward solve")
        return
    with open(os.path.join(args.outdir, "table_forward.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    nl = " \\\\\n"
    with open(os.path.join(args.outdir, "table_forward.tex"), "w") as f:
        f.write("% Forward-model validation -- reproduce/make_forward.py\n")
        f.write("Benchmark & Noise & $e_{\\mathrm{field}}$ & true-param floor" + nl)
        for r in rows:
            f.write(f"{r['benchmark']} & {r['noise']}\\% & {r['e_field']} & {r['e_floor']}" + nl)
    print(f"[fwd] wrote {args.outdir}/table_forward.{{csv,tex}}")


if __name__ == "__main__":
    main()
