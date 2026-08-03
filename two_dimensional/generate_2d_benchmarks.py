"""Generate the two-dimensional anisotropic Weak-Pareto benchmarks.

The data satisfy

    C_0 D_t^alpha u = -c_x d_x u
                      + nu_x D_x^beta_x u + nu_y D_y^beta_y u

on a doubly periodic square. The directional spectral operator uses the
principal-branch multiplier (i k)^beta and annihilates the zero mode. Each
Fourier mode evolves through E_{alpha,1}(lambda t^alpha), so no temporal
forward discretisation is used to generate the reported fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mpmath as mp
import numpy as np

mp.mp.dps = 40

BENCHMARKS = {
    "A": dict(alpha=0.85, bx=1.70, by=1.40, cx=0.0, nux=0.30, nuy=0.20),
    "B": dict(alpha=0.85, bx=1.70, by=1.40, cx=0.6, nux=0.30, nuy=0.20),
}

# Five conjugate mode pairs (ten active Fourier coefficients).
DEFAULT_MODES = (
    (1, 1, 1.00, 0.0),
    (2, 1, 0.60, 0.7),
    (1, 3, 0.45, 1.2),
    (3, 2, 0.30, 2.0),
    (2, 4, 0.20, 0.5),
)


def mittag_leffler_e_alpha_1(alpha: float, z: np.ndarray) -> np.ndarray:
    """Evaluate E_{alpha,1}(z) for a complex array in extended precision."""
    z = np.asarray(z, dtype=complex)
    out = np.empty_like(z)
    for i, value in enumerate(z.ravel()):
        zz = mp.mpc(value)
        out.ravel()[i] = complex(
            mp.nsum(lambda n: zz**n / mp.gamma(alpha * n + 1), [0, mp.inf])
        )
    return out


def directional_multiplier(k: np.ndarray, beta: float) -> np.ndarray:
    """Return (i k)^beta on the principal branch, with zero mode set to zero."""
    k = np.asarray(k, dtype=float)
    out = np.zeros(k.shape, dtype=complex)
    nz = k != 0
    out[nz] = np.abs(k[nz]) ** beta * np.exp(
        0.5j * np.pi * beta * np.sign(k[nz])
    )
    return out


def make_benchmark(
    *,
    alpha: float,
    bx: float,
    by: float,
    cx: float,
    nux: float,
    nuy: float,
    nt: int = 90,
    ng: int = 80,
    T: float = 3.0,
    L: float = 2 * np.pi,
    modes=DEFAULT_MODES,
):
    """Return ``(t, x, y, U, zmax, n_active)`` for one benchmark."""
    if nt < 4 or ng < 8:
        raise ValueError("nt must be at least 4 and ng at least 8")
    t = np.linspace(0.0, T, nt)
    x = np.linspace(0.0, L, ng, endpoint=False)
    y = x.copy()
    k = 2 * np.pi * np.fft.fftfreq(ng, d=x[1] - x[0])
    KX, KY = np.meshgrid(k, k, indexing="ij")

    lam = (
        -cx * directional_multiplier(KX, 1.0)
        + nux * directional_multiplier(KX, bx)
        + nuy * directional_multiplier(KY, by)
    )

    U0 = np.zeros((ng, ng), dtype=float)
    for mx, my, amplitude, phase in modes:
        U0 += amplitude * np.sin(mx * x[:, None] + my * y[None, :] + phase)
    U0_hat = np.fft.fft2(U0)
    active = np.abs(U0_hat) > 1e-9 * np.abs(U0_hat).max()

    zmax = float(np.max(np.abs(lam[active])) * T**alpha)
    if zmax > 12.0:
        raise ValueError(
            f"max |lambda| T^alpha={zmax:.3f} exceeds the validated evaluation range"
        )

    U_hat = np.zeros((nt, ng, ng), dtype=complex)
    lam_active = lam[active]
    for i, ti in enumerate(t):
        U_hat[i][active] = U0_hat[active] * mittag_leffler_e_alpha_1(
            alpha, lam_active * ti**alpha
        )
    U_complex = np.fft.ifft2(U_hat, axes=(1, 2))
    imag_max = float(np.max(np.abs(U_complex.imag)))
    if imag_max >= 1e-10:
        raise RuntimeError(f"generated field is not real to tolerance: {imag_max:.3e}")
    return t, x, y, U_complex.real, zmax, int(np.count_nonzero(active))


def l1_caputo(f: np.ndarray, alpha: float, h: float) -> np.ndarray:
    """Independent L1 Caputo derivative along time, used only for verification."""
    from scipy import special

    f = np.asarray(f, dtype=float)
    n = f.shape[0]
    out = np.zeros_like(f)
    j = np.arange(n, dtype=float)
    weights = (j + 1) ** (1 - alpha) - j ** (1 - alpha)
    factor = h ** (-alpha) / special.gamma(2 - alpha)
    for i in range(1, n):
        out[i] = factor * np.tensordot(
            weights[:i][::-1], f[1 : i + 1] - f[:i], axes=(0, 0)
        )
    return out


def verification_residual(spec: dict, t, x, y, U) -> float:
    """Relative L1 residual on the final three quarters of the time interval."""
    del y  # identical periodic grid; retained in the signature for clarity
    k = 2 * np.pi * np.fft.fftfreq(x.size, d=x[1] - x[0])
    KX, KY = np.meshgrid(k, k, indexing="ij")
    lam = (
        -spec["cx"] * directional_multiplier(KX, 1.0)
        + spec["nux"] * directional_multiplier(KX, spec["bx"])
        + spec["nuy"] * directional_multiplier(KY, spec["by"])
    )
    lhs = l1_caputo(U, spec["alpha"], t[1] - t[0])
    rhs = np.fft.ifft2(
        lam[None] * np.fft.fft2(U, axes=(1, 2)), axes=(1, 2)
    ).real
    sl = slice(U.shape[0] // 4, None)
    return float(np.linalg.norm(lhs[sl] - rhs[sl]) / np.linalg.norm(rhs[sl]))


def generate_one(name: str, spec: dict, *, nt: int, ng: int, T: float):
    t, x, y, U, zmax, n_active = make_benchmark(nt=nt, ng=ng, T=T, **spec)
    residual = verification_residual(spec, t, x, y, U)
    metadata = {
        "spec": spec,
        "nt": int(nt),
        "ng": int(ng),
        "T": float(T),
        "active_fourier_coefficients": n_active,
        "max_abs_lambda_T_alpha": zmax,
        "l1_relative_residual": residual,
        "field_range": [float(U.min()), float(U.max())],
    }
    print(
        f"[{name}] {U.shape}; active={n_active}; "
        f"max|lambda|T^alpha={zmax:.3f}; L1 residual={residual:.6g}",
        flush=True,
    )
    return t, x, y, U, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="data")
    parser.add_argument("--nt", type=int, default=90)
    parser.add_argument("--ng", type=int, default=80)
    parser.add_argument("--T", type=float, default=3.0)
    parser.add_argument("--which", default="A,B")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = [name.strip() for name in args.which.split(",") if name.strip()]
    invalid = sorted(set(names) - set(BENCHMARKS))
    if invalid:
        raise SystemExit(f"unknown benchmark(s): {', '.join(invalid)}")

    manifest = {}
    for name in names:
        t, x, y, U, metadata = generate_one(
            name, BENCHMARKS[name], nt=args.nt, ng=args.ng, T=args.T
        )
        np.savez_compressed(outdir / f"benchmark_{name}.npz", t=t, x=x, y=y, U=U)
        manifest[name] = metadata

    (outdir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
