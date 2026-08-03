"""Derivative backends for fractional PDE discovery benchmarks.

The benchmark code separates the *model-selection idea* from the derivative
estimator.  This is deliberate: in practice each dataset may deserve a
problem-matched derivative backend (regularized inverse derivatives for noisy
bounded data, spectral derivatives for periodic synthetic data, and L1 Caputo
for time-fractional simulations).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

try:
    from regfracdiff import fracdiff_reg
except Exception:  # pragma: no cover
    fracdiff_reg = None  # type: ignore

EPS = 1e-14


def integer_time_derivative(U: ArrayLike, order: int, dt: float) -> NDArray[np.float64]:
    """Finite-difference time derivative for order 0, 1, or 2."""
    U = np.asarray(U, dtype=float)
    if order == 0:
        return U.copy()
    if order == 1:
        out = np.empty_like(U, dtype=float)
        out[1:-1] = (U[2:] - U[:-2]) / (2.0 * dt)
        out[0] = (U[1] - U[0]) / dt
        out[-1] = (U[-1] - U[-2]) / dt
        return out
    if order == 2:
        out = np.empty_like(U, dtype=float)
        out[1:-1] = (U[2:] - 2.0 * U[1:-1] + U[:-2]) / (dt * dt)
        out[0] = out[1]
        out[-1] = out[-2]
        return out
    raise ValueError("only integer time derivative orders 0, 1 and 2 are supported")


def caputo_l1_time(U: ArrayLike, alpha: float, dt: float) -> NDArray[np.float64]:
    """Caputo L1 derivative for 0 <= alpha <= 2 on a uniform time grid.

    For 1 < alpha < 2, the implementation applies an L1 derivative of order
    alpha-1 to a finite-difference first derivative.  This is adequate for the
    benchmark generator/evaluator; publication experiments should use the same
    backend consistently in both data generation and discovery diagnostics.
    """
    U = np.asarray(U, dtype=float)
    alpha = float(alpha)
    if not (0.0 <= alpha <= 2.0):
        raise ValueError("alpha must lie in [0, 2]")
    if abs(alpha - round(alpha)) < 1e-11:
        return integer_time_derivative(U, int(round(alpha)), dt)
    if alpha < 1.0:
        nt = U.shape[0]
        out = np.full_like(U, np.nan, dtype=float)
        weights = np.array([(k + 1.0) ** (1.0 - alpha) - k ** (1.0 - alpha) for k in range(nt)], dtype=float)
        coef = dt ** (-alpha) / special.gamma(2.0 - alpha)
        dU = np.diff(U, axis=0)
        for n in range(1, nt):
            out[n] = coef * np.tensordot(weights[:n], dU[n - 1 :: -1], axes=(0, 0))
        return out
    return caputo_l1_time(integer_time_derivative(U, 1, dt), alpha - 1.0, dt)


def spectral_space_derivative(U: ArrayLike, beta: float, Lx: float, riesz: bool = False) -> NDArray[np.float64]:
    """Periodic spectral derivative along axis=1.

    Parameters
    ----------
    beta:
        Derivative order.  For directional derivatives the Fourier multiplier is
        (i k)^beta.  If ``riesz=True`` the multiplier is -|k|^beta, useful for
        isotropic fractional diffusion / fractional Laplacian terms.
    Lx:
        Period length.
    """
    U = np.asarray(U, dtype=float)
    beta = float(beta)
    if not (0.0 <= beta <= 4.0):
        raise ValueError("beta must lie in [0, 4]")
    if abs(beta) < 1e-14:
        return U.copy()
    nx = U.shape[1]
    dx = float(Lx) / nx
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    if riesz:
        multiplier = -(np.abs(k) ** beta)
    else:
        multiplier = (1j * k).astype(complex) ** beta
        multiplier[np.isclose(k, 0.0)] = 0.0
    return np.fft.ifft(np.fft.fft(U, axis=1) * multiplier[None, :], axis=1).real


def regularized_time_derivative(
    U: ArrayLike,
    alpha: float,
    dt: float,
    *,
    lam: float = 1e-8,
    kind: str = "caputo",
    reg_order: int = 2,
    initial: str = "first",
) -> NDArray[np.float64]:
    """Regularized inverse derivative along time; returns shape (nt-1, nx)."""
    if fracdiff_reg is None:  # pragma: no cover
        raise RuntimeError("regfracdiff.py is required for regularized derivatives")
    return fracdiff_reg(
        np.asarray(U, dtype=float),
        alpha=float(alpha),
        dt=float(dt),
        axis=0,
        lam=float(lam),
        kind=kind,  # type: ignore[arg-type]
        side="left",
        reg_order=int(reg_order),
        initial=initial,  # type: ignore[arg-type]
    )


def regularized_space_derivative(
    U: ArrayLike,
    beta: float,
    dx: float,
    *,
    lam: float = 1e-6,
    kind: str = "caputo",
    side: str = "left",
    reg_order: int = 2,
    initial: str = "first",
) -> NDArray[np.float64]:
    """Regularized inverse derivative along space; returns shape (nt, nx-1)."""
    if fracdiff_reg is None:  # pragma: no cover
        raise RuntimeError("regfracdiff.py is required for regularized derivatives")
    return fracdiff_reg(
        np.asarray(U, dtype=float),
        alpha=float(beta),
        dt=float(dx),
        axis=1,
        lam=float(lam),
        kind=kind,  # type: ignore[arg-type]
        side=side,  # type: ignore[arg-type]
        reg_order=int(reg_order),
        initial=initial,  # type: ignore[arg-type]
    )
