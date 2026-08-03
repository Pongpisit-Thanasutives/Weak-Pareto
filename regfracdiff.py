"""Regularized fractional differentiation from samples.

This module provides data-only, noise-robust fractional derivative estimators
with a NumPy-like API.  The main routine, ``fracdiff_reg``, estimates a
fractional derivative by solving a regularized fractional-integral inverse
problem rather than applying a raw high-pass fractional-difference stencil.

For a 1-D signal y sampled at uniform spacing dt, it estimates g such that

    I^alpha g ~= y - boundary_polynomial

where I^alpha is the left-sided Riemann-Liouville fractional integral.  The
least-squares problem is regularized with a finite-difference penalty.  This is
usually much more noise-stable than direct Grünwald-Letnikov differencing.

The implementation is intentionally lightweight: NumPy + SciPy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, comb
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gamma as gamma_fn
from scipy.sparse import diags, eye, csr_matrix

Kind = Literal["caputo", "rl", "rl-zero"]
Side = Literal["left", "right", "both", "centered"]
Initial = Union[Literal["first", "mean", "median", "zero"], float]


@lru_cache(maxsize=256)
def fractional_integral_kernel(m: int, alpha: float, dt: float) -> np.ndarray:
    """Piecewise-constant product-integration weights for I^alpha.

    The resulting lower-triangular convolution maps interval-centered samples
    g[0:m] to y[1:] values:

        (I^alpha g)(t_{i+1}) ≈ sum_{j=0}^i w[i-j] g[j].
    """
    if m <= 0:
        raise ValueError("m must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    k = np.arange(1, m + 1, dtype=float)
    w = (k**alpha - (k - 1.0) ** alpha) * (dt**alpha) / gamma_fn(alpha + 1.0)
    return w.astype(float, copy=False)


@lru_cache(maxsize=256)
def fractional_integral_matrix(m: int, alpha: float, dt: float) -> np.ndarray:
    """Dense lower-triangular product-integration matrix for I^alpha."""
    w = fractional_integral_kernel(m, alpha, dt)
    A = np.zeros((m, m), dtype=float)
    for i in range(m):
        A[i, : i + 1] = w[i::-1]
    return A


@lru_cache(maxsize=256)
def difference_matrix(m: int, order: int, dt: float) -> csr_matrix:
    """Forward finite-difference matrix used as a regularizer."""
    if order < 0:
        raise ValueError("reg_order must be nonnegative")
    if order == 0:
        return eye(m, format="csr")
    if order >= m:
        raise ValueError("reg_order must be smaller than the number of output samples")
    coeffs = np.array([(-1.0) ** (order - k) * comb(order, k) for k in range(order + 1)])
    coeffs = coeffs / (dt**order)
    offsets = np.arange(order + 1)
    return diags(coeffs, offsets, shape=(m - order, m), format="csr")


@lru_cache(maxsize=256)
def _normal_solver_factor(m: int, alpha: float, dt: float, lam: float, reg_order: int):
    """Cholesky factorization of A.T A + lam L.T L for repeated RHS solves."""
    A = fractional_integral_matrix(m, alpha, dt)
    normal = A.T @ A
    if lam > 0:
        L = difference_matrix(m, reg_order, dt)
        normal = normal + lam * (L.T @ L).toarray()
    # A tiny nugget handles exact/near-singular boundary cases without changing
    # the requested regularization in any meaningful way.
    normal = normal + 1e-14 * np.eye(m)
    return cho_factor(normal, lower=True, check_finite=False)


def _validate_signal_axis(y: ArrayLike, axis: int) -> tuple[np.ndarray, int]:
    """Validate an input array and normalize the differentiated axis index."""
    arr = np.asarray(y, dtype=float)
    if arr.ndim < 1:
        raise ValueError("input must have at least one dimension")
    if not np.all(np.isfinite(arr)):
        raise ValueError("input contains NaN or inf")
    axis = int(axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("axis out of bounds")
    if arr.shape[axis] < 3:
        raise ValueError("need at least 3 samples along the differentiated axis")
    return arr, axis


def _initial_values(Y: np.ndarray, initial: Initial, k: int) -> np.ndarray:
    """Return one boundary value per column for Y with shape (n, batch)."""
    if isinstance(initial, (int, float, np.floating)):
        return np.full(Y.shape[1], float(initial), dtype=float)
    if initial == "zero":
        return np.zeros(Y.shape[1], dtype=float)
    if initial == "first":
        return Y[0].copy()
    k = max(1, min(int(k), Y.shape[0]))
    if initial == "mean":
        return np.mean(Y[:k], axis=0)
    if initial == "median":
        return np.median(Y[:k], axis=0)
    raise ValueError("initial must be 'first', 'mean', 'median', 'zero', or a scalar")


def _boundary_slopes(Y: np.ndarray, dt: float, k: int) -> np.ndarray:
    """Vectorized least-squares slope estimate from the first k samples."""
    k = max(2, min(int(k), Y.shape[0]))
    t = np.arange(k, dtype=float) * dt
    tc = t - t.mean()
    denom = float(np.dot(tc, tc))
    if denom == 0:
        return np.zeros(Y.shape[1], dtype=float)
    Yk = Y[:k]
    return (tc[:, None] * (Yk - Yk.mean(axis=0, keepdims=True))).sum(axis=0) / denom


def _rhs_for_kind(
    Y: np.ndarray,
    *,
    alpha: float,
    dt: float,
    kind: Kind,
    initial: Initial,
    initial_window: int,
) -> np.ndarray:
    """Build RHS B with shape (n-1, batch) for I^alpha g ≈ B."""
    n = Y.shape[0]
    t_tail = np.arange(1, n, dtype=float)[:, None] * dt

    kind_norm = "rl" if kind == "rl-zero" else kind
    if kind_norm == "rl":
        # Left-sided R-L inverse relation under zero fractional-boundary terms.
        # This is often suitable for spatial transport data whose lower-boundary
        # contribution is negligible after trimming.  Nonzero boundary terms can
        # create a singular component near the boundary, so trim boundaries in
        # PDE discovery.
        return Y[1:].copy()

    if kind_norm != "caputo":
        raise ValueError("kind must be 'caputo', 'rl', or 'rl-zero'")

    order = int(ceil(alpha - 1e-12))
    y0 = _initial_values(Y, initial, initial_window)
    poly = y0[None, :]
    if order >= 2:
        slope0 = _boundary_slopes(Y, dt, initial_window)
        poly = poly + t_tail * slope0[None, :]
    return Y[1:] - poly


def _solve_fractional_inverse_batch(
    Y: np.ndarray,
    *,
    alpha: float,
    dt: float,
    lam: float,
    reg_order: int,
    kind: Kind,
    initial: Initial,
    initial_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve all columns of Y simultaneously.

    Parameters
    ----------
    Y:
        Array with shape (n, batch).

    Returns
    -------
    G:
        Fractional derivative estimates, shape (n-1, batch).
    Yhat:
        Reconstructed signal implied by G and the selected boundary convention,
        shape (n, batch).  Mainly useful for diagnostics.
    """
    n, batch = Y.shape
    m = n - 1
    if reg_order >= m:
        raise ValueError("reg_order must be smaller than len(y)-1")

    B = _rhs_for_kind(Y, alpha=alpha, dt=dt, kind=kind, initial=initial, initial_window=initial_window)
    A = fractional_integral_matrix(m, alpha, dt)
    rhs = A.T @ B
    cfac = _normal_solver_factor(m, float(alpha), float(dt), float(lam), int(reg_order))
    G = cho_solve(cfac, rhs, check_finite=False)

    # Reconstruction for diagnostics.
    tail = A @ G
    kind_norm = "rl" if kind == "rl-zero" else kind
    if kind_norm == "caputo":
        order = int(ceil(alpha - 1e-12))
        y0 = _initial_values(Y, initial, initial_window)
        t_tail = np.arange(1, n, dtype=float)[:, None] * dt
        poly = y0[None, :]
        if order >= 2:
            slope0 = _boundary_slopes(Y, dt, initial_window)
            poly = poly + t_tail * slope0[None, :]
        yhat = np.vstack([Y[:1], tail + poly])
    else:
        yhat = np.vstack([np.zeros((1, batch)), tail])
    return G, yhat


@dataclass
class RegularizedFractionalDifferentiator:
    """Noise-robust data-only fractional derivative estimator.

    Parameters
    ----------
    alpha:
        Derivative order. Supports 0 < alpha <= 2.  ``alpha=1`` gives a robust
        first derivative; ``alpha=2`` gives a robust second derivative.
    dt:
        Uniform sample spacing.
    lam:
        Tikhonov regularization strength. Larger values give smoother estimates.
    reg_order:
        Finite-difference order in the penalty.  Typical choices: 1 or 2.
    kind:
        ``"caputo"`` subtracts the boundary polynomial before inversion.
        ``"rl"`` assumes zero Riemann-Liouville fractional boundary terms and
        solves I^alpha g ≈ y directly.
    side:
        ``"left"`` is causal/lower-limit derivative. ``"right"`` flips the
        axis. ``"centered"`` averages left and right. ``"both"`` returns a pair.
    initial:
        Boundary value convention for Caputo mode.
    """

    alpha: float = 0.5
    dt: float = 1.0
    lam: float = 1e-3
    reg_order: int = 2
    kind: Kind = "caputo"
    side: Side = "left"
    initial: Initial = "median"
    initial_window: int = 5

    def __post_init__(self) -> None:
        """Validate derivative order, spacing, regularization, derivative kind, and side."""
        if not (0.0 < float(self.alpha) <= 2.0):
            raise ValueError("this implementation supports 0 < alpha <= 2")
        if float(self.dt) <= 0.0:
            raise ValueError("dt must be positive")
        if float(self.lam) < 0.0:
            raise ValueError("lam must be nonnegative")
        if int(self.reg_order) < 0:
            raise ValueError("reg_order must be nonnegative")
        if self.kind not in ("caputo", "rl", "rl-zero"):
            raise ValueError("kind must be 'caputo', 'rl', or 'rl-zero'")
        if self.side not in ("left", "right", "both", "centered"):
            raise ValueError("side must be 'left', 'right', 'both', or 'centered'")

    def _left_diff(self, y: ArrayLike, *, axis: int, return_smoothed: bool = False):
        """Compute the left-sided regularized fractional derivative along one axis.
        
        The input is moved to the first axis, reshaped into a batch of 1-D signals,
        solved in one vectorized linear algebra call, then reshaped back to match the
        original array with one fewer sample along the differentiated axis.
        """
        arr, axis = _validate_signal_axis(y, axis)
        moved = np.moveaxis(arr, axis, 0)
        original_shape = moved.shape
        Y = moved.reshape(original_shape[0], -1)
        G, Yhat = _solve_fractional_inverse_batch(
            Y,
            alpha=float(self.alpha),
            dt=float(self.dt),
            lam=float(self.lam),
            reg_order=int(self.reg_order),
            kind=self.kind,
            initial=self.initial,
            initial_window=int(self.initial_window),
        )
        out_shape = (original_shape[0] - 1,) + original_shape[1:]
        G_arr = G.reshape(out_shape)
        G_arr = np.moveaxis(G_arr, 0, axis)
        if not return_smoothed:
            return G_arr
        Yhat_arr = Yhat.reshape(original_shape)
        Yhat_arr = np.moveaxis(Yhat_arr, 0, axis)
        return G_arr, Yhat_arr

    def __call__(self, y: ArrayLike, *, axis: int = -1, return_smoothed: bool = False):
        """Apply the configured derivative estimator to an array.
        
        Use ``axis`` to choose the differentiated dimension. ``side='left'`` computes a
        left-sided derivative, ``'right'`` flips the axis, ``'both'`` returns both, and
        ``'centered'`` averages the two.
        """
        if self.side == "left":
            return self._left_diff(y, axis=axis, return_smoothed=return_smoothed)
        if self.side == "right":
            arr, axis = _validate_signal_axis(y, axis)
            flipped = np.flip(arr, axis=axis)
            res = self._left_diff(flipped, axis=axis, return_smoothed=return_smoothed)
            if return_smoothed:
                g, yhat = res
                return np.flip(g, axis=axis), np.flip(yhat, axis=axis)
            return np.flip(res, axis=axis)
        if self.side == "both":
            left = RegularizedFractionalDifferentiator(
                self.alpha, self.dt, self.lam, self.reg_order, self.kind, "left", self.initial, self.initial_window
            )(y, axis=axis, return_smoothed=False)
            right = RegularizedFractionalDifferentiator(
                self.alpha, self.dt, self.lam, self.reg_order, self.kind, "right", self.initial, self.initial_window
            )(y, axis=axis, return_smoothed=False)
            return left, right
        # centered
        left, right = RegularizedFractionalDifferentiator(
            self.alpha, self.dt, self.lam, self.reg_order, self.kind, "both", self.initial, self.initial_window
        )(y, axis=axis, return_smoothed=False)
        return 0.5 * (left + right)


def fracdiff_reg(
    y: ArrayLike,
    alpha: float = 0.5,
    *,
    dt: float = 1.0,
    lam: float = 1e-3,
    reg_order: int = 2,
    axis: int = -1,
    kind: Kind = "caputo",
    side: Side = "left",
    initial: Initial = "median",
    initial_window: int = 5,
    return_smoothed: bool = False,
    maxiter: Optional[int] = None,
    tol: float = 1e-6,
):
    """Fractional derivative from samples, analogous to ``np.diff``.

    Parameters ``maxiter`` and ``tol`` are accepted for backward compatibility
    with earlier LSQR-based versions; the current implementation uses a cached
    dense Cholesky solve and ignores them.

    Examples
    --------
    >>> d05 = fracdiff_reg(x, alpha=0.5, dt=0.01, lam=1e-3)
    >>> dxb = fracdiff_reg(U, alpha=1.7, dt=dx, axis=1, kind="caputo")
    >>> dleft, dright = fracdiff_reg(U, alpha=1.4, dt=dx, axis=1, kind="rl", side="both")
    """
    return RegularizedFractionalDifferentiator(
        alpha=float(alpha),
        dt=float(dt),
        lam=float(lam),
        reg_order=int(reg_order),
        kind=kind,
        side=side,
        initial=initial,
        initial_window=int(initial_window),
    )(y, axis=axis, return_smoothed=return_smoothed)


def fracdiff_caputo(y: ArrayLike, alpha: float = 0.5, **kwargs):
    """Convenience wrapper for Caputo-style regularized fractional derivative."""
    kwargs.setdefault("kind", "caputo")
    return fracdiff_reg(y, alpha=alpha, **kwargs)


def fracdiff_rl(y: ArrayLike, alpha: float = 0.5, **kwargs):
    """Convenience wrapper for left/right R-L-zero-boundary derivative.

    This solves I^alpha g ≈ y without subtracting a boundary polynomial.  For
    bounded spatial data, trim boundaries when using the resulting derivative in
    regression because nonzero R-L boundary terms can create singular behavior.
    """
    kwargs.setdefault("kind", "rl")
    return fracdiff_reg(y, alpha=alpha, **kwargs)
