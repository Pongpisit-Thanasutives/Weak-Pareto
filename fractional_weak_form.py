"""Weak-form candidate libraries for noise-robust fractional PDE/FDE discovery.

The routines in this module implement the central weak-form identity

    < L u, phi > = < u, L^* phi > + boundary/initial corrections,

for the fractional definitions used elsewhere in this repository:

* Caputo: Riemann--Liouville adjoint on the test function, with the initial
  polynomial subtracted from the data-side field.
* Riemann--Liouville: left/right Grünwald-Letnikov product-integration matrix as
  a consistent discrete realization; the test side uses the matrix transpose.
* Grünwald--Letnikov: exact discrete adjoint of the selected GL stencil.
* Riesz/fractional Laplacian on periodic domains: self-adjoint Fourier
  multiplier -|k|^alpha.

The implementation is intentionally NumPy/SciPy-only and is designed to be used
as a drop-in, integrated-feature alternative to strong-form candidate libraries.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse, special

FractionalKind = Literal["caputo", "riemann_liouville", "grunwald_letnikov", "riesz", "identity", "integer"]
Side = Literal["left", "right", "symmetric"]
AxisName = Literal["t", "x"]

EPS = 1e-14


@dataclass(frozen=True)
class FractionalOperatorSpec:
    """Describe one weak fractional operator.

    Parameters
    ----------
    kind:
        Fractional derivative definition. ``"caputo"`` and
        ``"riemann_liouville"`` are represented through the corresponding
        left/right GL product-integration discretization and differ by the
        Caputo initial-polynomial correction. ``"grunwald_letnikov"`` means the
        discrete GL stencil itself is the model operator. ``"riesz"`` is the
        periodic self-adjoint Riesz/fractional-Laplacian multiplier.
    order:
        Derivative order. This module supports 0 <= order <= 4 for Riesz and
        0 <= order <= 2 for one-sided Caputo/RL/GL operators.
    axis:
        Axis label used by :class:`SeparableWeakLibrary2D`.
    side:
        Side of a one-sided operator. ``"symmetric"`` averages left and right
        GL stencils. Riesz ignores this parameter because it is self-adjoint.
    caputo_initial:
        Boundary convention for Caputo mode. ``"first"`` subtracts the first
        grid value along the operator axis. ``"zero"`` gives a zero initial
        polynomial. For 1 < order <= 2, the first derivative in the initial
        polynomial is estimated by a first-order boundary difference.
    """

    kind: FractionalKind
    order: float = 0.0
    axis: AxisName = "x"
    side: Side = "left"
    caputo_initial: Literal["first", "zero"] = "first"

    def __post_init__(self) -> None:
        if self.kind not in {"caputo", "riemann_liouville", "grunwald_letnikov", "riesz", "identity", "integer"}:
            raise ValueError(f"unknown operator kind {self.kind!r}")
        if self.side not in {"left", "right", "symmetric"}:
            raise ValueError("side must be 'left', 'right', or 'symmetric'")
        if self.axis not in {"t", "x"}:
            raise ValueError("axis must be 't' or 'x'")
        if float(self.order) < 0.0:
            raise ValueError("order must be nonnegative")
        if self.kind in {"caputo", "riemann_liouville", "grunwald_letnikov"} and float(self.order) > 2.0 + 1e-12:
            raise ValueError("one-sided Caputo/RL/GL operators currently support order <= 2")
        if self.kind == "riesz" and float(self.order) > 4.0 + 1e-12:
            raise ValueError("Riesz operators currently support order <= 4")


@dataclass(frozen=True)
class WeakTerm:
    """One candidate weak-library term.

    The term represents ``operator(transform(U))`` in strong form and is
    assembled weakly as ``<transform(U), operator^* phi>``.
    """

    name: str
    operator: FractionalOperatorSpec
    transform: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None

    def apply_transform(self, U: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the optional nonlinear transform on a data field.

        If ``transform`` is omitted, the term uses ``U`` directly.  Otherwise
        the callable must return an array with the same shape as ``U``; this
        guards against accidentally constructing a malformed library column.
        """
        if self.transform is None:
            return U
        out = np.asarray(self.transform(U), dtype=float)
        if out.shape != U.shape:
            raise ValueError(f"term {self.name!r} transform returned {out.shape}, expected {U.shape}")
        return out


def _as_float_array(a: ArrayLike, *, name: str) -> NDArray[np.float64]:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size < 3:
        raise ValueError(f"{name} must have at least three points")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _uniform_spacing(grid: NDArray[np.float64], *, name: str) -> float:
    d = np.diff(grid)
    if not np.all(d > 0):
        raise ValueError(f"{name} must be strictly increasing")
    h = float(np.mean(d))
    if not np.allclose(d, h, rtol=1e-5, atol=1e-12):
        raise ValueError(f"{name} must be uniformly spaced")
    return h


@lru_cache(maxsize=256)
def gl_weights(n: int, alpha: float, h: float) -> NDArray[np.float64]:
    """Return signed Grünwald weights ``(-1)^m binom(alpha,m) h^-alpha``."""
    if n <= 0:
        raise ValueError("n must be positive")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if h <= 0:
        raise ValueError("h must be positive")
    w = np.empty(int(n), dtype=float)
    w[0] = 1.0
    for m in range(1, int(n)):
        w[m] = -w[m - 1] * (float(alpha) - m + 1.0) / float(m)
    if alpha == 0:
        return w
    return w / (float(h) ** float(alpha))


@lru_cache(maxsize=128)
def gl_matrix(n: int, alpha: float, h: float, side: Side = "left") -> sparse.csr_matrix:
    """Sparse GL derivative matrix on a uniform grid.

    ``side='left'`` gives the causal lower-triangular stencil; ``'right'`` gives
    the anti-causal upper-triangular stencil; ``'symmetric'`` averages both.
    """
    n = int(n)
    alpha = float(alpha)
    h = float(h)
    if side not in {"left", "right", "symmetric"}:
        raise ValueError("side must be 'left', 'right', or 'symmetric'")
    if alpha == 0.0:
        return sparse.eye(n, format="csr")
    if side == "symmetric":
        return 0.5 * (gl_matrix(n, alpha, h, "left") + gl_matrix(n, alpha, h, "right"))
    w = gl_weights(n, alpha, h)
    offsets = np.arange(n) if side == "left" else -np.arange(n)
    # scipy.diags uses offset k>0 for upper diagonals.  For the left stencil,
    # D[i, j] = w[i-j] with j <= i, hence negative offsets.
    if side == "left":
        offsets = -np.arange(n)
    else:
        offsets = np.arange(n)
    diagonals = [np.full(n - abs(k), w[abs(k)], dtype=float) for k in offsets]
    return sparse.diags(diagonals, offsets, shape=(n, n), format="csr")




@lru_cache(maxsize=128)
def fractional_integral_weights(n: int, alpha: float, h: float) -> NDArray[np.float64]:
    """Return Grünwald product-integration weights for a left fractional integral.

    The discrete convolution approximates ``I_left^alpha f`` by
    ``h^alpha * sum_m (-1)^m binom(-alpha, m) f[i-m]``.  This is the natural
    inverse-family companion to the GL derivative matrix and gives an exact
    matrix-transpose adjoint for weak integral formulations.
    """
    n = int(n)
    alpha = float(alpha)
    h = float(h)
    if n <= 0:
        raise ValueError("n must be positive")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if h <= 0:
        raise ValueError("h must be positive")
    w = np.empty(n, dtype=float)
    w[0] = 1.0
    for m in range(1, n):
        # Recurrence for (-1)^m binom(-alpha,m), which is positive for alpha>0.
        w[m] = w[m - 1] * (alpha + m - 1.0) / float(m)
    return w * (h ** alpha)


@lru_cache(maxsize=128)
def fractional_integral_matrix(n: int, alpha: float, h: float, side: Side = "left") -> sparse.csr_matrix:
    """Sparse GL-style fractional integral matrix on a uniform grid.

    ``side='left'`` is causal/lower triangular; ``side='right'`` is anti-causal;
    ``'symmetric'`` averages both.  For weak forms, the adjoint tests are
    obtained by row-wise multiplication ``phi @ J``.
    """
    n = int(n)
    alpha = float(alpha)
    h = float(h)
    if side not in {"left", "right", "symmetric"}:
        raise ValueError("side must be 'left', 'right', or 'symmetric'")
    if alpha == 0.0:
        return sparse.eye(n, format="csr")
    if side == "symmetric":
        return 0.5 * (fractional_integral_matrix(n, alpha, h, "left") + fractional_integral_matrix(n, alpha, h, "right"))
    w = fractional_integral_weights(n, alpha, h)
    offsets = -np.arange(n) if side == "left" else np.arange(n)
    diagonals = [np.full(n - abs(int(k)), w[abs(int(k))], dtype=float) for k in offsets]
    return sparse.diags(diagonals, offsets, shape=(n, n), format="csr")


def fractional_integral_adjoint_tests(
    tests: ArrayLike,
    grid: ArrayLike,
    alpha: float,
    side: Side = "left",
) -> NDArray[np.float64]:
    """Apply the exact discrete adjoint of a one-sided fractional integral.

    If the model contains ``I_left^alpha f`` then the weak test side is
    represented by ``tests @ J_left``.  This is used for the Volterra/Caputo
    integral form ``u-u0 = I_t^alpha N(u)``.
    """
    phi = np.asarray(tests, dtype=float)
    x = _as_float_array(grid, name="grid")
    h = _uniform_spacing(x, name="grid")
    if phi.ndim != 2 or phi.shape[1] != x.size:
        raise ValueError("tests must have shape (n_tests, n_grid)")
    J = fractional_integral_matrix(x.size, float(alpha), h, side)
    return phi @ J

def caputo_corrected_field(U: NDArray[np.float64], axis: int, order: float, h: float, initial: str = "first") -> NDArray[np.float64]:
    """Return ``U - P_initial`` for Caputo weak identities.

    For 0 < order <= 1, subtract the initial value. For 1 < order <= 2,
    subtract the initial value plus a one-sided boundary-slope estimate from the
    first two samples times the axis coordinate. This helper belongs to the
    optional Volterra target; the reported L1-adjoint superunit experiment does
    not call it. The operation is vectorized over all other dimensions.
    """
    if initial not in {"first", "zero"}:
        raise ValueError("initial must be 'first' or 'zero'")
    arr = np.asarray(U, dtype=float)
    if order <= EPS:
        return arr.copy()
    if initial == "zero":
        return arr.copy()
    axis = int(axis)
    if axis < 0:
        axis += arr.ndim
    moved = np.moveaxis(arr, axis, 0)
    corrected = moved.copy()
    p0 = moved[0:1]
    corrected = corrected - p0
    if float(order) > 1.0 + 1e-12:
        slope0 = (moved[1:2] - moved[0:1]) / float(h)
        coord = (np.arange(moved.shape[0], dtype=float) * float(h)).reshape((-1,) + (1,) * (moved.ndim - 1))
        corrected = corrected - coord * slope0
    return np.moveaxis(corrected, 0, axis)


def gaussian_test_matrix(
    grid: ArrayLike,
    centers: Sequence[float] | int,
    width: float,
    *,
    periodic: bool = False,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Build smooth Gaussian test functions on a 1-D grid.

    The output has shape ``(n_tests, n_grid)``.  Gaussian tests are not compactly
    supported, but for periodic Riesz operators and interior time windows they
    provide stable, spectrally localized weak features.  Their derivatives and
    fractional adjoints are computed numerically by this module.
    """
    x = _as_float_array(grid, name="grid")
    if width <= 0:
        raise ValueError("width must be positive")
    if isinstance(centers, int):
        if centers <= 0:
            raise ValueError("number of centers must be positive")
        lo, hi = float(x[0]), float(x[-1])
        if periodic:
            c = np.linspace(lo, hi + (x[1] - x[0]), centers, endpoint=False)
        else:
            pad = 2.5 * float(width)
            c = np.linspace(lo + pad, hi - pad, centers)
    else:
        c = np.asarray(list(centers), dtype=float)
    tests = np.empty((len(c), x.size), dtype=float)
    period = float((x[1] - x[0]) * x.size)
    for i, ci in enumerate(c):
        dist = x - float(ci)
        if periodic:
            dist = (dist + 0.5 * period) % period - 0.5 * period
        tests[i] = np.exp(-0.5 * (dist / float(width)) ** 2)
        if normalize:
            norm = np.sqrt(np.sum(tests[i] ** 2) + EPS)
            tests[i] /= norm
    return tests


def fourier_test_matrix(
    grid: ArrayLike,
    max_mode: int,
    *,
    include_constant: bool = True,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Build real periodic Fourier test functions on a 1-D uniform grid.

    The rows are ``1, cos(2*pi*m*x/L), sin(2*pi*m*x/L)`` up to
    ``max_mode``.  Fourier tests are useful for periodic Riesz/fractional
    Laplacian discovery because the operator is diagonal in this basis, so
    high-order fractional terms are not washed out by overly broad Gaussian
    windows.
    """
    x = _as_float_array(grid, name="grid")
    h = _uniform_spacing(x, name="grid")
    n = x.size
    max_mode = int(max_mode)
    if max_mode < 0:
        raise ValueError("max_mode must be nonnegative")
    max_mode = min(max_mode, max(0, n // 2 - 1))
    L = float(h * n)
    rows: list[NDArray[np.float64]] = []
    if include_constant:
        rows.append(np.ones(n, dtype=float))
    for m in range(1, max_mode + 1):
        theta = 2.0 * np.pi * m * x / L
        rows.append(np.cos(theta))
        rows.append(np.sin(theta))
    tests = np.vstack(rows) if rows else np.zeros((0, n), dtype=float)
    if normalize and tests.size:
        norms = np.linalg.norm(tests, axis=1)
        norms[norms < EPS] = 1.0
        tests = tests / norms[:, None]
    return tests


def periodic_riesz_on_tests(tests: ArrayLike, grid: ArrayLike, order: float) -> NDArray[np.float64]:
    """Apply the periodic self-adjoint Riesz operator to row-wise test functions.

    The convention matches ``fpde_derivatives.spectral_space_derivative(...,
    riesz=True)`` for order > 0: multiplier ``-|k|^order``.  At order zero the
    identity is returned, matching the existing discovery convention that
    ``D_x^0 u = u``.
    """
    phi = np.asarray(tests, dtype=float)
    if phi.ndim != 2:
        raise ValueError("tests must have shape (n_tests, n_grid)")
    x = _as_float_array(grid, name="grid")
    h = _uniform_spacing(x, name="grid")
    if abs(float(order)) < EPS:
        return phi.copy()
    k = 2.0 * np.pi * np.fft.fftfreq(x.size, d=h)
    multiplier = -(np.abs(k) ** float(order))
    return np.fft.ifft(np.fft.fft(phi, axis=1) * multiplier[None, :], axis=1).real


def periodic_spectral_directional_adjoint_on_tests(tests: ArrayLike, grid: ArrayLike, order: float) -> NDArray[np.float64]:
    """Apply the adjoint of the periodic directional spectral derivative to tests.

    This matches ``fpde_derivatives.spectral_space_derivative(..., riesz=False)``,
    whose model-side Fourier multiplier is ``(i k)^order``.  For the real
    Euclidean inner product, the weak adjoint acting on the test function uses
    the conjugate multiplier ``conj((i k)^order)``.  This function is needed for
    FFT-generated directional fractional benchmarks such as ``tsfade_fft``;
    using a finite-domain one-sided Grünwald/Riemann--Liouville adjoint there
    introduces a boundary-convention mismatch.
    """
    phi = np.asarray(tests, dtype=float)
    if phi.ndim != 2:
        raise ValueError("tests must have shape (n_tests, n_grid)")
    x = _as_float_array(grid, name="grid")
    h = _uniform_spacing(x, name="grid")
    if phi.shape[1] != x.size:
        raise ValueError("test length and grid length do not match")
    if abs(float(order)) < EPS:
        return phi.copy()
    k = 2.0 * np.pi * np.fft.fftfreq(x.size, d=h)
    multiplier = (1j * k).astype(complex) ** float(order)
    multiplier[np.isclose(k, 0.0)] = 0.0
    return np.fft.ifft(np.fft.fft(phi, axis=1) * np.conjugate(multiplier)[None, :], axis=1).real


@lru_cache(maxsize=128)
def caputo_l1_matrix(n: int, alpha: float, h: float) -> sparse.csr_matrix:
    """Discrete Caputo-L1 time-derivative matrix on a uniform grid.

    This matrix is useful when the data generator/evaluator uses the standard
    L1 Caputo discretization.  The weak feature can then use the exact discrete
    adjoint ``phi @ D_L1`` instead of a Grünwald approximation to the continuous
    adjoint.  Constants are in the null space for ``0 < alpha <= 1``.  For
    ``1 < alpha < 2`` the implementation composes ``L1(alpha-1) @ D1``.  Its
    transpose therefore contains an endpoint-dominated opposite-sign weight
    pair inherited from the one-sided first row of ``D1``; this is an implicit
    initial-rate treatment and can amplify noise near the initial time.
    """
    n = int(n)
    alpha = float(alpha)
    h = float(h)
    if n <= 1:
        raise ValueError("n must be at least 2")
    if not (0.0 <= alpha <= 2.0):
        raise ValueError("alpha must lie in [0, 2]")
    if h <= 0:
        raise ValueError("h must be positive")
    if alpha <= EPS:
        return sparse.eye(n, format="csr")
    if abs(alpha - 1.0) < 1e-11:
        D = sparse.lil_matrix((n, n), dtype=float)
        D[0, 0] = -1.0 / h; D[0, 1] = 1.0 / h
        for i in range(1, n - 1):
            D[i, i - 1] = -0.5 / h; D[i, i + 1] = 0.5 / h
        D[n - 1, n - 2] = -1.0 / h; D[n - 1, n - 1] = 1.0 / h
        return D.tocsr()
    if alpha > 1.0:
        # Compose L1(alpha-1) with the same first-derivative matrix.  This
        # mirrors fpde_derivatives.caputo_l1_time for 1 < alpha < 2.
        return caputo_l1_matrix(n, alpha - 1.0, h) @ caputo_l1_matrix(n, 1.0, h)
    weights = np.array([(k + 1.0) ** (1.0 - alpha) - k ** (1.0 - alpha) for k in range(n)], dtype=float)
    coef = h ** (-alpha) / special.gamma(2.0 - alpha)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    # row 0 remains zero, matching caputo_l1_time's invalid/unused initial row
    # while preserving the constant-null-space property in weak products.
    for row in range(1, n):
        # U[row]
        rows.append(row); cols.append(row); vals.append(coef * weights[0])
        # U[0]
        rows.append(row); cols.append(0); vals.append(-coef * weights[row - 1])
        for col in range(1, row):
            k_pos = row - col
            vals_col = coef * (weights[k_pos] - weights[k_pos - 1])
            rows.append(row); cols.append(col); vals.append(vals_col)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def integer_time_derivative_matrix(n: int, h: float, order: int = 1) -> sparse.csr_matrix:
    """Matrix matching fpde_derivatives.integer_time_derivative for order 0/1/2."""
    n = int(n); h = float(h); order = int(order)
    if order == 0:
        return sparse.eye(n, format="csr")
    if order == 1:
        D = sparse.lil_matrix((n, n), dtype=float)
        D[0, 0] = -1.0 / h; D[0, 1] = 1.0 / h
        for i in range(1, n - 1):
            D[i, i - 1] = -0.5 / h; D[i, i + 1] = 0.5 / h
        D[n - 1, n - 2] = -1.0 / h; D[n - 1, n - 1] = 1.0 / h
        return D.tocsr()
    if order == 2:
        D = sparse.lil_matrix((n, n), dtype=float)
        for i in range(1, n - 1):
            D[i, i - 1] = 1.0 / (h * h); D[i, i] = -2.0 / (h * h); D[i, i + 1] = 1.0 / (h * h)
        D[0] = D[1]; D[n - 1] = D[n - 2]
        return D.tocsr()
    raise ValueError("only orders 0, 1, and 2 are supported")


def caputo_l1_adjoint_tests(tests: ArrayLike, grid: ArrayLike, alpha: float) -> NDArray[np.float64]:
    """Apply the exact discrete adjoint of the Caputo-L1 derivative to tests.

    For superunit orders the underlying matrix is ``L1(alpha-1) @ D1``.  The
    returned adjoint tests consequently retain the endpoint sensitivity of the
    one-sided initial row of ``D1``; applying the transpose does not remove that
    implicit initial-rate contribution.
    """
    phi = np.asarray(tests, dtype=float)
    x = _as_float_array(grid, name="grid")
    h = _uniform_spacing(x, name="grid")
    if phi.ndim != 2 or phi.shape[1] != x.size:
        raise ValueError("tests must have shape (n_tests, n_grid)")
    if abs(float(alpha) - round(float(alpha))) < 1e-11:
        D = integer_time_derivative_matrix(x.size, h, int(round(float(alpha))))
    else:
        D = caputo_l1_matrix(x.size, float(alpha), h)
    return phi @ D


def adjoint_tests_1d(
    tests: ArrayLike,
    grid: ArrayLike,
    spec: FractionalOperatorSpec,
) -> NDArray[np.float64]:
    """Apply the adjoint fractional operator to row-wise 1-D test functions."""
    phi = np.asarray(tests, dtype=float)
    if phi.ndim != 2:
        raise ValueError("tests must have shape (n_tests, n_grid)")
    x = _as_float_array(grid, name="grid")
    if phi.shape[1] != x.size:
        raise ValueError("test length and grid length do not match")
    h = _uniform_spacing(x, name="grid")
    if spec.kind == "identity" or abs(float(spec.order)) < EPS:
        return phi.copy()
    if spec.kind == "riesz":
        return periodic_riesz_on_tests(phi, x, spec.order)
    if spec.kind == "integer":
        D = integer_time_derivative_matrix(x.size, h, int(round(float(spec.order))))
        return phi @ D
    # For left model operators, the weak adjoint is represented by D_left.T on
    # column vectors.  With row-wise tests this is phi @ D_left.  The same exact
    # discrete-adjoint rule applies to R-L, Caputo, and GL; Caputo differs only
    # by data-side initial-polynomial correction during assembly.
    D = gl_matrix(x.size, float(spec.order), h, spec.side)
    return phi @ D


class SeparableWeakLibrary2D:
    """Efficient tensor-product weak library for data on a uniform (t, x) grid.

    For row-wise time tests ``rho_k(t)`` and spatial tests ``psi_l(x)``, each
    weak feature is assembled as matrix products

        rho @ F(U) @ psi_adj.T,

    then flattened over all ``(k, l)`` test windows.  This avoids looping over
    all grid points for every candidate term.
    """

    def __init__(
        self,
        t: ArrayLike,
        x: ArrayLike,
        time_tests: ArrayLike,
        space_tests: ArrayLike,
    ) -> None:
        self.t = _as_float_array(t, name="t")
        self.x = _as_float_array(x, name="x")
        self.dt = _uniform_spacing(self.t, name="t")
        self.dx = _uniform_spacing(self.x, name="x")
        self.time_tests = np.asarray(time_tests, dtype=float)
        self.space_tests = np.asarray(space_tests, dtype=float)
        if self.time_tests.ndim != 2 or self.time_tests.shape[1] != self.t.size:
            raise ValueError("time_tests must have shape (n_time_tests, nt)")
        if self.space_tests.ndim != 2 or self.space_tests.shape[1] != self.x.size:
            raise ValueError("space_tests must have shape (n_space_tests, nx)")
        self._adjoint_cache: dict[FractionalOperatorSpec, NDArray[np.float64]] = {}

    @property
    def n_rows(self) -> int:
        """Number of weak regression rows.

        With ``K_t`` time tests and ``K_x`` spatial tests, the library contains
        ``K_t * K_x`` weak integral equations.
        """
        return int(self.time_tests.shape[0] * self.space_tests.shape[0])

    def _tests_for_axis(self, spec: FractionalOperatorSpec) -> NDArray[np.float64]:
        if spec.axis == "t":
            return self.time_tests
        return self.space_tests

    def _grid_for_axis(self, spec: FractionalOperatorSpec) -> NDArray[np.float64]:
        return self.t if spec.axis == "t" else self.x

    def adjoint_tests(self, spec: FractionalOperatorSpec) -> NDArray[np.float64]:
        """Return cached test functions after applying ``spec``'s adjoint.

        For example, a left Riemann--Liouville derivative on the data side is
        represented by the corresponding right/discrete-transpose action on
        the test side.  This method hides the definition-specific adjoint logic
        from the library-building code.
        """
        if spec not in self._adjoint_cache:
            self._adjoint_cache[spec] = adjoint_tests_1d(self._tests_for_axis(spec), self._grid_for_axis(spec), spec)
        return self._adjoint_cache[spec]

    def _correct_for_caputo(self, F: NDArray[np.float64], spec: FractionalOperatorSpec) -> NDArray[np.float64]:
        if spec.kind != "caputo" or float(spec.order) <= EPS:
            return F
        axis = 0 if spec.axis == "t" else 1
        h = self.dt if spec.axis == "t" else self.dx
        return caputo_corrected_field(F, axis=axis, order=float(spec.order), h=h, initial=spec.caputo_initial)

    def weak_inner(
        self,
        F: ArrayLike,
        *,
        time_factor: ArrayLike | None = None,
        space_factor: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Return all separable weak inner products ``<F, rho_k psi_l>``.

        ``time_factor`` and ``space_factor`` may be replaced by adjoint test
        matrices.  The returned vector is flattened in time-test-major order.
        """
        arr = np.asarray(F, dtype=float)
        if arr.shape != (self.t.size, self.x.size):
            raise ValueError(f"F has shape {arr.shape}; expected {(self.t.size, self.x.size)}")
        T = self.time_tests if time_factor is None else np.asarray(time_factor, dtype=float)
        X = self.space_tests if space_factor is None else np.asarray(space_factor, dtype=float)
        if T.shape[1] != self.t.size or X.shape[1] != self.x.size:
            raise ValueError("test-factor shapes are incompatible with F")
        vals = T @ arr @ X.T
        return (vals * self.dt * self.dx).reshape(-1)

    def weak_operator_feature(self, F: ArrayLike, spec: FractionalOperatorSpec) -> NDArray[np.float64]:
        """Assemble ``<operator(F), phi>`` by applying ``operator^*`` to tests."""
        arr = np.asarray(F, dtype=float)
        arr = self._correct_for_caputo(arr, spec)
        if spec.kind == "identity" or abs(float(spec.order)) < EPS:
            return self.weak_inner(arr)
        if spec.axis == "t":
            return self.weak_inner(arr, time_factor=self.adjoint_tests(spec))
        return self.weak_inner(arr, space_factor=self.adjoint_tests(spec))

    def build_library(self, U: ArrayLike, terms: Sequence[WeakTerm]) -> tuple[NDArray[np.float64], list[str]]:
        """Build a weak candidate matrix for terms ``operator(transform(U))``."""
        arr = np.asarray(U, dtype=float)
        if arr.shape != (self.t.size, self.x.size):
            raise ValueError(f"U has shape {arr.shape}; expected {(self.t.size, self.x.size)}")
        cols = []
        names = []
        for term in terms:
            F = term.apply_transform(arr)
            cols.append(self.weak_operator_feature(F, term.operator))
            names.append(term.name)
        if not cols:
            return np.zeros((self.n_rows, 0), dtype=float), []
        return np.column_stack(cols), names

    def target(self, U: ArrayLike, lhs_operator: FractionalOperatorSpec) -> NDArray[np.float64]:
        """Build a weak target vector for the left-hand-side operator on ``U``."""
        return self.weak_operator_feature(np.asarray(U, dtype=float), lhs_operator)


def fit_least_squares(
    Theta: ArrayLike,
    b: ArrayLike,
    *,
    ridge: float = 0.0,
    normalize: bool = True,
) -> tuple[NDArray[np.float64], dict[str, float | int]]:
    """Fit coefficients for a weak or vanilla candidate library.

    Columns may be normalized internally for conditioning; coefficients are
    returned in the original physical scaling.
    """
    X = np.asarray(Theta, dtype=float)
    y = np.asarray(b, dtype=float).reshape(-1)
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xf = X[finite]
    yf = y[finite]
    if Xf.ndim != 2 or Xf.shape[0] == 0:
        raise ValueError("no finite regression rows")
    scales = np.ones(Xf.shape[1], dtype=float)
    Xn = Xf
    if normalize and Xf.shape[1] > 0:
        scales = np.linalg.norm(Xf, axis=0)
        scales[scales < EPS] = 1.0
        Xn = Xf / scales[None, :]
    if ridge > 0:
        coef_n = np.linalg.solve(Xn.T @ Xn + float(ridge) * np.eye(Xn.shape[1]), Xn.T @ yf)
    else:
        coef_n, *_ = np.linalg.lstsq(Xn, yf, rcond=None)
    coef = coef_n / scales
    pred = Xf @ coef
    resid = yf - pred
    rel_rmse = float(np.linalg.norm(resid) / (np.linalg.norm(yf) + EPS))
    return coef.astype(float), {
        "n_rows": int(Xf.shape[0]),
        "rank": int(np.linalg.matrix_rank(Xf)),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "rel_rmse": rel_rmse,
        "cond": float(np.linalg.cond(Xf)) if Xf.shape[1] else 0.0,
    }


def vanilla_riesz_library(
    U: ArrayLike,
    t: ArrayLike,
    x: ArrayLike,
    beta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Strong-form baseline for ``u_t = c0 u + c1 Riesz_x^beta u``.

    This intentionally differentiates the noisy data directly and is included
    for diagnostics against the weak candidate construction.
    """
    arr = np.asarray(U, dtype=float)
    tt = _as_float_array(t, name="t")
    xx = _as_float_array(x, name="x")
    dt = _uniform_spacing(tt, name="t")
    if arr.shape != (tt.size, xx.size):
        raise ValueError("U, t, and x have incompatible shapes")
    ut = np.empty_like(arr)
    ut[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
    ut[0] = (arr[1] - arr[0]) / dt
    ut[-1] = (arr[-1] - arr[-2]) / dt
    ux_beta = periodic_riesz_on_tests(arr, xx, beta)
    # Trim one time row and a small spatial boundary strip for fairer finite
    # difference behavior, even though the Riesz operator is periodic.
    sl_t = slice(1, -1)
    X = np.column_stack([arr[sl_t].reshape(-1), ux_beta[sl_t].reshape(-1)])
    y = ut[sl_t].reshape(-1)
    return X, y, ["u", f"Riesz_x^{beta:g} u"]


# =====================================================================
# Blended extensions (compact-support tests, Riesz-Feller skew, order polish)
# Added to combine the strengths of an independent weak-FDE derivation with
# this package.  All functions are additive; existing behaviour is unchanged.
# =====================================================================
def compact_bump_test_matrix(
    grid: ArrayLike,
    centers: Sequence[float] | int,
    width: float,
    *,
    power: int = 6,
    periodic: bool = False,
    normalize: bool = True,
) -> NDArray[np.float64]:
    r"""Compactly-supported C^{power-1} test functions ``(1-s^2)^power``.

    ``s = (grid-center)/(support_factor*width)`` with ``support_factor=3`` so the
    half-support is ``3*width`` (comparable spectral localization to
    :func:`gaussian_test_matrix` at the same ``width``).  Unlike Gaussian tests,
    these vanish *exactly* outside their support together with all derivatives up
    to order ``power-1``.  For the derivative-form (GL/RL/Caputo-L1) *temporal*
    weak targets this makes the omitted boundary functional ``B_L(f,phi)`` exactly
    zero rather than merely small, which removes a residual end-of-record bias in
    fractional-order identification under noise and on short time horizons.  For
    periodic spatial operators the boundary term already vanishes, so there the
    two families are interchangeable.
    """
    x = _as_float_array(grid, name="grid")
    if width <= 0:
        raise ValueError("width must be positive")
    if int(power) < 1:
        raise ValueError("power must be >= 1")
    half = 3.0 * float(width)
    if isinstance(centers, int):
        if centers <= 0:
            raise ValueError("number of centers must be positive")
        lo, hi = float(x[0]), float(x[-1])
        if periodic:
            c = np.linspace(lo, hi + (x[1] - x[0]), centers, endpoint=False)
        else:
            pad = half * 1.02  # keep full support strictly inside the domain
            if hi - lo <= 2.0 * pad:
                raise ValueError("width too large for a non-periodic compact test grid")
            c = np.linspace(lo + pad, hi - pad, centers)
    else:
        c = np.asarray(list(centers), dtype=float)
    period = float((x[1] - x[0]) * x.size)
    tests = np.zeros((len(c), x.size), dtype=float)
    for i, ci in enumerate(c):
        dist = x - float(ci)
        if periodic:
            dist = (dist + 0.5 * period) % period - 0.5 * period
        s = dist / half
        mask = np.abs(s) < 1.0
        tests[i, mask] = (1.0 - s[mask] ** 2) ** int(power)
        if normalize:
            nrm = np.sqrt(np.sum(tests[i] ** 2) + EPS)
            tests[i] /= nrm
    return tests


def periodic_riesz_feller_adjoint_on_tests(
    tests: ArrayLike,
    grid: ArrayLike,
    order: float,
    theta: float = 0.0,
) -> NDArray[np.float64]:
    r"""Adjoint of the periodic Riesz--Feller operator applied to row-wise tests.

    Fourier multiplier (Feller / skewed fractional Laplacian, skewness angle
    ``theta``):

        \widehat{D^{order}_{theta} f}(k)
            = -|k|^{order} \exp\!\big(i\,\operatorname{sgn}(k)\,\theta\,\pi/2\big)\,\widehat f(k).

    For the real Euclidean inner product the weak adjoint on the test side uses
    the conjugate multiplier, followed by the real part after the inverse FFT.
    ``theta=0`` recovers the symmetric self-adjoint Riesz operator
    (:func:`periodic_riesz_on_tests`); ``|theta|>0`` models the *asymmetric*
    space-fractional transport of skewed alpha-stable / Levy flights (e.g.
    ``S_{order}(...,skew,...)`` benchmarks) that a symmetric Riesz column cannot
    represent.  The admissible range is ``|theta| <= min(order, 2-order)``.
    """
    phi = np.asarray(tests, dtype=float)
    if phi.ndim != 2:
        raise ValueError("tests must have shape (n_tests, n_grid)")
    x = _as_float_array(grid, name="grid")
    h = _uniform_spacing(x, name="grid")
    if phi.shape[1] != x.size:
        raise ValueError("test length and grid length do not match")
    order = float(order)
    theta = float(theta)
    if abs(order) < EPS:
        return phi.copy()
    bound = min(order, 2.0 - order) if 0.0 < order < 2.0 else 0.0
    if abs(theta) > bound + 1e-9:
        raise ValueError(f"|theta|={abs(theta):g} exceeds admissible bound {bound:g} for order={order:g}")
    k = 2.0 * np.pi * np.fft.fftfreq(x.size, d=h)
    mult = -(np.abs(k) ** order) * np.exp(1j * np.sign(k) * theta * np.pi / 2.0)
    mult[np.isclose(k, 0.0)] = 0.0
    return np.fft.ifft(np.fft.fft(phi, axis=1) * np.conjugate(mult)[None, :], axis=1).real


def refine_orders_local(
    residual_fn: Callable[[NDArray[np.float64]], float],
    z0: ArrayLike,
    bounds: Sequence[tuple[float, float]],
    *,
    method: str = "Nelder-Mead",
    maxiter: int = 200,
) -> tuple[NDArray[np.float64], float]:
    r"""Local polish of fractional orders on the (smooth) weak-form residual.

    ``residual_fn`` maps an order vector ``z=(alpha,beta_1,...)`` to a scalar
    relative residual ``||b(z)-Theta(z) xi*||/||b(z)||`` with ``xi*`` refit
    inside.  Because the *weak* residual is smooth in the continuous orders --
    unlike the strong-form thresholded-regression order loss, which is piecewise
    discontinuous as the support jumps -- a few derivative-free simplex steps
    refine a coarse differential-evolution / grid optimum to higher order
    precision at negligible cost.  Intended as an optional per-support-size
    *polish* after the existing best-subset Pareto-DE search.  Returns the
    refined orders (clipped to ``bounds``) and the achieved residual.
    """
    from scipy.optimize import minimize

    z0 = np.asarray(z0, dtype=float)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    if z0.shape[0] != lo.shape[0]:
        raise ValueError("z0 and bounds must have matching length")

    def obj(z: NDArray[np.float64]) -> float:
        return float(residual_fn(np.clip(z, lo, hi)))

    res = minimize(obj, z0, method=method,
                   options={"maxiter": int(maxiter), "xatol": 1e-4, "fatol": 1e-10})
    z = np.clip(np.asarray(res.x, dtype=float), lo, hi)
    return z, float(residual_fn(z))
