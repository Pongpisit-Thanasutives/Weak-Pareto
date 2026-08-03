"""Branch-aware temporal-order utilities for Caputo discovery.

The Caputo family changes definition when ``n = ceil(alpha)`` changes.  In
particular, the operators on ``0 < alpha < 1``, ``alpha = 1`` and
``1 < alpha < 2`` must be treated as distinct modes.  This module centralises
that bookkeeping so feature banks never interpolate across an integer order and
the optimiser compares branch minima explicitly.
"""
from __future__ import annotations

from typing import Literal, Sequence
import numpy as np
from numpy.typing import NDArray

TemporalAlphaMode = Literal[
    "fractional_subunit",
    "integer",
    "fractional_superunit",
]

VALID_ALPHA_MODES: tuple[TemporalAlphaMode, ...] = (
    "fractional_subunit",
    "integer",
    "fractional_superunit",
)


def infer_alpha_mode(alpha: float, *, integer_tol: float = 1e-11) -> TemporalAlphaMode:
    """Return the Caputo mode associated with ``alpha``."""
    a = float(alpha)
    if abs(a - 1.0) <= float(integer_tol):
        return "integer"
    return "fractional_subunit" if a < 1.0 else "fractional_superunit"


def available_alpha_modes(
    alpha_grid: Sequence[float],
    *,
    branch_epsilon: float = 1e-3,
    integer_tol: float = 1e-11,
) -> tuple[TemporalAlphaMode, ...]:
    """Return the temporal modes admitted by the declared search interval."""
    grid = np.asarray(alpha_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("alpha_grid must be a non-empty one-dimensional sequence")
    lo, hi = float(np.min(grid)), float(np.max(grid))
    eps = float(branch_epsilon)
    modes: list[TemporalAlphaMode] = []
    if lo < 1.0 - eps:
        modes.append("fractional_subunit")
    if lo - integer_tol <= 1.0 <= hi + integer_tol:
        modes.append("integer")
    if hi > 1.0 + eps:
        modes.append("fractional_superunit")
    if not modes:
        # Degenerate intervals very close to one are interpreted as the exact
        # integer candidate rather than an artificial fractional sliver.
        if lo - integer_tol <= 1.0 <= hi + integer_tol:
            return ("integer",)
        modes.append(infer_alpha_mode(0.5 * (lo + hi), integer_tol=integer_tol))
    return tuple(modes)


def alpha_bounds_for_mode(
    alpha_grid: Sequence[float],
    mode: TemporalAlphaMode,
    *,
    branch_epsilon: float = 1e-3,
) -> tuple[float, float]:
    """Return the closed numerical bounds used for one temporal mode.

    The fractional modes stop at ``1 +/- branch_epsilon``.  The exact integer
    candidate is represented by the degenerate interval ``(1, 1)``.
    """
    grid = np.asarray(alpha_grid, dtype=float)
    lo, hi = float(np.min(grid)), float(np.max(grid))
    eps = float(branch_epsilon)
    if mode == "integer":
        if not (lo <= 1.0 <= hi):
            raise ValueError("the declared alpha interval does not contain alpha=1")
        return 1.0, 1.0
    if mode == "fractional_subunit":
        upper = min(hi, 1.0 - eps)
        if upper <= lo:
            raise ValueError("no subunit fractional interval is admitted")
        return lo, upper
    if mode == "fractional_superunit":
        lower = max(lo, 1.0 + eps)
        if hi <= lower:
            raise ValueError("no superunit fractional interval is admitted")
        return lower, hi
    raise ValueError(f"unknown temporal mode: {mode!r}")


def branch_grid(
    alpha_grid: Sequence[float],
    mode: TemporalAlphaMode,
    *,
    branch_epsilon: float = 1e-3,
) -> NDArray[np.float64]:
    """Construct an interpolation grid confined to one temporal mode.

    Declared endpoints are inserted so differential evolution can evaluate the
    complete numerical branch without borrowing feature slices from another
    Caputo mode.  At least two nodes are returned for a non-degenerate
    fractional branch.
    """
    lo, hi = alpha_bounds_for_mode(alpha_grid, mode, branch_epsilon=branch_epsilon)
    if mode == "integer":
        return np.array([1.0], dtype=float)
    base = np.asarray(alpha_grid, dtype=float)
    nodes = base[(base >= lo - 1e-14) & (base <= hi + 1e-14)]
    nodes = np.concatenate([nodes, np.array([lo, hi], dtype=float)])
    nodes = np.unique(np.round(nodes.astype(float), 15))
    nodes.sort()
    if nodes.size == 1:
        nodes = np.array([lo, hi], dtype=float)
    if nodes.size < 2 or not np.all(np.diff(nodes) > 0):
        raise ValueError(f"could not construct a valid interpolation grid for {mode}")
    return nodes.astype(float)


def model_mode(alpha: float, alpha_mode: str | None) -> TemporalAlphaMode:
    """Resolve a stored/optional model mode while remaining backward compatible."""
    if alpha_mode in VALID_ALPHA_MODES:
        return alpha_mode  # type: ignore[return-value]
    return infer_alpha_mode(float(alpha))
