"""Pareto-front Differential Evolution for flexible fractional PDE discovery.

Model class
-----------
    D_t^alpha u = sum_{j=1}^c xi_j * u^{p_j} * D_x^{beta_j} u

The important design choice is that support size ``c`` is *not* a gene in one
large mixed chromosome.  Instead, the algorithm performs a cardinality sweep:
for c = 1, 2, ..., Cmax it searches for the best continuous fractional orders
and coefficients at that fixed complexity.  The resulting validation-error vs.
complexity curve is an empirical Pareto front from which the number of terms can
be selected by elbow, heuristic BIC/AIC-type scores, or a sparse-relaxed rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence
import csv
import itertools
import json
import math
import time

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.optimize import differential_evolution

from fpde_derivatives import (
    caputo_l1_time,
    regularized_space_derivative,
    regularized_time_derivative,
    spectral_space_derivative,
)
from fpde_datasets import GridDataset, add_multiplicative_uniform_noise
from selected_fde_io import save_selected_fde
from temporal_modes import (
    TemporalAlphaMode,
    alpha_bounds_for_mode,
    available_alpha_modes,
    branch_grid,
    infer_alpha_mode,
    model_mode,
)

EPS = 1e-14


def _tqdm(iterable=None, *, enabled: bool = True, **kwargs):
    """Return a tqdm progress iterator when available; otherwise return iterable.

    The project should remain usable without tqdm installed, so progress bars are
    optional.  Install tqdm for notebook/terminal progress tracking:

        pip install tqdm

    When ``enabled`` is false or tqdm is missing, this function degrades to the
    original iterable without changing program behavior.
    """
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def _tqdm_context(*, enabled: bool = True, **kwargs):
    """Return a tqdm progress-bar context manager or a no-op context manager."""
    if not enabled:
        from contextlib import nullcontext
        return nullcontext(None)
    try:
        from tqdm.auto import tqdm
        return tqdm(**kwargs)
    except Exception:
        from contextlib import nullcontext
        return nullcontext(None)


def _progress_write(message: str, *, enabled: bool = True) -> None:
    """Write a message without breaking active tqdm bars when tqdm is installed."""
    if enabled:
        try:
            from tqdm.auto import tqdm
            tqdm.write(message)
            return
        except Exception:
            pass
    print(message)


SelectionRule = Literal["elbow", "bic", "aic", "sparse_relaxed"]
DerivativeBackend = Literal["regularized", "spectral_l1"]


@dataclass
class DiscoveryConfig:
    """Runtime and candidate-library configuration for one discovery run.

    The most important fields are:

    ``alpha_grid``
        Temporal fractional orders used to precompute/interpolate candidate
        targets ``D_t^alpha u``.  The optimizer may propose continuous alpha
        values inside this grid.
    ``beta_grid``
        Spatial fractional orders used to precompute/interpolate RHS candidate
        terms ``D_x^beta u``.
    ``cmax``
        Maximum RHS support size in the Pareto sweep.  In the packaged paper
        benchmarks this is set to 4 so the optimizer can test extra, possibly
        spurious RHS terms; all internal comparison pipelines receive the same cap.
    ``p_values``
        Allowed powers in terms ``u**p * D_x^beta u``.  The paper benchmarks
        use ``(0,1,2)`` so competing nonlinear candidate terms are present for the mostly
        linear benchmarks and the nonlinear Burgers term is representable.
    ``backend`` and derivative-kind fields
        Define how vanilla strong-form derivatives are computed.  Weak methods
        read the same config and choose the corresponding weak adjoint operator.
    ``maxiter``/``popsize``
        Differential-evolution budget.  These are runtime controls and may
        change between notebook and paper profiles without changing the candidate
        hypothesis class.
    """

    backend: DerivativeBackend = "spectral_l1"
    alpha_grid: Sequence[float] = tuple(np.linspace(0.7, 1.2, 21))
    beta_grid: Sequence[float] = tuple(np.linspace(0.5, 2.2, 35))
    cmax: int = 4
    p_values: Sequence[int] = (0, 1, 2)
    max_patterns_per_c: int | None = None
    maxiter: int = 16
    popsize: int = 5
    polish: bool = False
    seed: int = 0
    val_fraction: float = 0.25
    trim_t: int = 3
    trim_x: int = 4
    noise_percent: float = 0.0
    smooth_sigma_t: float = 0.0
    smooth_sigma_x: float = 0.0
    lam_t: float = 1e-8
    lam_x: float = 1e-6
    regularized_time_kind: str = "caputo"
    regularized_space_kind: str = "caputo"
    regularized_space_side: str = "left"
    power_mode: Literal["raw", "positive"] = "raw"
    spectral_riesz: bool = False
    selection: SelectionRule = "elbow"
    sparse_relaxed_tol: float = 0.08

    # Post-selection pruning for numerically inactive terms. With broad paper
    # libraries, DE may add a competing nonlinear term whose fitted contribution to
    # the RHS is negligible but whose validation error is microscopically smaller.
    # This is not a real discovered physics term. We prune such terms and refit
    # the remaining structure before exporting/reporting the selected equation.
    # The main rule is scale-aware contribution pruning, not raw coefficient size:
    # ||xi_j theta_j||_2 / ||Theta xi||_2 <= selected_contribution_prune_tol.
    selected_contribution_prune_tol: float = 1e-4
    selected_coef_prune_abs_tol: float = 1e-10
    # Legacy fallback used only if feature columns are unavailable.
    selected_coef_prune_rel_tol: float = 1e-3
    # If True, refit the selected support's coefficients by evaluating its final
    # continuous orders directly (no order-grid interpolation).  This changes the
    # final conditional fit, not the preceding support/mode selection; that selection
    # can still depend on the grid and search trajectory.
    exact_order_refit: bool = True
    # If True, additionally polish (alpha, beta) on the directly evaluated weak
    # residual through a small local search around the DE optimum.  This removes
    # interpolation error from the final conditional estimates but does not make
    # the selected support, temporal mode, or optimisation basin grid-independent.
    # Off by default to preserve runtime; enabled in publication runs.
    exact_order_polish: bool = False

    # Caputo temporal orders are branch-aware.  The operators on (0,1), {1}
    # and (1,2) are distinct modes because n=ceil(alpha) changes at the integer.
    # Fractional feature interpolation therefore stops at 1 +/- this gap, while
    # alpha=1 is evaluated as an exact discrete candidate.
    alpha_branch_epsilon: float = 1e-3
    branch_aware_time: bool = True

    duplicate_beta_gap: float = 0.04
    duplicate_penalty: float = 0.02

    # Tikhonov (ridge) regularisation applied to the column-normalised weak
    # design when solving for the coefficients, both during the order search and
    # in the final refit.  Fractional columns at nearby orders are highly
    # collinear, so a negligible ridge admits large cancelling coefficients; a
    # small but non-trivial value keeps the coefficient estimates well posed
    # without materially biasing well-separated benchmarks.
    ridge: float = 1e-3

    # Smallest fractional order treated as a genuine derivative.  The identity
    # operator sits at exactly beta=0 as a distinct discrete candidate; proposed
    # orders below this threshold (or below the first positive grid point, if
    # larger) are resolved to the identity so that the identity and the signed
    # Riesz family are never blended by interpolation across beta=0.
    beta_min_positive: float = 0.0
    # Automatic complexity stopping for the support-size sweep.  This is the
    # publication-oriented mechanism: keep solving the fixed-cardinality DE
    # problems until increasing c stops buying meaningful validation improvement.
    # The stop is deliberately based on the best validation error at each c, not
    # on a single coefficient threshold as in STRidge.
    auto_stop: bool = False
    auto_stop_min_c: int = 3
    auto_stop_patience: int = 1
    auto_stop_rel_improvement: float = 0.03
    auto_stop_log10_improvement: float = 0.02
    auto_stop_use_selection_stability: bool = True
    auto_stop_selection_patience: int = 1

    # Progress reporting. ``progress`` shows feature-precomputation and
    # support-pattern progress bars. ``progress_de`` additionally shows a nested
    # generation-level progress bar for every fixed-pattern DE solve; useful for
    # long runs, but noisy for notebooks with many patterns.
    progress: bool = True
    progress_de: bool = False
    progress_leave: bool = False


@dataclass
class PDEModel:
    """Container for one discovered FPDE candidate.
    
    A ``PDEModel`` represents a single equation of the form
    
        D_t^alpha u = sum_j xi_j * u**p_j * D_x^beta_j u
    
    with fixed support size ``c``. It stores the nonlinear orders, discrete powers,
    least-squares coefficients, train/validation errors, heuristic information
    criteria, the selection objective, and a printable equation string.

    ``train_rel_mse`` and ``val_rel_mse`` retain their selection-stage meanings.
    ``objective`` is the selection objective (normally
    ``log10(val_rel_mse)``).  A post-selection exact/full-data refit is reported
    separately in ``full_data_rel_l2`` so that it cannot overwrite validation or
    objective fields with a differently defined quantity.
    
    You usually do not instantiate this class manually. It is returned by
    ``ParetoFDEOptimizer.evaluate``, ``optimize_fixed_pattern``,
    ``support_size_sweep``, and ``run_pareto_discovery``.
    """
    c: int
    alpha: float
    p_tuple: tuple[int, ...]
    beta_tuple: tuple[float, ...]
    coefficients: NDArray[np.float64]
    train_mse: float
    val_mse: float
    train_rel_mse: float
    val_rel_mse: float
    aic: float
    bic: float
    objective: float
    n_train: int
    n_val: int
    backend: str = ""
    alpha_mode: str = "auto"
    full_data_rel_l2: float = math.nan

    def canonicalized(self) -> "PDEModel":
        """Return a copy with terms sorted into a stable order.
        
        Differential evolution treats terms as exchangeable, so two equivalent models
        may list terms in different orders. This method sorts by ``p``, ``beta``, and
        coefficient value so printing, CSV export, and equality-style inspection are
        more stable. The mathematical equation is unchanged.
        """
        order = sorted(range(self.c), key=lambda i: (self.p_tuple[i], self.beta_tuple[i], float(self.coefficients[i])))
        return PDEModel(
            c=int(self.c),
            alpha=float(self.alpha),
            p_tuple=tuple(int(self.p_tuple[i]) for i in order),
            beta_tuple=tuple(float(self.beta_tuple[i]) for i in order),
            coefficients=np.array([float(self.coefficients[i]) for i in order]),
            train_mse=float(self.train_mse),
            val_mse=float(self.val_mse),
            train_rel_mse=float(self.train_rel_mse),
            val_rel_mse=float(self.val_rel_mse),
            aic=float(self.aic),
            bic=float(self.bic),
            objective=float(self.objective),
            n_train=int(self.n_train),
            n_val=int(self.n_val),
            backend=str(self.backend),
            alpha_mode=str(model_mode(self.alpha, self.alpha_mode)),
            full_data_rel_l2=float(self.full_data_rel_l2),
        )

    def equation(self, digits: int = 5, tol: float = 1e-12) -> str:
        """Format the model as a human-readable FPDE string.
        
        Parameters
        ----------
        digits:
            Number of decimal digits used for derivative orders and coefficients.
        tol:
            Coefficients with absolute value below this tolerance are omitted from the
            printed RHS.
        
        Returns
        -------
        str
            Text such as ``D_t^1.00000 u = (-1.0)*D_x^1.00000 u``.
        """
        rec = self.canonicalized()
        pieces: list[str] = []
        for xi, p, beta in zip(rec.coefficients, rec.p_tuple, rec.beta_tuple):
            if abs(float(xi)) <= tol:
                continue
            if p == 0:
                term = f"D_x^{beta:.{digits}f} u"
            elif p == 1:
                term = f"u D_x^{beta:.{digits}f} u"
            else:
                term = f"u^{p} D_x^{beta:.{digits}f} u"
            pieces.append(f"({xi:.{digits}g})*{term}")
        rhs = " + ".join(pieces) if pieces else "0"
        lhs = "partial_t u" if model_mode(rec.alpha, rec.alpha_mode) == "integer" else f"D_t^{rec.alpha:.{digits}f} u"
        return f"{lhs} = {rhs}"

    def to_dict(self) -> dict:
        """Convert the model to a JSON-serializable dictionary.
        
        Use this when saving benchmark outputs, passing results to notebooks, or
        inspecting models without NumPy arrays. The dictionary includes ``equation`` and
        ``terms`` fields for convenience.
        """
        rec = self.canonicalized()
        return {
            "c": int(rec.c),
            "alpha": float(rec.alpha),
            "alpha_mode": str(model_mode(rec.alpha, rec.alpha_mode)),
            "p_tuple": [int(v) for v in rec.p_tuple],
            "beta_tuple": [float(v) for v in rec.beta_tuple],
            "coefficients": [float(v) for v in rec.coefficients],
            "train_mse": float(rec.train_mse),
            "val_mse": float(rec.val_mse),
            "train_rel_mse": float(rec.train_rel_mse),
            "val_rel_mse": float(rec.val_rel_mse),
            "aic": float(rec.aic),
            "bic": float(rec.bic),
            "objective": float(rec.objective),
            "full_data_rel_l2": float(rec.full_data_rel_l2),
            "n_train": int(rec.n_train),
            "n_val": int(rec.n_val),
            "backend": rec.backend,
            "equation": rec.equation(),
            "terms": [[int(p), float(b)] for p, b in zip(rec.p_tuple, rec.beta_tuple)],
        }


class FractionalFeatureBank:
    """Precomputed/interpolated target and RHS derivative features.

    ``backend='regularized'`` uses the regularized inverse derivative from
    ``regfracdiff`` and aligns all quantities on spacetime cells.  This is useful
    for bounded/noisy data.

    ``backend='spectral_l1'`` uses Caputo-L1 in time and periodic spectral space
    derivatives on nodes.  This is fast and accurate for periodic synthetic data.
    """

    def __init__(self, data: GridDataset, config: DiscoveryConfig):
        """Initialize the feature bank from a gridded dataset and config.
        
        This stores a possibly noised/smoothed copy of ``data.U`` and the configured
        alpha/beta grids. Heavy derivative arrays are not computed here; call
        ``precompute()`` before accessing ``target``, ``spatial``, ``u_power``, or
        ``library``.
        """
        U = np.asarray(data.U, dtype=float)
        U = add_multiplicative_uniform_noise(U, config.noise_percent, seed=config.seed)
        if config.smooth_sigma_t > 0 or config.smooth_sigma_x > 0:
            U = gaussian_filter(U, sigma=(config.smooth_sigma_t, config.smooth_sigma_x), mode="nearest")
        self.data = data
        self.U = U
        self.config = config
        self.alpha_grid = np.asarray(config.alpha_grid, dtype=float)
        self.beta_grid = np.asarray(config.beta_grid, dtype=float)
        if np.any(np.diff(self.alpha_grid) <= 0) or np.any(np.diff(self.beta_grid) <= 0):
            raise ValueError("alpha_grid and beta_grid must be strictly increasing")
        self.alpha_branch_epsilon = float(getattr(config, "alpha_branch_epsilon", 1e-3))
        self.alpha_modes = available_alpha_modes(
            self.alpha_grid, branch_epsilon=self.alpha_branch_epsilon
        ) if bool(getattr(config, "branch_aware_time", True)) else tuple()
        self.alpha_mode_grids: dict[str, NDArray[np.float64]] = {}
        self.time_mode_banks: dict[str, NDArray[np.float64]] = {}
        self.time_bank: NDArray[np.float64] | None = None
        self.space_bank: NDArray[np.float64] | None = None
        self.u_flat: NDArray[np.float64] | None = None
        self.mask_flat: NDArray[np.bool_] | None = None
        self.u_power_cache: dict[int, NDArray[np.float64]] = {}

    def _cell_mask(self, shape: tuple[int, int]) -> NDArray[np.bool_]:
        """Build the valid spacetime mask after boundary trimming.
        
        The mask removes the first/last ``trim_t`` time indices and first/last
        ``trim_x`` spatial indices. Trimming is important because fractional derivatives
        are boundary-sensitive and, for regularized derivatives, arrays are cell-aligned
        rather than node-aligned.
        """
        nt, nx = shape
        mask = np.ones((nt, nx), dtype=bool)
        tt = max(0, int(self.config.trim_t))
        xx = max(0, int(self.config.trim_x))
        if tt:
            mask[:tt, :] = False
            mask[-tt:, :] = False
        if xx:
            mask[:, :xx] = False
            mask[:, -xx:] = False
        return mask

    @staticmethod
    def _interp(order: float, grid: NDArray[np.float64], bank: NDArray[np.float64]) -> NDArray[np.float64]:
        """Linearly interpolate a precomputed derivative bank to an arbitrary order.
        
        The DE optimizer proposes continuous alpha/beta values. Instead of recomputing
        derivatives at every proposal, the code precomputes derivatives on a grid and
        interpolates between neighboring orders.
        """
        z = float(np.clip(order, grid[0], grid[-1]))
        i = int(np.searchsorted(grid, z, side="right") - 1)
        if i < 0:
            return bank[0]
        if i >= len(grid) - 1:
            return bank[-1]
        lo, hi = float(grid[i]), float(grid[i + 1])
        w = (z - lo) / (hi - lo + EPS)
        return (1.0 - w) * bank[i] + w * bank[i + 1]

    def precompute(self, verbose: bool = True) -> None:
        """Precompute all target and spatial derivative features.
        
        After this call, the following are available:
        
        ``time_bank``
            Rows are flattened/trimmed ``D_t^alpha u`` for every alpha in
            ``config.alpha_grid``.
        ``space_bank``
            Rows are flattened/trimmed ``D_x^beta u`` for every beta in
            ``config.beta_grid``.
        ``u_flat``
            Flattened/trimmed version of ``u`` aligned to the derivative features.
        
        This is the method to call before manually building candidate terms such as
        ``bank.u_power(2) * bank.spatial(1.5)``.
        """
        t0 = time.time()
        cfg = self.config
        U = self.U
        if cfg.backend == "regularized":
            # Time derivative: (nt-1, nx), space derivative: (nt, nx-1).  Align
            # on cells (nt-1, nx-1) by averaging over the remaining coordinate.
            mask = self._cell_mask((U.shape[0] - 1, U.shape[1] - 1))
            self.mask_flat = mask.reshape(-1)
            u_cell = 0.25 * (U[:-1, :-1] + U[1:, :-1] + U[:-1, 1:] + U[1:, 1:])
            self.u_flat = u_cell.reshape(-1)[self.mask_flat]
            t_cols = []
            if verbose:
                print(f"[{self.data.name}] precomputing branch-aware regularized time derivatives")
            self.alpha_mode_grids = {
                mode: branch_grid(self.alpha_grid, mode, branch_epsilon=self.alpha_branch_epsilon)
                for mode in self.alpha_modes
            }
            for mode, grid_mode in self.alpha_mode_grids.items():
                rows = []
                for a in _tqdm(grid_mode, enabled=verbose and cfg.progress, desc=f"time {mode}", leave=cfg.progress_leave):
                    raw_t = regularized_time_derivative(
                        U, float(a), self.data.dt, lam=cfg.lam_t,
                        kind=cfg.regularized_time_kind, initial="first",
                    )
                    cell_t = 0.5 * (raw_t[:, :-1] + raw_t[:, 1:])
                    rows.append(cell_t.reshape(-1)[self.mask_flat])
                self.time_mode_banks[mode] = np.vstack(rows)
            # Backward-compatible aggregate bank on the declared grid.
            for a in self.alpha_grid:
                t_cols.append(self._target_from_mode(float(a), infer_alpha_mode(float(a))))
            if verbose:
                print(f"[{self.data.name}] precomputing regularized spatial derivatives: {len(self.beta_grid)} orders")
            x_cols = []
            for b in _tqdm(self.beta_grid, enabled=verbose and cfg.progress, desc="space orders", leave=cfg.progress_leave):
                raw_x = regularized_space_derivative(
                    U,
                    float(b),
                    self.data.dx,
                    lam=cfg.lam_x,
                    kind=cfg.regularized_space_kind,
                    side=cfg.regularized_space_side,
                    initial="first",
                )
                cell_x = 0.5 * (raw_x[:-1, :] + raw_x[1:, :])
                x_cols.append(cell_x.reshape(-1)[self.mask_flat])
        elif cfg.backend == "spectral_l1":
            mask = self._cell_mask(U.shape)
            self.mask_flat = mask.reshape(-1)
            self.u_flat = U.reshape(-1)[self.mask_flat]
            t_cols = []
            if verbose:
                print(f"[{self.data.name}] precomputing branch-aware Caputo-L1 time derivatives")
            self.alpha_mode_grids = {
                mode: branch_grid(self.alpha_grid, mode, branch_epsilon=self.alpha_branch_epsilon)
                for mode in self.alpha_modes
            }
            for mode, grid_mode in self.alpha_mode_grids.items():
                rows = []
                for a in _tqdm(grid_mode, enabled=verbose and cfg.progress, desc=f"time {mode}", leave=cfg.progress_leave):
                    raw_t = caputo_l1_time(U, float(a), self.data.dt)
                    rows.append(raw_t.reshape(-1)[self.mask_flat])
                self.time_mode_banks[mode] = np.vstack(rows)
            for a in self.alpha_grid:
                t_cols.append(self._target_from_mode(float(a), infer_alpha_mode(float(a))))
            x_cols = []
            if verbose:
                print(f"[{self.data.name}] precomputing spectral spatial derivatives: {len(self.beta_grid)} orders")
            for b in _tqdm(self.beta_grid, enabled=verbose and cfg.progress, desc="space orders", leave=cfg.progress_leave):
                raw_x = spectral_space_derivative(U, float(b), self.data.Lx, riesz=cfg.spectral_riesz)
                x_cols.append(raw_x.reshape(-1)[self.mask_flat])
        else:  # pragma: no cover
            raise ValueError(f"unknown backend {cfg.backend!r}")
        self.time_bank = np.vstack(t_cols)
        self.space_bank = np.vstack(x_cols)
        if verbose:
            print(f"[{self.data.name}] feature bank ready: {self.n_points} points in {time.time() - t0:.2f}s")

    @property
    def n_points(self) -> int:
        """Number of valid aligned spacetime samples in the feature bank.
        
        This is the row count used in regression. It may be smaller than
        ``data.U.size`` because of trimming and, for regularized derivatives, because
        ``np.diff``-like outputs have one fewer point along a differentiated axis.
        """
        if self.u_flat is None:
            raise RuntimeError("call precompute() first")
        return int(self.u_flat.size)

    def available_alpha_modes(self) -> tuple[TemporalAlphaMode, ...]:
        """Temporal Caputo modes admitted by the declared search interval."""
        if self.alpha_modes:
            return tuple(self.alpha_modes)  # type: ignore[return-value]
        return (infer_alpha_mode(float(np.mean(self.alpha_grid))),)

    def alpha_grid_for_mode(self, mode: TemporalAlphaMode) -> NDArray[np.float64]:
        if mode in self.alpha_mode_grids:
            return self.alpha_mode_grids[mode]
        return branch_grid(self.alpha_grid, mode, branch_epsilon=self.alpha_branch_epsilon)

    def alpha_bounds(self, mode: TemporalAlphaMode) -> tuple[float, float]:
        return alpha_bounds_for_mode(self.alpha_grid, mode, branch_epsilon=self.alpha_branch_epsilon)

    def _target_from_mode(self, alpha: float, mode: TemporalAlphaMode) -> NDArray[np.float64]:
        if mode not in self.time_mode_banks:
            raise RuntimeError(f"time mode {mode!r} was not precomputed")
        grid = self.alpha_grid_for_mode(mode)
        bank = self.time_mode_banks[mode]
        if mode == "integer":
            return bank[0]
        return self._interp(float(alpha), grid, bank)

    def target(self, alpha: float, alpha_mode: str | None = None) -> NDArray[np.float64]:
        """Return the aligned LHS vector without crossing a Caputo branch.

        ``alpha=1`` is an exact integer candidate. Fractional values are
        interpolated only within their subunit or superunit branch.
        """
        if self.time_bank is None:
            raise RuntimeError("call precompute() first")
        mode = model_mode(float(alpha), alpha_mode)
        return self._target_from_mode(float(alpha), mode)

    def spatial(self, beta: float) -> NDArray[np.float64]:
        """Return the aligned spatial derivative vector ``D_x^beta u``.
        
        The returned array has shape ``(bank.n_points,)``. Multiply it only by other
        bank-aligned vectors, such as ``bank.u_power(p)``. Do not multiply it directly
        by ``data.U`` because the shapes/alignments may differ.
        """
        if self.space_bank is None:
            raise RuntimeError("call precompute() first")
        return self._interp(float(beta), self.beta_grid, self.space_bank)

    def u_power(self, p: int) -> NDArray[np.float64]:
        """Return the aligned nonlinear factor ``u**p``.
        
        ``p=0`` returns ones, so ``bank.u_power(0) * bank.spatial(beta)`` represents a
        pure derivative term. ``p=2`` gives the aligned factor needed for
        ``u^2 D_x^beta u``. Results are cached by power.
        """
        if self.u_flat is None:
            raise RuntimeError("call precompute() first")
        p = int(p)
        if p not in self.u_power_cache:
            if p == 0:
                self.u_power_cache[p] = np.ones_like(self.u_flat)
            elif self.config.power_mode == "positive":
                self.u_power_cache[p] = np.maximum(self.u_flat, 0.0) ** p
            else:
                self.u_power_cache[p] = self.u_flat**p
        return self.u_power_cache[p]

    def library(self, p_tuple: Sequence[int], beta_tuple: Sequence[float]) -> NDArray[np.float64]:
        """Build a design matrix for specific candidate terms.
        
        Parameters
        ----------
        p_tuple:
            Powers ``p_j`` in terms ``u**p_j D_x^beta_j u``.
        beta_tuple:
            Spatial derivative orders ``beta_j``. Must have the same length as
            ``p_tuple``.
        
        Returns
        -------
        Theta:
            Matrix with shape ``(bank.n_points, len(p_tuple))``. Column ``j`` is
            ``bank.u_power(p_j) * bank.spatial(beta_j)``.
        """
        if len(p_tuple) != len(beta_tuple):
            raise ValueError("p_tuple and beta_tuple must have the same length")
        cols = [self.u_power(int(p)) * self.spatial(float(beta)) for p, beta in zip(p_tuple, beta_tuple)]
        return np.column_stack(cols) if cols else np.zeros((self.n_points, 0))


def train_val_split(n: int, val_fraction: float, seed: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Create a deterministic train/validation split over bank rows.
    
    The split is by flattened spacetime sample index. Use the returned indices to
    fit coefficients on train rows and score candidate FPDEs on validation rows.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(int(n))
    n_val = max(1, int(round(float(val_fraction) * n)))
    return np.sort(perm[n_val:]).astype(np.int64), np.sort(perm[:n_val]).astype(np.int64)


class ParetoFDEOptimizer:
    """Optimizer for fixed-cardinality FPDE structures.
    
    This class owns the regression/evaluation logic after a ``FractionalFeatureBank``
    has been precomputed. It can evaluate one candidate equation, anchor a DE search
    with a coarse grid, and optimize continuous derivative orders for a fixed
    ``p_tuple``.
    
    The optimizer does not decide how many terms should be active. That is handled
    by ``support_size_sweep`` and Pareto selection.
    """
    def __init__(self, bank: FractionalFeatureBank, train_idx: NDArray[np.int64], val_idx: NDArray[np.int64], config: DiscoveryConfig):
        """Store the feature bank, train/validation indices, and config.
        
        Construct this after ``bank.precompute()`` and ``train_val_split``. Most users
        call the higher-level ``run_pareto_discovery`` wrapper instead.
        """
        self.bank = bank
        self.train_idx = np.asarray(train_idx, dtype=np.int64)
        self.val_idx = np.asarray(val_idx, dtype=np.int64)
        self.config = config

    def _fit(self, X: NDArray[np.float64], y: NDArray[np.float64], *, assume_finite: bool = False) -> NDArray[np.float64]:
        """Fit linear coefficients by ridge least squares on finite rows.
        
        This is the variable-projection step: for any proposed nonlinear orders
        ``(alpha, beta_1, ..., beta_c)``, the best linear coefficients ``xi`` are solved
        analytically rather than evolved by DE.  A Tikhonov ridge on the
        column-normalised design (``config.ridge``) keeps nearly collinear
        fractional columns from producing large cancelling coefficients.
        """
        if assume_finite:
            Xf, yf = X, y
        else:
            finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if not np.any(finite):
                return np.zeros(X.shape[1])
            Xf, yf = X[finite], y[finite]
        lam = float(getattr(self.config, "ridge", 1e-3))
        if lam <= 0.0:
            coef, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
            return coef
        s = np.linalg.norm(Xf, axis=0); s[s == 0.0] = 1.0
        Xn = Xf / s
        coef_n = np.linalg.solve(Xn.T @ Xn + lam * np.eye(Xn.shape[1]), Xn.T @ yf)
        return coef_n / s

    @staticmethod
    def _mse_rel(X: NDArray[np.float64], y: NDArray[np.float64], coef: NDArray[np.float64], *, assume_finite: bool = False) -> tuple[float, float]:
        """Compute MSE and variance-normalized MSE for a fixed coefficient vector.
        
        Rows containing NaN/inf in the target or library are ignored. The relative MSE
        divides by NumPy's empirical target variance with ``ddof=0`` (divisor
        equal to the number of retained rows), allowing comparison across
        datasets with different derivative scales.
        """
        if assume_finite:
            resid = y - X @ coef
            mse = float(np.mean(resid**2))
            return mse, float(mse / (np.var(y) + EPS))
        finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if not np.any(finite):
            return float("inf"), float("inf")
        resid = y[finite] - X[finite] @ coef
        mse = float(np.mean(resid**2))
        return mse, float(mse / (np.var(y[finite]) + EPS))

    def _snap_betas(self, beta_tuple: Sequence[float]) -> tuple[float, ...]:
        """Resolve near-zero orders to the identity and keep derivatives strictly positive.

        The identity operator ($\\beta=0$) and the signed Riesz operator have
        opposite limits as $\\beta\\downarrow0$, so linearly interpolating the
        precomputed order bank across $\\beta=0$ would blend them into a spurious
        near-zero column.  We therefore treat the identity as a distinct discrete
        candidate at exactly $\\beta=0$ and restrict derivative orders to the
        strictly positive grid: any proposed order below the first positive grid
        point (or ``config.beta_min_positive`` if larger) is snapped to $0$, so
        the two operator families are never mixed by interpolation.
        """
        grid = np.asarray(self.bank.beta_grid, dtype=float)
        has_identity = bool(np.any(np.isclose(grid, 0.0)))
        if not has_identity:
            return tuple(float(b) for b in beta_tuple)
        positive = grid[grid > 1e-9]
        thr = float(positive[0]) if positive.size else 0.0
        cfg_thr = float(getattr(self.config, "beta_min_positive", 0.0) or 0.0)
        thr = max(thr, cfg_thr)
        return tuple(0.0 if float(b) < thr else float(b) for b in beta_tuple)

    def evaluate(self, alpha: float, p_tuple: Sequence[int], beta_tuple: Sequence[float], *, alpha_mode: str | None = None) -> PDEModel:
        """Evaluate one FPDE candidate without running DE.
        
        Parameters
        ----------
        alpha:
            Temporal derivative order on the LHS.
        p_tuple, beta_tuple:
            RHS structure ``u**p_j D_x**beta_j u``.
        
        Returns
        -------
        PDEModel
            A fully scored candidate with least-squares coefficients,
            train/validation errors, heuristic AIC/BIC-type scores, and a
            formatted equation.
        
        This is the best method for debugging one manually chosen equation.
        """
        p_tuple = tuple(int(p) for p in p_tuple)
        beta_tuple = self._snap_betas(beta_tuple)
        c = len(p_tuple)
        resolved_alpha_mode = model_mode(float(alpha), alpha_mode)
        try:
            y = self.bank.target(alpha, alpha_mode=resolved_alpha_mode)
        except TypeError:  # backward-compatible custom banks used in tests/examples
            y = self.bank.target(alpha)
        X = self.bank.library(p_tuple, beta_tuple)
        # The precomputed spectral/weak-row tables are finite in the publication
        # experiments.  Check this once per candidate instead of rebuilding the
        # same row mask independently in the fit, training score, and validation
        # score.  The non-finite fallback is unchanged.
        finite_all = bool(np.isfinite(y).all() and np.isfinite(X).all())
        Xtr, ytr = X[self.train_idx], y[self.train_idx]
        Xva, yva = X[self.val_idx], y[self.val_idx]
        coef = self._fit(Xtr, ytr, assume_finite=finite_all)
        train_mse, train_rel = self._mse_rel(Xtr, ytr, coef, assume_finite=finite_all)
        val_mse, val_rel = self._mse_rel(Xva, yva, coef, assume_finite=finite_all)
        k_eff = c + (0 if resolved_alpha_mode == "integer" else 1) + c
        n_val = max(2, len(self.val_idx))
        # These are AIC/BIC-type heuristics rather than conventional likelihood
        # comparisons across every alpha: changing alpha changes the target
        # D_t^alpha u.  The dimensionless validation criterion remains the
        # principal objective for DE, Pareto dominance, elbow selection, and
        # stopping.  The published raw-MSE formulas are retained unchanged.
        aic = float(n_val * math.log(max(val_mse, np.finfo(float).tiny)) + 2.0 * k_eff)
        bic = float(n_val * math.log(max(val_mse, np.finfo(float).tiny)) + math.log(n_val) * k_eff)
        return PDEModel(
            c=c,
            alpha=float(alpha),
            p_tuple=p_tuple,
            beta_tuple=beta_tuple,
            coefficients=coef,
            train_mse=train_mse,
            val_mse=val_mse,
            train_rel_mse=train_rel,
            val_rel_mse=val_rel,
            aic=aic,
            bic=bic,
            objective=math.log10(max(val_rel, np.finfo(float).tiny)),
            n_train=int(len(self.train_idx)),
            n_val=int(len(self.val_idx)),
            backend="pareto-de",
            alpha_mode=resolved_alpha_mode,
        ).canonicalized()

    def duplicate_penalty(self, p_tuple: Sequence[int], beta_tuple: Sequence[float]) -> float:
        """Penalty for nearly duplicate terms with the same power and close beta values.
        
        Fractional derivative columns with nearby orders can be highly collinear. This
        small penalty discourages DE from using two terms that merely split one physical
        operator into adjacent beta values.
        """
        penalty = 0.0
        gap_min = float(self.config.duplicate_beta_gap)
        for i in range(len(p_tuple)):
            for j in range(i + 1, len(p_tuple)):
                if int(p_tuple[i]) == int(p_tuple[j]):
                    gap = abs(float(beta_tuple[i]) - float(beta_tuple[j]))
                    if gap < gap_min:
                        penalty += float(self.config.duplicate_penalty) * (gap_min - gap) / max(gap_min, EPS)
        return penalty

    def _anchor_values(self, grid: NDArray[np.float64], n_interior: int = 5) -> list[float]:
        """Truth-agnostic coarse anchor values on the declared search interval.

        The anchors are the interval endpoints, ``n_interior`` uniformly spaced
        interior points, and the a-priori integer orders (1 and 2) that fall
        inside the interval.  No dataset-specific fractional values are used, so
        the anchor grid is independent of the benchmark ground truth; it only
        gives differential evolution a spread of reasonable starting points.
        """
        lo, hi = float(grid[0]), float(grid[-1])
        vals = list(np.linspace(lo, hi, max(2, int(n_interior))))
        vals += [v for v in (1.0, 2.0) if lo <= v <= hi]
        out: list[float] = []
        for v in sorted(float(x) for x in vals):
            if lo <= v <= hi and not any(abs(v - w) < 1e-9 for w in out):
                out.append(v)
        return out

    def anchor_start(self, p_tuple: Sequence[int], *, alpha_mode: TemporalAlphaMode) -> tuple[NDArray[np.float64], PDEModel]:
        """Find a truth-agnostic coarse start within one temporal mode."""
        p_tuple = tuple(int(p) for p in p_tuple)
        if alpha_mode == "integer":
            av = [1.0]
        else:
            try:
                agrid = self.bank.alpha_grid_for_mode(alpha_mode)
            except AttributeError:
                agrid = branch_grid(self.bank.alpha_grid, alpha_mode, branch_epsilon=float(getattr(self.config, "alpha_branch_epsilon", 1e-3)))
            av = self._anchor_values(np.asarray(agrid, dtype=float), n_interior=5)
            av = [a for a in av if infer_alpha_mode(a) == alpha_mode]
        bv = self._anchor_values(self.bank.beta_grid, n_interior=6)
        candidates = [(a, betas) for a in av for betas in itertools.product(bv, repeat=len(p_tuple))]
        max_anchor = 800
        if len(candidates) > max_anchor:
            rng = np.random.default_rng(4242 + 37 * len(p_tuple) + sum(p_tuple))
            idx = set(rng.choice(len(candidates), size=max_anchor, replace=False).tolist())
            candidates = [candidates[i] for i in sorted(idx)[:max_anchor]]
        best_rec: PDEModel | None = None
        best_z: NDArray[np.float64] | None = None
        for a, betas in candidates:
            if self.duplicate_penalty(p_tuple, betas) > 0:
                continue
            rec = self.evaluate(float(a), p_tuple, betas, alpha_mode=alpha_mode)
            score = rec.objective + self.duplicate_penalty(p_tuple, betas)
            if best_rec is None or score < best_rec.objective + self.duplicate_penalty(p_tuple, best_rec.beta_tuple):
                best_rec = rec
                best_z = np.array(([float(a)] if alpha_mode != "integer" else []) + list(map(float, betas)))
        if best_rec is None or best_z is None:
            a0 = 1.0 if alpha_mode == "integer" else float(np.mean(av))
            best_z = np.array(([a0] if alpha_mode != "integer" else []) + [float(np.mean(self.bank.beta_grid))] * len(p_tuple))
            best_rec = self.evaluate(a0, p_tuple, best_z[-len(p_tuple):], alpha_mode=alpha_mode)
        best_rec.backend = f"anchor-grid:{alpha_mode}"
        return best_z, best_rec.canonicalized()

    def _optimize_fixed_pattern_mode(self, p_tuple: Sequence[int], *, seed: int, alpha_mode: TemporalAlphaMode) -> PDEModel:
        p_tuple = tuple(int(p) for p in p_tuple)
        beta_bounds = [(float(self.bank.beta_grid[0]), float(self.bank.beta_grid[-1]))] * len(p_tuple)
        if alpha_mode == "integer":
            bounds = beta_bounds
        else:
            try:
                a_bounds = self.bank.alpha_bounds(alpha_mode)
            except AttributeError:
                a_bounds = alpha_bounds_for_mode(self.bank.alpha_grid, alpha_mode, branch_epsilon=float(getattr(self.config, "alpha_branch_epsilon", 1e-3)))
            bounds = [a_bounds] + beta_bounds
        x0, anchor = self.anchor_start(p_tuple, alpha_mode=alpha_mode)
        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        x0 = np.minimum(np.maximum(np.asarray(x0, dtype=float), lower), upper)

        def decode(z: Sequence[float]) -> tuple[float, Sequence[float]]:
            if alpha_mode == "integer":
                return 1.0, z
            return float(z[0]), z[1:]

        def obj(z: Sequence[float]) -> float:
            alpha, betas = decode(z)
            return self.evaluate(alpha, p_tuple, betas, alpha_mode=alpha_mode).objective + self.duplicate_penalty(p_tuple, betas)

        if not bounds:
            return anchor
        kwargs = dict(
            func=obj, bounds=bounds, maxiter=int(self.config.maxiter),
            popsize=int(self.config.popsize), seed=int(seed), tol=1e-5,
            polish=bool(self.config.polish), workers=1, updating="immediate",
        )
        with _tqdm_context(
            enabled=bool(self.config.progress_de), total=int(self.config.maxiter),
            desc=f"DE {alpha_mode} p={p_tuple}", leave=bool(self.config.progress_leave),
        ) as de_bar:
            def _callback(*_args, **_kwargs):
                if de_bar is not None:
                    de_bar.update(1)
                return False
            if bool(self.config.progress_de):
                kwargs["callback"] = _callback
            try:
                res = differential_evolution(**kwargs, x0=x0)
            except (TypeError, ValueError):
                res = differential_evolution(**kwargs)
        alpha, betas = decode(res.x)
        de = self.evaluate(alpha, p_tuple, betas, alpha_mode=alpha_mode)
        de.backend = f"scipy-de:{alpha_mode}"
        return (de if de.objective < anchor.objective else anchor).canonicalized()

    def optimize_fixed_pattern(self, p_tuple: Sequence[int], *, seed: int) -> PDEModel:
        """Optimise each admitted Caputo mode separately and compare minima."""
        try:
            modes = self.bank.available_alpha_modes()
        except AttributeError:
            modes = available_alpha_modes(
                self.bank.alpha_grid,
                branch_epsilon=float(getattr(self.config, "alpha_branch_epsilon", 1e-3)),
            )
        candidates = [
            self._optimize_fixed_pattern_mode(p_tuple, seed=int(seed) + 1009 * i, alpha_mode=mode)
            for i, mode in enumerate(modes)
        ]
        return min(candidates, key=lambda m: (m.objective, m.val_rel_mse)).canonicalized()


def generate_p_patterns(c: int, p_values: Sequence[int], *, include: Sequence[tuple[int, ...]] = (), max_patterns: int | None = None, seed: int = 0) -> list[tuple[int, ...]]:
    """Generate unordered RHS power patterns for a support size.
    
    For example, with ``p_values=(0,1,2)`` and ``c=2`` this returns patterns such as
    ``(0,0)``, ``(0,1)``, and ``(1,2)`` while avoiding duplicate permutations like
    both ``(0,1)`` and ``(1,0)``. Important patterns can be forced in with
    ``include``.
    """
    combos = list(itertools.combinations_with_replacement(tuple(int(p) for p in p_values), int(c)))
    out: list[tuple[int, ...]] = []
    for pat in include:
        pat = tuple(int(v) for v in pat)
        if len(pat) == c and all(v in p_values for v in pat) and pat not in out:
            out.append(pat)
    for pat in combos:
        if pat not in out:
            out.append(pat)
    if max_patterns is not None and len(out) > max_patterns:
        forced = [p for p in out if p in include]
        rest = [p for p in out if p not in forced]
        n = max(0, max_patterns - len(forced))
        rng = np.random.default_rng(seed)
        if n < len(rest):
            rest = [rest[i] for i in sorted(rng.choice(len(rest), size=n, replace=False))]
        out = (forced + rest)[:max_patterns]
    return out


def support_size_sweep(
    optimizer: ParetoFDEOptimizer,
    config: DiscoveryConfig,
    verbose: bool = True,
) -> tuple[list[PDEModel], dict[int, PDEModel], dict]:
    """Run the proposed outer sweep over RHS support size.
    
    For each ``c = 1, 2, ..., cmax`` this function generates candidate power
    patterns, optimizes each fixed-cardinality model with DE, and stores the best
    model for that ``c``. It also implements automatic stopping when validation
    improvement plateaus or the Pareto-selected model remains stable after adding a
    larger support size.
    
    Returns ``(all_models, best_by_c, stop_info)``.
    """
    important = {
        1: [(0,)],
        2: [(0, 0), (0, 1), (1, 1)],
        3: [(0, 0, 0), (0, 0, 1), (0, 1, 1)],
        4: [(0, 0, 0, 0), (0, 0, 0, 1)],
        5: [(0, 0, 0, 0, 0), (0, 0, 0, 0, 1)],
    }
    all_models: list[PDEModel] = []
    best_by_c: dict[int, PDEModel] = {}
    stop_info = {
        "auto_stop_enabled": bool(config.auto_stop),
        "stopped_early": False,
        "stop_c": None,
        "reason": "completed_cmax",
        "history": [],
    }
    plateau_count = 0
    selection_stability_count = 0
    prev_best_rel: float | None = None

    for c in range(1, int(config.cmax) + 1):
        pats = generate_p_patterns(c, config.p_values, include=important.get(c, ()), max_patterns=config.max_patterns_per_c, seed=config.seed + c * 1009)
        if verbose:
            print(f"\n=== support size c={c}: {len(pats)} p-pattern(s) ===")
        pat_iter = _tqdm(
            list(enumerate(pats, start=1)),
            enabled=verbose and config.progress,
            desc=f"c={c} patterns",
            total=len(pats),
            leave=config.progress_leave,
        )
        for j, pat in pat_iter:
            rec = optimizer.optimize_fixed_pattern(pat, seed=config.seed + 7919 * c + 103 * j)
            all_models.append(rec)
            if c not in best_by_c or rec.val_rel_mse < best_by_c[c].val_rel_mse:
                best_by_c[c] = rec
            if verbose:
                _progress_write(f"c={c} p={pat} val_rel={rec.val_rel_mse:.3e} {rec.equation(digits=4)}", enabled=config.progress)
        if verbose:
            _progress_write(f"best c={c}: {best_by_c[c].equation(digits=5)}", enabled=config.progress)

        curr = float(best_by_c[c].val_rel_mse)
        rel_improvement = None
        log10_improvement = None
        plateau = False
        if prev_best_rel is not None and np.isfinite(prev_best_rel) and np.isfinite(curr):
            rel_improvement = float((prev_best_rel - curr) / max(prev_best_rel, EPS))
            log10_improvement = float(math.log10(max(prev_best_rel, EPS)) - math.log10(max(curr, EPS)))
            plateau = (
                rel_improvement < float(config.auto_stop_rel_improvement)
                and log10_improvement < float(config.auto_stop_log10_improvement)
            )
            plateau_count = plateau_count + 1 if plateau else 0

        # Selection-stability stopping: after seeing the curve up to the current
        # support size, apply the same model-selection rule that will be used in
        # the final report.  If adding larger c values does not move the selected
        # model to a higher c, the extra capacity is probably just fitting
        # collinear derivative columns.
        selected_c_so_far = None
        selection_stable = False
        if bool(config.auto_stop_use_selection_stability) and len(best_by_c) >= 2:
            try:
                current_pareto = pareto_front(best_by_c)
                selected_so_far = select_model(current_pareto, config.selection, config.sparse_relaxed_tol)
                selected_c_so_far = int(selected_so_far.c)
                selection_stable = selected_c_so_far < int(c)
                selection_stability_count = selection_stability_count + 1 if selection_stable else 0
            except Exception:
                selected_c_so_far = None
                selection_stable = False

        stop_info["history"].append({
            "c": int(c),
            "best_val_rel_mse": curr,
            "relative_improvement_from_previous_c": rel_improvement,
            "log10_improvement_from_previous_c": log10_improvement,
            "plateau": bool(plateau),
            "plateau_count": int(plateau_count),
            "selected_c_so_far": selected_c_so_far,
            "selection_stable": bool(selection_stable),
            "selection_stability_count": int(selection_stability_count),
        })

        should_stop_by_plateau = (
            plateau_count >= int(config.auto_stop_patience)
        )
        should_stop_by_selection = (
            bool(config.auto_stop_use_selection_stability)
            and selection_stability_count >= int(config.auto_stop_selection_patience)
        )
        if (
            bool(config.auto_stop)
            and c >= int(config.auto_stop_min_c)
            and (should_stop_by_plateau or should_stop_by_selection)
            and c < int(config.cmax)
        ):
            if should_stop_by_selection:
                reason = (
                    "model-selection stability: adding larger support did not "
                    f"move the selected Pareto model beyond c={selected_c_so_far} "
                    f"for {selection_stability_count} consecutive step(s)"
                )
            else:
                reason = (
                    "validation improvement plateaued: "
                    f"relative improvement < {config.auto_stop_rel_improvement:g} and "
                    f"log10 improvement < {config.auto_stop_log10_improvement:g} "
                    f"for {plateau_count} consecutive step(s)"
                )
            stop_info.update({
                "stopped_early": True,
                "stop_c": int(c),
                "reason": reason,
            })
            if verbose:
                _progress_write(f"auto-stop at c={c}: {stop_info['reason']}", enabled=config.progress)
            break

        if prev_best_rel is None or curr < prev_best_rel:
            prev_best_rel = curr

    return all_models, dict(sorted(best_by_c.items())), stop_info


def pareto_front(best_by_c: dict[int, PDEModel]) -> dict[int, PDEModel]:
    """Return non-dominated models in the normalized-error/complexity plane.
    
    A model is dominated if another model has no larger variance-normalized
    validation MSE (``val_rel_mse``) and no larger support size, with at least
    one strict improvement.  Using the dimensionless criterion is important
    because the target ``D_t^alpha u`` changes scale with ``alpha``.  The result
    is the empirical Pareto front used for complexity selection.
    """
    keep: dict[int, PDEModel] = {}
    for ci, ri in best_by_c.items():
        dominated = False
        for cj, rj in best_by_c.items():
            if ci == cj:
                continue
            if (rj.val_rel_mse <= ri.val_rel_mse and cj <= ci) and (
                rj.val_rel_mse < ri.val_rel_mse or cj < ci
            ):
                dominated = True
                break
        if not dominated:
            keep[ci] = ri
    return dict(sorted(keep.items()))


def term_contribution_ratios(
    optimizer: ParetoFDEOptimizer,
    model: PDEModel,
) -> NDArray[np.float64]:
    """Return scale-aware RHS contribution ratios for the selected terms.

    For a fitted equation ``b approx Theta xi``, the contribution of term ``j``
    is measured by

        ||xi_j theta_j||_2 / ||Theta xi||_2.

    This is preferable to raw coefficient magnitude because candidate-library
    columns can have very different scales, especially across fractional orders
    and nonlinear powers.  The computation uses the optimizer's feature bank when
    available.  If a minimal dummy optimizer without a bank is passed in tests,
    the function falls back to normalized coefficient magnitudes.
    """
    rec = model.canonicalized()
    coefs = np.asarray(rec.coefficients, dtype=float)
    if coefs.size == 0:
        return np.zeros(0, dtype=float)

    try:
        X = optimizer.bank.library(rec.p_tuple, rec.beta_tuple)
        X = np.asarray(X, dtype=float)
        finite = np.all(np.isfinite(X), axis=1)
        if not np.any(finite):
            raise ValueError("no finite rows")
        Xf = X[finite]
        contributions = Xf * coefs[None, :]
        fitted = np.sum(contributions, axis=1)
        denom = float(np.linalg.norm(fitted))
        if not np.isfinite(denom) or denom <= EPS:
            denom = float(np.sum(np.linalg.norm(contributions, axis=0))) + EPS
        ratios = np.linalg.norm(contributions, axis=0) / max(denom, EPS)
        return np.asarray(ratios, dtype=float)
    except Exception:
        scale = float(np.max(np.abs(coefs)))
        return np.abs(coefs) / max(scale, EPS)


def prune_inactive_selected_terms(
    optimizer: ParetoFDEOptimizer,
    model: PDEModel,
    *,
    contribution_tol: float = 1e-4,
    abs_tol: float = 1e-10,
    coef_rel_fallback_tol: float = 1e-3,
    rel_tol: float | None = None,
) -> PDEModel:
    """Remove numerically inactive terms from a selected model and refit.

    Broad paper candidate libraries include competing nonlinear candidate terms such as
    ``u D_x^beta u`` and ``u^2 D_x^beta u``.  In clean or low-noise data, a
    higher-cardinality candidate can sometimes reduce validation error by a tiny
    numerical amount while giving the extra term a negligible fitted contribution.
    Reporting that term as part of the discovered equation is misleading: the
    active symbolic structure is the lower-cardinality equation.

    This function is a post-selection cleanup, not an oracle.  It uses only the
    fitted model and the same candidate-library columns used by the optimizer.
    Term ``j`` is dropped if its absolute coefficient is numerically zero or if

        ||xi_j theta_j||_2 / ||Theta xi||_2 <= contribution_tol.

    If feature columns are unavailable, the fallback rule uses normalized
    coefficient magnitudes.  The remaining structure is re-evaluated/refit by the
    same optimizer.  The rule is applied equally to vanilla and weak Pareto-DE
    methods.
    """
    if rel_tol is not None:
        # Backward-compatible alias used by older notebooks/scripts.
        coef_rel_fallback_tol = float(rel_tol)
    rec = model.canonicalized()
    coefs = np.asarray(rec.coefficients, dtype=float)
    if coefs.size == 0:
        return model
    ratios = term_contribution_ratios(optimizer, rec)
    if ratios.size != coefs.size:
        scale = float(np.max(np.abs(coefs)))
        ratios = np.abs(coefs) / max(scale, EPS)
    keep = [
        i
        for i, c in enumerate(coefs)
        if abs(float(c)) > float(abs_tol)
        and float(ratios[i]) > float(contribution_tol)
    ]
    if len(keep) == len(coefs) or len(keep) == 0:
        return model
    p_tuple = tuple(int(rec.p_tuple[i]) for i in keep)
    beta_tuple = tuple(float(rec.beta_tuple[i]) for i in keep)
    try:
        pruned = optimizer.evaluate(
            float(rec.alpha), p_tuple, beta_tuple, alpha_mode=rec.alpha_mode
        ).canonicalized()
    except TypeError:
        pruned = optimizer.evaluate(float(rec.alpha), p_tuple, beta_tuple).canonicalized()
    pruned.backend = model.backend
    return pruned

def select_model(candidates: dict[int, PDEModel], rule: SelectionRule = "elbow", sparse_relaxed_tol: float = 0.08) -> PDEModel:
    """Select one model from the Pareto candidates.
    
    Rules
    -----
    ``elbow``:
        Choose the concave knee of the error-complexity curve. Support sizes and
        ``-log10`` validation error are min-max normalised, and the selected size
        maximises the signed vertical excess above the chord joining the first
        and last points. If no interior point lies above the chord, choose the
        smallest model.
    ``bic`` / ``aic``:
        Choose the model with the smallest heuristic BIC/AIC-type score.  These
        raw-MSE diagnostics are not conventional fixed-response likelihood
        comparisons across different temporal orders.
    ``sparse_relaxed``:
        Choose the smallest model whose normalized validation error is within a
        tolerance of the best normalized validation error.
    """
    if not candidates:
        raise ValueError("empty candidate set")
    cs = np.array(sorted(candidates), dtype=int)
    models = [candidates[int(c)] for c in cs]
    if rule == "bic":
        return min(models, key=lambda m: m.bic)
    if rule == "aic":
        return min(models, key=lambda m: m.aic)
    if rule == "sparse_relaxed":
        best = min(models, key=lambda m: m.val_rel_mse)
        cutoff = best.val_rel_mse * (1.0 + float(sparse_relaxed_tol))
        feasible = [m for m in models if m.val_rel_mse <= cutoff]
        return min(feasible, key=lambda m: (m.c, m.val_rel_mse))
    if len(models) == 1:
        return models[0]
    if len(models) == 2:
        improvement = math.log10(models[0].val_rel_mse + EPS) - math.log10(models[1].val_rel_mse + EPS)
        return models[1] if improvement > 0.15 else models[0]
    x = (cs.astype(float) - cs.min()) / (cs.max() - cs.min() + EPS)
    y = np.array([math.log10(m.val_rel_mse + EPS) for m in models])
    benefit = -y
    yn = (benefit - benefit.min()) / (benefit.max() - benefit.min() + EPS)
    # The Pareto front is monotone, so its end points normalise to (0, 0)
    # and (1, 1). A genuine parsimony elbow is an interior point above this
    # chord, not merely far from it on either side. The signed score prevents a
    # convex ``anti-elbow`` below the chord from being selected.
    signed_excess = yn - x
    interior = np.arange(1, len(models) - 1, dtype=int)
    positive = interior[signed_excess[interior] > 0.0]
    if positive.size == 0:
        return models[0]
    best_i = int(positive[np.argmax(signed_excess[positive])])
    return models[best_i]


def run_pareto_discovery(data: GridDataset, config: DiscoveryConfig, output_dir: str | Path | None = None, verbose: bool = True, export_selected_fde: bool = True) -> dict:
    """High-level wrapper for the full proposed discovery framework.
    
    This is the function to call from notebooks and scripts. It precomputes the
    feature bank, creates the validation split, runs the support-size sweep, builds
    the Pareto front, selects the final model, and optionally writes results to
    ``summary.json``, ``all_models.csv``, and ``best_by_c.csv``.

    When ``output_dir`` is provided and ``export_selected_fde=True``, this also
    writes ``selected_fde.json``. That file is the bridge to the
    ``fractional_pinn`` package: it can be loaded as a ``ModelSpec`` for the
    PINN + Gauss--Jacobi refinement stage.
    """
    bank = FractionalFeatureBank(data, config)
    bank.precompute(verbose=verbose)
    train_idx, val_idx = train_val_split(bank.n_points, config.val_fraction, config.seed)
    opt = ParetoFDEOptimizer(bank, train_idx, val_idx, config)
    all_models, best_by_c, stop_info = support_size_sweep(opt, config, verbose=verbose)
    pareto = pareto_front(best_by_c)
    selected_raw = select_model(pareto, config.selection, config.sparse_relaxed_tol)
    selected = prune_inactive_selected_terms(
        opt,
        selected_raw,
        contribution_tol=config.selected_contribution_prune_tol,
        abs_tol=config.selected_coef_prune_abs_tol,
        coef_rel_fallback_tol=config.selected_coef_prune_rel_tol,
    )
    summary = {
        "dataset": data.name,
        "truth": data.truth,
        "config": config_to_dict(config),
        "best_by_c": {str(c): m.to_dict() for c, m in best_by_c.items()},
        "pareto": {str(c): m.to_dict() for c, m in pareto.items()},
        "selected": selected.to_dict(),
        "auto_stop": stop_info,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if export_selected_fde:
            selected_path = save_selected_fde(
                selected,
                output_dir / "selected_fde.json",
                dataset_name=data.name,
                config=config,
                metadata={"truth": data.truth, "selection": config.selection},
            )
            summary["selected_fde_path"] = str(selected_path)
        write_json(output_dir / "summary.json", summary)
        write_models_csv(output_dir / "all_models.csv", all_models)
        write_models_csv(output_dir / "best_by_c.csv", list(best_by_c.values()))
    return summary


def config_to_dict(config: DiscoveryConfig) -> dict:
    """Convert ``DiscoveryConfig`` to a JSON-friendly dictionary."""
    d = dict(config.__dict__)
    d["alpha_grid"] = [float(v) for v in config.alpha_grid]
    d["beta_grid"] = [float(v) for v in config.beta_grid]
    d["p_values"] = [int(v) for v in config.p_values]
    return d


def write_json(path: str | Path, obj: object) -> None:
    """Write a JSON file, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def write_models_csv(path: str | Path, models: Sequence[PDEModel]) -> None:
    """Write a list of ``PDEModel`` objects to a compact CSV table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fields = [
            "c", "alpha", "alpha_mode", "p_tuple", "beta_tuple", "coefficients",
            "train_mse", "val_mse", "train_rel_mse", "val_rel_mse", "objective",
            "full_data_rel_l2", "aic", "bic", "equation", "backend",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in models:
            d = m.to_dict()
            writer.writerow({
                "c": d["c"],
                "alpha": d["alpha"],
                "alpha_mode": d.get("alpha_mode", infer_alpha_mode(float(d["alpha"]))),
                "p_tuple": json.dumps(d["p_tuple"]),
                "beta_tuple": json.dumps(d["beta_tuple"]),
                "coefficients": json.dumps(d["coefficients"]),
                "train_mse": d["train_mse"],
                "val_mse": d["val_mse"],
                "train_rel_mse": d["train_rel_mse"],
                "val_rel_mse": d["val_rel_mse"],
                "objective": d["objective"],
                "full_data_rel_l2": d["full_data_rel_l2"],
                "aic": d["aic"],
                "bic": d["bic"],
                "equation": d["equation"],
                "backend": d["backend"],
            })
