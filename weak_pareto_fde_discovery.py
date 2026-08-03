"""Weak-form Pareto discovery for fractional PDE/FDE benchmarks.

This module integrates :mod:`fractional_weak_form` with the existing
support-size swept Pareto-DE optimizer.  It is deliberately API-compatible with
``FractionalFeatureBank`` where the optimizer expects ``target(alpha)`` and
``library(p_tuple, beta_tuple)`` methods, but the features are weak integral
features rather than pointwise strong-form derivatives.

For a candidate term ``u**p * L_x^beta u`` the weak feature is assembled as

    <u**p L_x^beta u, phi> = <L_x^beta u, u**p phi>
                         = <u, (L_x^beta)^*(u**p phi)>.

For ``p=0`` this reduces to the standard fixed-test adjoint feature
``<u, L_x^* phi>``. For ``p>0`` the exact discrete-adjoint identity makes the
feature algebraically equal to the test-function projection
``<u**p L_x u, phi>`` of the corresponding strong-form feature. Its benefit is
therefore averaging across a weak row, not complete removal of differentiation
from the noisy data path.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence
import csv
import json
import math
import time

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from denoising import DenoiseConfig, DenoiseMethod, denoise_spacetime_field
from fpde_datasets import GridDataset, add_multiplicative_uniform_noise
from fractional_weak_form import (
    EPS,
    FractionalOperatorSpec,
    SeparableWeakLibrary2D,
    adjoint_tests_1d,
    caputo_corrected_field,
    caputo_l1_adjoint_tests,
    gaussian_test_matrix,
    fourier_test_matrix,
    fractional_integral_adjoint_tests,
    periodic_riesz_on_tests,
    periodic_spectral_directional_adjoint_on_tests,
    refine_orders_local,
)
from pareto_fde_discovery import (
    DiscoveryConfig,
    ParetoFDEOptimizer,
    PDEModel,
    config_to_dict,
    pareto_front,
    prune_inactive_selected_terms,
    select_model,
    support_size_sweep,
    train_val_split,
    write_json,
    write_models_csv,
)
from selected_fde_io import save_selected_fde
from temporal_modes import (
    TemporalAlphaMode,
    alpha_bounds_for_mode,
    available_alpha_modes,
    branch_grid,
    infer_alpha_mode,
    model_mode,
)

SpaceWeakKind = Literal["riesz", "riemann_liouville", "grunwald_letnikov", "integer"]
TimeWeakKind = Literal["caputo", "riemann_liouville", "grunwald_letnikov", "integer"]
WeakTimeForm = Literal["derivative", "caputo_integral"]


def _interp(order: float, grid: NDArray[np.float64], bank: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linearly interpolate a precomputed order bank.

    Differential evolution proposes continuous fractional orders, but features
    are expensive to precompute.  ``grid`` contains the discrete orders, and
    ``bank[i]`` is the feature array at ``grid[i]``.  The return value has the
    same shape as one bank slice.
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


def _default_test_counts(nt: int, nx: int, *, budget: Literal["smoke", "standard", "paper"] = "standard") -> tuple[int, int]:
    """Choose the number of time/space weak test functions for a run budget.

    ``smoke`` is for notebooks and CI, ``standard`` is for exploratory runs,
    and ``paper`` uses more weak rows for publication-quality experiments.
    The budget affects numerical resolution only; it does not change the
    candidate equation class.
    """
    if budget == "smoke":
        return max(8, min(14, nt // 5)), max(10, min(24, nx // 4))
    if budget == "paper":
        return max(20, min(44, nt // 3)), max(32, min(80, nx // 2))
    return max(14, min(26, nt // 4)), max(20, min(48, nx // 3))


def weak_operator_kinds_for_config(config: DiscoveryConfig, dataset_name: str | None = None) -> tuple[TimeWeakKind, SpaceWeakKind]:
    """Choose weak operator definitions matching a dataset/config convention."""
    # Time: existing spectral_l1 backend uses Caputo-L1; regularized backend may
    # be Caputo/RL/GL according to config.regularized_time_kind.
    tk = str(getattr(config, "regularized_time_kind", "caputo")).lower()
    time_kind: TimeWeakKind = "caputo" if tk not in {"riemann_liouville", "grunwald_letnikov", "integer"} else tk  # type: ignore[assignment]
    if str(config.backend) == "spectral_l1":
        time_kind = "caputo"

    # Space: Riesz synthetic data should use the self-adjoint Riesz operator.
    # The paper FADE benchmark and bounded ADE control use one-sided GL/RL weak
    # operators as the closest weak analogue of the directional finite-domain
    # derivative used in the original candidate library.
    if bool(getattr(config, "spectral_riesz", False)):
        space_kind: SpaceWeakKind = "riesz"
    elif str(config.backend) == "spectral_l1":
        # The paper FADE benchmark is named ``tsfade_fft`` and the vanilla
        # backend evaluates D_x^beta with the periodic Fourier multiplier
        # (i k)^beta.  The weak library must use the exact discrete adjoint of
        # that same operator, not a one-sided finite-domain GL/RL adjoint.
        space_kind = "spectral_directional"
    else:
        sk = str(getattr(config, "regularized_space_kind", "riemann_liouville")).lower()
        if sk in {"caputo", "riemann_liouville"}:
            space_kind = "riemann_liouville"
        elif sk in {"grunwald_letnikov", "integer"}:
            space_kind = sk  # type: ignore[assignment]
        else:
            space_kind = "riemann_liouville"
    return time_kind, space_kind


class WeakFractionalFeatureBank:
    """Precompute weak target and RHS features for Pareto-DE.

    Purpose
    -------
    This is the weak-form replacement for ``FractionalFeatureBank``.  The
    vanilla feature bank creates pointwise derivative columns such as
    ``D_t^alpha u`` and ``D_x^beta u``.  This class instead creates rows of weak
    integrals, one row per tensor-product test function ``phi_k(t) psi_l(x)``.

    Mathematical contract
    ---------------------
    For a linear candidate term ``D_x^beta u``, the stored feature is
    ``<u, (D_x^beta)^* phi>``.  For the existing Pareto model term
    ``u**p * D_x^beta u``, the stored feature is
    ``<u, (D_x^beta)^*(u**p phi)>``.  Thus fractional derivatives are applied to
    smooth/data-weighted test functions rather than directly to noisy data.

    Main arguments
    --------------
    ``data`` and ``config``
        Clean gridded data plus the shared canonical candidate configuration.
        Noise is injected inside the feature bank using ``config.noise_percent``
        and ``config.seed`` so all methods see the same noisy realization.
    ``test_budget``
        ``smoke``/``standard``/``paper`` controls how many weak test functions
        are used.  It changes numerical resolution, not the candidate library.
    ``time_kind``/``space_kind``
        Optional overrides for Caputo/Riemann--Liouville/GL/Riesz conventions.
        If omitted, they are inferred from the dataset config.
    ``denoise``
        Optional preprocessing used only by the practical high-noise variant.
    ``time_form``
        ``derivative`` for the main weak PDE residual. With the ``spectral_l1``
        backend this uses the exact transpose of the discrete L1 matrix on the
        temporal tests.  For superunit orders that matrix is
        ``L1(alpha-1) @ D1``; its transpose has an endpoint-dominated weight
        pair and therefore treats the initial rate implicitly.
        ``caputo_integral`` is an optional Volterra residual; for orders above
        one it subtracts an initial slope estimated explicitly from the first
        two samples and is not used by the reported superunit experiment.
    """

    def __init__(
        self,
        data: GridDataset,
        config: DiscoveryConfig,
        *,
        time_tests: NDArray[np.float64] | None = None,
        space_tests: NDArray[np.float64] | None = None,
        test_budget: Literal["smoke", "standard", "paper"] = "standard",
        test_counts: tuple[int, int] | None = None,
        time_width: float | None = None,
        space_width: float | None = None,
        time_kind: TimeWeakKind | None = None,
        space_kind: SpaceWeakKind | None = None,
        space_side: Literal["left", "right", "symmetric"] | None = None,
        denoise: DenoiseConfig | None = None,
        time_form: WeakTimeForm = "derivative",
    ) -> None:
        U = np.asarray(data.U, dtype=float)
        U = add_multiplicative_uniform_noise(U, config.noise_percent, seed=config.seed)
        self.denoise_config = denoise or DenoiseConfig(method="none")
        U, self.denoise_metadata = denoise_spacetime_field(
            U,
            noise_percent=float(config.noise_percent),
            config=self.denoise_config,
        )
        if config.smooth_sigma_t > 0 or config.smooth_sigma_x > 0:
            U = gaussian_filter(U, sigma=(config.smooth_sigma_t, config.smooth_sigma_x), mode="nearest")
        self.data = data
        self.U = U
        self.config = config
        self.test_budget = test_budget
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
        nt, nx = U.shape
        if test_counts is not None:
            # Explicit override of the number of time/space test functions
            # (K = n_t * n_x weak rows); used by the K-sensitivity study.
            n_t, n_x = int(test_counts[0]), int(test_counts[1])
        else:
            n_t, n_x = _default_test_counts(nt, nx, budget=test_budget)
        if time_width is None:
            # Broad enough to smooth noise, narrow enough to give many independent rows.
            time_width = max(3.0 * data.dt, 0.070 * (float(data.t[-1]) - float(data.t[0]) + data.dt))
        if space_width is None:
            space_width = max(2.0 * data.dx, data.Lx / max(18.0, 0.75 * n_x))
        self.time_tests = np.asarray(
            time_tests if time_tests is not None else gaussian_test_matrix(data.t, centers=n_t, width=time_width, periodic=False),
            dtype=float,
        )
        auto_time_kind, auto_space_kind = weak_operator_kinds_for_config(config, data.name)
        if space_tests is not None:
            default_space_tests = space_tests
        elif auto_space_kind == "riesz" and len(self.beta_grid) and float(np.max(self.beta_grid)) > 2.4:
            # High-order periodic Riesz terms are sensitive to high-frequency
            # content.  Broad Gaussian windows intentionally smooth noise, but
            # they can also erase the high modes needed to distinguish, e.g.,
            # beta=0.55 from beta=2.80.  Use Fourier tests for those periodic
            # high-order Riesz search spaces; the Riesz adjoint is diagonal in
            # this basis and remains an exact weak projection.
            max_mode = min(max(8, n_x), max(1, data.x.size // 2 - 2))
            default_space_tests = fourier_test_matrix(data.x, max_mode=max_mode, include_constant=True)
        else:
            default_space_tests = gaussian_test_matrix(data.x, centers=n_x, width=space_width, periodic=True)
        self.space_tests = np.asarray(default_space_tests, dtype=float)
        self.weak = SeparableWeakLibrary2D(data.t, data.x, self.time_tests, self.space_tests)
        self.time_kind: TimeWeakKind = auto_time_kind if time_kind is None else time_kind
        self.space_kind: SpaceWeakKind = auto_space_kind if space_kind is None else space_kind
        self.time_form: WeakTimeForm = time_form
        if self.time_form == "caputo_integral" and self.time_kind not in {"caputo", "integer"}:
            raise ValueError("caputo_integral time_form is valid only for Caputo/integer time operators")
        self._active_alpha = float(self.alpha_grid[0]) if self.alpha_grid.size else 0.0
        self.space_side = str(config.regularized_space_side if space_side is None else space_side)
        if self.space_side not in {"left", "right", "symmetric"}:
            self.space_side = "left"
        self.time_bank: NDArray[np.float64] | None = None
        self.time_discretization = (
            "caputo_volterra_integral" if self.time_form == "caputo_integral"
            else "caputo_l1_adjoint" if (str(config.backend) == "spectral_l1" and self.time_kind == "caputo")
            else "fractional_weak_adjoint"
        )
        # space_feature_cache[p] has shape (n_beta, n_weak_rows)
        self.space_feature_cache: dict[int, NDArray[np.float64]] = {}
        self.u_power_cache: dict[int, NDArray[np.float64]] = {}
        # Compatibility attributes for existing diagnostics.
        self.mask_flat = np.ones(self.weak.n_rows, dtype=bool)
        self.u_flat = np.ones(self.weak.n_rows, dtype=float)

    @property
    def n_points(self) -> int:
        """Number of weak regression rows.

        Each row corresponds to one tensor-product test function
        ``rho_k(t) psi_l(x)``.  This is the weak-form analogue of the number
        of spacetime points used by a vanilla strong-form regression.
        """
        return self.weak.n_rows

    def _time_spec(self, alpha: float) -> FractionalOperatorSpec:
        kind = "caputo" if self.time_kind == "integer" else self.time_kind
        return FractionalOperatorSpec(kind=kind, order=float(alpha), axis="t", side="left")

    def _space_spec(self, beta: float) -> FractionalOperatorSpec:
        kind = self.space_kind
        return FractionalOperatorSpec(kind=kind, order=float(beta), axis="x", side=self.space_side)  # type: ignore[arg-type]

    def precompute(self, verbose: bool = True) -> None:
        """Precompute weak target vectors and common RHS feature columns.

        The alpha grid is used for left-hand-side weak targets.  The beta grid
        is used for right-hand-side weak spatial features.  The common linear
        case ``p=0`` is precomputed immediately; nonlinear/data-weighted terms
        are computed lazily when requested by ``library``.
        """
        t0 = time.time()
        if verbose:
            print(
                f"[{self.data.name}] precomputing weak features: "
                f"{len(self.alpha_grid)} alpha orders, {len(self.beta_grid)} beta orders, "
                f"{self.n_points} weak rows"
            )
        self.alpha_mode_grids = {
            mode: branch_grid(self.alpha_grid, mode, branch_epsilon=self.alpha_branch_epsilon)
            for mode in self.alpha_modes
        }
        for mode, grid_mode in self.alpha_mode_grids.items():
            self.time_mode_banks[mode] = np.vstack([self._target_direct(float(a)) for a in grid_mode])
        # Backward-compatible aggregate bank; branch-aware target() does not use
        # interpolation across alpha=1.
        self.time_bank = np.vstack([
            self._target_from_mode(float(a), infer_alpha_mode(float(a))) for a in self.alpha_grid
        ])
        # Precompute p=0 for the derivative form. In the optional Caputo-integral
        # diagnostic, alpha also enters the RHS and the feature is evaluated
        # exactly at the active alpha to avoid cross-branch interpolation.
        if self.time_form != "caputo_integral":
            self.space_feature_cache[0] = np.vstack([self._space_feature_direct(0, float(b)) for b in self.beta_grid])
        if verbose:
            print(f"[{self.data.name}] weak feature bank ready in {time.time() - t0:.2f}s")

    def _target_direct(self, alpha: float) -> NDArray[np.float64]:
        # Optional Volterra/Caputo integral form: D_t^alpha u = N(u) is
        # rewritten as u - P_initial = I_t^alpha N(u). The target contains no
        # temporal derivative; alpha enters the RHS through the adjoint
        # fractional integral on time tests. For alpha > 1,
        # caputo_corrected_field estimates the initial slope from the first two
        # samples. The reported superunit experiment does not use this path.
        if self.time_form == "caputo_integral":
            corrected = caputo_corrected_field(self.U, axis=0, order=float(alpha), h=self.data.dt, initial="first")
            return self.weak.weak_inner(corrected)
        # When the benchmark/evaluator uses Caputo-L1 in time, use its exact
        # discrete adjoint on the test functions.  For alpha > 1 the matrix is
        # L1(alpha-1) @ D1, so its transpose retains an endpoint-dominated
        # opposite-sign pair inherited from D1.  This is an implicit treatment
        # of the initial rate, not a derivative-free initial boundary.
        if self.time_discretization == "caputo_l1_adjoint":
            T_adj = caputo_l1_adjoint_tests(self.time_tests, self.data.t, float(alpha))
            return self.weak.weak_inner(self.U, time_factor=T_adj)
        return self.weak.target(self.U, self._time_spec(alpha))

    def _time_factor_for_rhs(self, alpha: float | None = None) -> NDArray[np.float64]:
        if self.time_form == "caputo_integral":
            return fractional_integral_adjoint_tests(self.time_tests, self.data.t, float(self._active_alpha if alpha is None else alpha), side="left")
        return self.time_tests

    def _space_feature_p0(self, beta: float, alpha: float | None = None) -> NDArray[np.float64]:
        if self.space_kind == "riesz":
            space_factor = periodic_riesz_on_tests(self.space_tests, self.data.x, beta)
        elif self.space_kind == "spectral_directional":
            space_factor = periodic_spectral_directional_adjoint_on_tests(self.space_tests, self.data.x, beta)
        else:
            space_factor = adjoint_tests_1d(self.space_tests, self.data.x, self._space_spec(beta))
        if self.time_form == "caputo_integral":
            return self.weak.weak_inner(
                self.U,
                time_factor=self._time_factor_for_rhs(alpha),
                space_factor=space_factor,
            )
        return self.weak.weak_inner(self.U, space_factor=space_factor)

    def _adjoint_space_product(self, weighted_tests: NDArray[np.float64], beta: float) -> NDArray[np.float64]:
        """Apply L_x^* to row-wise data-weighted spatial tests."""
        if self.space_kind == "riesz":
            return periodic_riesz_on_tests(weighted_tests, self.data.x, beta)
        if self.space_kind == "spectral_directional":
            return periodic_spectral_directional_adjoint_on_tests(weighted_tests, self.data.x, beta)
        spec = self._space_spec(beta)
        return adjoint_tests_1d(weighted_tests, self.data.x, spec)

    def _space_feature_direct(self, p: int, beta: float, alpha: float | None = None) -> NDArray[np.float64]:
        p = int(p)
        if p == 0:
            return self._space_feature_p0(beta, alpha=alpha)
        U = self.U
        if self.config.power_mode == "positive":
            Up = np.maximum(U, 0.0) ** p
        else:
            Up = U ** p
        nT = self.time_tests.shape[0]
        nX = self.space_tests.shape[0]
        vals = np.empty((nT, nX), dtype=float)
        for ell, psi in enumerate(self.space_tests):
            weighted = Up * psi[None, :]
            adj_weighted = self._adjoint_space_product(weighted, beta)
            # integrate over x first for every time level
            v_t = np.sum(U * adj_weighted, axis=1) * self.weak.dx
            vals[:, ell] = self._time_factor_for_rhs(alpha) @ v_t * self.weak.dt
        return vals.reshape(-1)

    def _ensure_p(self, p: int) -> None:
        p = int(p)
        if p not in self.space_feature_cache and self.time_form != "caputo_integral":
            self.space_feature_cache[p] = np.vstack([self._space_feature_direct(p, float(b)) for b in self.beta_grid])

    def available_alpha_modes(self) -> tuple[TemporalAlphaMode, ...]:
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
        bank = self.time_mode_banks[mode]
        if mode == "integer":
            return bank[0]
        return _interp(float(alpha), self.alpha_grid_for_mode(mode), bank)

    def target(self, alpha: float, alpha_mode: str | None = None) -> NDArray[np.float64]:
        """Return the weak temporal target within one Caputo mode."""
        if self.time_bank is None:
            raise RuntimeError("call precompute() first")
        mode = model_mode(float(alpha), alpha_mode)
        self._active_alpha = 1.0 if mode == "integer" else float(alpha)
        return self._target_from_mode(self._active_alpha, mode)

    def spatial_feature(self, p: int, beta: float) -> NDArray[np.float64]:
        """Return one weak RHS column for ``u**p * D_x^beta u``.

        ``p=0`` is the linear benchmark term and is precomputed for speed.
        ``p>0`` uses the mathematically consistent data-weighted weak form
        ``<u, (D_x^beta)^*(u**p phi)>`` and is cached on first use.
        """
        if self.time_form == "caputo_integral":
            # Alpha enters the RHS in this form. Evaluate at the active order
            # directly so no interpolation can cross the integer Caputo branch.
            return self._space_feature_direct(int(p), float(beta), alpha=float(self._active_alpha))
        self._ensure_p(int(p))
        bank = self.space_feature_cache[int(p)]
        return _interp(float(beta), self.beta_grid, bank)

    def spatial(self, beta: float) -> NDArray[np.float64]:
        """Compatibility alias for the original optimizer's linear RHS column."""
        return self.spatial_feature(0, beta)

    def u_power(self, p: int) -> NDArray[np.float64]:
        """Return weak-row placeholder powers for optimizer compatibility.

        In the weak implementation nonlinear powers are handled inside
        ``spatial_feature`` using data-weighted tests.  This method is retained
        because the original Pareto optimizer expects the feature bank API to
        expose ``u_power``.
        """
        # Compatibility only: weak rows do not carry pointwise u values.
        p = int(p)
        if p not in self.u_power_cache:
            self.u_power_cache[p] = np.ones(self.n_points, dtype=float)
        return self.u_power_cache[p]

    def library(self, p_tuple: Sequence[int], beta_tuple: Sequence[float]) -> NDArray[np.float64]:
        """Build the weak RHS matrix for a proposed support pattern.

        ``p_tuple`` and ``beta_tuple`` are parallel sequences.  Column ``j`` is
        the weak feature for ``u**p_tuple[j] * D_x**beta_tuple[j] u``.
        """
        if len(p_tuple) != len(beta_tuple):
            raise ValueError("p_tuple and beta_tuple must have the same length")
        cols = [self.spatial_feature(int(p), float(beta)) for p, beta in zip(p_tuple, beta_tuple)]
        return np.column_stack(cols) if cols else np.zeros((self.n_points, 0), dtype=float)

    # ----- exact (non-interpolated) features at arbitrary continuous orders -----
    # ``target``/``spatial_feature``/``library`` interpolate precomputed features
    # across the order grid for differential-evolution speed.  The methods below
    # evaluate the weak operators *exactly* at the requested continuous orders, with
    # no grid and no interpolation.  They are used to refit the final selected model
    # (and the truth-structure diagnostic) so that the reported coefficients, orders
    # and residual do not depend on the precompute grid (see refit_selected_exact).
    def target_exact(self, alpha: float, alpha_mode: str | None = None) -> NDArray[np.float64]:
        """Weak LHS target evaluated exactly within the selected temporal mode."""
        mode = model_mode(float(alpha), alpha_mode)
        self._active_alpha = 1.0 if mode == "integer" else float(alpha)
        return self._target_direct(self._active_alpha)

    def library_exact(self, alpha: float, p_tuple: Sequence[int], beta_tuple: Sequence[float], alpha_mode: str | None = None) -> NDArray[np.float64]:
        """Weak RHS matrix evaluated exactly at ``(alpha, beta_tuple)`` (no interpolation)."""
        if len(p_tuple) != len(beta_tuple):
            raise ValueError("p_tuple and beta_tuple must have the same length")
        mode = model_mode(float(alpha), alpha_mode)
        self._active_alpha = 1.0 if mode == "integer" else float(alpha)
        cols = [self._space_feature_direct(int(p), float(beta), alpha=float(self._active_alpha))
                for p, beta in zip(p_tuple, beta_tuple)]
        return np.column_stack(cols) if cols else np.zeros((self.n_points, 0), dtype=float)


@dataclass(frozen=True)
class WeakStabilityConfig:
    """Controls the optional repeated weak-Pareto stability utility.

    The paper's proposed method is ``run_weak_pareto_discovery``: weak library
    + best-subset Pareto-DE.  This stability configuration is retained for
    ablation/diagnostic experiments where repeated weak runs over scales/splits
    are useful, but it is not the primary method in the final paper framing.
    

    Each weak row is an integral over a spacetime test window.  A single weak
    Pareto run may select a spurious fractional order if several derivative
    columns are nearly collinear.  This utility repeats weak Pareto over
    multiple train/validation groups and test-function scales, clusters the
    selected equations, and refits the most stable cluster.

    Important fields
    ----------------
    ``test_budget``
        Number of weak tests: ``smoke`` for debugging, ``standard`` for normal
        experiments, ``paper`` for final benchmark runs.
    ``width_scales``
        Multipliers applied to the default weak-test widths.  Stable structures
        should survive moderate changes in weak window size.
    ``n_splits`` and ``group_axis``
        Number and style of grouped validation splits.  Grouped splits hold out
        whole time/space weak windows instead of random rows.
    ``alpha_cluster_tol``/``beta_cluster_tol``
        Tolerances used to cluster repeated discoveries into the same structure.
    ``selection``
        How to choose among clusters.  ``stability_score`` balances validation
        error, recurrence frequency, and complexity.
    ``alpha_profile_refit``
        Optional refinement step for temporal order after support selection.
        This option is not part of the lean proposed method used in the paper
        benchmark script.
    """

    test_budget: Literal["smoke", "standard", "paper"] = "standard"
    width_scales: tuple[float, ...] = (0.75, 1.0, 1.35)
    n_splits: int = 3
    group_axis: Literal["time", "space", "checkerboard", "random"] = "time"
    alpha_cluster_tol: float = 0.04
    beta_cluster_tol: float = 0.08
    refit_width_scale: float = 1.0
    smoothing_sigmas: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.75, 0.0), (1.5, 0.0))
    denoise_methods: tuple[DenoiseMethod, ...] = ("none",)
    denoise_sigma_factors: tuple[float, ...] = (1.0,)
    denoise_transform: str = "standardize"
    time_form: WeakTimeForm = "derivative"
    # Use "auto" to profile alpha with the Volterra integral form for periodic
    # Riesz/spectral datasets, but keep the derivative form for bounded one-sided
    # paper benchmarks where the integral boundary convention is less reliable.
    alpha_profile_time_form: str = "auto"
    # After the RHS support/spatial orders are selected by stability voting,
    # re-estimate the temporal fractional order by profiling alpha with support
    # fixed. This specifically addresses the noisy-alpha identifiability issue.
    alpha_profile_refit: bool = True
    alpha_profile_rule: Literal["median_error", "fold_stability", "one_se_high"] = "fold_stability"
    alpha_profile_extra_points: int = 41
    alpha_profile_rel_tol: float = 0.02
    alpha_snap_values: tuple[float, ...] = (1.0,)
    alpha_snap_rel_tol: float = 0.05
    fold_selection_rule: Literal["elbow", "bic", "aic", "sparse_relaxed"] | None = "bic"
    selection: Literal["frequency", "frequency_then_error", "stability_score"] = "stability_score"
    frequency_weight: float = 0.15
    complexity_weight: float = 0.0


def weak_grouped_train_val_split(
    n_time_tests: int,
    n_space_tests: int,
    val_fraction: float,
    seed: int,
    *,
    group_axis: Literal["time", "space", "checkerboard", "random"] = "time",
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split weak rows while holding out whole test-function groups.

    Rows are flattened as ``time_test_major``: row ``k*n_space_tests + ell``.
    Holding out full time or space groups gives a harder validation problem than
    random weak-row splitting and reduces overfitting to a particular local window.
    """
    nt = int(n_time_tests)
    nx = int(n_space_tests)
    n = nt * nx
    rng = np.random.default_rng(int(seed))
    if group_axis == "random":
        perm = rng.permutation(n)
        n_val = max(1, int(round(float(val_fraction) * n)))
        return np.sort(perm[n_val:]).astype(np.int64), np.sort(perm[:n_val]).astype(np.int64)
    if group_axis == "space":
        groups = np.arange(nx)
        rng.shuffle(groups)
        n_val_groups = max(1, int(round(float(val_fraction) * nx)))
        val_groups = set(int(g) for g in groups[:n_val_groups])
        val = [k * nx + ell for k in range(nt) for ell in range(nx) if ell in val_groups]
    elif group_axis == "checkerboard":
        # Hold out diagonal blocks so both time and space extrapolation are tested.
        offsets = np.arange(max(nt, nx))
        rng.shuffle(offsets)
        period = max(2, int(round(1.0 / max(float(val_fraction), EPS))))
        held = set(int(o % period) for o in offsets[: max(1, period // 3)])
        val = [k * nx + ell for k in range(nt) for ell in range(nx) if ((k + ell) % period) in held]
        if not val:
            val = [int(i) for i in rng.choice(n, size=max(1, int(round(float(val_fraction) * n))), replace=False)]
    else:
        groups = np.arange(nt)
        rng.shuffle(groups)
        n_val_groups = max(1, int(round(float(val_fraction) * nt)))
        val_groups = set(int(g) for g in groups[:n_val_groups])
        val = [k * nx + ell for k in range(nt) for ell in range(nx) if k in val_groups]
    val_idx = np.array(sorted(set(val)), dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    mask[val_idx] = False
    train_idx = np.flatnonzero(mask).astype(np.int64)
    return train_idx, val_idx


def _base_weak_widths(data: GridDataset, test_budget: Literal["smoke", "standard", "paper"]) -> tuple[float, float]:
    nt, nx = data.U.shape
    _, n_x = _default_test_counts(nt, nx, budget=test_budget)
    time_width = max(3.0 * data.dt, 0.070 * (float(data.t[-1]) - float(data.t[0]) + data.dt))
    space_width = max(2.0 * data.dx, data.Lx / max(18.0, 0.75 * n_x))
    return float(time_width), float(space_width)


def _order_bin(value: float, tol: float) -> float:
    tol = max(float(tol), EPS)
    return float(round(float(value) / tol) * tol)


def weak_model_signature(model: PDEModel, *, alpha_tol: float = 0.04, beta_tol: float = 0.08) -> tuple[Any, ...]:
    """Return a tolerance-binned structure/order signature for stability voting."""
    rec = model.canonicalized()
    terms = tuple((int(p), _order_bin(float(b), beta_tol)) for p, b in zip(rec.p_tuple, rec.beta_tuple))
    return (int(rec.c), _order_bin(float(rec.alpha), alpha_tol), terms)


def _median_model_from_cluster(models: Sequence[PDEModel]) -> tuple[float, tuple[int, ...], tuple[float, ...]]:
    if not models:
        raise ValueError("empty stability cluster")
    recs = [m.canonicalized() for m in models]
    p_tuple = tuple(int(p) for p in recs[0].p_tuple)
    alpha = float(np.median([m.alpha for m in recs]))
    beta = tuple(float(v) for v in np.median(np.array([m.beta_tuple for m in recs], dtype=float), axis=0))
    return alpha, p_tuple, beta


def _build_scaled_weak_bank(
    data: GridDataset,
    config: DiscoveryConfig,
    *,
    test_budget: Literal["smoke", "standard", "paper"],
    width_scale: float,
    time_kind: TimeWeakKind | None = None,
    space_kind: SpaceWeakKind | None = None,
    denoise: DenoiseConfig | None = None,
    time_form: WeakTimeForm = "derivative",
) -> WeakFractionalFeatureBank:
    tw, xw = _base_weak_widths(data, test_budget)
    return WeakFractionalFeatureBank(
        data,
        config,
        test_budget=test_budget,
        time_width=tw * float(width_scale),
        space_width=xw * float(width_scale),
        time_kind=time_kind,
        space_kind=space_kind,
        denoise=denoise,
        time_form=time_form,
    )



def _stable_mode(values: Sequence[Any], default: Any = None) -> Any:
    if not values:
        return default
    counts: dict[str, tuple[int, Any]] = {}
    for v in values:
        key = json.dumps(v, sort_keys=True) if isinstance(v, (list, tuple, dict)) else str(v)
        count, _ = counts.get(key, (0, v))
        counts[key] = (count + 1, v)
    return max(counts.values(), key=lambda item: item[0])[1]


def _alpha_profile_grid(config: DiscoveryConfig, extra_points: int) -> NDArray[np.float64]:
    base = np.asarray(config.alpha_grid, dtype=float)
    if base.size == 0:
        raise ValueError("alpha_grid is empty")
    if int(extra_points) > base.size:
        dense = np.linspace(float(np.min(base)), float(np.max(base)), int(extra_points))
        vals = np.concatenate([base, dense])
    else:
        vals = base
    vals = np.array(sorted({round(float(v), 12) for v in vals if np.isfinite(v)}), dtype=float)
    return vals


def _profile_alpha_fixed_support(
    data: GridDataset,
    config: DiscoveryConfig,
    *,
    stability: WeakStabilityConfig,
    p_tuple: Sequence[int],
    beta_tuple: Sequence[float],
    winning_records: Sequence[dict[str, Any]],
    time_kind: TimeWeakKind | None = None,
    space_kind: SpaceWeakKind | None = None,
) -> tuple[float, dict[str, Any]]:
    """Profile alpha after support/spatial orders are fixed.

    This intentionally separates temporal-order identification from support-size
    discovery.  In high noise, the DE stage can use a low alpha to compensate for
    derivative noise.  With the RHS support fixed, repeated weak validation folds
    provide a cleaner profile for alpha.
    """
    alphas = _alpha_profile_grid(config, stability.alpha_profile_extra_points)
    cfg_profile = replace(config, alpha_grid=tuple(float(a) for a in alphas))
    profile_time_form = str(stability.alpha_profile_time_form)
    if profile_time_form == "auto":
        profile_time_form = "caputo_integral" if bool(getattr(config, "spectral_riesz", False)) else "derivative"
    if profile_time_form not in {"derivative", "caputo_integral"}:
        raise ValueError("alpha_profile_time_form must be 'auto', 'derivative', or 'caputo_integral'")
    # Use the winning-cluster folds when available; otherwise fall back to a
    # compact default design.  Deduplicate to avoid overweighting identical folds.
    designs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    source_records = list(winning_records)
    if not source_records:
        for method in stability.denoise_methods:
            for sigma_factor in stability.denoise_sigma_factors:
                for smooth_sigma_t, smooth_sigma_x in stability.smoothing_sigmas:
                    for width_scale in stability.width_scales:
                        for split_id in range(int(stability.n_splits)):
                            source_records.append({
                                "width_scale": float(width_scale),
                                "smooth_sigma_t": float(smooth_sigma_t),
                                "smooth_sigma_x": float(smooth_sigma_x),
                                "split_id": int(split_id),
                                "denoise_method": method,
                                "denoise_sigma_factor": sigma_factor,
                            })
    for r in source_records:
        key = (
            round(float(r.get("width_scale", stability.refit_width_scale)), 8),
            round(float(r.get("smooth_sigma_t", config.smooth_sigma_t)), 8),
            round(float(r.get("smooth_sigma_x", config.smooth_sigma_x)), 8),
            int(r.get("split_id", 0)),
            str(r.get("denoise_method", "none")),
            round(float(r.get("denoise_sigma_factor", 1.0)), 8),
        )
        if key not in seen:
            seen.add(key)
            designs.append(r)
    # Keep the profile bounded in runtime while still diverse.
    designs = designs[: max(1, min(16, len(designs)))]

    fold_records: list[dict[str, Any]] = []
    for j, r in enumerate(designs):
        denoise_cfg = DenoiseConfig(
            method=str(r.get("denoise_method", "none")),  # type: ignore[arg-type]
            sigma_factor=float(r.get("denoise_sigma_factor", 1.0)),
            transform=str(stability.denoise_transform),  # type: ignore[arg-type]
        )
        cfg_fold = replace(
            cfg_profile,
            smooth_sigma_t=float(r.get("smooth_sigma_t", cfg_profile.smooth_sigma_t)),
            smooth_sigma_x=float(r.get("smooth_sigma_x", cfg_profile.smooth_sigma_x)),
        )
        bank = _build_scaled_weak_bank(
            data,
            cfg_fold,
            test_budget=stability.test_budget,
            width_scale=float(r.get("width_scale", stability.refit_width_scale)),
            time_kind=time_kind,
            space_kind=space_kind,
            denoise=denoise_cfg,
            time_form=profile_time_form,  # type: ignore[arg-type]
        )
        bank.precompute(verbose=False)
        tr, va = weak_grouped_train_val_split(
            bank.time_tests.shape[0],
            bank.space_tests.shape[0],
            cfg_fold.val_fraction,
            seed=int(cfg_fold.seed) + 50021 * int(r.get("split_id", j)) + 17 * j,
            group_axis=stability.group_axis,
        )
        opt = ParetoFDEOptimizer(bank, tr, va, cfg_fold)  # type: ignore[arg-type]
        vals = []
        for a in alphas:
            m = opt.evaluate(float(a), p_tuple, beta_tuple)
            vals.append(float(m.val_rel_mse))
        vals_arr = np.asarray(vals, dtype=float)
        best_i = int(np.nanargmin(vals_arr))
        fold_records.append({
            "fold": int(j + 1),
            "width_scale": float(r.get("width_scale", stability.refit_width_scale)),
            "smooth_sigma_t": float(r.get("smooth_sigma_t", cfg_profile.smooth_sigma_t)),
            "smooth_sigma_x": float(r.get("smooth_sigma_x", cfg_profile.smooth_sigma_x)),
            "split_id": int(r.get("split_id", j)),
            "denoise_method": str(r.get("denoise_method", "none")),
            "denoise_effective_method": str(getattr(bank, "denoise_metadata", {}).get("effective_method", "unknown")),
            "denoise_sigma_factor": float(r.get("denoise_sigma_factor", 1.0)),
            "best_alpha": float(alphas[best_i]),
            "best_val_rel_mse": float(vals_arr[best_i]),
            "values": [float(v) for v in vals_arr],
        })
    M = np.asarray([r["values"] for r in fold_records], dtype=float)
    logM = np.log10(np.maximum(M, EPS))
    median_log = np.nanmedian(logM, axis=0)
    q25 = np.nanpercentile(logM, 25, axis=0)
    q75 = np.nanpercentile(logM, 75, axis=0)
    iqr = q75 - q25
    # A candidate is stable in a fold if it is within a small relative band of
    # the best alpha for that fold.  This avoids over-selecting a single alpha by
    # numerical noise on a flat residual profile.
    tol_log = math.log10(1.0 + max(float(stability.alpha_profile_rel_tol), 0.0))
    within = logM <= (np.nanmin(logM, axis=1, keepdims=True) + tol_log)
    stable_count = np.sum(within, axis=0)
    stable_fraction = stable_count / max(1, logM.shape[0])
    if stability.alpha_profile_rule == "median_error":
        idx = int(np.nanargmin(median_log))
    elif stability.alpha_profile_rule == "one_se_high":
        best = float(np.nanmin(median_log))
        feasible = np.flatnonzero(median_log <= best + tol_log)
        idx = int(feasible[-1]) if feasible.size else int(np.nanargmin(median_log))
    else:  # fold_stability
        # Prefer recurring plateaus, then median error, then sharper profile.
        keys = [(-float(stable_count[i]), float(median_log[i]), float(iqr[i])) for i in range(len(alphas))]
        idx = int(min(range(len(alphas)), key=lambda i: keys[i]))

    # Parsimonious temporal-order snapping.  If an integer/special order is
    # statistically indistinguishable from the best profiled alpha, report that
    # simpler order.  This is important for noisy data where fractional orders
    # near 1 can otherwise win by tiny residual differences while representing
    # a less parsimonious memory model.
    snap_tol_log = math.log10(1.0 + max(float(stability.alpha_snap_rel_tol), 0.0))
    best_log = float(np.nanmin(median_log))
    snapped_to: float | None = None
    for snap_value in stability.alpha_snap_values:
        sv = float(snap_value)
        if sv < float(np.min(alphas)) - 1e-12 or sv > float(np.max(alphas)) + 1e-12:
            continue
        j = int(np.argmin(np.abs(alphas - sv)))
        if abs(float(alphas[j]) - sv) <= max(1e-10, 0.5 * float(np.min(np.diff(alphas))) if len(alphas) > 1 else 1e-10):
            if float(median_log[j]) <= best_log + snap_tol_log:
                idx = j
                snapped_to = float(alphas[j])
                break
    profile_tol_log = max(tol_log, snap_tol_log)
    feasible_profile = np.flatnonzero(median_log <= best_log + profile_tol_log)
    if feasible_profile.size:
        confidence_interval = [float(alphas[int(feasible_profile[0])]), float(alphas[int(feasible_profile[-1])])]
    else:
        confidence_interval = [float(alphas[idx]), float(alphas[idx])]
    rows = []
    for i, a in enumerate(alphas):
        rows.append({
            "alpha": float(a),
            "median_log10_val_rel_mse": float(median_log[i]),
            "iqr_log10_val_rel_mse": float(iqr[i]),
            "stable_count": int(stable_count[i]),
            "stable_fraction": float(stable_fraction[i]),
            "median_val_rel_mse": float(10 ** median_log[i]),
        })
    summary = {
        "selected_alpha": float(alphas[idx]),
        "rule": stability.alpha_profile_rule,
        "time_form": profile_time_form,
        "snapped_to": snapped_to,
        "confidence_interval": confidence_interval,
        "snap_values": [float(v) for v in stability.alpha_snap_values],
        "snap_rel_tol": float(stability.alpha_snap_rel_tol),
        "support_p_tuple": [int(p) for p in p_tuple],
        "support_beta_tuple": [float(b) for b in beta_tuple],
        "alpha_grid": [float(a) for a in alphas],
        "profile_rows": rows,
        "fold_records": fold_records,
    }
    return float(alphas[idx]), summary

def run_stability_selected_weak_pareto_discovery(
    data: GridDataset,
    config: DiscoveryConfig,
    output_dir: str | Path | None = None,
    verbose: bool = True,
    export_selected_fde: bool = True,
    *,
    stability: WeakStabilityConfig | None = None,
    time_kind: TimeWeakKind | None = None,
    space_kind: SpaceWeakKind | None = None,
) -> dict[str, Any]:
    """Run the optional repeated weak-Pareto stability utility.

    This function is kept for diagnostics and ablations.  The main proposed
    method used by the paper scripts is ``run_weak_pareto_discovery``.
    

    It runs the existing Pareto support-size sweep several times with perturbed
    weak test widths and grouped validation folds, clusters the selected models
    by tolerance-binned fractional orders, and refits the most stable cluster on
    all weak rows.
    """
    stab = stability or WeakStabilityConfig()
    records: list[dict[str, Any]] = []
    selected_models: list[PDEModel] = []
    fold = 0
    denoise_methods = tuple(stab.denoise_methods or ("none",))
    denoise_sigma_factors = tuple(stab.denoise_sigma_factors or (1.0,))
    for denoise_i, denoise_method in enumerate(denoise_methods):
        for sigma_i, denoise_sigma_factor in enumerate(denoise_sigma_factors):
            denoise_cfg = DenoiseConfig(
                method=denoise_method,
                sigma_factor=float(denoise_sigma_factor),
                transform=str(stab.denoise_transform),  # type: ignore[arg-type]
            )
            for smooth_sigma_t, smooth_sigma_x in stab.smoothing_sigmas:
                cfg_fold = replace(config, smooth_sigma_t=float(smooth_sigma_t), smooth_sigma_x=float(smooth_sigma_x))
                for width_scale in stab.width_scales:
                    for split_id in range(int(stab.n_splits)):
                        fold += 1
                        bank = _build_scaled_weak_bank(
                            data,
                            cfg_fold,
                            test_budget=stab.test_budget,
                            width_scale=float(width_scale),
                            time_kind=time_kind,
                            space_kind=space_kind,
                            denoise=denoise_cfg,
                            time_form=stab.time_form,
                        )
                        bank.precompute(verbose=False)
                        tr, va = weak_grouped_train_val_split(
                            bank.time_tests.shape[0],
                            bank.space_tests.shape[0],
                            cfg_fold.val_fraction,
                            seed=int(cfg_fold.seed)
                            + 10007 * split_id
                            + int(round(1000 * float(width_scale)))
                            + int(round(100 * float(smooth_sigma_t)))
                            + 131 * denoise_i
                            + 977 * sigma_i,
                            group_axis=stab.group_axis,
                        )
                        opt = ParetoFDEOptimizer(bank, tr, va, cfg_fold)  # type: ignore[arg-type]
                        all_models, best_by_c, stop_info = support_size_sweep(opt, cfg_fold, verbose=False)
                        pareto = pareto_front(best_by_c)
                        chosen = select_model(pareto, stab.fold_selection_rule or cfg_fold.selection, cfg_fold.sparse_relaxed_tol).canonicalized()
                        chosen.backend = "weak-stability-fold"
                        selected_models.append(chosen)
                        records.append({
                            "fold": fold,
                            "width_scale": float(width_scale),
                            "smooth_sigma_t": float(smooth_sigma_t),
                            "smooth_sigma_x": float(smooth_sigma_x),
                            "split_id": int(split_id),
                            "group_axis": stab.group_axis,
                            "denoise_method": str(denoise_method),
                            "denoise_effective_method": str(getattr(bank, "denoise_metadata", {}).get("effective_method", "unknown")),
                            "denoise_sigma_factor": float(denoise_sigma_factor),
                            "denoise_sigma": float(getattr(bank, "denoise_metadata", {}).get("sigma", 0.0)),
                            "selected": chosen.to_dict(),
                            "best_by_c": {str(c): m.to_dict() for c, m in best_by_c.items()},
                            "pareto": {str(c): m.to_dict() for c, m in pareto.items()},
                            "auto_stop": stop_info,
                        })
                        if verbose:
                            print(
                                f"stability fold {fold}: denoise={denoise_method}/{getattr(bank, 'denoise_metadata', {}).get('effective_method', 'unknown')} "
                                f"scale={float(width_scale):.3g} smooth=({float(smooth_sigma_t):.2g},{float(smooth_sigma_x):.2g}) "
                                f"split={split_id} val_rel={chosen.val_rel_mse:.3e} {chosen.equation(digits=4)}"
                            )
    clusters: dict[tuple[Any, ...], list[PDEModel]] = {}
    for m in selected_models:
        sig = weak_model_signature(m, alpha_tol=stab.alpha_cluster_tol, beta_tol=stab.beta_cluster_tol)
        clusters.setdefault(sig, []).append(m)
    if not clusters:
        raise RuntimeError("stability selection produced no candidate models")

    def cluster_key(item: tuple[tuple[Any, ...], list[PDEModel]]) -> tuple[float, float, float]:
        sig, ms = item
        freq = len(ms)
        freq_frac = float(freq / max(1, len(selected_models)))
        med_err = float(np.median([m.val_rel_mse for m in ms]))
        c = int(ms[0].c)
        if stab.selection == "frequency":
            return (-float(freq), float(c), med_err)
        if stab.selection == "frequency_then_error":
            return (-float(freq), med_err, float(c))
        stability_score = math.log10(max(med_err, EPS)) - float(stab.frequency_weight) * freq_frac + float(stab.complexity_weight) * float(c)
        return (stability_score, -float(freq), float(c))

    winning_sig, winning_models = min(clusters.items(), key=cluster_key)
    alpha, p_tuple, beta_tuple = _median_model_from_cluster(winning_models)
    # Final coefficient refit on a default-scale weak-row set using all rows.  The
    # support/orders are those stabilized by the ensemble; coefficients are not
    # averaged across incompatible weak scales.  Use the median smoothing level
    # among folds in the winning cluster because smoothing can be part of the
    # noise-robust structural vote.
    win_records = []
    for r, m in zip(records, selected_models):
        if weak_model_signature(m, alpha_tol=stab.alpha_cluster_tol, beta_tol=stab.beta_cluster_tol) == winning_sig:
            win_records.append(r)
    alpha_profile_summary: dict[str, Any] | None = None
    if bool(stab.alpha_profile_refit):
        alpha, alpha_profile_summary = _profile_alpha_fixed_support(
            data,
            config,
            stability=stab,
            p_tuple=p_tuple,
            beta_tuple=beta_tuple,
            winning_records=win_records,
            time_kind=time_kind,
            space_kind=space_kind,
        )

    refit_cfg = replace(
        config,
        alpha_grid=tuple(float(a) for a in sorted({*map(float, config.alpha_grid), float(alpha)})),
        smooth_sigma_t=float(np.median([r.get("smooth_sigma_t", config.smooth_sigma_t) for r in win_records])) if win_records else float(config.smooth_sigma_t),
        smooth_sigma_x=float(np.median([r.get("smooth_sigma_x", config.smooth_sigma_x) for r in win_records])) if win_records else float(config.smooth_sigma_x),
    )
    refit_denoise_method = _stable_mode([r.get("denoise_method", "none") for r in win_records], "none")
    refit_denoise_factor = float(np.median([float(r.get("denoise_sigma_factor", 1.0)) for r in win_records])) if win_records else 1.0
    refit_denoise = DenoiseConfig(
        method=str(refit_denoise_method),  # type: ignore[arg-type]
        sigma_factor=refit_denoise_factor,
        transform=str(stab.denoise_transform),  # type: ignore[arg-type]
    )
    final_bank = _build_scaled_weak_bank(
        data,
        refit_cfg,
        test_budget=stab.test_budget,
        width_scale=float(stab.refit_width_scale),
        time_kind=time_kind,
        space_kind=space_kind,
        denoise=refit_denoise,
        time_form=stab.time_form,
    )
    final_bank.precompute(verbose=verbose)
    all_idx = np.arange(final_bank.n_points, dtype=np.int64)
    final_opt = ParetoFDEOptimizer(final_bank, all_idx, all_idx, refit_cfg)  # type: ignore[arg-type]
    selected = final_opt.evaluate(alpha, p_tuple, beta_tuple).canonicalized()
    selected.backend = "weak-stability-pareto"

    cluster_summaries = []
    for sig, ms in sorted(clusters.items(), key=cluster_key):
        a_med, p_med, b_med = _median_model_from_cluster(ms)
        cluster_summaries.append({
            "signature": repr(sig),
            "frequency": int(len(ms)),
            "frequency_fraction": float(len(ms) / max(1, len(selected_models))),
            "median_val_rel_mse": float(np.median([m.val_rel_mse for m in ms])),
            "median_alpha": float(a_med),
            "p_tuple": [int(p) for p in p_med],
            "median_beta_tuple": [float(b) for b in b_med],
            "examples": [m.to_dict() for m in ms[:3]],
        })
    summary: dict[str, Any] = {
        "dataset": data.name,
        "truth": data.truth,
        "config": config_to_dict(refit_cfg),
        "weak_stability_config": {
            "test_budget": stab.test_budget,
            "width_scales": [float(v) for v in stab.width_scales],
            "n_splits": int(stab.n_splits),
            "group_axis": stab.group_axis,
            "alpha_cluster_tol": float(stab.alpha_cluster_tol),
            "beta_cluster_tol": float(stab.beta_cluster_tol),
            "refit_width_scale": float(stab.refit_width_scale),
            "smoothing_sigmas": [[float(a), float(b)] for a, b in stab.smoothing_sigmas],
            "denoise_methods": [str(v) for v in stab.denoise_methods],
            "denoise_sigma_factors": [float(v) for v in stab.denoise_sigma_factors],
            "denoise_transform": str(stab.denoise_transform),
            "time_form": stab.time_form,
            "alpha_profile_time_form": str(stab.alpha_profile_time_form),
            "alpha_profile_refit": bool(stab.alpha_profile_refit),
            "alpha_profile_rule": stab.alpha_profile_rule,
            "alpha_profile_extra_points": int(stab.alpha_profile_extra_points),
            "alpha_profile_rel_tol": float(stab.alpha_profile_rel_tol),
            "alpha_snap_values": [float(v) for v in stab.alpha_snap_values],
            "alpha_snap_rel_tol": float(stab.alpha_snap_rel_tol),
            "fold_selection_rule": stab.fold_selection_rule,
            "selection": stab.selection,
            "frequency_weight": float(stab.frequency_weight),
            "complexity_weight": float(stab.complexity_weight),
        },
        "n_folds": int(len(records)),
        "folds": records,
        "clusters": cluster_summaries,
        "winning_signature": repr(winning_sig),
        "selected": selected.to_dict(),
        "alpha_profile": alpha_profile_summary,
        "consensus_frequency": int(len(winning_models)),
        "consensus_fraction": float(len(winning_models) / max(1, len(selected_models))),
        "weak_feature_bank": {
            "time_kind": final_bank.time_kind,
            "space_kind": final_bank.space_kind,
            "space_side": final_bank.space_side,
            "n_time_tests": int(final_bank.time_tests.shape[0]),
            "n_space_tests": int(final_bank.space_tests.shape[0]),
            "n_weak_rows": int(final_bank.n_points),
            "test_budget": stab.test_budget,
            "time_discretization": final_bank.time_discretization,
            "time_form": final_bank.time_form,
            "denoise": getattr(final_bank, "denoise_metadata", {}),
        },
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
                metadata={
                    "truth": data.truth,
                    "selection": config.selection,
                    "feature_form": "weak_stability",
                    "winning_signature": repr(winning_sig),
                },
            )
            summary["selected_fde_path"] = str(selected_path)
        write_json(output_dir / "summary.json", summary)
        with (output_dir / "stability_folds.csv").open("w", newline="") as f:
            fieldnames = ["fold", "width_scale", "smooth_sigma_t", "smooth_sigma_x", "split_id", "group_axis", "denoise_method", "denoise_effective_method", "denoise_sigma_factor", "selected_equation", "selected_c", "selected_alpha", "selected_terms", "val_rel_mse"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                sel = r["selected"]
                writer.writerow({
                    "fold": r["fold"],
                    "width_scale": r["width_scale"],
                    "smooth_sigma_t": r.get("smooth_sigma_t", 0.0),
                    "smooth_sigma_x": r.get("smooth_sigma_x", 0.0),
                    "split_id": r["split_id"],
                    "group_axis": r["group_axis"],
                    "denoise_method": r.get("denoise_method", "none"),
                    "denoise_effective_method": r.get("denoise_effective_method", "unknown"),
                    "denoise_sigma_factor": r.get("denoise_sigma_factor", 1.0),
                    "selected_equation": sel["equation"],
                    "selected_c": sel["c"],
                    "selected_alpha": sel["alpha"],
                    "selected_terms": json.dumps(sel["terms"]),
                    "val_rel_mse": sel["val_rel_mse"],
                })
        with (output_dir / "stability_clusters.csv").open("w", newline="") as f:
            fieldnames = ["signature", "frequency", "frequency_fraction", "median_val_rel_mse", "median_alpha", "p_tuple", "median_beta_tuple"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in cluster_summaries:
                writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in c.items() if k in fieldnames})
        if alpha_profile_summary is not None:
            with (output_dir / "alpha_profile.csv").open("w", newline="") as f:
                fieldnames = ["alpha", "median_log10_val_rel_mse", "iqr_log10_val_rel_mse", "stable_count", "stable_fraction", "median_val_rel_mse"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(alpha_profile_summary.get("profile_rows", []))
    return summary



@dataclass
class WeakParetoProblem:
    """Separated objects used by the proposed weak Pareto-DE method.

    This small container makes the proposed method transparent in notebooks.
    The one-line wrapper :func:`run_weak_pareto_discovery` is still available
    for scripts, but educational workflows can now perform the two conceptual
    steps explicitly:

    1. build the weak candidate library/feature bank; and
    2. run the best-subset Pareto-DE optimizer on that weak library.

    Attributes
    ----------
    bank:
        Precomputed :class:`WeakFractionalFeatureBank`.  It provides
        ``target(alpha)`` and ``library(p_tuple, beta_tuple)`` just like the
        vanilla feature bank, except rows are weak integrals instead of
        pointwise derivative samples.
    train_idx, val_idx:
        Deterministic train/validation row split used by the optimizer.
    optimizer:
        :class:`pareto_fde_discovery.ParetoFDEOptimizer` configured to use the
        weak-row set.  Calling ``optimizer.evaluate(...)`` is useful for debugging
        one hand-selected structure; calling :func:`run_best_subset_pareto_de`
        performs the full best-subset cardinality sweep.
    """

    bank: WeakFractionalFeatureBank
    train_idx: NDArray[np.int64]
    val_idx: NDArray[np.int64]
    optimizer: ParetoFDEOptimizer


def build_weak_candidate_library(
    data: GridDataset,
    config: DiscoveryConfig,
    *,
    test_budget: Literal["smoke", "standard", "paper"] = "standard",
    test_counts: tuple[int, int] | None = None,
    time_kind: TimeWeakKind | None = None,
    space_kind: SpaceWeakKind | None = None,
    time_form: WeakTimeForm = "derivative",
    verbose: bool = True,
) -> WeakFractionalFeatureBank:
    """Build and precompute the weak candidate library only.

    This is step 1 of the proposed method.  It creates weak regression rows
    using smooth test functions and fractional adjoints.  It does **not** run
    any sparse model selection or differential evolution.

    Use this function in notebooks when you want to inspect the weak target,
    RHS columns, feature correlations, or manually evaluate a candidate before
    running the best-subset search.  The default ``time_form="derivative"`` is
    the publication path and applies the transpose of the discrete Caputo--L1
    matrix to the temporal tests.  The optional ``"caputo_integral"`` form is
    not used by the reported superunit experiment.
    """
    bank = WeakFractionalFeatureBank(
        data,
        config,
        test_budget=test_budget,
        test_counts=test_counts,
        time_kind=time_kind,
        space_kind=space_kind,
        time_form=time_form,
    )
    bank.precompute(verbose=verbose)
    return bank


def build_best_subset_pareto_problem(
    bank: WeakFractionalFeatureBank,
    config: DiscoveryConfig,
) -> WeakParetoProblem:
    """Create the best-subset Pareto-DE problem for a precomputed weak-row set.

    This is the bridge between weak-library construction and the optimizer.  It
    is intentionally small: create the same deterministic train/validation split
    used by the scripts and wrap the bank in ``ParetoFDEOptimizer``.
    """
    train_idx, val_idx = train_val_split(bank.n_points, config.val_fraction, config.seed)
    opt = ParetoFDEOptimizer(bank, train_idx, val_idx, config)  # type: ignore[arg-type]
    return WeakParetoProblem(bank=bank, train_idx=train_idx, val_idx=val_idx, optimizer=opt)


def _support_progress_rows(best_by_c: dict[int, PDEModel]) -> list[dict[str, Any]]:
    """Convert the best model at each support size into JSON/CSV rows."""
    rows: list[dict[str, Any]] = []
    prev_val = None
    best_so_far = math.inf
    for c in sorted(best_by_c):
        m = best_by_c[c]
        val = float(m.val_rel_mse)
        if prev_val is None:
            rel_improvement = None
            log10_improvement = None
        else:
            rel_improvement = float((prev_val - val) / max(prev_val, EPS))
            log10_improvement = float(math.log10(max(prev_val, EPS)) - math.log10(max(val, EPS)))
        best_so_far = min(best_so_far, val)
        rows.append({
            "c": int(c),
            "alpha": float(m.alpha),
            "alpha_mode": str(model_mode(m.alpha, getattr(m, "alpha_mode", None))),
            "p_tuple": [int(v) for v in m.p_tuple],
            "beta_tuple": [float(v) for v in m.beta_tuple],
            "coefficients": [float(v) for v in m.coefficients],
            "train_rel_mse": float(m.train_rel_mse),
            "val_rel_mse": float(m.val_rel_mse),
            "best_val_rel_mse_so_far": float(best_so_far),
            "relative_improvement_from_previous_c": rel_improvement,
            "log10_improvement_from_previous_c": log10_improvement,
            "equation": m.equation(digits=6),
            "backend": str(m.backend),
        })
        prev_val = val
    return rows


def write_support_progress(path: str | Path, best_by_c: dict[int, PDEModel]) -> None:
    """Write ``support_size_progress.csv`` for inspecting the Pareto sweep.

    The file answers the practical question: *what is the best equation found
    at each support size?*  It is useful in notebooks and paper debugging
    because it shows whether the true equation appears at ``c=1``, ``c=2``,
    etc., and how much validation error improves when complexity increases.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _support_progress_rows(best_by_c)
    fields = [
        "c", "alpha", "alpha_mode", "p_tuple", "beta_tuple", "coefficients",
        "train_rel_mse", "val_rel_mse", "best_val_rel_mse_so_far",
        "relative_improvement_from_previous_c", "log10_improvement_from_previous_c",
        "equation", "backend",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: json.dumps(v) if isinstance(v, list) else v
                for k, v in row.items()
            })


def run_best_subset_pareto_de(
    problem: WeakParetoProblem,
    config: DiscoveryConfig,
    *,
    data: GridDataset | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = True,
    export_selected_fde: bool = True,
) -> dict[str, Any]:
    """Run step 2 of the proposed method: best-subset Pareto-DE.

    Parameters
    ----------
    problem:
        Output of :func:`build_best_subset_pareto_problem`, which contains a
        precomputed weak candidate library and the optimizer.
    config:
        Discovery configuration.  It controls support-size range, selection
        rule, and DE runtime budget.
    data:
        Optional dataset metadata used only for labels/truth in output files.
    output_dir:
        If provided, writes ``summary.json``, ``all_models.csv``,
        ``best_by_c.csv``, ``support_size_progress.csv``, and
        ``support_size_progress.json``.

    Returns
    -------
    dict
        Same summary structure as :func:`run_weak_pareto_discovery`, now with a
        first-class ``support_size_progress`` table.
    """
    all_models, best_by_c, stop_info = support_size_sweep(problem.optimizer, config, verbose=verbose)
    pareto = pareto_front(best_by_c)
    selected_raw = select_model(pareto, config.selection, config.sparse_relaxed_tol)
    selected_raw.backend = "weak-pareto-de"
    selected = prune_inactive_selected_terms(
        problem.optimizer,
        selected_raw,
        contribution_tol=config.selected_contribution_prune_tol,
        abs_tol=config.selected_coef_prune_abs_tol,
        coef_rel_fallback_tol=config.selected_coef_prune_rel_tol,
    )
    selected.backend = "weak-pareto-de"
    # Recompute the final conditional coefficients/residual directly at the polished
    # continuous orders, without order-grid interpolation.  This removes interpolation
    # error from the final conditional refit, but the selected support, temporal mode,
    # and optimization basin can still depend on the order grid and search trajectory.
    # Selection-stage validation/objective fields remain unchanged.
    exact_refit_info: dict[str, Any] | None = None
    if getattr(config, "exact_order_refit", True):
        coef_before = np.asarray(selected.coefficients, dtype=float).copy()
        alpha_before = float(selected.alpha)
        beta_before = tuple(float(b) for b in selected.beta_tuple)
        # Optional local polishing uses the directly evaluated weak residual.  It
        # refines the selected basin; it does not make the final equation
        # unconditionally independent of the preceding grid/search trajectory.
        if getattr(config, "exact_order_polish", False) and selected.c >= 1:
            selected = _polish_selected_orders_exact(problem.bank, selected, config, ridge=float(getattr(config, 'ridge', 1e-3)))
            selected.backend = "weak-pareto-de"
        selected, exact_rel = refit_selected_exact(problem.bank, selected, ridge=float(getattr(config, 'ridge', 1e-3)))
        selected.backend = "weak-pareto-de"
        coef_after = np.asarray(selected.coefficients, dtype=float)
        max_shift = float(np.max(np.abs(coef_after - coef_before))) if (coef_after.size and coef_after.shape == coef_before.shape) else None
        exact_refit_info = {
            "applied": True,
            "order_polished": bool(getattr(config, "exact_order_polish", False)),
            "full_data_rel_l2": float(exact_rel),
            "selection_metrics_preserved": True,
            "alpha_shift_vs_de": float(selected.alpha) - alpha_before,
            "beta_shift_vs_de": [float(b) - b0 for b, b0 in zip(selected.beta_tuple, beta_before)],
            "max_abs_coefficient_shift_vs_interpolated": max_shift,
        }
    rows = _support_progress_rows(best_by_c)
    dataset_name = data.name if data is not None else getattr(problem.bank.data, "name", "unknown")
    truth = data.truth if data is not None else getattr(problem.bank.data, "truth", "")
    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "truth": truth,
        "config": config_to_dict(config),
        "weak_feature_bank": {
            "time_kind": problem.bank.time_kind,
            "space_kind": problem.bank.space_kind,
            "space_side": problem.bank.space_side,
            "n_time_tests": int(problem.bank.time_tests.shape[0]),
            "n_space_tests": int(problem.bank.space_tests.shape[0]),
            "n_weak_rows": int(problem.bank.n_points),
            "test_budget": getattr(problem.bank, "test_budget", None),
            "time_discretization": problem.bank.time_discretization,
        },
        "support_size_progress": rows,
        "best_by_c": {str(c): m.to_dict() for c, m in best_by_c.items()},
        "pareto": {str(c): m.to_dict() for c, m in pareto.items()},
        "selected": selected.to_dict(),
        "selected_exact_refit": exact_refit_info,
        "auto_stop": stop_info,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if export_selected_fde and data is not None:
            selected_path = save_selected_fde(
                selected,
                output_dir / "selected_fde.json",
                dataset_name=data.name,
                config=config,
                metadata={"truth": data.truth, "selection": config.selection, "feature_form": "weak"},
            )
            summary["selected_fde_path"] = str(selected_path)
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "support_size_progress.json", rows)
        write_models_csv(output_dir / "all_models.csv", all_models)
        write_models_csv(output_dir / "best_by_c.csv", list(best_by_c.values()))
        write_support_progress(output_dir / "support_size_progress.csv", best_by_c)
    return summary


def run_weak_pareto_discovery(
    data: GridDataset,
    config: DiscoveryConfig,
    output_dir: str | Path | None = None,
    verbose: bool = True,
    export_selected_fde: bool = True,
    *,
    test_budget: Literal["smoke", "standard", "paper"] = "standard",
    test_counts: tuple[int, int] | None = None,
    time_kind: TimeWeakKind | None = None,
    space_kind: SpaceWeakKind | None = None,
    time_form: WeakTimeForm = "derivative",
) -> dict[str, Any]:
    """Run the proposed method: weak library + best-subset Pareto-DE.

    This one-line wrapper is used by scripts.  It is exactly equivalent to the
    explicit notebook workflow::

        bank = build_weak_candidate_library(data, config, test_budget=...)
        problem = build_best_subset_pareto_problem(bank, config)
        summary = run_best_subset_pareto_de(problem, config, data=data)

    The returned/written summary includes ``support_size_progress`` so users can
    inspect the best equation found at each support size.  The publication
    default is ``time_form="derivative"``; this uses the transposed discrete
    Caputo--L1 operator.  For superunit orders the operator is the composition
    ``L1(alpha-1) @ D1``, whose transpose treats the initial rate implicitly
    through an endpoint-dominated opposite-sign weight pair.
    """
    bank = build_weak_candidate_library(
        data,
        config,
        test_budget=test_budget,
        test_counts=test_counts,
        time_kind=time_kind,
        space_kind=space_kind,
        time_form=time_form,
        verbose=verbose,
    )
    problem = build_best_subset_pareto_problem(bank, config)
    return run_best_subset_pareto_de(
        problem,
        config,
        data=data,
        output_dir=output_dir,
        verbose=verbose,
        export_selected_fde=export_selected_fde,
    )

def fit_coefficients_for_structure(bank: Any, alpha: float, terms: Sequence[tuple[int, float]], ridge: float = 1e-3, alpha_mode: str | None = None) -> tuple[NDArray[np.float64], float]:
    """Fit coefficients for a fixed truth/probe structure and return rel RMSE.

    Uses *exact* (non-interpolated) weak features at the requested orders when the
    bank supports them, so the truth-structure diagnostic does not inherit any
    order-grid interpolation error; falls back to the interpolated API otherwise.
    """
    if hasattr(bank, "precompute"):
        # Caller should have precomputed, but doing it twice is prevented by tests.
        pass
    if hasattr(bank, "target_exact") and hasattr(bank, "library_exact"):
        try:
            y = bank.target_exact(float(alpha), alpha_mode=alpha_mode)
            X = bank.library_exact(float(alpha), [p for p, _ in terms], [b for _, b in terms], alpha_mode=alpha_mode)
        except TypeError:
            y = bank.target_exact(float(alpha))
            X = bank.library_exact(float(alpha), [p for p, _ in terms], [b for _, b in terms])
    else:
        y = bank.target(float(alpha))
        X = bank.library([p for p, _ in terms], [b for _, b in terms])
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xf = X[finite]
    yf = y[finite]
    scales = np.linalg.norm(Xf, axis=0)
    scales[scales < EPS] = 1.0
    Xn = Xf / scales[None, :]
    coef_n = np.linalg.solve(Xn.T @ Xn + float(ridge) * np.eye(Xn.shape[1]), Xn.T @ yf)
    coef = coef_n / scales
    resid = yf - Xf @ coef
    rel = float(np.linalg.norm(resid) / (np.linalg.norm(yf) + EPS))
    return coef.astype(float), rel


def _polish_selected_orders_exact(bank: Any, model: PDEModel, config: DiscoveryConfig, *, ridge: float | None = None) -> PDEModel:
    """Locally polish ``(alpha, beta_tuple)`` on a directly evaluated weak residual.

    A bounded simplex search refines the differential-evolution optimum using
    features evaluated without order-grid interpolation.  This is a conditional
    refinement of the selected basin; the selected support, temporal mode, and
    basin can still depend on the precompute grid and search trajectory.  Two
    constraints keep the polish faithful to the selected model rather than
    letting it drift into a different operator:

    * a term selected as the identity/reaction operator (``beta`` snapped to
      zero, i.e. below the first strictly-positive order node) keeps that
      discrete mode -- its order stays exactly zero and it is not polished, so
      an identity term can no longer be converted into a low-order Riesz
      derivative; and
    * each genuine derivative order is polished only within a declared local
      trust region about its selected value, so the polish refines the
      selected basin instead of roaming the whole declared interval.

    Coefficients are refit afterwards by :func:`refit_selected_exact`.
    """
    if ridge is None:
        ridge = float(getattr(config, "ridge", 1e-3))
    if not (hasattr(bank, "target_exact") and hasattr(bank, "library_exact")) or model.c < 1:
        return model
    p_tuple = tuple(int(p) for p in model.p_tuple)
    selected_alpha_mode = model_mode(float(model.alpha), getattr(model, "alpha_mode", None))
    if selected_alpha_mode == "integer":
        a_lo = a_hi = 1.0
    elif hasattr(bank, "alpha_bounds"):
        a_lo, a_hi = bank.alpha_bounds(selected_alpha_mode)
    else:
        a_lo, a_hi = alpha_bounds_for_mode(
            bank.alpha_grid, selected_alpha_mode,
            branch_epsilon=float(getattr(config, "alpha_branch_epsilon", 1e-3)),
        )
    bg = np.asarray(bank.beta_grid, dtype=float)
    b_hi = float(bg.max())
    pos = bg[bg > 1e-9]
    b_pos_min = float(pos.min()) if pos.size else float(bg.min())
    uq = np.unique(bg)
    spacing = float(np.median(np.diff(uq))) if uq.size > 1 else 0.05
    delta = max(3.0 * spacing, 0.15)

    betas0 = [float(b) for b in model.beta_tuple]
    is_ident = [b < b_pos_min for b in betas0]
    free_idx = [j for j, idj in enumerate(is_ident) if not idj]

    def _full_betas(z: NDArray[np.float64]) -> list:
        betas = list(betas0)
        for k, j in enumerate(free_idx):
            betas[j] = float(z[1 + k])
        for j, idj in enumerate(is_ident):
            if idj:
                betas[j] = 0.0
        return betas

    def residual(z: NDArray[np.float64]) -> float:
        a = float(z[0]); betas = _full_betas(z)
        try:
            y = bank.target_exact(a, alpha_mode=selected_alpha_mode)
            X = bank.library_exact(a, p_tuple, betas, alpha_mode=selected_alpha_mode)
        except TypeError:
            y = bank.target_exact(a)
            X = bank.library_exact(a, p_tuple, betas)
        finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        Xf, yf = X[finite], y[finite]
        if Xf.shape[0] == 0:
            return 1e6
        s = np.linalg.norm(Xf, axis=0); s[s < EPS] = 1.0
        Xn = Xf / s[None, :]
        coef = np.linalg.solve(Xn.T @ Xn + float(ridge) * np.eye(Xn.shape[1]), Xn.T @ yf) / s
        return float(np.linalg.norm(yf - Xf @ coef) / (np.linalg.norm(yf) + EPS))

    beta_bounds = [
        (max(b_pos_min, betas0[j] - delta), min(b_hi, betas0[j] + delta)) for j in free_idx
    ]
    try:
        if selected_alpha_mode == "integer":
            z0_beta = np.array([betas0[j] for j in free_idx], dtype=float)
            def residual_integer(z_beta: NDArray[np.float64]) -> float:
                z_full = np.concatenate(([1.0], np.asarray(z_beta, dtype=float)))
                return residual(z_full)
            if z0_beta.size:
                z_beta, _ = refine_orders_local(residual_integer, z0_beta, beta_bounds, maxiter=120)
                z = np.concatenate(([1.0], np.asarray(z_beta, dtype=float)))
            else:
                z = np.array([1.0], dtype=float)
        else:
            z0 = np.array([float(model.alpha)] + [betas0[j] for j in free_idx], dtype=float)
            bounds = [(a_lo, a_hi)] + beta_bounds
            z, _ = refine_orders_local(residual, z0, bounds, maxiter=120)
    except Exception:
        return model
    betas_new = _full_betas(z)
    return PDEModel(
        c=int(model.c), alpha=float(z[0]), p_tuple=p_tuple,
        beta_tuple=tuple(float(v) for v in betas_new),
        coefficients=np.asarray(model.coefficients, dtype=float),
        train_mse=float(model.train_mse), val_mse=float(model.val_mse),
        train_rel_mse=float(model.train_rel_mse), val_rel_mse=float(model.val_rel_mse),
        aic=float(model.aic), bic=float(model.bic), objective=float(model.objective),
        n_train=int(model.n_train), n_val=int(model.n_val), backend=str(model.backend),
        alpha_mode=selected_alpha_mode,
        full_data_rel_l2=float(model.full_data_rel_l2),
    )

def refit_selected_exact(bank: Any, model: PDEModel, *, ridge: float = 1e-3) -> tuple[PDEModel, float]:
    """Refit a selected model's coefficients at its EXACT continuous orders.

    Differential evolution stores continuous ``alpha``/``beta`` but obtains the
    coefficients from features *interpolated* across the precompute order grid.
    This recomputes the weak target and library directly at
    ``(alpha, beta_tuple)`` and refits the coefficients by the same normalized
    ridge least squares.  Direct evaluation removes interpolation error from the
    final conditional refit.  The selected support, temporal mode, and
    optimization basin can nevertheless still depend on the order grid and search
    trajectory.

    The selection-stage meanings of ``train_rel_mse``, ``val_rel_mse``,
    ``objective``, and the heuristic AIC/BIC-type fields are preserved.  The
    full-data relative L2 residual is stored separately in
    ``full_data_rel_l2``.  Returns ``(refit_model, full_data_rel_l2)``.
    """
    if not (hasattr(bank, "target_exact") and hasattr(bank, "library_exact")):
        return model, float(model.full_data_rel_l2)
    terms = list(zip(model.p_tuple, model.beta_tuple))
    if not terms:
        return model, float(model.full_data_rel_l2)
    selected_alpha_mode = model_mode(float(model.alpha), getattr(model, "alpha_mode", None))
    coef, rel = fit_coefficients_for_structure(
        bank, float(model.alpha), terms, ridge=ridge, alpha_mode=selected_alpha_mode
    )
    refit = PDEModel(
        c=int(model.c), alpha=float(model.alpha),
        p_tuple=tuple(int(p) for p in model.p_tuple),
        beta_tuple=tuple(float(b) for b in model.beta_tuple),
        coefficients=np.asarray(coef, dtype=float),
        train_mse=float(model.train_mse), val_mse=float(model.val_mse),
        train_rel_mse=float(model.train_rel_mse),
        val_rel_mse=float(model.val_rel_mse),
        aic=float(model.aic), bic=float(model.bic), objective=float(model.objective),
        n_train=int(model.n_train), n_val=int(model.n_val),
        backend=str(model.backend), alpha_mode=selected_alpha_mode,
        full_data_rel_l2=float(rel),
    )
    return refit, float(rel)


def coefficient_truth(dataset_name: str) -> list[float] | None:
    """Return nominal benchmark coefficients in truth-term order, if known.

    These values are used only for post-hoc reporting/tolerance sweeps, never
    during fitting or model selection.  ``None`` means coefficient scoring is
    unavailable or not meaningful for that dataset/operator convention.
    """
    return {
        "paper_ADE_Convection_diffusion": [-1.0, 0.25],
        "paper_FADE_tsfade_fft": [-1.0, 0.5],
        "synthetic_space_fractional_RD": [0.04, 0.18],
        "synthetic_time_space_fractional_RD": [0.03, 0.12],
        "synthetic_two_fractional_rhs": [0.05, 0.005],
        "synthetic_fractional_burgers": [-1.0, 0.25],
    }.get(dataset_name)


def model_order_metrics(
    model: dict[str, Any],
    expected_alpha: float,
    expected_terms: Sequence[tuple[int, float]],
    *,
    alpha_tol: float = 0.075,
    beta_tol: float = 0.10,
    coef_rel_tol: float = 1e-3,
    coef_abs_tol: float = 1e-8,
) -> dict[str, float | int | str | bool]:
    """Compare a discovered equation with benchmark truth metadata.

    This scorer deliberately separates **symbolic structure recovery** from
    **continuous-order/ coefficient accuracy**.

    ``full_structure_recovered`` answers only the structural question:

    * Did the selected equation use the expected number of active RHS terms?
    * Did the active RHS terms have the expected symbolic powers ``p``?

    It does **not** require ``alpha`` or ``beta`` to be within a tolerance.
    The continuous fractional orders are reported separately through
    ``alpha_abs_error``, ``mean_matched_beta_abs_error``, and
    ``max_matched_beta_abs_error``.  The optional boolean diagnostics
    ``alpha_hit`` and ``beta_order_recovered`` use the visible tolerances only
    as convenience summaries; they are not used to define full structural
    recovery.

    This convention matches the paper framing: first ask whether the candidate
    equation has the right symbolic form/support, then quantify how accurate the
    recovered fractional orders and coefficients are.
    """
    raw_terms = [(int(p), float(b)) for p, b in model.get("terms", [])]
    raw_coef = [float(c) for c in model.get("coefficients", [])]

    # Drop numerically zero selected terms before scoring support size.  This is
    # not an oracle threshold; it only prevents an exactly/near-zero least-squares
    # coefficient from being counted as an active discovered term.
    if len(raw_coef) == len(raw_terms) and raw_terms:
        scale = max(abs(c) for c in raw_coef)
        threshold = max(float(coef_abs_tol), float(coef_rel_tol) * scale)
        terms = [tb for tb, c in zip(raw_terms, raw_coef) if abs(c) > threshold]
    else:
        threshold = float(coef_abs_tol)
        terms = raw_terms

    unmatched = terms.copy()
    beta_errors: list[float] = []
    structural_hits = 0
    beta_order_hits = 0
    matched_terms: list[dict[str, float | int | bool]] = []

    for p_true, b_true in expected_terms:
        # Structure matching uses symbolic power only.  Beta closeness is an
        # accuracy metric, not a prerequisite for full_structure_recovered.
        candidates = [(i, abs(b - b_true)) for i, (p, b) in enumerate(unmatched) if p == p_true]
        if candidates:
            i, err = min(candidates, key=lambda z: z[1])
            p_sel, b_sel = unmatched.pop(i)
            structural_hits += 1
            beta_ok = bool(err <= beta_tol)
            beta_order_hits += int(beta_ok)
            beta_errors.append(float(err))
            matched_terms.append({
                "expected_p": int(p_true),
                "expected_beta": float(b_true),
                "selected_p": int(p_sel),
                "selected_beta": float(b_sel),
                "beta_abs_error": float(err),
                "structural_hit": True,
                "beta_hit": beta_ok,
            })
        else:
            beta_errors.append(float("inf"))
            matched_terms.append({
                "expected_p": int(p_true),
                "expected_beta": float(b_true),
                "selected_p": -1,
                "selected_beta": float("nan"),
                "beta_abs_error": float("inf"),
                "structural_hit": False,
                "beta_hit": False,
            })

    selected_c = int(len(terms))
    expected_c = len(expected_terms)
    support_size_match = bool(selected_c == expected_c)

    structural_precision = float(structural_hits / max(1, selected_c))
    structural_recall = float(structural_hits / max(1, expected_c))
    structural_f1 = float(
        0.0 if structural_precision + structural_recall == 0
        else 2.0 * structural_precision * structural_recall / (structural_precision + structural_recall)
    )

    beta_precision = float(beta_order_hits / max(1, selected_c))
    beta_recall = float(beta_order_hits / max(1, expected_c))
    beta_f1 = float(0.0 if beta_precision + beta_recall == 0 else 2.0 * beta_precision * beta_recall / (beta_precision + beta_recall))

    alpha_abs_error = float(abs(float(model.get("alpha", np.nan)) - float(expected_alpha)))
    alpha_hit = bool(np.isfinite(alpha_abs_error) and alpha_abs_error <= alpha_tol)
    beta_order_recovered = bool(support_size_match and beta_order_hits == expected_c)

    rhs_structure_recovered = bool(support_size_match and structural_hits == expected_c)
    full_structure_recovered = rhs_structure_recovered
    structure_and_orders_recovered = bool(full_structure_recovered and alpha_hit and beta_order_recovered)

    if full_structure_recovered:
        label = "full_structure_recovered"
    elif structural_hits > 0:
        label = "partial_structure_recovered"
    else:
        label = "structure_not_recovered"

    finite_beta_errors = [e for e in beta_errors if np.isfinite(e)]
    return {
        "alpha_abs_error": alpha_abs_error,
        "alpha_hit": alpha_hit,
        "alpha_tol": float(alpha_tol),
        "mean_matched_beta_abs_error": float(np.mean(finite_beta_errors)) if finite_beta_errors else float("inf"),
        "max_matched_beta_abs_error": float(np.max(finite_beta_errors)) if finite_beta_errors else float("inf"),
        "beta_hits": int(beta_order_hits),  # backward-compatible name for order hits
        "beta_order_hits": int(beta_order_hits),
        "beta_tol": float(beta_tol),
        "beta_order_recovered": beta_order_recovered,
        "structural_hits": int(structural_hits),
        "rhs_precision": structural_precision,
        "rhs_recall": structural_recall,
        "rhs_f1": structural_f1,
        "beta_order_precision": beta_precision,
        "beta_order_recall": beta_recall,
        "beta_order_f1": beta_f1,
        "selected_c": selected_c,
        "raw_selected_c": int(model.get("c", len(raw_terms))),
        "effective_terms_json": json.dumps([[int(p), float(b)] for p, b in terms]),
        "coefficient_prune_threshold": float(threshold),
        "expected_c": int(expected_c),
        "support_size_error": int(abs(selected_c - expected_c)),
        "support_size_match": support_size_match,
        "rhs_structure_recovered": rhs_structure_recovered,
        "full_structure_recovered": full_structure_recovered,
        "structure_and_orders_recovered": structure_and_orders_recovered,
        "recovery_label": label,
        "matched_terms_json": json.dumps(matched_terms),
    }

