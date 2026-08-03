"""Regression tests for branch-aware Caputo temporal-order handling."""
from __future__ import annotations

import numpy as np

from fpde_datasets import GridDataset
from fpde_derivatives import caputo_l1_time
from pareto_fde_discovery import (
    DiscoveryConfig,
    FractionalFeatureBank,
    PDEModel,
    ParetoFDEOptimizer,
)
from temporal_modes import available_alpha_modes, branch_grid


def test_declared_interval_is_split_without_cross_integer_interpolation():
    grid = np.linspace(0.8, 1.15, 15)  # deliberately does not contain 1 exactly
    modes = available_alpha_modes(grid, branch_epsilon=1e-3)
    assert modes == ("fractional_subunit", "integer", "fractional_superunit")
    below = branch_grid(grid, "fractional_subunit", branch_epsilon=1e-3)
    above = branch_grid(grid, "fractional_superunit", branch_epsilon=1e-3)
    assert below[-1] == 0.999
    assert above[0] == 1.001
    assert np.all(below < 1.0)
    assert np.all(above > 1.0)


def test_strong_feature_bank_inserts_exact_integer_mode_off_grid():
    t = np.linspace(0.0, 1.0, 25)
    x = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    U = (t[:, None] ** 2) * np.sin(x)[None, :]
    data = GridDataset(U=U, t=t, x=x, name="branch_test", truth="", recommended_backend="spectral_l1")
    cfg = DiscoveryConfig(
        alpha_grid=tuple(np.linspace(0.8, 1.15, 14)),
        beta_grid=(1.0, 2.0),
        trim_t=0,
        trim_x=0,
        noise_percent=0.0,
        maxiter=1,
        popsize=2,
        progress=False,
    )
    assert not np.any(np.isclose(np.asarray(cfg.alpha_grid), 1.0))
    bank = FractionalFeatureBank(data, cfg)
    bank.precompute(verbose=False)
    assert "integer" in bank.available_alpha_modes()
    expected = caputo_l1_time(U, 1.0, data.dt).reshape(-1)
    obtained = bank.target(1.0, alpha_mode="integer")
    np.testing.assert_allclose(obtained, expected, rtol=0.0, atol=1e-13)
    assert bank.alpha_grid_for_mode("fractional_subunit").max() <= 1.0 - cfg.alpha_branch_epsilon
    assert bank.alpha_grid_for_mode("fractional_superunit").min() >= 1.0 + cfg.alpha_branch_epsilon


class _SyntheticBranchBank:
    """Small deterministic bank whose optimum temporal mode is prescribed."""

    def __init__(self, desired: str):
        self.desired = desired
        self.alpha_grid = np.linspace(0.85, 1.15, 13)
        self.beta_grid = np.array([1.0, 1.5])
        self._x = np.linspace(-1.0, 1.0, 24)
        self._z = np.cos(np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False))

    @property
    def n_points(self):
        return self._x.size

    def available_alpha_modes(self):
        return ("fractional_subunit", "integer", "fractional_superunit")

    def alpha_grid_for_mode(self, mode):
        return branch_grid(self.alpha_grid, mode, branch_epsilon=1e-3)

    def alpha_bounds(self, mode):
        g = self.alpha_grid_for_mode(mode)
        return float(g[0]), float(g[-1])

    def target(self, alpha, alpha_mode=None):
        mode = alpha_mode or ("integer" if abs(alpha - 1.0) < 1e-11 else ("fractional_subunit" if alpha < 1.0 else "fractional_superunit"))
        if self.desired == "integer":
            mismatch = 0.0 if mode == "integer" else 0.25 + abs(float(alpha) - 1.0)
        else:
            mismatch = abs(float(alpha) - 0.97) if mode == "fractional_subunit" else 0.35
        return self._x + mismatch * self._z

    def library(self, p_tuple, beta_tuple):
        return np.column_stack([self._x for _ in p_tuple])


def _select_mode(desired: str) -> PDEModel:
    cfg = DiscoveryConfig(
        alpha_grid=tuple(np.linspace(0.85, 1.15, 13)),
        beta_grid=(1.0, 1.5),
        maxiter=3,
        popsize=3,
        progress=False,
        progress_de=False,
        ridge=1e-10,
    )
    bank = _SyntheticBranchBank(desired)
    idx = np.arange(bank.n_points)
    opt = ParetoFDEOptimizer(bank, idx[:18], idx[18:], cfg)
    return opt.optimize_fixed_pattern((0,), seed=0)


def test_optimizer_selects_exact_integer_mode_off_fractional_branches():
    model = _select_mode("integer")
    assert model.alpha_mode == "integer"
    assert model.alpha == 1.0
    assert model.equation().startswith("partial_t u")


def test_optimizer_keeps_near_integer_fractional_truth_on_fractional_branch():
    model = _select_mode("fractional_subunit")
    assert model.alpha_mode == "fractional_subunit"
    assert model.alpha < 1.0
    assert abs(model.alpha - 0.97) < 0.04


def test_model_serialisation_records_temporal_mode():
    model = PDEModel(
        c=1,
        alpha=1.0,
        p_tuple=(0,),
        beta_tuple=(2.0,),
        coefficients=np.array([0.25]),
        train_mse=0.0,
        val_mse=0.0,
        train_rel_mse=0.0,
        val_rel_mse=0.0,
        aic=0.0,
        bic=0.0,
        objective=0.0,
        n_train=10,
        n_val=5,
        alpha_mode="integer",
    )
    payload = model.to_dict()
    assert payload["alpha_mode"] == "integer"
    assert payload["equation"].startswith("partial_t u")


def test_equation_renderer_preserves_fractional_riesz_order_and_integer_time_mode():
    from reproduce.make_equations import equation_latex

    text = equation_latex(
        "synthetic_space_fractional_RD",
        1.0,
        [(0, 1.015, -0.2)],
        "integer",
    )
    assert text.startswith(r"\partial_t u")
    assert r"\mathcal{R}_{1.01}u" in text
    assert "u_x" not in text

    nonlinear = equation_latex(
        "synthetic_fractional_burgers",
        1.0,
        [(1, 1.0, -1.0)],
        "integer",
    )
    assert r"u\,u_x" in nonlinear
    assert "u^{1}" not in nonlinear
