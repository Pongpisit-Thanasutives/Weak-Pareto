from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines import grid_stridge_baseline, weak_grid_stridge_baseline, weak_fixed_library_stability_baseline
from fpde_datasets import make_time_space_fractional_advection_dispersion
from pareto_fde_discovery import DiscoveryConfig, run_pareto_discovery


def _small_config(seed: int = 7) -> DiscoveryConfig:
    return DiscoveryConfig(
        backend="spectral_l1",
        spectral_riesz=True,
        alpha_grid=(0.78, 0.82, 0.9, 1.0),
        beta_grid=(0.0, 1.0, 1.55, 1.9),
        cmax=2,
        p_values=(0,),
        max_patterns_per_c=1,
        maxiter=0,
        popsize=2,
        trim_t=5,
        trim_x=0,
        power_mode="raw",
        selection="elbow",
        seed=seed,
        progress=False,
    )


def test_vanilla_pareto_smoke_runs(tmp_path):
    data = make_time_space_fractional_advection_dispersion(nt=36, nx=40)
    cfg = _small_config()
    summary = run_pareto_discovery(data, cfg, output_dir=tmp_path / "pareto", verbose=False)
    selected = summary["selected"]
    assert selected["c"] in {1, 2}
    assert np.isfinite(selected["val_rel_mse"])
    assert (tmp_path / "pareto" / "summary.json").exists()


def test_grid_stridge_smoke_runs():
    data = make_time_space_fractional_advection_dispersion(nt=36, nx=40)
    cfg = _small_config(seed=8)
    result = grid_stridge_baseline(data, cfg, verbose=False, max_terms=3)
    assert result.support_size >= 0
    assert np.isfinite(result.val_rel_mse)


def test_weak_grid_stridge_smoke_runs():
    data = make_time_space_fractional_advection_dispersion(nt=36, nx=40)
    cfg = _small_config(seed=9)
    result = weak_grid_stridge_baseline(data, cfg, verbose=False, max_terms=2, test_budget="smoke")
    assert result.support_size >= 0
    assert np.isfinite(result.val_rel_mse)


def test_weak_fixed_stability_smoke_runs():
    data = make_time_space_fractional_advection_dispersion(nt=36, nx=40)
    cfg = _small_config(seed=10)
    result = weak_fixed_library_stability_baseline(
        data, cfg, verbose=False, max_terms=2, test_budget="smoke", width_scales=(1.0,), n_splits=1
    )
    assert result.support_size >= 0
    assert np.isfinite(result.val_rel_mse)


def test_benchmark_spec_loads_one_dataset_by_name():
    from dataset_configs import benchmark_spec, benchmark_specs, available_benchmark_dataset_names, load_benchmark

    assert "synthetic_space_fractional_RD" in available_benchmark_dataset_names()
    spec = benchmark_spec(
        "synthetic_space_fractional_RD",
        profile="notebook",
        noise_percent=0.1,
        seed=0,
        maxiter=0,
        popsize=2,
    )
    assert spec["dataset_name"] == "synthetic_space_fractional_RD"
    assert spec["data"].U.ndim == 2
    assert spec["config"].noise_percent == 0.1
    assert spec["truth_spec"].dataset_name == "synthetic_space_fractional_RD"

    data, cfg, truth = load_benchmark("synthetic_space_fractional_RD", profile="notebook", seed=0)
    assert data.U.ndim == 2
    assert cfg.cmax == 4
    assert truth.dataset_name == "synthetic_space_fractional_RD"

    specs = benchmark_specs(dataset_names=["synthetic_space_fractional_RD"], profile="notebook", seed=0)
    assert len(specs) == 1
    assert specs[0]["dataset_name"] == "synthetic_space_fractional_RD"


def test_weak_pareto_separated_steps_and_progress(tmp_path):
    from weak_pareto_fde_discovery import (
        build_weak_candidate_library,
        build_best_subset_pareto_problem,
        run_best_subset_pareto_de,
    )

    data = make_time_space_fractional_advection_dispersion(nt=36, nx=40)
    cfg = _small_config(seed=11)
    bank = build_weak_candidate_library(data, cfg, test_budget="smoke", verbose=False)
    assert bank.n_points > 0
    problem = build_best_subset_pareto_problem(bank, cfg)
    summary = run_best_subset_pareto_de(problem, cfg, data=data, output_dir=tmp_path / "weak_split", verbose=False)
    assert "support_size_progress" in summary
    assert len(summary["support_size_progress"]) >= 1
    assert (tmp_path / "weak_split" / "support_size_progress.csv").exists()
    assert (tmp_path / "weak_split" / "support_size_progress.json").exists()


def test_selected_model_prunes_low_contribution_terms():
    from pareto_fde_discovery import PDEModel, prune_inactive_selected_terms

    class DummyBank:
        def library(self, p_tuple, beta_tuple):
            # Three same-scale columns; the third term has negligible contribution
            # because its coefficient is tiny.
            x = np.linspace(0.0, 1.0, 20)
            return np.column_stack([np.ones_like(x), x, x**2])[:, : len(p_tuple)]

    class DummyOptimizer:
        bank = DummyBank()

        def evaluate(self, alpha, p_tuple, beta_tuple):
            return PDEModel(
                c=len(p_tuple),
                alpha=float(alpha),
                p_tuple=tuple(p_tuple),
                beta_tuple=tuple(beta_tuple),
                coefficients=np.ones(len(p_tuple)),
                train_mse=1.0,
                val_mse=1.0,
                train_rel_mse=1.0,
                val_rel_mse=1.0,
                aic=1.0,
                bic=1.0,
                objective=0.0,
                n_train=10,
                n_val=5,
                backend="dummy",
            )

    model = PDEModel(
        c=3,
        alpha=0.8,
        p_tuple=(0, 0, 2),
        beta_tuple=(1.0, 1.7, 0.83),
        coefficients=np.array([-1.0, 0.5, -1e-5]),
        train_mse=1.0,
        val_mse=1.0,
        train_rel_mse=1.0,
        val_rel_mse=1.0,
        aic=1.0,
        bic=1.0,
        objective=0.0,
        n_train=10,
        n_val=5,
    )
    pruned = prune_inactive_selected_terms(DummyOptimizer(), model, contribution_tol=1e-4, abs_tol=1e-10)
    assert pruned.c == 2
    assert pruned.p_tuple == (0, 0)
    assert pruned.beta_tuple == (1.0, 1.7)


def test_contribution_pruning_is_not_raw_coefficient_pruning():
    from pareto_fde_discovery import PDEModel, prune_inactive_selected_terms, term_contribution_ratios

    class DummyBank:
        def library(self, p_tuple, beta_tuple):
            # The first coefficient is small, but its column has a huge norm, so
            # its fitted contribution is not negligible and must not be pruned.
            x = np.ones(20)
            return np.column_stack([1.0e6 * x, x])[:, : len(p_tuple)]

    class DummyOptimizer:
        bank = DummyBank()

        def evaluate(self, alpha, p_tuple, beta_tuple):
            raise AssertionError("no pruning should occur, so evaluate should not be called")

    model = PDEModel(
        c=2,
        alpha=1.0,
        p_tuple=(0, 0),
        beta_tuple=(1.0, 2.0),
        coefficients=np.array([1e-5, 1.0]),
        train_mse=1.0,
        val_mse=1.0,
        train_rel_mse=1.0,
        val_rel_mse=1.0,
        aic=1.0,
        bic=1.0,
        objective=0.0,
        n_train=10,
        n_val=5,
    )
    ratios = term_contribution_ratios(DummyOptimizer(), model)
    assert ratios[0] > 0.1
    kept = prune_inactive_selected_terms(DummyOptimizer(), model, contribution_tol=1e-4, abs_tol=1e-10)
    assert kept.c == 2
    assert kept.p_tuple == (0, 0)


def test_structure_recovery_flag_uses_symbolic_support_after_pruning():
    """full_structure_recovered should be a symbolic support flag only.

    Alpha/beta/coefficient accuracy is reported separately.  Numerically inactive
    extra terms should not cause a correct two-term structure to be marked false.
    Significant extra terms should still fail the support-size check.
    """
    from weak_pareto_fde_discovery import model_order_metrics

    expected_terms = [(0, 1.0), (0, 1.7)]

    nearly_zero_extra = {
        "alpha": 0.8,
        "c": 3,
        "terms": [(0, 0.998), (0, 1.72), (2, 0.83)],
        "coefficients": [-0.998, 0.493, -1.0e-5],
    }
    metrics = model_order_metrics(nearly_zero_extra, 0.8, expected_terms, coef_rel_tol=1e-3)
    assert metrics["full_structure_recovered"] is True
    assert metrics["selected_c"] == 2
    assert metrics["raw_selected_c"] == 3

    significant_extra = {
        "alpha": 0.8,
        "c": 3,
        "terms": [(0, 0.998), (0, 1.72), (2, 0.83)],
        "coefficients": [-0.998, 0.493, -0.05],
    }
    metrics = model_order_metrics(significant_extra, 0.8, expected_terms, coef_rel_tol=1e-3)
    assert metrics["full_structure_recovered"] is False
    assert metrics["selected_c"] == 3


def test_structure_flag_does_not_require_beta_tolerance():
    """A model with the right symbolic powers/support is structurally recovered.

    The bad beta value should appear in the beta-error columns instead of changing
    full_structure_recovered to False.
    """
    from weak_pareto_fde_discovery import model_order_metrics

    metrics = model_order_metrics(
        {
            "alpha": 0.91,
            "c": 2,
            "terms": [(0, 0.1), (0, 3.0)],
            "coefficients": [-1.0, 0.5],
        },
        expected_alpha=0.8,
        expected_terms=[(0, 1.0), (0, 1.7)],
        alpha_tol=0.01,
        beta_tol=0.01,
    )
    assert metrics["full_structure_recovered"] is True
    assert metrics["structure_and_orders_recovered"] is False
    assert metrics["max_matched_beta_abs_error"] > 1.0
    assert metrics["alpha_abs_error"] > 0.1
