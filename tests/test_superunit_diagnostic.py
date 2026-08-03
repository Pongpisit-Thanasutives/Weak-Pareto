"""Focused tests for the semi-analytic superunit Caputo diagnostic."""
from __future__ import annotations

import dataclasses

import numpy as np

from fpde_datasets import make_superunit_fractional_diffusion
from pareto_fde_discovery import DiscoveryConfig
from fractional_weak_form import caputo_l1_adjoint_tests, caputo_l1_matrix
from weak_pareto_fde_discovery import (
    WeakFractionalFeatureBank,
    build_weak_candidate_library,
    run_weak_pareto_discovery,
)
from reproduce.run_superunit_experiment import WEAK_TIME_DISCRETIZATION, WEAK_TIME_FORM


def _smoke_config(noise_percent: float) -> DiscoveryConfig:
    return DiscoveryConfig(
        backend="spectral_l1",
        alpha_grid=tuple(np.linspace(0.65, 1.85, 35)),
        beta_grid=tuple(np.linspace(1.50, 2.50, 25)),
        cmax=1,
        p_values=(0,),
        maxiter=6,
        popsize=4,
        polish=False,
        seed=0,
        noise_percent=float(noise_percent),
        val_fraction=0.25,
        trim_t=2,
        trim_x=0,
        spectral_riesz=False,
        selection="elbow",
        exact_order_refit=True,
        exact_order_polish=True,
        auto_stop=False,
        progress=False,
        progress_de=False,
        ridge=1e-4,
    )


def test_superunit_generator_has_expected_initial_conditions() -> None:
    data = make_superunit_fractional_diffusion(nt=80, nx=64)
    expected_u0 = (
        np.sin(data.x)
        + 0.35 * np.sin(2.0 * data.x + 0.3)
        + 0.15 * np.sin(3.0 * data.x - 0.2)
    )
    np.testing.assert_allclose(data.U[0], expected_u0, rtol=0.0, atol=2e-14)
    # E_{alpha,1}(-lambda t^alpha) has zero first derivative at t=0 for alpha>1.
    dt = data.t[1] - data.t[0]
    initial_slope = (data.U[1] - data.U[0]) / dt
    assert np.linalg.norm(initial_slope) / np.linalg.norm(data.U[0]) < 0.08
    assert data.t[-1] == 2.0
    assert data.x[-1] < 2.0 * np.pi


def test_weak_superunit_smoke_recovers_clean_and_noisy_branch() -> None:
    data = make_superunit_fractional_diffusion(nt=100, nx=64)
    for noise in (0.0, 0.5):
        cfg = _smoke_config(noise)
        summary = run_weak_pareto_discovery(
            data,
            cfg,
            output_dir=None,
            verbose=False,
            test_budget="standard",
            test_counts=(12, 16),
        )
        selected = summary["selected"]
        assert selected["alpha_mode"] == "fractional_superunit"
        assert abs(float(selected["alpha"]) - 1.65) < 0.15
        assert abs(float(selected["beta_tuple"][0]) - 2.0) < 0.15


def test_reported_superunit_protocol_uses_composed_l1_with_implicit_initial_rate() -> None:
    data = make_superunit_fractional_diffusion(nt=48, nx=32)
    cfg = _smoke_config(0.0)
    bank = build_weak_candidate_library(
        data,
        cfg,
        test_budget="smoke",
        test_counts=(6, 8),
        time_form=WEAK_TIME_FORM,
        verbose=False,
    )
    assert bank.time_form == "derivative"
    assert bank.time_discretization == WEAK_TIME_DISCRETIZATION == "caputo_l1_adjoint"

    h = float(data.t[1] - data.t[0])
    direct = caputo_l1_matrix(data.t.size, 1.65, h)
    composed = caputo_l1_matrix(data.t.size, 0.65, h) @ caputo_l1_matrix(data.t.size, 1.0, h)
    np.testing.assert_allclose(direct.toarray(), composed.toarray(), rtol=1e-12, atol=2e-12)


def test_superunit_adjoint_weights_are_endpoint_dominated() -> None:
    data = make_superunit_fractional_diffusion(nt=120, nx=64)
    cfg = _smoke_config(0.0)
    bank = WeakFractionalFeatureBank(
        data,
        cfg,
        test_budget="paper",
        test_counts=(24, 32),
        time_form=WEAK_TIME_FORM,
    )
    superunit = caputo_l1_adjoint_tests(bank.time_tests, data.t, 1.65)
    subunit = caputo_l1_adjoint_tests(bank.time_tests, data.t, 0.80)

    def endpoint_share(weights: np.ndarray) -> np.ndarray:
        return np.sum(weights[:, :2] ** 2, axis=1) / np.sum(weights**2, axis=1)

    super_share = endpoint_share(superunit)
    sub_share = endpoint_share(subunit)
    assert float(np.min(super_share)) > 0.95
    assert float(np.median(super_share)) > 0.98
    assert float(np.median(super_share)) > 5.0 * float(np.median(sub_share))
    assert superunit[0, 0] * superunit[0, 1] < 0.0
