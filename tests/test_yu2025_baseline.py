from pathlib import Path

import numpy as np
import pytest

from dataset_configs import benchmark_spec
from external_baselines.yu2025 import YuBaselineConfig, resolve_device, run_yu_baseline


def test_device_auto_resolves() -> None:
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}


def test_yu_spline_smoke_writes_common_schema(tmp_path: Path) -> None:
    spec = benchmark_spec(
        "paper_FADE_tsfade_fft",
        profile="notebook",
        noise_percent=0.0,
        seed=0,
    )
    cfg = YuBaselineConfig.for_profile(
        "smoke",
        surrogate="spline",
        seed=0,
        noise_percent=0.0,
        verbose=False,
    )
    result = run_yu_baseline(spec["data"], cfg, output_dir=tmp_path)
    selected = result.selected_model_dict()
    assert result.dataset == "paper_FADE_tsfade_fft"
    assert np.isfinite(result.alpha)
    assert np.isfinite(result.beta)
    assert selected["c"] == len(selected["terms"]) == len(selected["coefficients"])
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "selected_model.json").exists()
    assert (tmp_path / "shared_noisy_field.npy").exists()


def test_yu_rejects_periodic_riesz_dataset(tmp_path: Path) -> None:
    spec = benchmark_spec(
        "synthetic_space_fractional_RD",
        profile="notebook",
        noise_percent=0.0,
        seed=0,
    )
    cfg = YuBaselineConfig.for_profile("smoke", surrogate="spline", verbose=False)
    with pytest.raises(ValueError, match="outside the declared Yu adapter comparison scope"):
        run_yu_baseline(spec["data"], cfg, output_dir=tmp_path)

from scipy.special import gamma

from external_baselines.yu2025.yu_baseline import (
    _fractional_space_left,
    _fractional_time_caputo,
    _sparse_threshold_search,
)


class _PolynomialSurrogate:
    """Analytic u(x,t)=x^3+t^3 surrogate for quadrature tests."""

    def derivative(self, points: np.ndarray, *, dx: int = 0, dt: int = 0) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        x, t = p[:, 0], p[:, 1]
        if dx and dt:
            return np.zeros(len(p))
        if dx == 0 and dt == 0:
            return x**3 + t**3
        z, order = (x, dx) if dx else (t, dt)
        if order == 1:
            return 3.0 * z**2
        if order == 2:
            return 6.0 * z
        if order == 3:
            return np.full(len(p), 6.0)
        return np.zeros(len(p))


def test_gauss_jacobi_fractional_derivatives_match_polynomial_formula() -> None:
    surrogate = _PolynomialSurrogate()
    points = np.array([[0.8, 0.7], [1.2, 1.1], [1.8, 1.5]], dtype=float)
    alpha = 0.73
    beta = 1.61
    got_t = _fractional_time_caputo(surrogate, points, alpha, n_quad=5)
    got_x = _fractional_space_left(surrogate, points, beta, n_quad=5)
    expected_t = gamma(4.0) / gamma(4.0 - alpha) * points[:, 1] ** (3.0 - alpha)
    expected_x = gamma(4.0) / gamma(4.0 - beta) * points[:, 0] ** (3.0 - beta)
    np.testing.assert_allclose(got_t, expected_t, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(got_x, expected_x, rtol=2e-12, atol=2e-12)


def test_sparse_fit_uses_training_design_for_condition_penalty() -> None:
    # Validation rows are intentionally almost collinear and badly scaled. The
    # reported condition number must still equal the training-only design.
    x_train = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0], [-1.0, 2.0]])
    x_val = np.array([[1e8, 1e8], [2e8, 2e8 + 1e-6]])
    R = np.vstack([x_train, x_val])
    y = R @ np.array([2.0, -0.5])
    cfg = YuBaselineConfig.for_profile(
        "smoke",
        surrogate="spline",
        threshold_search_iters=0,
        l0_weight=0.0,
        verbose=False,
    )
    fit = _sparse_threshold_search(
        R,
        y,
        np.arange(len(x_train), dtype=np.int64),
        np.arange(len(x_train), len(R), dtype=np.int64),
        cfg,
    )
    np.testing.assert_allclose(fit.coefficients, [2.0, -0.5], rtol=1e-11, atol=1e-11)
    assert fit.condition_number == pytest.approx(np.linalg.cond(x_train))


def test_yu_spline_exact_order_path(tmp_path: Path) -> None:
    spec = benchmark_spec(
        "paper_FADE_tsfade_fft",
        profile="notebook",
        noise_percent=0.0,
        seed=2,
    )
    cfg = YuBaselineConfig.for_profile(
        "smoke",
        surrogate="spline",
        seed=2,
        order_mode="exact",
        exact_order_polish=True,
        exact_polish_maxiter=2,
        verbose=False,
    )
    result = run_yu_baseline(spec["data"], cfg, output_dir=tmp_path)
    assert np.isfinite(result.objective)
    assert result.metadata["order_mode"] == "exact"
    assert "exact_polish_seconds" in result.runtime_breakdown
