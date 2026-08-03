from __future__ import annotations

import numpy as np
from scipy.special import gamma

from fractional_weak_form import caputo_l1_matrix, periodic_riesz_on_tests
from pareto_fde_discovery import PDEModel, select_model


def _model(c: int, error: float) -> PDEModel:
    return PDEModel(
        c=c,
        alpha=1.0,
        p_tuple=(0,) * c,
        beta_tuple=tuple(float(i + 1) for i in range(c)),
        coefficients=np.ones(c),
        train_mse=error,
        val_mse=error,
        train_rel_mse=error,
        val_rel_mse=error,
        aic=0.0,
        bic=0.0,
        objective=float(np.log10(error)),
        n_train=10,
        n_val=10,
    )


def test_signed_elbow_selects_concave_knee() -> None:
    candidates = {1: _model(1, 1e-1), 2: _model(2, 1e-4), 3: _model(3, 8e-5)}
    assert select_model(candidates, "elbow").c == 2


def test_signed_elbow_rejects_convex_anti_elbow() -> None:
    # The middle point lies below the endpoint chord in benefit-complexity space.
    candidates = {1: _model(1, 1e-1), 2: _model(2, 8e-2), 3: _model(3, 1e-4)}
    assert select_model(candidates, "elbow").c == 1


def test_periodic_riesz_matches_fourier_mode() -> None:
    n = 64
    beta = 1.7
    mode = 3
    x = np.arange(n, dtype=float) * (2.0 * np.pi / n)
    f = np.sin(mode * x)
    numerical = periodic_riesz_on_tests(f[None, :], x, beta)[0]
    exact = -(abs(mode) ** beta) * f
    assert np.linalg.norm(numerical - exact) / np.linalg.norm(exact) < 1e-11


def test_caputo_l1_converges_on_power() -> None:
    alpha = 0.7
    q = 3.0
    errors = []
    for n in (65, 129, 257):
        t = np.linspace(0.0, 1.0, n)
        h = float(t[1] - t[0])
        numerical = np.asarray(caputo_l1_matrix(n, alpha, h) @ (t**q), dtype=float)
        exact = gamma(q + 1.0) / gamma(q - alpha + 1.0) * t ** (q - alpha)
        errors.append(np.linalg.norm(numerical[1:] - exact[1:]) / np.linalg.norm(exact[1:]))
    assert errors[2] < errors[1] < errors[0]
    rate = np.log(errors[1] / errors[2]) / np.log(2.0)
    assert rate > 1.15

def test_caputo_l1_superunit_converges_on_power() -> None:
    alpha = 1.3
    q = 3.0
    errors = []
    for n in (65, 129, 257):
        t = np.linspace(0.0, 1.0, n)
        h = float(t[1] - t[0])
        numerical = np.asarray(caputo_l1_matrix(n, alpha, h) @ (t**q), dtype=float)
        exact = gamma(q + 1.0) / gamma(q - alpha + 1.0) * t ** (q - alpha)
        # The composed branch uses two data-side operations; omit the first two
        # rows, where the one-sided start-up stencils dominate the norm.
        errors.append(np.linalg.norm(numerical[2:] - exact[2:]) / np.linalg.norm(exact[2:]))
    assert errors[2] < errors[1] < errors[0]
    rate = np.log(errors[1] / errors[2]) / np.log(2.0)
    assert rate > 1.05

