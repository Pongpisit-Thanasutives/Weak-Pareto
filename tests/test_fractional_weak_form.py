from __future__ import annotations

import numpy as np

from fpde_datasets import add_multiplicative_uniform_noise, make_space_fractional_advection_dispersion
from fractional_weak_form import (
    FractionalOperatorSpec,
    SeparableWeakLibrary2D,
    WeakTerm,
    adjoint_tests_1d,
    fit_least_squares,
    gaussian_test_matrix,
    gl_matrix,
    vanilla_riesz_library,
    caputo_l1_adjoint_tests,
)


def test_gl_discrete_adjoint_identity_left_and_right():
    rng = np.random.default_rng(123)
    n = 41
    h = 0.05
    grid = np.arange(n) * h
    f = rng.normal(size=n)
    phi = rng.normal(size=(1, n))

    for side in ["left", "right", "symmetric"]:
        spec = FractionalOperatorSpec(kind="grunwald_letnikov", order=0.73, axis="t", side=side)
        D = gl_matrix(n, 0.73, h, side)
        lhs = float((D @ f) @ phi[0])
        adj_phi = adjoint_tests_1d(phi, grid, spec)[0]
        rhs = float(f @ adj_phi)
        assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_caputo_constant_has_zero_weak_derivative():
    t = np.linspace(0.0, 1.0, 50)
    x = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    U = np.ones((t.size, x.size)) * 3.7
    T = gaussian_test_matrix(t, centers=8, width=0.08, periodic=False)
    X = gaussian_test_matrix(x, centers=10, width=0.35, periodic=True)
    weak = SeparableWeakLibrary2D(t, x, T, X)

    caputo = FractionalOperatorSpec(kind="caputo", order=0.6, axis="t", side="left")
    target = weak.target(U, caputo)
    assert np.linalg.norm(target) < 1e-10


def test_caputo_l1_discrete_adjoint_matches_strong_l1():
    from fpde_derivatives import caputo_l1_time

    rng = np.random.default_rng(44)
    t = np.linspace(0.0, 1.0, 30)
    dt = t[1] - t[0]
    f = rng.normal(size=(t.size, 3))
    phi = rng.normal(size=(2, t.size))
    alpha = 0.72

    strong = caputo_l1_time(f, alpha, dt)
    # caputo_l1_time marks the initial row invalid in the strong-form API; the
    # weak matrix uses a zero initial derivative row, so replace NaN by zero.
    strong = np.nan_to_num(strong, nan=0.0)
    lhs = phi @ strong
    adj = caputo_l1_adjoint_tests(phi, t, alpha)
    rhs = adj @ f
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_weak_fractional_library_is_noise_robust_against_vanilla():
    beta = 1.65
    truth = np.array([0.04, 0.18])
    data = make_space_fractional_advection_dispersion(nt=80, nx=96, beta=beta, growth=truth[0], diff=truth[1], seed=11)
    U = add_multiplicative_uniform_noise(data.U, 10.0, seed=3)

    T = gaussian_test_matrix(data.t, centers=14, width=0.065, periodic=False)
    X = gaussian_test_matrix(data.x, centers=24, width=data.Lx / 36.0, periodic=True)
    weak = SeparableWeakLibrary2D(data.t, data.x, T, X)
    lhs = FractionalOperatorSpec(kind="caputo", order=1.0, axis="t", side="left")
    terms = [
        WeakTerm("u", FractionalOperatorSpec(kind="identity", order=0.0, axis="x")),
        WeakTerm(f"Riesz_x^{beta} u", FractionalOperatorSpec(kind="riesz", order=beta, axis="x")),
    ]
    bw = weak.target(U, lhs)
    Xw, _ = weak.build_library(U, terms)
    coef_w, _ = fit_least_squares(Xw, bw, ridge=1e-12, normalize=True)
    err_w = np.linalg.norm(coef_w - truth) / np.linalg.norm(truth)

    Xv, bv, _ = vanilla_riesz_library(U, data.t, data.x, beta=beta)
    coef_v, _ = fit_least_squares(Xv, bv, ridge=1e-12, normalize=True)
    err_v = np.linalg.norm(coef_v - truth) / np.linalg.norm(truth)

    assert err_w < 0.12
    assert err_v > 0.5
    assert err_v / err_w > 5.0


def test_fractional_integral_discrete_adjoint_identity():
    from fractional_weak_form import fractional_integral_matrix, fractional_integral_adjoint_tests

    rng = np.random.default_rng(2026)
    n = 32
    h = 0.04
    grid = np.arange(n) * h
    f = rng.normal(size=n)
    phi = rng.normal(size=(3, n))
    alpha = 0.63
    J = fractional_integral_matrix(n, alpha, h, "left")
    lhs = phi @ (J @ f)
    adj = fractional_integral_adjoint_tests(phi, grid, alpha, "left")
    rhs = adj @ f
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)
