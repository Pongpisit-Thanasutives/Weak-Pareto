from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from two_dimensional import run_2d_experiments as runner
from two_dimensional import weak_pareto_2d as wp2


ROOT = Path(__file__).resolve().parents[1]
TWOD = ROOT / "two_dimensional"


def test_tensor_contraction_matches_einsum():
    rng = np.random.default_rng(3)
    U = rng.normal(size=(5, 6, 7))
    A = rng.normal(size=(2, 5))
    B = rng.normal(size=(3, 6))
    C = rng.normal(size=(4, 7))
    scale = 0.17
    got = wp2.contract(U, A, B, C, scale).reshape(2, 3, 4)
    expected = scale * np.einsum("txy,at,bx,cy->abc", U, A, B, C)
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)


def test_directional_discrete_adjoint_identity():
    rng = np.random.default_rng(5)
    n = 48
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(n, d=x[1] - x[0])
    u = rng.normal(size=n)
    tests = rng.normal(size=(4, n))
    beta = 1.37
    multiplier = np.zeros(n, dtype=complex)
    nz = k != 0
    multiplier[nz] = np.abs(k[nz]) ** beta * np.exp(
        0.5j * np.pi * beta * np.sign(k[nz])
    )
    Xu = np.fft.ifft(np.fft.fft(u) * multiplier).real
    adjoint_tests = wp2.conj_multiplier_adjoint(tests, k, beta)
    np.testing.assert_allclose(tests @ Xu, adjoint_tests @ u, rtol=1e-12, atol=1e-11)


def test_gram_objective_equals_explicit_interpolated_objective():
    with np.load(TWOD / "data" / "benchmark_A.npz") as archive:
        t, x, y, U = (archive[key] for key in ("t", "x", "y", "U"))
    lib = wp2.Library2D(
        U, t, x, y,
        np.linspace(0.55, 1.25, 9),
        np.linspace(0.5, 2.5, 11),
        Kt=5, Kx=6, Ky=6,
        spatial_width=0.08,
    )
    rng = np.random.default_rng(7)
    idx = rng.permutation(lib.K)
    va, tr = idx[: lib.K // 4], idx[lib.K // 4 :]
    lib.set_split(tr, va)
    alpha = 0.83
    terms = [("x", 0, 1.67), ("y", 0, 1.43)]
    _, xi_gram, eval_gram = wp2.objective(lib, alpha, terms, tr, va)

    columns = np.stack([lib.column(direction, beta) for direction, _, beta in terms], axis=1)
    target = lib.target(alpha)
    xi_explicit = wp2.ridge_fit(columns[tr], target[tr])
    residual = target[va] - columns[va] @ xi_explicit
    eval_explicit = np.mean(residual**2) / (np.var(target[va]) + wp2.EPS)
    np.testing.assert_allclose(xi_gram, xi_explicit, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(eval_gram, eval_explicit, rtol=1e-10, atol=1e-12)


def test_matching_is_optimal_within_direction():
    model = {
        "alpha": 0.85,
        "terms": [("x", 0, 1.65), ("x", 0, 1.02), ("y", 0, 1.41)],
        "xi": [0.31, -0.59, 0.19],
    }
    score = runner.match_and_score(model, runner.TRUTH["B"])
    assert score is not None
    assert score["e_beta_max"] < 0.06
    assert score["e_xi_max"] < 0.06


def test_reference_manifest_and_fields_are_consistent():
    manifest = json.loads((TWOD / "data" / "generation_manifest.json").read_text())
    assert set(manifest) == {"A", "B"}
    for benchmark in ("A", "B"):
        with np.load(TWOD / "data" / f"benchmark_{benchmark}.npz") as archive:
            assert archive["U"].shape == (90, 80, 80)
            assert np.isfinite(archive["U"]).all()
        assert manifest[benchmark]["nt"] == 90
        assert manifest[benchmark]["ng"] == 80
        assert manifest[benchmark]["l1_relative_residual"] < 0.01


def test_paper_rule_gives_reported_weak_row_count():
    kt, kx = wp2.paper_test_counts(90, 80)
    _, ky = wp2.paper_test_counts(90, 80)
    assert (kt, kx, ky) == (30, 40, 40)
    assert kt * kx * ky == 48_000


def test_reported_reference_archive_is_complete():
    records = runner.load_records(TWOD / "reference_results")
    assert len(records) == 70
    main = [row for row in records if row["experiment"] in {"A", "B"}]
    assert len(main) == 50
    assert all(row["support_recovered"] for row in main)
    assert all(row["operator_recovered"] for row in main)


def test_refined_grid_reference_archive_is_complete():
    records = runner.load_records(TWOD / "resolution_112" / "results")
    assert len(records) == 25
    assert {row["weak_rows"] for row in records} == {94_080}
    assert all(row["operator_recovered"] for row in records)


def test_paper_width_is_applied_on_full_periodic_length():
    with np.load(TWOD / "data" / "benchmark_A.npz") as archive:
        t, x, y, U = (archive[key] for key in ("t", "x", "y", "U"))
    kt, kx = wp2.paper_test_counts(t.size, x.size)
    _, expected = wp2.paper_widths(t, x, kx)
    L = (x[1] - x[0]) * x.size
    lib = wp2.Library2D(
        U, t, x, y, np.linspace(0.55, 1.25, 3), np.linspace(0.5, 2.5, 3),
        Kt=kt, Kx=kx, Ky=kx, spatial_width=expected / L,
    )
    assert np.isclose(lib.spatial_width_x, expected)
    assert np.isclose(lib.spatial_width_y, expected)


def test_reported_high_noise_summary_values_are_locked():
    import csv

    with (TWOD / "reference_results" / "summary_2d.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {
        (row["experiment"], row["width"], float(row["noise"])): row for row in rows
    }
    a20 = by_key[("A", "paper-rule", 0.2)]
    b20 = by_key[("B", "paper-rule", 0.2)]
    assert int(a20["operator_recovered"]) == 5
    assert int(b20["operator_recovered"]) == 5
    assert np.isclose(float(a20["e_alpha_mean"]), 0.0008127108351294154)
    assert np.isclose(float(a20["e_beta_max_mean"]), 0.002170364093191335)
    assert np.isclose(float(a20["e_xi_max_mean"]), 0.011120916193762823)
    assert np.isclose(float(b20["e_alpha_mean"]), 0.002074359462522746)
    assert np.isclose(float(b20["e_beta_max_mean"]), 0.012208403177883787)
    assert np.isclose(float(b20["e_xi_max_mean"]), 0.04797697134191103)
