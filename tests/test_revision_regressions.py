from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

import fpde_datasets
from fpde_datasets import GridDataset, load_npz_dataset, prepare_all_datasets, save_dataset_npz
from pareto_fde_discovery import (
    DiscoveryConfig, PDEModel, ParetoFDEOptimizer, pareto_front, select_model,
    support_size_sweep,
)
from scripts.run_yu2025_comparison import _input_hash_manifest
from weak_pareto_fde_discovery import refit_selected_exact


def _model(c: int, *, raw: float, normalized: float) -> PDEModel:
    return PDEModel(
        c=c,
        alpha=0.8,
        p_tuple=(0,) * c,
        beta_tuple=tuple(float(i + 1) for i in range(c)),
        coefficients=np.ones(c),
        train_mse=raw,
        val_mse=raw,
        train_rel_mse=normalized,
        val_rel_mse=normalized,
        aic=float(c),
        bic=float(c),
        objective=float(np.log10(normalized)),
        n_train=20,
        n_val=10,
    )


def test_pareto_front_uses_normalized_validation_error() -> None:
    # Raw MSE would keep c=2 because 0.5 < 1.0.  The dimensionless criterion
    # correctly removes it because c=1 is both smaller and better normalized.
    candidates = {
        1: _model(1, raw=1.0, normalized=0.10),
        2: _model(2, raw=0.5, normalized=0.20),
        3: _model(3, raw=0.4, normalized=0.05),
    }
    assert list(pareto_front(candidates)) == [1, 3]


def test_sparse_relaxed_uses_normalized_validation_error() -> None:
    candidates = {
        1: _model(1, raw=10.0, normalized=0.105),
        2: _model(2, raw=0.1, normalized=0.100),
    }
    # Within 8% of the best normalized score, so the smaller support wins. Raw
    # MSE would incorrectly choose c=2.
    assert select_model(candidates, "sparse_relaxed", sparse_relaxed_tol=0.08).c == 1




def test_de_objective_is_log10_normalized_validation_error() -> None:
    class Bank:
        beta_grid = np.array([0.5, 2.0])

        def target(self, alpha, alpha_mode=None):
            return np.array([0.0, 2.0, 1.0, 5.0])

        def library(self, p_tuple, beta_tuple):
            return np.zeros((4, len(p_tuple)))

    cfg = DiscoveryConfig(alpha_grid=(0.7, 0.9), beta_grid=(0.5, 2.0), p_values=(0,))
    opt = ParetoFDEOptimizer(Bank(), np.array([0, 1]), np.array([2, 3]), cfg)
    model = opt.evaluate(0.8, (0,), (1.0,), alpha_mode="fractional_subunit")
    assert model.objective == pytest.approx(np.log10(model.val_rel_mse))
    assert model.objective != pytest.approx(np.log10(model.val_mse))


def test_elbow_selection_uses_normalized_validation_error() -> None:
    candidates = {
        1: _model(1, raw=1.0, normalized=1.0),
        2: _model(2, raw=0.8, normalized=0.1),
        3: _model(3, raw=0.1, normalized=0.09),
    }
    # The normalized curve has a clear elbow at c=2. The raw-MSE curve would
    # have no positive interior elbow and would fall back to c=1.
    assert select_model(candidates, "elbow").c == 2


def test_support_stopping_uses_normalized_validation_error() -> None:
    class FakeOptimizer:
        def optimize_fixed_pattern(self, pattern, seed):
            c = len(pattern)
            return _model(
                c,
                raw={1: 1.0, 2: 0.1, 3: 0.01}[c],
                normalized={1: 1.0, 2: 0.99, 3: 0.98}[c],
            )

    cfg = DiscoveryConfig(
        cmax=3,
        p_values=(0,),
        max_patterns_per_c=1,
        auto_stop=True,
        auto_stop_min_c=2,
        auto_stop_patience=1,
        auto_stop_rel_improvement=0.02,
        auto_stop_log10_improvement=0.02,
        auto_stop_use_selection_stability=False,
        progress=False,
    )
    _, best, stop = support_size_sweep(FakeOptimizer(), cfg, verbose=False)
    assert list(best) == [1, 2]
    assert stop["stopped_early"] is True
    assert stop["stop_c"] == 2
    assert stop["history"][-1]["best_val_rel_mse"] == pytest.approx(0.99)


def test_exact_refit_preserves_selection_metrics_and_sets_full_data_field() -> None:
    class ExactBank:
        def target_exact(self, alpha, alpha_mode=None):
            return np.array([1.0, 2.0, 3.0, 4.0])

        def library_exact(self, alpha, p_tuple, beta_tuple, alpha_mode=None):
            return np.array([[1.0], [2.0], [3.0], [4.0]])

    original = PDEModel(
        c=1,
        alpha=0.8,
        p_tuple=(0,),
        beta_tuple=(1.7,),
        coefficients=np.array([0.2]),
        train_mse=7.0,
        val_mse=8.0,
        train_rel_mse=0.7,
        val_rel_mse=0.8,
        aic=11.0,
        bic=12.0,
        objective=np.log10(0.8),
        n_train=20,
        n_val=10,
    )
    refit, rel = refit_selected_exact(ExactBank(), original, ridge=0.0)
    assert refit.train_rel_mse == original.train_rel_mse
    assert refit.val_rel_mse == original.val_rel_mse
    assert refit.objective == original.objective
    assert refit.aic == original.aic
    assert refit.bic == original.bic
    assert refit.full_data_rel_l2 == pytest.approx(rel)
    assert refit.full_data_rel_l2 < 1e-12
    assert refit.to_dict()["full_data_rel_l2"] == pytest.approx(rel)


def _tiny_dataset(name: str) -> GridDataset:
    t = np.linspace(0.0, 1.0, 4)
    x = np.linspace(0.0, 2.0, 5, endpoint=False)
    return GridDataset(
        U=np.add.outer(t, x),
        t=t,
        x=x,
        name=name,
        truth="u_t = 0",
        recommended_backend="spectral_l1_riesz",
    )


def test_dataset_save_load_round_trip(tmp_path: Path) -> None:
    original = _tiny_dataset("roundtrip")
    path = save_dataset_npz(original, tmp_path / "nested" / "sample.npz")
    loaded = load_npz_dataset(path)
    np.testing.assert_allclose(loaded.U, original.U)
    np.testing.assert_allclose(loaded.t, original.t)
    np.testing.assert_allclose(loaded.x, original.x)
    assert loaded.name == original.name
    assert loaded.truth == original.truth
    assert loaded.recommended_backend == original.recommended_backend


def test_prepare_all_datasets_completes_and_writes_expected_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fpde_datasets, "make_space_fractional_advection_dispersion", lambda: _tiny_dataset("synthetic_space_fractional_RD"))
    monkeypatch.setattr(fpde_datasets, "make_time_space_fractional_advection_dispersion", lambda: _tiny_dataset("synthetic_time_space_fractional_RD"))
    monkeypatch.setattr(fpde_datasets, "make_two_fractional_rhs_dataset", lambda: _tiny_dataset("synthetic_two_fractional_rhs"))
    monkeypatch.setattr(fpde_datasets, "make_fractional_burgers", lambda: _tiny_dataset("synthetic_fractional_burgers"))
    monkeypatch.setattr(
        fpde_datasets,
        "make_superunit_fractional_diffusion",
        lambda: _tiny_dataset("synthetic_superunit_fractional_diffusion"),
    )
    paths = prepare_all_datasets(tmp_path)
    expected = {
        "synthetic_space_fractional_RD.npz",
        "synthetic_time_space_fractional_RD.npz",
        "synthetic_two_fractional_rhs.npz",
        "synthetic_fractional_burgers.npz",
        "synthetic_superunit_fractional_diffusion.npz",
    }
    assert {p.name for p in paths} == expected
    assert all(p.is_file() for p in paths)


def test_periodic_initial_condition_uses_endpoint_excluded_domain_length(monkeypatch: pytest.MonkeyPatch) -> None:
    class ZeroNoiseRng:
        def normal(self, size):
            return np.zeros(size)

    monkeypatch.setattr(np.random, "default_rng", lambda seed=0: ZeroNoiseRng())
    x = np.linspace(0.0, 20.0, 32, endpoint=False)
    got = fpde_datasets._periodic_initial_condition(x, seed=3)
    Lx = (x[1] - x[0]) * len(x)
    base = (
        0.8 * np.sin(2 * np.pi * x / Lx)
        + 0.45 * np.cos(4 * np.pi * x / Lx)
        + 0.25 * np.sin(6 * np.pi * x / Lx + 0.4)
    )
    fft = np.fft.fft(base)
    fft *= np.exp(-40.0 * np.fft.fftfreq(len(x)) ** 2)
    expected = np.fft.ifft(fft).real
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-13)


def test_burgers_time_grid_is_endpoint_excluded() -> None:
    data = fpde_datasets.make_fractional_burgers(
        nt=6, nx=8, nx_fine=16, t_end=0.04, dt_fine=0.002, Lx=4.0
    )
    np.testing.assert_allclose(data.t, np.arange(6) * (0.04 / 6))
    assert data.t[-1] < 0.04


def test_yu_manifest_does_not_claim_absent_upstream_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = _input_hash_manifest(["paper_FADE_tsfade_fft"], root / "data")
    row = manifest["paper_FADE_tsfade_fft"]
    assert row["yu_original_sha256"] is None
    assert row["byte_identical"] is False
    assert row["upstream_snapshot_included"] is False
    assert row["upstream_byte_identity_verified"] is False
    assert row["verification_status"] == "not_verified_upstream_snapshot_absent"


def _fake_python(tmp_path: Path, *, fail_on: str | None = None) -> tuple[Path, Path]:
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    log = tmp_path / "python_calls.log"
    script = bindir / "python3"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f"echo \"$*\" >> {str(log)!r}\n"
        + (f"[[ \"$*\" == *{fail_on!r}* ]] && exit 17\n" if fail_on else "")
        + "exit 0\n"
    )
    script.chmod(0o755)
    return bindir, log


def _run_smoke_with_fake_python(tmp_path: Path, *, fail_on: str | None = None) -> tuple[subprocess.CompletedProcess[str], str]:
    root = Path(__file__).resolve().parents[1]
    bindir, log = _fake_python(tmp_path, fail_on=fail_on)
    env = os.environ.copy()
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["FRACTIONAL_PARETO_DIR"] = str(root)
    proc = subprocess.run(
        [
            "bash", str(root / "reproduce" / "run_all.sh"), "--smoke", "--jobs", "1",
            "--outdir", str(tmp_path / "out"), "--figdir", str(tmp_path / "fig"),
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc, log.read_text() if log.exists() else ""


def test_smoke_workflow_is_burgers_only_and_skips_long_stages(tmp_path: Path) -> None:
    proc, calls = _run_smoke_with_fake_python(tmp_path)
    assert proc.returncode == 0, proc.stdout
    assert "-m pytest -q" in calls
    assert "smoke_burgers.py" in calls
    assert "--outdir" in calls and "--figdir" in calls
    assert "precompute_campaign.py" not in calls
    assert "make_tables.py" not in calls
    assert "make_figures.py" not in calls
    assert "make_ksens.py" not in calls
    assert "make_forward.py" not in calls
    assert "make_equations.py" not in calls
    assert "make_ablation.py" not in calls
    assert "skipping publication-scale stages" in proc.stdout


def test_smoke_workflow_propagates_stage_failure(tmp_path: Path) -> None:
    proc, calls = _run_smoke_with_fake_python(tmp_path, fail_on="smoke_burgers.py")
    assert "smoke_burgers.py" in calls
    assert proc.returncode == 17


def _run_full_with_fake_python(
    tmp_path: Path, *job_args: str, env_jobs: str | None = None
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Exercise the publication driver without launching numerical work."""
    root = Path(__file__).resolve().parents[1]
    bindir, log = _fake_python(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["FRACTIONAL_PARETO_DIR"] = str(root)
    if env_jobs is None:
        env.pop("FPDE_REPRO_JOBS", None)
    else:
        env["FPDE_REPRO_JOBS"] = env_jobs
    proc = subprocess.run(
        [
            "bash",
            str(root / "reproduce" / "run_all.sh"),
            *job_args,
            "--outdir",
            str(tmp_path / "out"),
            "--figdir",
            str(tmp_path / "fig"),
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc, log.read_text() if log.exists() else ""


def test_run_all_default_budget_handles_empty_optional_flags(tmp_path: Path) -> None:
    """Regression for Bash 3.2/`set -u` empty-array expansion failures."""
    proc, calls = _run_full_with_fake_python(tmp_path)
    assert proc.returncode == 0, proc.stdout
    assert "precompute_campaign.py --jobs 2" in calls
    assert "make_tables.py --outdir" in calls
    assert "make_figures.py --figdir" in calls
    assert "--fast" not in calls
    assert "workers      : 2" in proc.stdout


def test_run_all_jobs_cli_controls_parallel_workers(tmp_path: Path) -> None:
    proc, calls = _run_full_with_fake_python(tmp_path, "--jobs=3")
    assert proc.returncode == 0, proc.stdout
    assert "precompute_campaign.py --jobs 3" in calls
    assert "make_ksens.py --jobs 3" in calls
    assert "workers      : 3" in proc.stdout


def test_run_all_jobs_environment_and_short_option(tmp_path: Path) -> None:
    proc_env, calls_env = _run_full_with_fake_python(tmp_path / "env", env_jobs="4")
    assert proc_env.returncode == 0, proc_env.stdout
    assert "precompute_campaign.py --jobs 4" in calls_env

    proc_cli, calls_cli = _run_full_with_fake_python(tmp_path / "cli", "-j", "5", env_jobs="4")
    assert proc_cli.returncode == 0, proc_cli.stdout
    assert "precompute_campaign.py --jobs 5" in calls_cli
    assert "make_ksens.py --jobs 5" in calls_cli


def test_run_all_rejects_invalid_jobs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for args in (("--jobs", "0"), ("--jobs=abc",), ("--jobs",)):
        proc = subprocess.run(
            ["bash", str(root / "reproduce" / "run_all.sh"), *args],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert proc.returncode == 2, (args, proc.stdout)
        assert "jobs" in proc.stdout.lower()

