"""One canonical discovery configuration per benchmark dataset.

The project has several discovery methods: the proposed support-size-swept
Pareto-DE method, Grid-STRidge, fixed-cardinality DE, and the optional GDE3P
one-shot baseline.  A common source of unfair comparisons is letting each method
use a different derivative grid, backend, trimming rule, or power set.

This module prevents that problem.  For every built-in dataset, it defines a
single *dataset-tailored overcomplete search space* and returns it as a
``DiscoveryConfig``.  All methods should receive this same config for a given
``dataset_name``.

Only runtime controls such as ``maxiter`` and ``popsize`` should be adjusted for
quick notebook runs versus publication runs.  The candidate search space itself
(``alpha_grid``, ``beta_grid``, ``p_values``, backend, Riesz/directional flag,
trimming, and power mode) remains dataset-specific and shared across baselines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from fpde_datasets import GridDataset, available_dataset_names, load_dataset_by_name, resolve_data_dir
from pareto_fde_discovery import DiscoveryConfig, config_to_dict

ConfigProfile = Literal["notebook", "paper"]


@dataclass(frozen=True)
class DatasetTruthSpec:
    """Ground-truth metadata for one benchmark dataset.

    The optimizer must **not** use this object.  It is written to run manifests
    and used only to score whether a selected equation matches the known
    benchmark equation within declared tolerances.

    The tolerances are intentionally visible because structure recovery is never
    an exact string match for continuous-order models.  For example, recovering
    beta=1.70 as beta=1.68 may be scientifically correct, while recovering
    beta=0.50 instead of beta=1.70 is not.
    """

    dataset_name: str
    expected_c: int
    expected_alpha: float
    expected_terms: tuple[tuple[int, float], ...]
    description: str
    alpha_tol: float = 0.15  # declared recovery tolerance tau (matches the manuscript)
    beta_tol: float = 0.15
    note: str = (
        "Truth metadata is for evaluation only. The discovery methods receive "
        "only the data and the canonical overcomplete candidate configuration."
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize truth metadata for JSON manifests and result tables."""
        return {
            "dataset_name": self.dataset_name,
            "expected_c": int(self.expected_c),
            "expected_alpha": float(self.expected_alpha),
            "expected_terms": [[int(p), float(b)] for p, b in self.expected_terms],
            "alpha_tol": float(self.alpha_tol),
            "beta_tol": float(self.beta_tol),
            "description": self.description,
            "note": self.note,
        }


def _unique_sorted(values) -> tuple[float, ...]:
    """Return sorted floats with near-duplicates removed."""
    out: list[float] = []
    for v in sorted(float(x) for x in values):
        if not out or abs(v - out[-1]) > 1e-12:
            out.append(v)
    return tuple(out)


def _grid_with_required(start: float, stop: float, num: int, required: tuple[float, ...]) -> tuple[float, ...]:
    """Create a base/convenience grid with requested anchors. Publication runs replace it with a truth-agnostic uniform grid."""
    return _unique_sorted([*np.linspace(start, stop, num), *required])


DATASET_TRUTH_SPECS: dict[str, DatasetTruthSpec] = {
    "paper_ADE_Convection_diffusion": DatasetTruthSpec(
        dataset_name="paper_ADE_Convection_diffusion",
        expected_c=2,
        expected_alpha=1.0,
        expected_terms=((0, 1.0), (0, 2.0)),
        description="Paper integer-order ADE control: D_t u = -D_x u + 0.25 D_x^2 u.",
    ),
    "paper_FADE_tsfade_fft": DatasetTruthSpec(
        dataset_name="paper_FADE_tsfade_fft",
        expected_c=2,
        expected_alpha=0.8,
        expected_terms=((0, 1.0), (0, 1.7)),
        description="Paper time-space fractional FADE benchmark: D_t^0.8 u = -D_x u + 0.5 D_x^1.7 u.",
    ),
    "synthetic_space_fractional_RD": DatasetTruthSpec(
        dataset_name="synthetic_space_fractional_RD",
        expected_c=2,
        expected_alpha=1.0,
        expected_terms=((0, 0.0), (0, 1.65)),
        description="Synthetic Riesz reaction-diffusion: D_t u = a u + b Riesz_x^1.65 u.",
    ),
    "synthetic_time_space_fractional_RD": DatasetTruthSpec(
        dataset_name="synthetic_time_space_fractional_RD",
        expected_c=2,
        expected_alpha=0.82,
        expected_terms=((0, 0.0), (0, 1.55)),
        description="Synthetic Caputo-time/Riesz-space benchmark: D_t^0.82 u = a u + b Riesz_x^1.55 u.",
    ),
    "synthetic_two_fractional_rhs": DatasetTruthSpec(
        dataset_name="synthetic_two_fractional_rhs",
        expected_c=2,
        expected_alpha=1.0,
        expected_terms=((0, 0.55), (0, 2.80)),
        description="Synthetic benchmark with two true fractional RHS operators.",
    ),
    "synthetic_superunit_fractional_diffusion": DatasetTruthSpec(
        dataset_name="synthetic_superunit_fractional_diffusion",
        expected_c=1,
        expected_alpha=1.65,
        expected_terms=((0, 2.0),),
        description=(
            "Focused superunit Caputo diagnostic: "
            "D_t^1.65 u = 0.12 D_x^2 u with zero initial velocity."
        ),
    ),
    "synthetic_fractional_burgers": DatasetTruthSpec(
        dataset_name="synthetic_fractional_burgers",
        expected_c=2,
        expected_alpha=1.0,
        expected_terms=((1, 1.0), (0, 1.7)),
        description="Synthetic NONLINEAR fractional Burgers: D_t u = -u u_x + 0.25 D_x^1.7 u.",
    ),
}


def dataset_truth_spec(dataset_name: str) -> DatasetTruthSpec:
    """Return expected structure metadata for ``dataset_name``."""
    try:
        return DATASET_TRUTH_SPECS[dataset_name]
    except KeyError as exc:
        valid = ", ".join(DATASET_TRUTH_SPECS)
        raise ValueError(f"Unknown dataset_name={dataset_name!r}. Valid names: {valid}") from exc


def dataset_config_philosophy() -> dict[str, str]:
    """Human-readable rules used to define the canonical dataset configs.

    These rules are also copied into experiment manifests so the benchmark is
    auditable.  The configs are tailored to the physics of each dataset, but
    they are not oracle models: they do not contain the true coefficients, they
    allow extra powers and support sizes, and they include competing derivative
    orders around the expected one.
    """
    return {
        "one_config_per_dataset": (
            "Every method uses the same dataset-specific candidate search space. "
            "The notebook/paper profile changes only compute budget and synthetic "
            "grid size, not the hypothesis class."
        ),
        "not_cheating_rule": (
            "The publication campaign uses dense, truth-agnostic uniform fractional-order "
            "grids over declared benchmark-specific ranges; no noninteger benchmark truth "
            "is inserted into a grid or optimiser initialisation. All methods share the "
            "same public model class: overcomplete terms u^p D_x^beta u with p in {0,1,2} "
            "and support size c <= 4. This class contains competing nonlinear candidates and extra "
            "support slots, so it is not oracle support. The optimizer must choose the "
            "support size, powers, beta values, alpha value, and coefficients from the "
            "shared search space."
        ),
        "truth_usage": (
            "Ground-truth structures are used only after fitting to score recovery "
            "and are written separately as truth_spec in the run manifest."
        ),
    }


def dataset_candidate_config(dataset_name: str) -> DiscoveryConfig:
    """Return the canonical overcomplete search space for one dataset.

    This function is the key fairness rule in the project: all discovery methods
    must use the returned candidate space for the dataset of interest.  It defines
    benchmark-specific ranges and a broad candidate class.  The publication
    campaign subsequently applies :func:`with_uniform_order_grids`, so no
    noninteger benchmark-truth value is deliberately inserted into the grids.

    Runtime parameters are intentionally moderate placeholders and are usually
    overwritten by :func:`recommended_discovery_config`.

    Most bundled benchmark equations are linear with two right-hand-side terms,
    while the fractional Burgers benchmark contains a nonlinear convective term.
    The publication search class is deliberately broader: ``cmax=4`` and
    ``p_values=(0,1,2)``.  This allows competing nonlinear candidate terms of the form
    ``u^p D_x^beta u`` and extra support sizes, while the optimizer must still
    select the parsimonious governing structure.  For stress tests, scripts may override ``cmax`` up to 5 with
    ``--cmax 5``; for fast debugging, scripts may override runtime budgets
    without changing the mathematical implementation.
    """
    if dataset_name == "paper_ADE_Convection_diffusion":
        return DiscoveryConfig(
            backend="regularized",
            alpha_grid=_grid_with_required(0.80, 1.20, 9, (1.00,)),
            beta_grid=_grid_with_required(0.70, 2.00, 27, (1.00, 2.00)),
            cmax=4,
            p_values=(0, 1, 2),
            max_patterns_per_c=None,
            maxiter=20,
            popsize=6,
            seed=11,
            val_fraction=0.25,
            trim_t=3,
            trim_x=6,
            lam_t=1e-8,
            lam_x=1e-6,
            power_mode="positive",
            spectral_riesz=False,
            selection="elbow",
            auto_stop=True,
            auto_stop_min_c=2,
            auto_stop_patience=1,
            auto_stop_rel_improvement=0.03,
            auto_stop_log10_improvement=0.02,
            auto_stop_use_selection_stability=True,
            auto_stop_selection_patience=1,
        )

    if dataset_name == "paper_FADE_tsfade_fft":
        return DiscoveryConfig(
            backend="spectral_l1",
            # Deliberately overcomplete but includes alpha=0.8 exactly.
            alpha_grid=_grid_with_required(0.60, 1.05, 19, (0.80,)),
            # Directional weak GL/RL operators are implemented up to order 2.
            # The grid is still overcomplete and includes beta=1 and 1.7 exactly.
            beta_grid=_grid_with_required(0.50, 2.00, 31, (1.00, 1.70)),
            cmax=4,
            p_values=(0, 1, 2),
            max_patterns_per_c=None,
            maxiter=20,
            popsize=6,
            seed=15,
            val_fraction=0.25,
            trim_t=5,
            trim_x=0,
            power_mode="positive",
            spectral_riesz=False,
            selection="elbow",
            auto_stop=True,
            auto_stop_min_c=2,
            auto_stop_patience=1,
            auto_stop_rel_improvement=0.03,
            auto_stop_log10_improvement=0.02,
            auto_stop_use_selection_stability=True,
            auto_stop_selection_patience=1,
        )

    if dataset_name == "synthetic_space_fractional_RD":
        return DiscoveryConfig(
            backend="spectral_l1",
            spectral_riesz=True,
            alpha_grid=_grid_with_required(0.80, 1.15, 15, (1.00,)),
            beta_grid=_grid_with_required(0.00, 2.10, 22, (0.00, 1.65)),
            cmax=4,
            p_values=(0, 1, 2),
            max_patterns_per_c=None,
            maxiter=16,
            popsize=5,
            seed=21,
            val_fraction=0.25,
            trim_t=3,
            trim_x=0,
            power_mode="raw",
            selection="elbow",
            auto_stop=True,
            auto_stop_min_c=2,
            auto_stop_patience=1,
            auto_stop_rel_improvement=0.03,
            auto_stop_log10_improvement=0.02,
            auto_stop_use_selection_stability=True,
            auto_stop_selection_patience=1,
        )

    if dataset_name == "synthetic_time_space_fractional_RD":
        return DiscoveryConfig(
            backend="spectral_l1",
            spectral_riesz=True,
            alpha_grid=_grid_with_required(0.65, 1.00, 15, (0.82,)),
            beta_grid=_grid_with_required(0.00, 1.90, 20, (0.00, 1.55)),
            cmax=4,
            p_values=(0, 1, 2),
            max_patterns_per_c=None,
            maxiter=16,
            popsize=5,
            seed=31,
            val_fraction=0.25,
            trim_t=5,
            trim_x=0,
            power_mode="raw",
            selection="elbow",
            auto_stop=True,
            auto_stop_min_c=2,
            auto_stop_patience=1,
            auto_stop_rel_improvement=0.03,
            auto_stop_log10_improvement=0.02,
            auto_stop_use_selection_stability=True,
            auto_stop_selection_patience=1,
        )

    if dataset_name == "synthetic_two_fractional_rhs":
        return DiscoveryConfig(
            backend="spectral_l1",
            spectral_riesz=True,
            alpha_grid=_grid_with_required(0.80, 1.15, 15, (1.00,)),
            beta_grid=_grid_with_required(0.30, 3.10, 57, (0.55, 2.80)),
            cmax=4,
            p_values=(0, 1, 2),
            max_patterns_per_c=None,
            maxiter=24,
            popsize=7,
            seed=41,
            val_fraction=0.25,
            trim_t=3,
            trim_x=0,
            power_mode="raw",
            selection="elbow",
            auto_stop=True,
            auto_stop_min_c=2,
            auto_stop_patience=1,
            auto_stop_rel_improvement=0.02,
            auto_stop_log10_improvement=0.015,
            auto_stop_use_selection_stability=True,
            auto_stop_selection_patience=1,
        )

    if dataset_name == "synthetic_superunit_fractional_diffusion":
        return DiscoveryConfig(
            backend="spectral_l1",
            spectral_riesz=False,
            alpha_grid=tuple(np.linspace(0.65, 1.85, 35)),
            beta_grid=tuple(np.linspace(0.50, 2.50, 33)),
            cmax=2,
            p_values=(0, 1, 2),
            max_patterns_per_c=None,
            maxiter=16,
            popsize=6,
            seed=61,
            val_fraction=0.25,
            trim_t=2,
            trim_x=0,
            power_mode="raw",
            selection="elbow",
            exact_order_refit=True,
            exact_order_polish=True,
            auto_stop=True,
            auto_stop_min_c=2,
            auto_stop_patience=1,
            auto_stop_rel_improvement=0.03,
            auto_stop_log10_improvement=0.02,
            auto_stop_use_selection_stability=True,
            auto_stop_selection_patience=1,
        )

    if dataset_name == "synthetic_fractional_burgers":
        return DiscoveryConfig(
            backend="spectral_l1",            # directional (i k)^beta weak operator, integer time
            spectral_riesz=False,
            alpha_grid=_grid_with_required(0.85, 1.15, 13, (1.00,)),
            beta_grid=_grid_with_required(0.50, 2.00, 31, (1.00, 1.70)),
            cmax=4,
            p_values=(0, 1, 2),               # the true advection term has p=1
            max_patterns_per_c=None,
            maxiter=24,
            popsize=7,
            seed=51,
            val_fraction=0.25,
            trim_t=5,
            trim_x=0,
            power_mode="raw",                 # field changes sign, so do not clamp u^p
            selection="elbow",
            auto_stop=True,
            auto_stop_min_c=2,
            auto_stop_patience=1,
            auto_stop_rel_improvement=0.03,
            auto_stop_log10_improvement=0.02,
            auto_stop_use_selection_stability=True,
            auto_stop_selection_patience=1,
        )

    valid = ", ".join(available_dataset_names())
    raise ValueError(f"Unknown dataset_name={dataset_name!r}. Valid names are: {valid}")


def recommended_discovery_config(
    dataset_name: str,
    *,
    profile: ConfigProfile = "notebook",
    cmax: int | None = None,
    p_values: tuple[int, ...] | list[int] | None = None,
    maxiter: int | None = None,
    popsize: int | None = None,
    selection: str | None = None,
    auto_stop: bool | None = None,
    noise_percent: float = 0.0,
    seed: int | None = None,
    progress: bool | None = None,
    progress_de: bool | None = None,
) -> DiscoveryConfig:
    """Return the shared tailored config for a dataset, with optional runtime overrides.

    The candidate library/search space is dataset-specific and shared across all
    methods.  ``profile='notebook'`` only reduces runtime defaults; it does not
    change the derivative-order grids, power set, backend, or trimming rules.
    """
    cfg = dataset_candidate_config(dataset_name)

    # Runtime profile: keep the same candidate space, just adjust DE budget.
    if profile == "notebook":
        profile_maxiter = 2
        profile_popsize = 3
    elif profile == "paper":
        profile_maxiter = cfg.maxiter
        profile_popsize = cfg.popsize
    else:
        raise ValueError("profile must be 'notebook' or 'paper'")

    cfg.maxiter = int(profile_maxiter if maxiter is None else maxiter)
    cfg.popsize = int(profile_popsize if popsize is None else popsize)
    if cmax is not None:
        cfg.cmax = int(cmax)
    if p_values is not None:
        cfg.p_values = tuple(int(v) for v in p_values)
    if selection is not None:
        cfg.selection = selection  # type: ignore[assignment]
    if auto_stop is not None:
        cfg.auto_stop = bool(auto_stop)
    if seed is not None:
        cfg.seed = int(seed)
    if progress is not None:
        cfg.progress = bool(progress)
    if progress_de is not None:
        cfg.progress_de = bool(progress_de)
    cfg.noise_percent = float(noise_percent)
    return cfg


def config_search_space_fingerprint(config: DiscoveryConfig) -> dict[str, Any]:
    """Return the fields that define the candidate library/search space.

    Use this in result files to verify that all methods used the same configured
    hypothesis class for the same dataset.
    """
    return {
        "backend": config.backend,
        "alpha_grid": [float(v) for v in config.alpha_grid],
        "beta_grid": [float(v) for v in config.beta_grid],
        "p_values": [int(v) for v in config.p_values],
        "cmax": int(config.cmax),
        "max_patterns_per_c": config.max_patterns_per_c,
        "trim_t": int(config.trim_t),
        "trim_x": int(config.trim_x),
        "power_mode": config.power_mode,
        "spectral_riesz": bool(config.spectral_riesz),
        "regularized_time_kind": config.regularized_time_kind,
        "regularized_space_kind": config.regularized_space_kind,
        "regularized_space_side": config.regularized_space_side,
        "lam_t": float(config.lam_t),
        "lam_x": float(config.lam_x),
    }


def load_dataset_with_config(
    dataset_name: str,
    *,
    data_dir: str | Path = "data",
    profile: ConfigProfile = "notebook",
    nt: int | None = None,
    nx: int | None = None,
    noise_percent: float = 0.0,
    seed: int = 0,
    cmax: int | None = None,
    p_values: tuple[int, ...] | list[int] | None = None,
    maxiter: int | None = None,
    popsize: int | None = None,
    selection: str | None = None,
    auto_stop: bool | None = None,
    progress: bool | None = None,
    progress_de: bool | None = None,
) -> tuple[GridDataset, DiscoveryConfig, DatasetTruthSpec]:
    """Load/generate a dataset and return its single shared tailored config."""
    # Important fairness convention: return a clean dataset and store the
    # requested noise level only in the DiscoveryConfig.  Each feature bank
    # injects noise internally using config.noise_percent and config.seed, so
    # vanilla/weak/STRidge methods see the same noisy realization.  Passing
    # noise_percent here as well would double-noise the observations.
    data = load_dataset_by_name(
        dataset_name,
        data_dir=data_dir,
        nt=nt,
        nx=nx,
        noise_percent=0.0,
        seed=seed,
    )
    cfg = recommended_discovery_config(
        dataset_name,
        profile=profile,
        cmax=cmax,
        p_values=p_values,
        maxiter=maxiter,
        popsize=popsize,
        selection=selection,
        auto_stop=auto_stop,
        noise_percent=noise_percent,
        seed=seed if seed is not None else None,
        progress=progress,
        progress_de=progress_de,
    )
    truth = dataset_truth_spec(dataset_name)
    return data, cfg, truth


def available_benchmark_dataset_names(
    *,
    data_dir: str | Path = "data",
    include_missing_paper: bool = False,
) -> tuple[str, ...]:
    """Return benchmark names that can be loaded in the current project.

    Parameters
    ----------
    data_dir:
        Directory containing optional paper ``.dat`` files.
    include_missing_paper:
        If ``False`` (default), paper datasets are returned only when their data
        files are present. Synthetic datasets are always returned. If ``True``,
        return every configured dataset name, even if loading a paper dataset
        would later raise ``FileNotFoundError``.

    Notes
    -----
    This is a benchmark-level helper. For the lower-level raw dataset loader,
    see ``fpde_datasets.available_dataset_names``.
    """
    data_dir = resolve_data_dir(data_dir)
    names: list[str] = []
    if include_missing_paper or (data_dir / "Convection_diffusion.dat").exists():
        names.append("paper_ADE_Convection_diffusion")
    if include_missing_paper or (data_dir / "tsfade_fft.dat").exists():
        names.append("paper_FADE_tsfade_fft")
    names.extend([
        "synthetic_space_fractional_RD",
        "synthetic_time_space_fractional_RD",
        "synthetic_two_fractional_rhs",
        "synthetic_fractional_burgers",
    ])
    return tuple(names)


def benchmark_grid_size(dataset_name: str, profile: ConfigProfile) -> tuple[int | None, int | None]:
    """Return the notebook/paper data grid override for one dataset.

    The profile changes only synthetic grid resolution and compute budget.  It
    does not change the mathematical candidate search space.
    """
    nt = nx = None
    if profile == "notebook":
        if dataset_name == "synthetic_space_fractional_RD":
            nt, nx = 50, 48
        elif dataset_name == "synthetic_time_space_fractional_RD":
            nt, nx = 50, 48
        elif dataset_name == "synthetic_two_fractional_rhs":
            nt, nx = 60, 64
        elif dataset_name == "synthetic_superunit_fractional_diffusion":
            nt, nx = 80, 48
        elif dataset_name == "synthetic_fractional_burgers":
            nt, nx = 60, 60
    elif profile != "paper":
        raise ValueError("profile must be 'notebook' or 'paper'")
    return nt, nx


def benchmark_spec(
    dataset_name: str,
    *,
    data_dir: str | Path = "data",
    profile: ConfigProfile = "notebook",
    maxiter: int | None = None,
    popsize: int | None = None,
    cmax: int | None = None,
    p_values: tuple[int, ...] | list[int] | None = None,
    noise_percent: float = 0.0,
    seed: int | None = None,
    nt: int | None = None,
    nx: int | None = None,
) -> dict[str, Any]:
    """Return the complete benchmark spec for exactly one dataset name.

    This is the recommended notebook API.  It directly returns the data, shared
    discovery config, truth metadata, and config fingerprint for one dataset:

    >>> spec = benchmark_spec("synthetic_space_fractional_RD", noise_percent=0.1)
    >>> data = spec["data"]
    >>> config = spec["config"]

    Unlike ``benchmark_specs(...)``, this function does not require filtering a
    list by hand.  The returned dictionary has the same schema as one element of
    ``benchmark_specs``.
    """
    if dataset_name not in DATASET_TRUTH_SPECS:
        valid = ", ".join(DATASET_TRUTH_SPECS)
        raise ValueError(f"Unknown dataset_name={dataset_name!r}. Valid names: {valid}")

    if nt is None and nx is None:
        nt, nx = benchmark_grid_size(dataset_name, profile)
    elif nt is None or nx is None:
        # Allow overriding just one dimension if needed; keep the other at its
        # profile default when available.
        default_nt, default_nx = benchmark_grid_size(dataset_name, profile)
        nt = default_nt if nt is None else nt
        nx = default_nx if nx is None else nx

    run_seed = truth_seed(dataset_name) if seed is None else int(seed)
    data, cfg, truth = load_dataset_with_config(
        dataset_name,
        data_dir=data_dir,
        profile=profile,
        nt=nt,
        nx=nx,
        noise_percent=noise_percent,
        seed=run_seed,
        maxiter=maxiter,
        popsize=popsize,
        cmax=cmax,
        p_values=p_values,
    )
    return {
        "dataset_name": dataset_name,
        "data": data,
        "config": cfg,
        "expected_c": truth.expected_c,
        "expected_alpha": truth.expected_alpha,
        "expected_terms": list(truth.expected_terms),
        "truth_spec": truth,
        "truth_spec_dict": truth.to_dict(),
        "config_search_space": config_search_space_fingerprint(cfg),
        "seed": int(run_seed),
    }


def load_benchmark(
    dataset_name: str,
    **kwargs: Any,
) -> tuple[GridDataset, DiscoveryConfig, DatasetTruthSpec]:
    """Return ``(data, config, truth)`` for one dataset name.

    This is a convenience wrapper around ``benchmark_spec`` for users who prefer
    tuple unpacking instead of a metadata dictionary.

    Examples
    --------
    >>> data, config, truth = load_benchmark("synthetic_space_fractional_RD")
    """
    spec = benchmark_spec(dataset_name, **kwargs)
    return spec["data"], spec["config"], spec["truth_spec"]


def benchmark_specs(
    *,
    data_dir: str | Path = "data",
    profile: ConfigProfile = "notebook",
    maxiter: int | None = None,
    popsize: int | None = None,
    cmax: int | None = None,
    p_values: tuple[int, ...] | list[int] | None = None,
    noise_percent: float = 0.0,
    seed: int | None = None,
    dataset_names: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return benchmark specs for many datasets.

    Use :func:`benchmark_spec` when you want a single dataset by name.  Use this
    plural helper for scripts that intentionally loop over all or several
    datasets.
    """
    names = list(dataset_names) if dataset_names is not None else list(
        available_benchmark_dataset_names(data_dir=data_dir)
    )
    return [
        benchmark_spec(
            name,
            data_dir=data_dir,
            profile=profile,
            maxiter=maxiter,
            popsize=popsize,
            cmax=cmax,
            p_values=p_values,
            noise_percent=noise_percent,
            seed=seed,
        )
        for name in names
    ]


def truth_seed(dataset_name: str) -> int:
    """Deterministic seed used for synthetic data/config examples."""
    return {
        "paper_ADE_Convection_diffusion": 11,
        "paper_FADE_tsfade_fft": 15,
        "synthetic_space_fractional_RD": 21,
        "synthetic_time_space_fractional_RD": 31,
        "synthetic_two_fractional_rhs": 41,
        "synthetic_fractional_burgers": 51,
        "synthetic_superunit_fractional_diffusion": 61,
    }.get(dataset_name, 0)


def expected_probe_structure(dataset_name: str) -> dict[str, object]:
    """Return a truth-like structure for manual notebook evaluation."""
    truth = dataset_truth_spec(dataset_name)
    return {
        "alpha": float(truth.expected_alpha),
        "p_tuple": tuple(p for p, _ in truth.expected_terms),
        "beta_tuple": tuple(beta for _, beta in truth.expected_terms),
    }


def config_summary(config: DiscoveryConfig) -> dict[str, Any]:
    """Compact JSON-friendly representation of a config."""
    return config_to_dict(config)


# --------------------------------------------------------------------------- #
#  Publication helper: truth-agnostic uniform order grids (item-1 optics fix)  #
# --------------------------------------------------------------------------- #
def uniform_order_grid(start: float, stop: float, num: int) -> tuple[float, ...]:
    """Dense uniform order grid with NO benchmark-true orders inserted."""
    return tuple(float(v) for v in np.linspace(float(start), float(stop), int(num)))


def with_uniform_order_grids(config: DiscoveryConfig, *, n_alpha: int = 47, n_beta: int = 59) -> DiscoveryConfig:
    """Return a copy of ``config`` with dense UNIFORM alpha/beta grids.

    The default ``dataset_candidate_config`` grids insert the benchmark-true
    orders via ``_grid_with_required`` so that fixed-grid baselines can probe them.
    For the proposed continuous-order method that insertion is unnecessary and, for
    publication, undesirable: a reviewer could object that a grid node sits exactly
    on the answer.  This helper rebuilds the grids uniformly over the SAME ranges
    without inserting any true order.  Used together with ``exact_order_refit=True``
    (the default), this removes interpolation error from the final conditional refit.
    The selected support, temporal mode, and optimisation basin can still depend on
    the grid and search trajectory.  Recommended for the publication runs.
    """
    import dataclasses

    a = np.asarray(config.alpha_grid, dtype=float)
    b = np.asarray(config.beta_grid, dtype=float)
    return dataclasses.replace(
        config,
        alpha_grid=uniform_order_grid(float(a.min()), float(a.max()), n_alpha),
        beta_grid=uniform_order_grid(float(b.min()), float(b.max()), n_beta),
    )
