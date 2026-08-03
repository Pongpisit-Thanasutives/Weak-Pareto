"""Datasets for publication-style fractional PDE discovery benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fpde_derivatives import spectral_space_derivative


def resolve_data_dir(data_dir: str | Path = "data") -> Path:
    """Return an existing data directory, robust to notebook working directories.

    Most examples pass ``data_dir="data"``.  That is convenient when the
    current working directory is the project root, but notebooks are often run
    with ``cwd`` set to ``notebooks/`` or from an IDE-specific directory.  This
    helper first checks the user-provided path, then falls back to the package
    root next to this file.
    """
    path = Path(data_dir)
    if path.exists():
        return path
    if not path.is_absolute():
        package_relative = Path(__file__).resolve().parent / path
        if package_relative.exists():
            return package_relative
    return path


@dataclass(frozen=True)
class GridDataset:
    """Uniform gridded field U(t, x)."""

    U: NDArray[np.float64]
    t: NDArray[np.float64]
    x: NDArray[np.float64]
    name: str
    truth: str
    recommended_backend: str = "regularized"

    @property
    def dt(self) -> float:
        """Uniform time spacing inferred from ``t``."""
        return float(self.t[1] - self.t[0])

    @property
    def dx(self) -> float:
        """Uniform spatial spacing inferred from ``x``."""
        return float(self.x[1] - self.x[0])

    @property
    def Lx(self) -> float:
        """Periodic domain length used by spectral spatial derivatives."""
        return float(self.dx * len(self.x))

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the gridded field ``U`` as ``(nt, nx)``."""
        return self.U.shape


def load_convection_diffusion(path: str | Path) -> GridDataset:
    """Load the paper repository's Convection_diffusion.dat.

    The FDE_discovery code uses t = arange(0, 15, 0.1) and x = arange(0, 30, 0.25),
    hence shape (150, 120).  The known governing equation is the classical
    advection-diffusion equation D_t u = -D_x u + 0.25 D_x^2 u.
    """
    U = np.loadtxt(path, dtype=float)
    nt, nx = U.shape
    t = np.arange(nt, dtype=float) * 0.1
    x = np.arange(nx, dtype=float) * 0.25
    return GridDataset(
        U=U,
        t=t,
        x=x,
        name="paper_ADE_Convection_diffusion",
        truth="D_t^1 u = -1.0 D_x^1 u + 0.25 D_x^2 u",
        recommended_backend="regularized",
    )


def load_paper_fade(path: str | Path) -> GridDataset:
    """Load the paper repository's time-space fractional FADE dataset.

    The file is usually named ``tsfade_fft.dat`` in the original
    ``yxn1019/FDE_discovery`` repository.  It is the synthetic fractional
    advection-diffusion equation benchmark from the paper:

        D_t^0.8 u = -1.0 D_x^1 u + 0.5 D_x^1.7 u.

    The grid convention follows the paper code: ``t = arange(0, 15, 0.1)``
    and ``x = arange(0, 30, 0.25)``, giving shape ``(150, 120)``.

    This dataset is the paper's true fractional PDE benchmark.  In contrast,
    ``Convection_diffusion.dat`` is an integer-order ADE control case.
    """
    U = np.loadtxt(path, dtype=float)
    nt, nx = U.shape
    t = np.arange(nt, dtype=float) * 0.1
    x = np.arange(nx, dtype=float) * 0.25
    return GridDataset(
        U=U,
        t=t,
        x=x,
        name="paper_FADE_tsfade_fft",
        truth="D_t^0.8 u = -1.0 D_x^1 u + 0.5 D_x^1.7 u",
        recommended_backend="spectral_l1_directional",
    )


def add_multiplicative_uniform_noise(U: NDArray[np.float64], percent: float, seed: int = 0) -> NDArray[np.float64]:
    """Noise convention used by the referenced FDE_discovery code."""
    if percent <= 0:
        return np.asarray(U, dtype=float).copy()
    rng = np.random.default_rng(seed)
    return np.asarray(U, dtype=float) * (1.0 + 0.01 * float(percent) * rng.uniform(-1.0, 1.0, size=U.shape))


def _periodic_initial_condition(x: NDArray[np.float64], seed: int = 0) -> NDArray[np.float64]:
    """Create a smooth random periodic initial condition for synthetic benchmarks."""
    if x.size < 2:
        raise ValueError("x must contain at least two points")
    rng = np.random.default_rng(seed)
    dx = float(x[1] - x[0])
    Lx = float(dx * len(x))
    u0 = (
        0.8 * np.sin(2 * np.pi * x / Lx)
        + 0.45 * np.cos(4 * np.pi * x / Lx)
        + 0.25 * np.sin(6 * np.pi * x / Lx + 0.4)
    )
    u0 += 0.05 * rng.normal(size=x.shape)
    # Smooth in Fourier space to avoid alias-heavy high modes.
    fft = np.fft.fft(u0)
    k = np.fft.fftfreq(len(x))
    fft *= np.exp(-40.0 * k**2)
    return np.fft.ifft(fft).real


def make_space_fractional_advection_dispersion(
    *,
    nt: int = 90,
    nx: int = 96,
    t_end: float = 1.0,
    Lx: float = 20.0,
    growth: float = 0.04,
    diff: float = 0.18,
    beta: float = 1.65,
    seed: int = 1,
) -> GridDataset:
    """Synthetic periodic space-fractional reaction-diffusion dataset.

    The generated equation is
        u_t = growth * u + diff * Riesz_x^beta u,
    where Riesz_x^beta has Fourier multiplier -|k|^beta.  The q=0 term in the
    discovery library is D_x^0 u = u, so the true support still has two RHS
    terms but avoids the complex-branch ambiguity of directional noninteger
    spatial derivatives on real-valued periodic data.
    """
    x = np.linspace(0.0, Lx, nx, endpoint=False)
    t = np.linspace(0.0, t_end, nt)
    u0 = _periodic_initial_condition(x, seed=seed)
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    lam = growth + diff * (-(np.abs(k) ** beta))
    lam[np.isclose(k, 0.0)] = growth
    u0hat = np.fft.fft(u0)
    U = np.empty((nt, nx), dtype=float)
    for i, ti in enumerate(t):
        U[i] = np.fft.ifft(np.exp(lam * ti) * u0hat).real
    return GridDataset(
        U=U,
        t=t,
        x=x,
        name="synthetic_space_fractional_RD",
        truth=f"D_t^1 u = {growth} u + {diff} Riesz_x^{beta} u",
        recommended_backend="spectral_l1_riesz",
    )

def make_time_space_fractional_advection_dispersion(
    *,
    alpha: float = 0.82,
    beta: float = 1.55,
    nt: int = 80,
    nx: int = 80,
    t_end: float = 3.0,
    Lx: float = 20.0,
    growth: float = 0.03,
    diff: float = 0.12,
    seed: int = 2,
) -> GridDataset:
    """Synthetic periodic time-and-space fractional reaction-diffusion dataset.

    The data are advanced using the same Caputo-L1 temporal discretization as
    the evaluator, mode-by-mode in Fourier space:
        D_t^alpha u = growth * u + diff * Riesz_x^beta u.

    Consequently, the clean time--space reaction--diffusion benchmark is a
    controlled recovery test with a shared temporal discretization, not an
    independent forward-solver validation.
    """
    from scipy import special

    x = np.linspace(0.0, Lx, nx, endpoint=False)
    t = np.linspace(0.0, t_end, nt)
    dt = float(t[1] - t[0])
    u0 = _periodic_initial_condition(x, seed=seed)
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    lam = growth + diff * (-(np.abs(k) ** beta))
    lam[np.isclose(k, 0.0)] = growth

    coef = dt ** (-alpha) / special.gamma(2.0 - alpha)
    weights = np.array([(j + 1.0) ** (1.0 - alpha) - j ** (1.0 - alpha) for j in range(nt)], dtype=float)
    Uhat = np.zeros((nt, nx), dtype=complex)
    Uhat[0] = np.fft.fft(u0)
    denom = coef - lam
    for n in range(1, nt):
        hist = np.zeros(nx, dtype=complex)
        for j in range(1, n):
            hist += weights[j] * (Uhat[n - j] - Uhat[n - j - 1])
        Uhat[n] = (coef * Uhat[n - 1] - coef * hist) / denom
    U = np.fft.ifft(Uhat, axis=1).real
    return GridDataset(
        U=U,
        t=t,
        x=x,
        name="synthetic_time_space_fractional_RD",
        truth=f"D_t^{alpha} u = {growth} u + {diff} Riesz_x^{beta} u",
        recommended_backend="spectral_l1_riesz",
    )


def make_two_fractional_rhs_dataset(
    *,
    beta1: float = 0.55,
    beta2: float = 2.80,
    coeff1: float = 0.05,
    coeff2: float = 0.005,
    nt: int = 90,
    nx: int = 128,
    t_end: float = 0.20,
    Lx: float = 20.0,
    seed: int = 4,
) -> GridDataset:
    """Synthetic PDE with two distinct fractional RHS operators.

    This benchmark is designed to make the benefit of the FPDE encoding
    visible.  The true equation has two active RHS terms, and **both** are
    fractional spatial derivatives:

        D_t^1 u = coeff1 * Riesz_x^beta1 u
                + coeff2 * Riesz_x^beta2 u.

    Since the two derivative orders are continuous nonlinear parameters, an
    overcomplete grid-STRidge library can easily spread mass across nearby
    columns.  The Pareto-DE encoding instead searches directly over two
    continuous orders for the two-term model.

    The data are generated by a periodic Fourier solution.  ``Riesz_x^beta``
    uses the multiplier ``-|k|^beta``, so positive coefficients represent
    dissipative fractional diffusion terms.

    The initial condition intentionally contains a broad set of Fourier modes.
    That makes the two fractional exponents identifiable: a single effective
    order cannot match both the low- and high-frequency decay rates.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, Lx, nx, endpoint=False)
    t = np.linspace(0.0, t_end, nt)

    u0 = np.zeros(nx, dtype=float)
    # Broad but still smooth-ish spectrum; the amplitude decay avoids aliasing
    # while preserving enough high-frequency information to distinguish beta2.
    for m in range(1, max(2, nx // 2 - 5)):
        amp = 1.0 / (m ** 0.35)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        u0 += amp * np.sin(2.0 * np.pi * m * x / Lx + phase)
    u0 /= np.std(u0) + 1e-14

    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    lam = coeff1 * (-(np.abs(k) ** beta1)) + coeff2 * (-(np.abs(k) ** beta2))
    lam[np.isclose(k, 0.0)] = 0.0
    u0hat = np.fft.fft(u0)
    U = np.empty((nt, nx), dtype=float)
    for i, ti in enumerate(t):
        U[i] = np.fft.ifft(np.exp(lam * ti) * u0hat).real
    return GridDataset(
        U=U,
        t=t,
        x=x,
        name="synthetic_two_fractional_rhs",
        truth=(
            f"D_t^1 u = {coeff1} Riesz_x^{beta1} u "
            f"+ {coeff2} Riesz_x^{beta2} u"
        ),
        recommended_backend="spectral_l1_riesz",
    )



def _mittag_leffler_e_alpha_1(
    alpha: float,
    z: NDArray[np.float64],
    *,
    tol: float = 1e-15,
    max_terms: int = 600,
) -> NDArray[np.float64]:
    """Evaluate ``E_{alpha,1}(z)`` by its convergent power series.

    The helper is intended for the modest negative arguments used by the
    superunit diagnostic generator.  It deliberately avoids a second forward
    discretisation of the Caputo operator, so recovery tests do not reuse the
    L1 temporal scheme employed by the discovery code.
    """
    from scipy import special

    if not (0.0 < float(alpha) < 2.0):
        raise ValueError("alpha must lie in (0, 2)")
    values = np.asarray(z, dtype=float)
    out = np.ones_like(values)
    for k in range(1, int(max_terms) + 1):
        term = np.power(values, k) / special.gamma(float(alpha) * k + 1.0)
        out += term
        scale = max(1.0, float(np.max(np.abs(out))))
        if float(np.max(np.abs(term))) <= float(tol) * scale:
            return out
    raise RuntimeError("Mittag--Leffler series did not converge within max_terms")


def make_superunit_fractional_diffusion(
    *,
    alpha: float = 1.65,
    diffusivity: float = 0.12,
    nt: int = 120,
    nx: int = 64,
    t_end: float = 2.0,
    Lx: float = 2.0 * np.pi,
) -> GridDataset:
    """Analytic periodic diagnostic with a superunit Caputo order.

    The field solves

        ``D_t^alpha u = diffusivity * D_x^2 u``,  ``1 < alpha < 2``,

    with a smooth three-harmonic initial field and zero initial velocity.  Each
    Fourier mode evolves as ``E_{alpha,1}(-diffusivity*k^2*t^alpha)``.  The
    resulting data are generated semi-analytically and therefore do not share
    the L1 temporal discretisation used by the discovery evaluator.

    This dataset is a focused branch-recovery diagnostic, not part of the
    frozen publication benchmark tables.
    """
    if not (1.0 < float(alpha) < 2.0):
        raise ValueError("superunit diagnostic requires 1 < alpha < 2")
    if float(diffusivity) <= 0.0:
        raise ValueError("diffusivity must be positive")
    if int(nt) < 3 or int(nx) < 8:
        raise ValueError("nt >= 3 and nx >= 8 are required")
    if float(t_end) <= 0.0 or float(Lx) <= 0.0:
        raise ValueError("t_end and Lx must be positive")

    t = np.linspace(0.0, float(t_end), int(nt))
    x = np.linspace(0.0, float(Lx), int(nx), endpoint=False)
    U = np.zeros((int(nt), int(nx)), dtype=float)
    harmonics = ((1, 1.0, 0.0), (2, 0.35, 0.3), (3, 0.15, -0.2))
    base_wavenumber = 2.0 * np.pi / float(Lx)
    for mode, amplitude, phase in harmonics:
        kappa = float(mode) * base_wavenumber
        temporal = _mittag_leffler_e_alpha_1(
            float(alpha),
            -float(diffusivity) * (kappa**2) * np.power(t, float(alpha)),
        )
        U += float(amplitude) * temporal[:, None] * np.sin(kappa * x + float(phase))[None, :]

    return GridDataset(
        U=U,
        t=t,
        x=x,
        name="synthetic_superunit_fractional_diffusion",
        truth=f"D_t^{alpha:g} u = {diffusivity:g} D_x^2 u",
        recommended_backend="spectral_l1_directional",
    )

def make_fractional_burgers(
    *,
    nu: float = 0.25,
    beta: float = 1.7,
    advection: float = -1.0,
    amplitude: float = 0.5,
    nt: int = 150,
    nx: int = 120,
    t_end: float = 12.0,
    Lx: float = 30.0,
    nx_fine: int = 480,
    dt_fine: float = 0.004,
) -> GridDataset:
    """Synthetic **nonlinear** fractional Burgers benchmark.

    True equation (periodic, directional spectral spatial operator):

        u_t = advection * u u_x + nu * D_x^beta u,

    i.e. one genuinely nonlinear advective term ``u^1 D_x^1 u`` and one linear
    fractional-diffusion term ``u^0 D_x^beta u``.  In the discovery encoding
    ``u^p D_x^beta u`` the truth is therefore ``{(p=1, beta=1.0): advection,
    (p=0, beta): nu}``.  This is the only benchmark here with a true ``p>0`` term,
    so it exercises the data-weighted weak form ``<u, (D_x^beta)^*(u^p phi)>``
    rather than only the linear ``p=0`` columns.

    The PDE is integrated pseudo-spectrally with explicit RK4 and 2/3 dealiasing
    on a fine grid (``nx_fine``), then subsampled to the reporting grid
    ``(nt, nx)`` on an endpoint-excluded time grid
    ``t_j = j t_end / nt`` for ``j=0,...,nt-1`` and a periodic, endpoint-excluded
    ``x`` grid.  The clean generator is deterministic and therefore has no seed
    argument.  Parameters are chosen so the solution stays smooth and
    spectrally band-limited well within the coarse-grid Nyquist (so the spatial
    operators remain accurate on the reporting grid) while the nonlinear term
    stays comparable in magnitude to the fractional-diffusion term.
    """
    x_fine = np.linspace(0.0, Lx, nx_fine, endpoint=False)
    k = 2.0 * np.pi * np.fft.fftfreq(nx_fine, d=Lx / nx_fine)
    L = nu * ((1j * k) ** float(beta))
    L[np.isclose(k, 0.0)] = 0.0
    ik = 1j * k
    kmax = float(np.max(np.abs(k)))
    mask = (np.abs(k) <= (2.0 / 3.0) * kmax)
    u0 = amplitude * (
        np.sin(2.0 * np.pi * x_fine / Lx)
        + 0.5 * np.cos(4.0 * np.pi * x_fine / Lx)
        + 0.35 * np.sin(6.0 * np.pi * x_fine / Lx + 0.5)
    )
    uh = np.fft.fft(u0)

    def Nhat(uh: NDArray[np.complex128]) -> NDArray[np.complex128]:
        # advection * u u_x = advection * 0.5 * d/dx(u^2); conservation form for stability.
        u = np.fft.ifft(uh).real
        return float(advection) * 0.5 * ik * np.fft.fft(u * u) * mask

    def rhs(uh: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return L * uh + Nhat(uh)

    nsteps = int(round(t_end / dt_fine))
    out_t = np.arange(nt, dtype=float) * (t_end / nt)
    snaps = np.empty((nt, nx_fine), dtype=float)
    saved = np.zeros(nt, dtype=bool)

    def save(n: int) -> None:
        tt = n * dt_fine
        idx = int(round(tt / (t_end / nt)))
        if 0 <= idx < nt and not saved[idx] and abs(tt - out_t[idx]) < dt_fine / 2:
            snaps[idx] = np.fft.ifft(uh).real
            saved[idx] = True

    save(0)
    for n in range(nsteps):
        k1 = rhs(uh)
        k2 = rhs(uh + 0.5 * dt_fine * k1)
        k3 = rhs(uh + 0.5 * dt_fine * k2)
        k4 = rhs(uh + dt_fine * k3)
        uh = uh + (dt_fine / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        save(n + 1)
    for i in range(nt):
        if not saved[i]:
            snaps[i] = snaps[i - 1]
    step = nx_fine // nx
    U = snaps[:, ::step][:, :nx]
    x = np.linspace(0.0, Lx, nx, endpoint=False)
    return GridDataset(
        U=U,
        t=out_t,
        x=x,
        name="synthetic_fractional_burgers",
        truth=f"D_t^1 u = {advection} u u_x + {nu} D_x^{beta} u",
        recommended_backend="spectral_l1_directional",
    )


def save_dataset_npz(data: GridDataset, path: str | Path) -> Path:
    """Save a :class:`GridDataset` as a compressed NumPy ``.npz`` file.

    Returns the normalized output path for convenient use by dataset-preparation
    scripts and tests.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        U=np.asarray(data.U, dtype=float),
        t=np.asarray(data.t, dtype=float),
        x=np.asarray(data.x, dtype=float),
        name=data.name,
        truth=data.truth,
        recommended_backend=data.recommended_backend,
    )
    return path


def load_npz_dataset(path: str | Path) -> GridDataset:
    """Load a ``GridDataset`` previously written by ``save_dataset_npz``."""
    with np.load(path, allow_pickle=True) as z:
        return GridDataset(
            U=z["U"].astype(float),
            t=z["t"].astype(float),
            x=z["x"].astype(float),
            name=str(z["name"].item()),
            truth=str(z["truth"].item()),
            recommended_backend=str(z["recommended_backend"].item()),
        )



def available_dataset_names() -> tuple[str, ...]:
    """Return the dataset names understood by :func:`load_dataset_by_name`.

    Use this in notebooks when you want a small dropdown/list of valid
    dataset identifiers.  These names are *input keys*; after loading, the
    returned ``GridDataset.name`` may be the same descriptive name.
    """
    return (
        "paper_ADE_Convection_diffusion",
        "paper_FADE_tsfade_fft",
        "synthetic_space_fractional_RD",
        "synthetic_time_space_fractional_RD",
        "synthetic_two_fractional_rhs",
        "synthetic_fractional_burgers",
        "synthetic_superunit_fractional_diffusion",
    )


def load_dataset_by_name(
    dataset_name: str,
    *,
    data_dir: str | Path = "data",
    nt: int | None = None,
    nx: int | None = None,
    noise_percent: float = 0.0,
    seed: int = 0,
) -> GridDataset:
    """Load or generate one of the built-in benchmark datasets by name.

    Parameters
    ----------
    dataset_name:
        One of ``available_dataset_names()``.  Common choices are
        ``"paper_ADE_Convection_diffusion"`` for the paper's classical
        advection-diffusion dataset, ``"paper_FADE_tsfade_fft"`` for the
        paper's fractional FADE benchmark, ``"synthetic_space_fractional_RD"``
        for a fast space-fractional synthetic benchmark,
        ``"synthetic_time_space_fractional_RD"`` for a time-and-space
        fractional benchmark, or ``"synthetic_two_fractional_rhs"`` for a
        synthetic equation with two true fractional RHS operators, or
        ``"synthetic_superunit_fractional_diffusion"`` for the focused
        superunit Caputo branch diagnostic.
    data_dir:
        Directory containing ``Convection_diffusion.dat`` when using the paper
        ADE dataset.  In the packaged project this is usually ``ROOT / "data"``.
    nt, nx:
        Optional grid sizes for synthetic datasets.  Ignored for the paper
        dataset, which is loaded from disk.
    noise_percent:
        Optional multiplicative uniform noise percentage, using the same noise
        convention as the referenced FDE discovery repository.
    seed:
        Random seed for stochastic synthetic initial conditions and optional
        noise. The clean fractional Burgers and superunit diagnostic generators are
        deterministic, so this argument affects them only when noise is
        requested.

    Returns
    -------
    GridDataset
        A uniform ``(t, x)`` gridded dataset ready for ``run_pareto_discovery``.

    Examples
    --------
    >>> data = load_dataset_by_name("synthetic_space_fractional_RD", nt=60, nx=64)
    >>> data.U.shape
    (60, 64)
    """
    data_dir = resolve_data_dir(data_dir)

    if dataset_name == "paper_ADE_Convection_diffusion":
        dat_path = data_dir / "Convection_diffusion.dat"
        if not dat_path.exists():
            raise FileNotFoundError(
                f"Could not find {dat_path}. Put Convection_diffusion.dat in the data directory "
                "or choose one of the synthetic dataset names."
            )
        data = load_convection_diffusion(dat_path)


    elif dataset_name == "paper_FADE_tsfade_fft":
        dat_path = data_dir / "tsfade_fft.dat"
        if not dat_path.exists():
            raise FileNotFoundError(
                f"Could not find {dat_path}. Put tsfade_fft.dat in the data directory "
                "or choose one of the synthetic dataset names. It can be downloaded "
                "from yxn1019/FDE_discovery/dataset/tsfade_fft.dat."
            )
        data = load_paper_fade(dat_path)

    elif dataset_name == "synthetic_space_fractional_RD":
        kwargs = {"seed": seed}
        if nt is not None:
            kwargs["nt"] = nt
        if nx is not None:
            kwargs["nx"] = nx
        data = make_space_fractional_advection_dispersion(**kwargs)

    elif dataset_name == "synthetic_time_space_fractional_RD":
        kwargs = {"seed": seed}
        if nt is not None:
            kwargs["nt"] = nt
        if nx is not None:
            kwargs["nx"] = nx
        data = make_time_space_fractional_advection_dispersion(**kwargs)


    elif dataset_name == "synthetic_two_fractional_rhs":
        kwargs = {"seed": seed}
        if nt is not None:
            kwargs["nt"] = nt
        if nx is not None:
            kwargs["nx"] = nx
        data = make_two_fractional_rhs_dataset(**kwargs)

    elif dataset_name == "synthetic_fractional_burgers":
        kwargs = {}
        if nt is not None:
            kwargs["nt"] = nt
        if nx is not None:
            kwargs["nx"] = nx
        data = make_fractional_burgers(**kwargs)

    elif dataset_name == "synthetic_superunit_fractional_diffusion":
        kwargs = {}
        if nt is not None:
            kwargs["nt"] = nt
        if nx is not None:
            kwargs["nx"] = nx
        data = make_superunit_fractional_diffusion(**kwargs)

    else:
        valid = ", ".join(available_dataset_names())
        raise ValueError(f"Unknown dataset_name={dataset_name!r}. Valid names are: {valid}")

    if noise_percent > 0:
        U_noisy = add_multiplicative_uniform_noise(data.U, noise_percent, seed=seed)
        data = GridDataset(
            U=U_noisy,
            t=data.t,
            x=data.x,
            name=f"{data.name}_noise{noise_percent:g}",
            truth=data.truth,
            recommended_backend=data.recommended_backend,
        )

    return data

def prepare_all_datasets(
    output_dir: str | Path,
    paper_dat: str | Path | None = None,
    paper_fade_dat: str | Path | None = None,
) -> list[Path]:
    """Write the recommended benchmark datasets to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if paper_dat is not None and Path(paper_dat).exists():
        data = load_convection_diffusion(paper_dat)
        p = output_dir / "paper_ADE_Convection_diffusion.npz"
        save_dataset_npz(data, p)
        paths.append(p)
    if paper_fade_dat is not None and Path(paper_fade_dat).exists():
        data = load_paper_fade(paper_fade_dat)
        p = output_dir / "paper_FADE_tsfade_fft.npz"
        save_dataset_npz(data, p)
        paths.append(p)
    data2 = make_space_fractional_advection_dispersion()
    p2 = output_dir / "synthetic_space_fractional_RD.npz"
    save_dataset_npz(data2, p2)
    paths.append(p2)
    data3 = make_time_space_fractional_advection_dispersion()
    p3 = output_dir / "synthetic_time_space_fractional_RD.npz"
    save_dataset_npz(data3, p3)
    paths.append(p3)
    data4 = make_two_fractional_rhs_dataset()
    p4 = output_dir / "synthetic_two_fractional_rhs.npz"
    save_dataset_npz(data4, p4)
    paths.append(p4)
    data5 = make_fractional_burgers()
    p5 = output_dir / "synthetic_fractional_burgers.npz"
    save_dataset_npz(data5, p5)
    paths.append(p5)
    data6 = make_superunit_fractional_diffusion()
    p6 = output_dir / "synthetic_superunit_fractional_diffusion.npz"
    save_dataset_npz(data6, p6)
    paths.append(p6)
    return paths
