"""Reproducible adaptation of Yu et al. (2025) for fair FPDE benchmarks.

This module preserves the essential method in the public Yu et al. code:

* a differentiable neural surrogate (or a spline ablation),
* Gauss--Jacobi quadrature for Caputo/Riemann--Liouville-style derivatives,
* STRidge for sparse linear coefficients, and
* differential evolution for the nonlinear fractional orders.

The adaptation removes notebook/global-state assumptions, freezes noise and
train/validation splits per seed, prevents validation leakage in STRidge, adds
CPU/GPU selection, vectorises derivative evaluation, and optionally precomputes
fractional-order banks followed by an exact-order final polish.

The packaged shared-field comparison is restricted to the paper FADE/ADE data.
Even there, the operator realizations are not mathematically identical:
Weak-Pareto uses its periodic directional spectral convention, while this Yu
adapter uses one-sided fractional derivatives with a finite lower terminal.  It
is not a drop-in operator-identical baseline for periodic signed-Riesz datasets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence
import csv
import json
import random
import time

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import differential_evolution, minimize
from scipy.special import gamma, roots_jacobi

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - handled at runtime for spline-only mode
    torch = None
    nn = None

from fpde_datasets import GridDataset, add_multiplicative_uniform_noise

EPS = np.finfo(float).eps

YuSurrogate = Literal["neural", "spline"]
YuProtocol = Literal["matched", "faithful"]
YuOrderMode = Literal["bank", "exact"]
YuDevice = Literal["auto", "cpu", "cuda", "mps"]


@dataclass
class YuBaselineConfig:
    """Configuration for the adapted neural fractional-discovery framework of Yu et al."""

    surrogate: YuSurrogate = "neural"
    protocol: YuProtocol = "matched"
    device: YuDevice = "auto"
    dtype: Literal["float32", "float64"] = "float32"
    seed: int = 0
    noise_percent: float = 0.0

    # Neural reconstruction.  FADE in the paper uses five hidden Gaussian layers.
    hidden_width: int = 20
    hidden_layers: int = 5
    activation: Literal["gaussian", "tanh"] = "gaussian"
    epochs: int = 2000
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 300
    min_delta: float = 1e-9
    paper_train_count: int = 1000
    paper_val_count: int = 2000
    matched_val_fraction: float = 0.20
    matched_max_points: int | None = None

    # Reconstructed interior grid used by the official FADE discovery script.
    eval_x_min: float = 4.0
    eval_x_max: float = 26.0
    eval_t_min: float = 3.0
    eval_t_max: float = 14.0
    eval_dx: float = 0.10
    eval_dt: float = 0.10
    discovery_points: int | None = None
    derivative_batch_size: int = 32768

    # Gauss--Jacobi fractional derivative calculation.
    quadrature_nodes: int = 5
    alpha_bounds: tuple[float, float] = (0.60, 0.999)
    beta_bounds: tuple[float, float] = (1.01, 1.999)
    order_mode: YuOrderMode = "bank"
    alpha_bank_size: int = 41
    beta_bank_size: int = 61

    # Yu alternating sparse/global optimisation.
    de_maxiter: int = 15
    de_popsize: int = 15
    de_tol: float = 1e-4
    de_polish: bool = True
    de_workers: int = 1
    exact_order_polish: bool = True
    exact_polish_maxiter: int = 40

    stridge_ridge: float = 2.0
    stridge_iters: int = 10
    threshold_initial: float = 0.1
    threshold_search_iters: int = 25
    l0_weight: float = 1e-3
    condition_cap: float = 1e12
    normalize_library: bool = False
    active_abs_tol: float = 1e-10

    # Optional output controls.
    save_order_banks: bool = False
    verbose: bool = True

    @classmethod
    def for_profile(
        cls,
        profile: Literal["smoke", "standard", "paper"],
        **overrides: Any,
    ) -> "YuBaselineConfig":
        """Construct a practical compute profile without changing the method."""
        if profile == "smoke":
            base = cls(
                hidden_width=8,
                hidden_layers=2,
                epochs=2,
                patience=2,
                matched_max_points=1200,
                batch_size=600,
                eval_dx=1.5,
                eval_dt=1.5,
                discovery_points=80,
                alpha_bank_size=2,
                beta_bank_size=2,
                de_maxiter=0,
                de_popsize=2,
                de_polish=False,
                exact_order_polish=False,
                threshold_search_iters=4,
                derivative_batch_size=4096,
            )
        elif profile == "standard":
            base = cls(
                epochs=1500,
                patience=250,
                eval_dx=0.20,
                eval_dt=0.20,
                discovery_points=6000,
                alpha_bank_size=31,
                beta_bank_size=41,
                de_maxiter=10,
                de_popsize=10,
                exact_polish_maxiter=25,
            )
        elif profile == "paper":
            base = cls(
                epochs=10000,
                patience=1200,
                eval_dx=0.10,
                eval_dt=0.10,
                discovery_points=None,
                alpha_bank_size=61,
                beta_bank_size=81,
                de_maxiter=15,
                de_popsize=15,
                exact_polish_maxiter=50,
            )
        else:  # pragma: no cover
            raise ValueError(f"unknown profile={profile!r}")
        for key, value in overrides.items():
            if value is not None:
                setattr(base, key, value)
        return base


@dataclass
class YuBaselineResult:
    """Selected equation and complete diagnostics for one Yu-framework run."""

    name: str
    dataset: str
    equation: str
    alpha: float
    beta: float
    terms: list[tuple[int, float]]
    term_names: list[str]
    coefficients: list[float]
    support_size: int
    train_mse: float
    val_mse: float
    val_rel_mse: float
    objective: float
    condition_number: float
    threshold: float
    runtime_seconds: float
    runtime_breakdown: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["terms"] = [[int(p), float(b)] for p, b in self.terms]
        return out

    def selected_model_dict(self) -> dict[str, Any]:
        return {
            "equation": self.equation,
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "terms": [[int(p), float(b)] for p, b in self.terms],
            "term_names": list(self.term_names),
            "coefficients": [float(v) for v in self.coefficients],
            "c": int(self.support_size),
            "val_mse": float(self.val_mse),
            "val_rel_mse": float(self.val_rel_mse),
            "objective": float(self.objective),
        }


@dataclass(frozen=True)
class _CandidateTerm:
    name: str
    p: int
    beta_kind: Literal["constant", "fixed", "fractional"]
    beta: float


# This is the exact candidate ordering in the public fSTRidge.py script.
_YU_TERMS: tuple[_CandidateTerm, ...] = (
    _CandidateTerm("1", -1, "constant", 0.0),
    _CandidateTerm("u", 0, "fixed", 0.0),
    _CandidateTerm("u_x", 0, "fixed", 1.0),
    _CandidateTerm("D_x^beta u", 0, "fractional", float("nan")),
    _CandidateTerm("u_xxx", 0, "fixed", 3.0),
    _CandidateTerm("u^2", 1, "fixed", 0.0),
    _CandidateTerm("u u_x", 1, "fixed", 1.0),
    _CandidateTerm("u u_xx", 1, "fixed", 2.0),
    _CandidateTerm("u u_xxx", 1, "fixed", 3.0),
    _CandidateTerm("u^2 u_x", 2, "fixed", 1.0),
    _CandidateTerm("u^2 u_xx", 2, "fixed", 2.0),
    _CandidateTerm("u^2 u_xxx", 2, "fixed", 3.0),
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(type(obj).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(requested: YuDevice = "auto") -> str:
    """Resolve a requested PyTorch device deterministically."""
    if requested == "cpu":
        return "cpu"
    if torch is None:
        if requested == "auto":
            return "cpu"
        raise RuntimeError("PyTorch is required for the requested neural device")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return "cuda"
    if requested == "mps":
        available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if not available:
            raise RuntimeError("--device mps was requested but MPS is unavailable")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return "mps"
    return "cpu"


if nn is not None:
    class LearnableGaussianActivation(nn.Module):
        """Learnable Gaussian activation used in the Yu FADE network."""

        def __init__(self) -> None:
            super().__init__()
            # Match the public Yu implementation: random mean and unit scale.
            # A tiny denominator floor prevents a singular activation without
            # changing the squared-scale parameterisation.
            self.mu = nn.Parameter(torch.randn(1))
            self.sigma = nn.Parameter(torch.ones(1))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            sigma_sq = self.sigma.square().clamp_min(1e-12)
            return torch.exp(-((x - self.mu) ** 2) / (2.0 * sigma_sq))


    class YuSurrogateNet(nn.Module):
        """Coordinate MLP matching the paper's 2-20-...-20-1 structure."""

        def __init__(self, width: int, hidden_layers: int, activation: str) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = 2
            for _ in range(int(hidden_layers)):
                layers.append(nn.Linear(in_dim, int(width)))
                layers.append(LearnableGaussianActivation() if activation == "gaussian" else nn.Tanh())
                in_dim = int(width)
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            return self.net(z)


class _NeuralSurrogate:
    def __init__(
        self,
        model: "YuSurrogateNet",
        *,
        device: str,
        dtype: "torch.dtype",
        x_min: float,
        x_max: float,
        t_min: float,
        t_max: float,
        derivative_batch_size: int,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.t_min, self.t_max = float(t_min), float(t_max)
        self.sx = 2.0 / max(self.x_max - self.x_min, EPS)
        self.st = 2.0 / max(self.t_max - self.t_min, EPS)
        self.batch_size = int(derivative_batch_size)

    def _normalise(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.empty_like(points, dtype=float)
        z[:, 0] = 2.0 * (points[:, 0] - self.x_min) / max(self.x_max - self.x_min, EPS) - 1.0
        z[:, 1] = 2.0 * (points[:, 1] - self.t_min) / max(self.t_max - self.t_min, EPS) - 1.0
        return z

    def derivative(self, points: NDArray[np.float64], *, dx: int = 0, dt: int = 0) -> NDArray[np.float64]:
        if dx < 0 or dt < 0 or dx + dt > 3:
            raise ValueError("derivative orders must be nonnegative with dx+dt<=3")
        z_all = self._normalise(np.asarray(points, dtype=float))
        out: list[NDArray[np.float64]] = []
        self.model.eval()
        for start in range(0, len(z_all), self.batch_size):
            z_np = z_all[start : start + self.batch_size]
            z = torch.as_tensor(z_np, device=self.device, dtype=self.dtype).clone().detach().requires_grad_(dx + dt > 0)
            y = self.model(z)
            g = y
            # Pure derivatives are all that the baseline requires.
            for _ in range(dx):
                g = torch.autograd.grad(g.sum(), z, create_graph=True)[0][:, 0:1]
            for _ in range(dt):
                g = torch.autograd.grad(g.sum(), z, create_graph=True)[0][:, 1:2]
            scale = (self.sx ** dx) * (self.st ** dt)
            out.append((g.detach().cpu().numpy().reshape(-1) * scale).astype(float))
            del z, y, g
        return np.concatenate(out) if out else np.empty(0, dtype=float)


class _SplineSurrogate:
    """CPU-only smooth surrogate used to isolate Yu's sparse/global component."""

    def __init__(self, t: NDArray[np.float64], x: NDArray[np.float64], U: NDArray[np.float64]) -> None:
        # RectBivariateSpline uses axis order (t, x).
        self.spline = RectBivariateSpline(np.asarray(t), np.asarray(x), np.asarray(U), kx=5, ky=5, s=0.0)

    def derivative(self, points: NDArray[np.float64], *, dx: int = 0, dt: int = 0) -> NDArray[np.float64]:
        p = np.asarray(points, dtype=float)
        return np.asarray(self.spline.ev(p[:, 1], p[:, 0], dx=int(dt), dy=int(dx)), dtype=float)


def _training_indices(total: int, cfg: YuBaselineConfig) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(total)
    if cfg.protocol == "faithful":
        n_train = min(int(cfg.paper_train_count), max(1, total - 1))
        n_val = min(int(cfg.paper_val_count), max(1, total - n_train))
    else:
        usable = total if cfg.matched_max_points is None else min(total, int(cfg.matched_max_points))
        order = order[:usable]
        n_val = max(1, int(round(float(cfg.matched_val_fraction) * usable)))
        n_train = max(1, usable - n_val)
    return order[:n_train], order[n_train : n_train + n_val]


def _train_neural_surrogate(
    data: GridDataset,
    U_noisy: NDArray[np.float64],
    cfg: YuBaselineConfig,
    output_dir: Path | None,
) -> tuple[_NeuralSurrogate, dict[str, Any]]:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for surrogate='neural'. Install requirements_yu2025.txt")
    _set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32

    X, T = np.meshgrid(data.x, data.t)
    points = np.column_stack([X.reshape(-1), T.reshape(-1)]).astype(float)
    values = np.asarray(U_noisy, dtype=float).reshape(-1, 1)
    train_idx, val_idx = _training_indices(len(points), cfg)

    x_min, x_max = float(data.x.min()), float(data.x.max())
    t_min, t_max = float(data.t.min()), float(data.t.max())
    z = np.empty_like(points)
    z[:, 0] = 2.0 * (points[:, 0] - x_min) / max(x_max - x_min, EPS) - 1.0
    z[:, 1] = 2.0 * (points[:, 1] - t_min) / max(t_max - t_min, EPS) - 1.0

    model = YuSurrogateNet(cfg.hidden_width, cfg.hidden_layers, cfg.activation).to(device=device, dtype=dtype)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed + 1729)
    batch_size = min(max(1, int(cfg.batch_size)), len(train_idx))

    z_val = torch.as_tensor(z[val_idx], device=device, dtype=dtype)
    y_val = torch.as_tensor(values[val_idx], device=device, dtype=dtype)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    report_every = max(1, int(cfg.epochs) // 20)

    for epoch in range(int(cfg.epochs)):
        model.train()
        shuffled = rng.permutation(train_idx)
        train_loss_sum = 0.0
        seen = 0
        for start in range(0, len(shuffled), batch_size):
            idx = shuffled[start : start + batch_size]
            xb = torch.as_tensor(z[idx], device=device, dtype=dtype)
            yb = torch.as_tensor(values[idx], device=device, dtype=dtype)
            optimiser.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite neural reconstruction loss")
            loss.backward()
            optimiser.step()
            train_loss_sum += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((model(z_val) - y_val) ** 2).detach().cpu())
        train_loss = train_loss_sum / max(1, seen)
        history.append({"epoch": epoch + 1, "train_mse": train_loss, "val_mse": val_loss})
        improved = val_loss < best_val - float(cfg.min_delta)
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if cfg.verbose and (epoch == 0 or (epoch + 1) % report_every == 0 or epoch + 1 == cfg.epochs):
            print(f"[Yu DNN] epoch={epoch+1} train_mse={train_loss:.6e} val_mse={val_loss:.6e} device={device}")
        if stale >= int(cfg.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "val_mse"])
            writer.writeheader()
            writer.writerows(history)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(cfg),
                "coordinate_bounds": {"x": [x_min, x_max], "t": [t_min, t_max]},
            },
            output_dir / "surrogate_checkpoint.pt",
        )

    surrogate = _NeuralSurrogate(
        model,
        device=device,
        dtype=dtype,
        x_min=x_min,
        x_max=x_max,
        t_min=t_min,
        t_max=t_max,
        derivative_batch_size=cfg.derivative_batch_size,
    )
    return surrogate, {
        "device": device,
        "dtype": cfg.dtype,
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "epochs_completed": int(len(history)),
        "best_val_mse": float(best_val),
        "activation": cfg.activation,
        "hidden_width": int(cfg.hidden_width),
        "hidden_layers": int(cfg.hidden_layers),
    }


def _evaluation_points(data: GridDataset, cfg: YuBaselineConfig) -> tuple[NDArray[np.float64], dict[str, Any]]:
    x_lo = max(float(cfg.eval_x_min), float(data.x.min()))
    x_hi = min(float(cfg.eval_x_max), float(data.x.max()) + 0.5 * data.dx)
    t_lo = max(float(cfg.eval_t_min), float(data.t.min()))
    t_hi = min(float(cfg.eval_t_max), float(data.t.max()) + 0.5 * data.dt)
    x = np.arange(x_lo, x_hi, float(cfg.eval_dx), dtype=float)
    t = np.arange(t_lo, t_hi, float(cfg.eval_dt), dtype=float)
    if len(x) < 2 or len(t) < 2:
        raise ValueError("evaluation grid is empty; check eval bounds and steps")
    X, T = np.meshgrid(x, t)
    points = np.column_stack([X.reshape(-1), T.reshape(-1)])
    total = len(points)
    if cfg.discovery_points is not None and int(cfg.discovery_points) < total:
        rng = np.random.default_rng(cfg.seed + 811)
        idx = np.sort(rng.choice(total, size=int(cfg.discovery_points), replace=False))
        points = points[idx]
    return points, {
        "x_range": [float(x[0]), float(x[-1])],
        "t_range": [float(t[0]), float(t[-1])],
        "eval_dx": float(cfg.eval_dx),
        "eval_dt": float(cfg.eval_dt),
        "full_grid_points": int(total),
        "discovery_points": int(len(points)),
    }


def _fractional_time_caputo(
    surrogate: Any,
    points: NDArray[np.float64],
    alpha: float,
    n_quad: int,
) -> NDArray[np.float64]:
    a = float(alpha)
    if abs(a - 1.0) < 1e-10:
        return surrogate.derivative(points, dt=1)
    if not (0.0 < a < 1.0):
        raise ValueError(f"Yu FADE time order must be in (0,1], got {a}")
    nodes, weights = roots_jacobi(int(n_quad), 0.0, -a)
    n = len(points)
    q_points = np.repeat(points, int(n_quad), axis=0)
    t = points[:, 1]
    q_points[:, 1] = (t[:, None] - 0.5 * t[:, None] * (nodes[None, :] + 1.0)).reshape(-1)
    vals = surrogate.derivative(q_points, dt=1).reshape(n, int(n_quad))
    factor = (0.5 * t) ** (1.0 - a) / gamma(1.0 - a)
    return (vals @ np.asarray(weights, dtype=float)) * factor


def _fractional_space_left(
    surrogate: Any,
    points: NDArray[np.float64],
    beta: float,
    n_quad: int,
) -> NDArray[np.float64]:
    b = float(beta)
    if abs(b - 1.0) < 1e-10:
        return surrogate.derivative(points, dx=1)
    if abs(b - 2.0) < 1e-10:
        return surrogate.derivative(points, dx=2)
    if not (1.0 < b < 2.0):
        raise ValueError(f"Yu FADE space order must be in [1,2], got {b}")
    nodes, weights = roots_jacobi(int(n_quad), 0.0, 1.0 - b)
    n = len(points)
    q_points = np.repeat(points, int(n_quad), axis=0)
    x = points[:, 0]
    q_points[:, 0] = (x[:, None] - 0.5 * x[:, None] * (nodes[None, :] + 1.0)).reshape(-1)
    vals = surrogate.derivative(q_points, dx=2).reshape(n, int(n_quad))
    factor = (0.5 * x) ** (2.0 - b) / gamma(2.0 - b)
    return (vals @ np.asarray(weights, dtype=float)) * factor


def _interp_bank(order: float, grid: NDArray[np.float64], bank: NDArray[np.float64]) -> NDArray[np.float64]:
    z = float(np.clip(order, grid[0], grid[-1]))
    i = int(np.searchsorted(grid, z, side="right") - 1)
    if i <= -1:
        return bank[0]
    if i >= len(grid) - 1:
        return bank[-1]
    lo, hi = float(grid[i]), float(grid[i + 1])
    w = (z - lo) / max(hi - lo, EPS)
    return (1.0 - w) * bank[i] + w * bank[i + 1]


def _base_library(surrogate: Any, points: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    u = surrogate.derivative(points)
    ux = surrogate.derivative(points, dx=1)
    uxx = surrogate.derivative(points, dx=2)
    uxxx = surrogate.derivative(points, dx=3)
    return {"u": u, "ux": ux, "uxx": uxx, "uxxx": uxxx}


def _assemble_library(base: dict[str, NDArray[np.float64]], frac_beta: NDArray[np.float64]) -> NDArray[np.float64]:
    u, ux, uxx, uxxx = base["u"], base["ux"], base["uxx"], base["uxxx"]
    return np.column_stack(
        [
            np.ones_like(u),
            u,
            ux,
            frac_beta,
            uxxx,
            u**2,
            u * ux,
            u * uxx,
            u * uxxx,
            (u**2) * ux,
            (u**2) * uxx,
            (u**2) * uxxx,
        ]
    )


def _stridge(
    X0: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    ridge: float,
    maxit: int,
    tol: float,
    normalize: bool,
) -> NDArray[np.float64]:
    """Sequential threshold ridge regression, faithful to the public implementation."""
    X0 = np.asarray(X0, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, d = X0.shape
    scales = np.ones(d, dtype=float)
    X = X0.copy()
    if normalize:
        norms = np.linalg.norm(X, axis=0)
        scales = np.where(norms > EPS, 1.0 / norms, 1.0)
        X = X * scales[None, :]
    A = X.T @ X + float(ridge) * np.eye(d)
    w = np.linalg.lstsq(A, X.T @ y, rcond=None)[0]
    big = np.flatnonzero(np.abs(w) > float(tol))
    previous = d
    for _ in range(int(maxit)):
        big = np.flatnonzero(np.abs(w) >= float(tol))
        if len(big) == previous:
            break
        previous = len(big)
        if len(big) == 0:
            break
        small = np.setdiff1d(np.arange(d), big, assume_unique=True)
        w[small] = 0.0
        A_big = X[:, big].T @ X[:, big] + float(ridge) * np.eye(len(big))
        w[big] = np.linalg.lstsq(A_big, X[:, big].T @ y, rcond=None)[0]
    if len(big) > 0:
        # The original code ends with an unregularized refit on the selected support.
        w[:] = 0.0
        w[big] = np.linalg.lstsq(X[:, big], y, rcond=None)[0]
    return scales * w if normalize else w


@dataclass
class _SparseFit:
    coefficients: NDArray[np.float64]
    objective: float
    threshold: float
    train_mse: float
    val_mse: float
    val_rel_mse: float
    condition_number: float


def _sparse_threshold_search(
    R: NDArray[np.float64],
    y: NDArray[np.float64],
    train_idx: NDArray[np.int64],
    val_idx: NDArray[np.int64],
    cfg: YuBaselineConfig,
) -> _SparseFit:
    finite = np.isfinite(y) & np.all(np.isfinite(R), axis=1)
    tr = train_idx[finite[train_idx]]
    va = val_idx[finite[val_idx]]
    if len(tr) < R.shape[1] + 2 or len(va) < 2:
        raise RuntimeError("insufficient finite rows for Yu STRidge")
    Xtr, ytr = R[tr], y[tr]
    Xv, yv = R[va], y[va]
    # The sparsity penalty is calibrated from the training design only.
    # Using validation rows here would leak validation information into model
    # selection even though the coefficients themselves are fit on Xtr.
    cond = float(np.linalg.cond(Xtr))
    cond_for_penalty = min(cond if np.isfinite(cond) else cfg.condition_cap, float(cfg.condition_cap))
    l0_penalty = float(cfg.l0_weight) * cond_for_penalty

    w_best = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    resid = yv - Xv @ w_best
    best_obj = float(np.linalg.norm(resid, 2) + l0_penalty * np.count_nonzero(w_best))
    best_tol = 0.0
    d_tol = float(cfg.threshold_initial)
    tol = d_tol

    for i in range(int(cfg.threshold_search_iters)):
        w = _stridge(
            Xtr,
            ytr,
            ridge=cfg.stridge_ridge,
            maxit=cfg.stridge_iters,
            tol=tol,
            normalize=cfg.normalize_library,
        )
        resid = yv - Xv @ w
        obj = float(np.linalg.norm(resid, 2) + l0_penalty * np.count_nonzero(w))
        if obj <= best_obj:
            best_obj, w_best, best_tol = obj, w.copy(), tol
            tol += d_tol
        else:
            tol = max(0.0, tol - 2.0 * d_tol)
            d_tol = 2.0 * d_tol / max(1, int(cfg.threshold_search_iters) - i)
            tol += d_tol

    tr_res = ytr - Xtr @ w_best
    va_res = yv - Xv @ w_best
    train_mse = float(np.mean(tr_res**2))
    val_mse = float(np.mean(va_res**2))
    val_rel = float(val_mse / (np.var(yv) + EPS))
    return _SparseFit(
        coefficients=np.asarray(w_best, dtype=float),
        objective=best_obj,
        threshold=float(best_tol),
        train_mse=train_mse,
        val_mse=val_mse,
        val_rel_mse=val_rel,
        condition_number=cond,
    )


def _active_result_terms(beta: float, coefficients: Sequence[float], tol: float) -> tuple[list[tuple[int, float]], list[str], list[float]]:
    terms: list[tuple[int, float]] = []
    names: list[str] = []
    coefs: list[float] = []
    for spec, coef in zip(_YU_TERMS, coefficients):
        c = float(coef)
        if abs(c) <= float(tol):
            continue
        b = float(beta) if spec.beta_kind == "fractional" else float(spec.beta)
        terms.append((int(spec.p), b))
        names.append(spec.name.replace("beta", f"{beta:.6g}"))
        coefs.append(c)
    return terms, names, coefs


def _format_equation(alpha: float, names: Sequence[str], coefficients: Sequence[float]) -> str:
    rhs: list[str] = []
    for i, (name, coef) in enumerate(zip(names, coefficients)):
        sign = "+" if coef >= 0 else "-"
        body = f"{abs(float(coef)):.6g} {name}"
        rhs.append(body if i == 0 and coef >= 0 else f"{sign} {body}")
    return f"D_t^{float(alpha):.6g} u = " + (" ".join(rhs) if rhs else "0")


def _dataset_scope_check(data: GridDataset) -> None:
    if data.name not in {"paper_FADE_tsfade_fft", "paper_ADE_Convection_diffusion"}:
        raise ValueError(
            f"{data.name!r} is outside the declared Yu adapter comparison scope. "
            "The adapter uses one-sided finite-terminal fractional derivatives, whereas "
            "the synthetic reaction-diffusion benchmarks use periodic signed-Riesz operators."
        )


def run_yu_baseline(
    data: GridDataset,
    config: YuBaselineConfig,
    output_dir: str | Path | None = None,
) -> YuBaselineResult:
    """Run the adapted neural fractional-discovery framework of Yu et al. on a predefined shared-field dataset."""
    _dataset_scope_check(data)
    cfg = config
    _set_seed(cfg.seed)
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "yu_config.json", asdict(cfg))

    started = time.perf_counter()
    phases: dict[str, float] = {}
    U_noisy = add_multiplicative_uniform_noise(data.U, cfg.noise_percent, seed=cfg.seed)
    if out is not None:
        np.save(out / "shared_noisy_field.npy", U_noisy)

    t0 = time.perf_counter()
    if cfg.surrogate == "neural":
        surrogate, surrogate_meta = _train_neural_surrogate(data, U_noisy, cfg, out)
    elif cfg.surrogate == "spline":
        surrogate = _SplineSurrogate(data.t, data.x, U_noisy)
        surrogate_meta = {
            "device": "cpu",
            "dtype": "float64",
            "train_count": int(U_noisy.size),
            "val_count": 0,
            "surrogate": "tensor-product quintic RectBivariateSpline",
        }
    else:  # pragma: no cover
        raise ValueError(f"unknown surrogate={cfg.surrogate!r}")
    phases["surrogate_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    points, eval_meta = _evaluation_points(data, cfg)
    base = _base_library(surrogate, points)
    phases["integer_library_seconds"] = time.perf_counter() - t0

    # Freeze one split for all order evaluations. This removes the stochastic DE
    # objective in the original global-state implementation.
    rng = np.random.default_rng(cfg.seed + 4242)
    perm = rng.permutation(len(points))
    n_val = max(2, int(round(0.20 * len(points))))
    val_idx = np.asarray(perm[:n_val], dtype=np.int64)
    train_idx = np.asarray(perm[n_val:], dtype=np.int64)

    alpha_grid = np.linspace(cfg.alpha_bounds[0], cfg.alpha_bounds[1], int(cfg.alpha_bank_size))
    beta_grid = np.linspace(cfg.beta_bounds[0], cfg.beta_bounds[1], int(cfg.beta_bank_size))
    alpha_bank: NDArray[np.float64] | None = None
    beta_bank: NDArray[np.float64] | None = None

    if cfg.order_mode == "bank":
        t0 = time.perf_counter()
        alpha_bank = np.vstack([
            _fractional_time_caputo(surrogate, points, float(a), cfg.quadrature_nodes)
            for a in alpha_grid
        ])
        beta_bank = np.vstack([
            _fractional_space_left(surrogate, points, float(b), cfg.quadrature_nodes)
            for b in beta_grid
        ])
        phases["fractional_bank_seconds"] = time.perf_counter() - t0
        if out is not None and cfg.save_order_banks:
            np.savez_compressed(
                out / "fractional_order_banks.npz",
                alpha_grid=alpha_grid,
                beta_grid=beta_grid,
                alpha_bank=alpha_bank,
                beta_bank=beta_bank,
                points=points,
            )

    evaluation_count = 0
    best_seen = float("inf")

    def evaluate(params: Sequence[float], exact: bool = False) -> tuple[_SparseFit, NDArray[np.float64], NDArray[np.float64]]:
        nonlocal evaluation_count, best_seen
        alpha, beta = float(params[0]), float(params[1])
        if exact or cfg.order_mode == "exact":
            target = _fractional_time_caputo(surrogate, points, alpha, cfg.quadrature_nodes)
            frac = _fractional_space_left(surrogate, points, beta, cfg.quadrature_nodes)
        else:
            assert alpha_bank is not None and beta_bank is not None
            target = _interp_bank(alpha, alpha_grid, alpha_bank)
            frac = _interp_bank(beta, beta_grid, beta_bank)
        R = _assemble_library(base, frac)
        fit = _sparse_threshold_search(R, target, train_idx, val_idx, cfg)
        evaluation_count += 1
        if cfg.verbose and fit.objective < best_seen:
            best_seen = fit.objective
            print(
                f"[Yu DE] eval={evaluation_count} objective={fit.objective:.6e} "
                f"alpha={alpha:.6f} beta={beta:.6f} active={np.count_nonzero(fit.coefficients)}"
            )
        return fit, target, R

    t0 = time.perf_counter()
    de_result = differential_evolution(
        lambda z: evaluate(z, exact=False)[0].objective,
        bounds=[tuple(cfg.alpha_bounds), tuple(cfg.beta_bounds)],
        maxiter=int(cfg.de_maxiter),
        popsize=int(cfg.de_popsize),
        tol=float(cfg.de_tol),
        seed=int(cfg.seed),
        polish=bool(cfg.de_polish),
        workers=int(cfg.de_workers),
        updating="deferred" if int(cfg.de_workers) != 1 else "immediate",
        disp=bool(cfg.verbose),
    )
    alpha_opt, beta_opt = map(float, de_result.x)
    phases["de_seconds"] = time.perf_counter() - t0

    if cfg.exact_order_polish:
        t0 = time.perf_counter()
        polished = minimize(
            lambda z: evaluate(z, exact=True)[0].objective,
            x0=np.array([alpha_opt, beta_opt], dtype=float),
            method="Powell",
            bounds=[tuple(cfg.alpha_bounds), tuple(cfg.beta_bounds)],
            options={"maxiter": int(cfg.exact_polish_maxiter), "xtol": 1e-5, "ftol": 1e-6},
        )
        if polished.success or np.isfinite(polished.fun):
            alpha_opt, beta_opt = map(float, polished.x)
        phases["exact_polish_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    final_fit, _, _ = evaluate((alpha_opt, beta_opt), exact=True)
    phases["exact_refit_seconds"] = time.perf_counter() - t0
    terms, names, coefs = _active_result_terms(beta_opt, final_fit.coefficients, cfg.active_abs_tol)
    equation = _format_equation(alpha_opt, names, coefs)
    runtime = time.perf_counter() - started

    result = YuBaselineResult(
        name="yu2025_full" if cfg.surrogate == "neural" else "yu2025_optimizer_only",
        dataset=data.name,
        equation=equation,
        alpha=alpha_opt,
        beta=beta_opt,
        terms=terms,
        term_names=names,
        coefficients=coefs,
        support_size=len(terms),
        train_mse=final_fit.train_mse,
        val_mse=final_fit.val_mse,
        val_rel_mse=final_fit.val_rel_mse,
        objective=final_fit.objective,
        condition_number=final_fit.condition_number,
        threshold=final_fit.threshold,
        runtime_seconds=runtime,
        runtime_breakdown=phases,
        metadata={
            "source_method": "Yu et al. (2025): DNN + Gauss-Jacobi + STRidge + DE",
            "surrogate": cfg.surrogate,
            "protocol": cfg.protocol,
            "order_mode": cfg.order_mode,
            "quadrature_nodes": int(cfg.quadrature_nodes),
            "alpha_bounds": list(cfg.alpha_bounds),
            "beta_bounds": list(cfg.beta_bounds),
            "candidate_terms": [term.name for term in _YU_TERMS],
            "evaluation_count": int(evaluation_count),
            "surrogate_metadata": surrogate_meta,
            "evaluation_grid": eval_meta,
            "adaptations": [
                "fixed noise and train/validation splits per seed",
                "STRidge fits training rows only; validation rows are not leaked into coefficient fitting",
                "normalized neural coordinates with exact chain-rule derivative scaling",
                "vectorized Gauss-Jacobi evaluation",
                "optional order-bank acceleration followed by exact-order polishing/refit",
                "explicit CPU/CUDA/MPS device selection",
            ],
            "operator_scope": "one-sided finite-terminal Caputo/RL-style derivatives; not periodic signed Riesz",
        },
    )
    if out is not None:
        _write_json(out / "summary.json", result.to_dict())
        _write_json(out / "selected_model.json", result.selected_model_dict())
    return result


__all__ = [
    "YuBaselineConfig",
    "YuBaselineResult",
    "run_yu_baseline",
    "resolve_device",
]
