"""Save/load selected governing FDEs as simple JSON records.

The file format is intentionally lightweight and independent of any downstream
refinement/simulation code:

    {
      "format": "selected_governing_fde_v1",
      "dataset_name": "paper_FADE_tsfade_fft",
      "alpha": 0.8,
      "terms": [
        {"power": 0, "beta": 1.0, "coefficient": -1.0, "side": "left"},
        {"power": 0, "beta": 1.7, "coefficient": 0.5, "side": "left"}
      ],
      "equation": "D_t^0.8 u = ..."
    }

The word ``side`` means the spatial derivative convention used for the RHS
operator, not the PDE equation side. All listed terms are RHS terms.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json
import time


FORMAT_VERSION = "selected_governing_fde_v1"


def _model_to_dict(model_or_dict: Any) -> dict[str, Any]:
    """Return a JSON-compatible selected-model dictionary.

    Accepts either a ``PDEModel`` object from ``pareto_fde_discovery`` or a dict
    such as ``summary["selected"]``.
    """
    if hasattr(model_or_dict, "to_dict"):
        return model_or_dict.to_dict()
    if isinstance(model_or_dict, dict):
        return dict(model_or_dict)
    raise TypeError("Expected a PDEModel or a selected-model dictionary")


def infer_spatial_side(*, config: Any | None = None, dataset_name: str | None = None, default: str = "left") -> str:
    """Infer the spatial derivative convention for exported terms.

    Parameters
    ----------
    config:
        Optional Pareto ``DiscoveryConfig``. If ``config.spectral_riesz`` is
        true, the exported terms use ``side='riesz_like'``. Otherwise this uses
        ``config.regularized_space_side`` when present.
    dataset_name:
        Optional benchmark name. Synthetic Riesz benchmark datasets default to
        ``'riesz_like'``; paper ADE/FADE datasets default to ``'left'``.
    default:
        Fallback spatial convention.
    """
    if config is not None:
        if bool(getattr(config, "spectral_riesz", False)):
            return "riesz_like"
        side = getattr(config, "regularized_space_side", None)
        if side:
            return str(side)
    if dataset_name in {
        "synthetic_space_fractional_RD",
        "synthetic_time_space_fractional_RD",
        "synthetic_two_fractional_rhs",
    }:
        return "riesz_like"
    return default


def selected_model_to_fde_dict(
    model_or_dict: Any,
    *,
    dataset_name: str | None = None,
    config: Any | None = None,
    side: str | None = None,
    sides: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a Pareto-selected model into a portable selected-FDE dict.

    Parameters
    ----------
    model_or_dict:
        Either the ``PDEModel`` returned by ``select_model`` or a dictionary such
        as ``summary['selected']``.
    dataset_name:
        Dataset identifier used for traceability.
    config:
        Pareto ``DiscoveryConfig`` used to infer spatial derivative side.
    side:
        One spatial operator side applied to all RHS terms. Use this when you
        know the selected terms should all be left-sided or Riesz-like.
    sides:
        Optional per-term spatial operator sides. Overrides ``side``.
    metadata:
        Extra JSON-serializable metadata to store with the file.
    """
    selected = _model_to_dict(model_or_dict)
    alpha = float(selected.get("alpha", selected.get("r", selected.get("time_order"))))
    raw_terms = selected.get("terms")
    if raw_terms is None:
        p_tuple = selected.get("p_tuple", [])
        beta_tuple = selected.get("beta_tuple", [])
        raw_terms = list(zip(p_tuple, beta_tuple))
    coefs = selected.get("coefficients", [])
    if sides is not None and len(sides) != len(raw_terms):
        raise ValueError("sides must have one entry per selected RHS term")
    default_side = side or infer_spatial_side(config=config, dataset_name=dataset_name)

    terms: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_terms):
        if isinstance(raw, dict):
            power = int(raw.get("power", raw.get("p", 0)))
            beta = float(raw.get("beta", raw.get("q", 0.0)))
            coef = float(raw.get("coefficient", coefs[i] if i < len(coefs) else 1.0))
            term_side = raw.get("side", default_side)
        else:
            power = int(raw[0])
            beta = float(raw[1])
            coef = float(coefs[i] if i < len(coefs) else 1.0)
            term_side = default_side
        if sides is not None:
            term_side = str(sides[i])
        terms.append({"power": power, "beta": beta, "coefficient": coef, "side": str(term_side)})

    out = {
        "format": FORMAT_VERSION,
        "dataset_name": dataset_name,
        "name": f"{dataset_name or 'unknown_dataset'}_pareto_selected",
        "alpha": alpha,
        "alpha_mode": selected.get("alpha_mode", "integer" if abs(alpha - 1.0) < 1e-11 else ("fractional_subunit" if alpha < 1.0 else "fractional_superunit")),
        "terms": terms,
        "equation": selected.get("equation", ""),
        "source": "pareto_discovery",
        "created_unix_time": time.time(),
        "pareto_selected": selected,
    }
    if metadata:
        out["metadata"] = dict(metadata)
    return out


def save_selected_fde(
    model_or_dict: Any,
    path: str | Path,
    *,
    dataset_name: str | None = None,
    config: Any | None = None,
    side: str | None = None,
    sides: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a Pareto-selected governing FDE to a JSON file.

    Returns the resolved ``Path`` to the written file.
    """
    path = Path(path)
    payload = selected_model_to_fde_dict(
        model_or_dict,
        dataset_name=dataset_name,
        config=config,
        side=side,
        sides=sides,
        metadata=metadata,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_selected_fde_dict(path: str | Path) -> dict[str, Any]:
    """Load a selected-FDE JSON file as a dictionary."""
    path = Path(path)
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    if payload.get("format") not in {FORMAT_VERSION, None}:
        raise ValueError(f"Unsupported selected FDE format: {payload.get('format')!r}")
    if "alpha" not in payload or "terms" not in payload:
        raise ValueError("Selected FDE JSON must contain top-level 'alpha' and 'terms'")
    return payload
