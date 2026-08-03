"""Operator-mode awareness of recovery scoring and order polishing.

These tests pin down the discrete operator-mode contract that the manuscript
states (Sec. 5.1) and that the previous review round required:

* a reaction (identity, beta=0) term reported as a low-order Riesz derivative
  must NOT count as operator-structure recovery, and vice versa;
* the exact-order polishing must preserve the selected mode -- an identity
  term stays exactly identity, and a derivative term is never pushed below the
  first strictly-positive grid node.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reproduce._repro_common import (  # noqa: E402
    IDENTITY_ORDER_TOL,
    coefficient_truth,
    matched_errors,
    _config_for,
)

# The space-fractional reaction--diffusion benchmark has a genuine identity
# (reaction) term, so it is the right vehicle for the mode-aware tests.
_RD = "synthetic_space_fractional_RD"


def _rd_truth():
    _, cfg, truth = _config_for(_RD, 0.0, 0, False, weak=True)
    return cfg, truth


def _selected(truth, betas):
    """A ``selected`` dict matching the truth's powers/alpha/coefficients but
    with the fractional orders set to ``betas`` (so we can probe scoring)."""
    terms = list(truth.expected_terms)
    coefs = list(coefficient_truth(_RD))
    return {
        "p_tuple": [int(p) for p, _ in terms],
        "beta_tuple": [float(b) for b in betas],
        "coefficients": [float(c) for c in coefs],
        "alpha": float(truth.expected_alpha),
    }


def test_identity_not_recovered_by_low_order_riesz():
    """beta*=0 with a selected order of 0.10 must NOT be an operator hit."""
    _, truth = _rd_truth()
    riesz_order = float(truth.expected_terms[1][1])
    sel = _selected(truth, betas=[0.10, riesz_order])  # identity -> low-order Riesz
    m = matched_errors(_RD, sel, truth)
    assert m["support_power_ok"] is True          # cardinality/powers still match
    assert m["operator_structure_ok"] is False    # but the identity is not recovered


def test_identity_recovered_by_identity():
    """The same configuration with an exact identity selection IS a recovery."""
    _, truth = _rd_truth()
    riesz_order = float(truth.expected_terms[1][1])
    sel = _selected(truth, betas=[0.0, riesz_order])
    m = matched_errors(_RD, sel, truth)
    assert m["operator_structure_ok"] is True


def test_derivative_truth_not_recovered_by_identity():
    """A derivative truth (beta*>0) reported as identity must NOT be a hit."""
    _, truth = _rd_truth()
    sel = _selected(truth, betas=[0.0, 0.0])  # Riesz term collapsed to identity
    m = matched_errors(_RD, sel, truth)
    assert m["operator_structure_ok"] is False


def _first_positive_node(cfg):
    bg = np.asarray(cfg.beta_grid, dtype=float)
    pos = bg[bg > 0.0]
    return float(pos.min())


# The clean space-fractional RD equation contains an identity (reaction) term
# and a Riesz derivative, so a clean discovery exercises the full
# build -> precompute -> exact-order polish path end to end.  Cache the run so
# both polishing contract tests share one discovery.
_clean_rd_cache: dict = {}


def _clean_rd_selected():
    if "sel" not in _clean_rd_cache:
        from reproduce._repro_common import run_weak
        cfg, _ = _rd_truth()
        summary, _ = run_weak(_RD, 0.0, 0, fast=True)
        _clean_rd_cache["sel"] = summary["selected"]
        _clean_rd_cache["cfg"] = cfg
    return _clean_rd_cache["sel"], _clean_rd_cache["cfg"]


def test_identity_preserved_through_polishing():
    """After polishing, the clean RD reaction term is exactly identity (beta=0)."""
    sel, _ = _clean_rd_selected()
    betas = [abs(float(b)) for b in sel["beta_tuple"]]
    idents = [b for b in betas if b <= IDENTITY_ORDER_TOL]
    # exactly one identity term survives, held at exactly zero (not a low Riesz)
    assert len(idents) == 1, sel["beta_tuple"]


def test_derivative_not_polished_below_first_node():
    """No polished derivative order sits below the first positive grid node."""
    sel, cfg = _clean_rd_selected()
    b_pos_min = _first_positive_node(cfg)
    for b in sel["beta_tuple"]:
        assert abs(b) <= IDENTITY_ORDER_TOL or abs(b) >= b_pos_min - 1e-9, sel["beta_tuple"]
