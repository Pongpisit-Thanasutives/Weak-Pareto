"""Regression checks for the archived manuscript-figure inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reproduce" / "reference_results" / "branch_aware_campaign" / "main"


def test_burgers_full_curve_matches_printed_table_rows() -> None:
    curve = pd.read_csv(ARCHIVE / "figure_burgers_curve.csv")
    table = pd.read_csv(ARCHIVE / "table_burgers.csv")

    assert curve["noise_percent"].tolist() == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    assert np.all(np.diff(curve["true_residual"].to_numpy()) > 0)
    assert np.all(curve["margin_ratio"].to_numpy() > 1.0)
    assert np.all(curve["e_xi_max"].to_numpy() < 0.02)
    assert set(curve["closest_competing_terms"]) == {"u u_x + u_xx"}

    printed = curve[curve["noise_percent"].isin([0.0, 10.0, 25.0])].reset_index(drop=True)
    table = table.reset_index(drop=True)
    np.testing.assert_allclose(printed["true_residual"], table["true_residual"])
    np.testing.assert_allclose(
        printed["closest_competing_residual"], table["closest_competing_residual"]
    )
    np.testing.assert_allclose(printed["margin_ratio"], table["margin_ratio"])
    np.testing.assert_allclose(printed["e_xi_max"], table["e_xi_max"])


def test_archived_renderer_preserves_intended_scales_and_point_counts(monkeypatch, tmp_path) -> None:
    from reproduce import render_archived_paper_figures as renderer

    captured = {}

    def capture(fig, _outdir, stem):
        captured[stem] = fig

    monkeypatch.setattr(renderer, "save", capture)

    renderer.pareto(tmp_path)
    pareto_ax = captured["Fig3"].axes[0]
    assert pareto_ax.get_yscale() == "linear"
    assert pareto_ax.lines[0].get_xdata().tolist() == [1, 2, 3]

    renderer.burgers(tmp_path)
    residual_ax = captured["Fig4"].axes[1]
    assert residual_ax.get_yscale() == "log"
    assert residual_ax.lines[0].get_xdata().tolist() == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    assert residual_ax.lines[1].get_xdata().tolist() == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]


def test_robustness_renderer_reads_the_archived_table(monkeypatch, tmp_path) -> None:
    from reproduce import render_archived_paper_figures as renderer

    captured = {}

    def capture(fig, _outdir, stem):
        captured[stem] = fig

    monkeypatch.setattr(renderer, "save", capture)
    renderer.robustness(tmp_path)
    figure = captured["Fig2"]
    archive = pd.read_csv(ARCHIVE / "table_robustness.csv")
    for ax, benchmark in zip(figure.axes[:2], ["FADE", "Fractional Burgers"]):
        rows = archive[archive["benchmark"] == benchmark].sort_values("noise_percent")
        np.testing.assert_allclose(ax.lines[0].get_xdata(), rows["noise_percent"])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), rows["weak_e_xi_max"])
        expected_strong = sorted(float(v) for v in rows["strong_e_xi_max"] if np.isfinite(v))
        plotted_strong = sorted(float(line.get_ydata()[0]) for line in ax.lines[1:])
        np.testing.assert_allclose(plotted_strong, expected_strong)


def test_pareto_renderer_reads_the_archived_table(monkeypatch, tmp_path) -> None:
    from reproduce import render_archived_paper_figures as renderer

    captured = {}

    def capture(fig, _outdir, stem):
        captured[stem] = fig

    monkeypatch.setattr(renderer, "save", capture)
    renderer.pareto(tmp_path)
    ax = captured["Fig3"].axes[0]
    archive = pd.read_csv(ARCHIVE / "table_progress.csv").sort_values("c")
    np.testing.assert_allclose(ax.lines[0].get_xdata(), archive["c"])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), archive["val_rel_mse"])
    selected = archive.loc[archive["selected"].astype(str).str.lower().isin(["true", "1"]), "c"]
    assert selected.tolist() == [2]


def test_graphical_overview_inputs_are_archived_results() -> None:
    from reproduce.graphical_overview_inputs import load_graphical_overview_inputs

    values = load_graphical_overview_inputs(twod_seed=1)
    assert values.support_sizes == (1, 2, 3)
    np.testing.assert_allclose(
        values.validation_errors,
        [5.494627814165509e-3, 1.1575309919953384e-4, 9.395185048249492e-5],
    )
    assert values.selected_support == 2
    assert (values.weak_support_recovery, values.strong_support_recovery, values.recovery_denominator) == (5, 0, 5)
    assert np.isclose(values.fade_alpha, 0.8038)
    np.testing.assert_allclose(values.fade_coefficients, [-1.11, 0.63])
    np.testing.assert_allclose(values.fade_orders, [1.03, 1.61])
    assert values.twod_seed == 1
    assert np.isclose(values.twod_alpha, 0.8481479956810346)
    np.testing.assert_allclose(values.twod_coefficients, [-0.6094133750553472, 0.3081305585107796, 0.20245039639813198])
    np.testing.assert_allclose(values.twod_orders, [1.0075731539565957, 1.6972801033828007, 1.3994630129635575])
