from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reproduce" / "reference_results" / "operator_tolerance_sensitivity"


def test_operator_tolerance_sensitivity_reference_counts() -> None:
    per_run = pd.read_csv(REFERENCE / "per_run.csv")
    summary = pd.read_csv(REFERENCE / "summary.csv")

    assert len(per_run) == 120
    assert per_run.groupby("method").size().to_dict() == {
        "Strong Pareto": 60,
        "Weak-Pareto": 60,
    }

    expected = {
        (0.125, "Weak-Pareto"): (45, 60),
        (0.150, "Weak-Pareto"): (47, 60),
        (0.175, "Weak-Pareto"): (47, 60),
        (0.125, "Strong Pareto"): (0, 60),
        (0.150, "Strong Pareto"): (0, 60),
        (0.175, "Strong Pareto"): (0, 60),
    }
    observed = {
        (round(float(row.absolute_order_tolerance), 3), str(row.method)): (
            int(row.recoveries),
            int(row.runs),
        )
        for row in summary.itertuples(index=False)
    }
    for key, value in expected.items():
        assert observed[key] == value
