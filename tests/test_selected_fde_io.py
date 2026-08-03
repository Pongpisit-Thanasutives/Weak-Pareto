from __future__ import annotations

from pathlib import Path

from selected_fde_io import load_selected_fde_dict, save_selected_fde


def test_selected_fde_json_roundtrip(tmp_path):
    selected = {
        "alpha": 0.82,
        "terms": [[0, 0.0], [0, 1.55]],
        "coefficients": [0.03, 0.12],
        "equation": "D_t^0.82 u = 0.03*D_x^0 u + 0.12*D_x^1.55 u",
    }
    path = save_selected_fde(selected, tmp_path / "selected_fde.json", dataset_name="synthetic_time_space_fractional_RD")
    payload = load_selected_fde_dict(Path(path))
    assert payload["format"] == "selected_governing_fde_v1"
    assert abs(payload["alpha"] - 0.82) < 1e-12
    assert len(payload["terms"]) == 2
