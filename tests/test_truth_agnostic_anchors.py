"""DE anchors must be truth-agnostic: derived only from the search interval and
the a-priori integers 1 and 2, never hard-coded benchmark fractional orders."""
import numpy as np

from dataset_configs import load_benchmark
from weak_pareto_fde_discovery import build_weak_candidate_library
from pareto_fde_discovery import ParetoFDEOptimizer


def _optimizer():
    data, cfg, _ = load_benchmark("paper_FADE_tsfade_fft", profile="paper", seed=0)
    bank = build_weak_candidate_library(data, cfg, test_budget="smoke", verbose=False)
    idx = np.arange(bank.n_points, dtype=np.int64)
    return ParetoFDEOptimizer(bank, idx, idx, cfg), bank


def test_anchor_values_are_uniform_plus_integers():
    opt, bank = _optimizer()
    grid = np.linspace(0.3, 3.1, 40)  # arbitrary interval, unrelated to any truth
    vals = opt._anchor_values(grid, n_interior=6)
    lo, hi = float(grid[0]), float(grid[-1])
    expected = sorted(set(
        [round(v, 9) for v in np.linspace(lo, hi, 6)]
        + [v for v in (1.0, 2.0) if lo <= v <= hi]
    ))
    assert [round(v, 9) for v in vals] == expected


def test_no_hardcoded_fractional_anchor_constants():
    # The notorious truth values must not appear as literal anchors in the source.
    import pareto_fde_discovery as mod
    src = open(mod.__file__).read()
    # isolate the anchor method bodies
    seg = src[src.index("def _anchor_values"):src.index("def optimize_fixed_pattern")]
    for truth in ("0.55", "2.8", "1.7", "0.82"):
        assert truth not in seg, f"hard-coded fractional anchor {truth} present"


def test_anchors_track_the_interval_not_the_truth():
    opt, bank = _optimizer()
    # Shifting the interval shifts the anchors (they are functions of the interval).
    a = opt._anchor_values(np.linspace(0.0, 1.0, 20), n_interior=5)
    b = opt._anchor_values(np.linspace(2.0, 3.0, 20), n_interior=5)
    assert max(a) <= 1.0 + 1e-9 and min(b) >= 2.0 - 1e-9
