"""The test_counts override must control the number of weak rows exactly."""
import dataclasses
import warnings

import numpy as np

from dataset_configs import load_benchmark
from weak_pareto_fde_discovery import build_weak_candidate_library

warnings.filterwarnings("ignore")


def test_test_counts_override_controls_K():
    data, cfg, _ = load_benchmark("paper_FADE_tsfade_fft", profile="paper", seed=0)
    # Tiny order grids keep precompute trivial; the override under test is
    # independent of the grids.
    cfg = dataclasses.replace(
        cfg, noise_percent=0.0, progress=False,
        alpha_grid=np.array([0.6, 0.8, 1.0]), beta_grid=np.array([1.0, 1.7, 2.0]),
    )
    bank = build_weak_candidate_library(
        data, cfg, test_budget="paper", test_counts=(5, 7), verbose=False
    )
    assert bank.time_tests.shape[0] == 5
    assert bank.space_tests.shape[0] == 7
    assert bank.time_tests.shape[0] * bank.space_tests.shape[0] == 35
