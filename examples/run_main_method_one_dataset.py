#!/usr/bin/env python3
"""Minimal example: run the proposed weak-library Pareto-DE method."""
from __future__ import annotations

from pathlib import Path

from dataset_configs import load_dataset_with_config
from weak_pareto_fde_discovery import run_weak_pareto_discovery

ROOT = Path(__file__).resolve().parents[1]

data, config, truth = load_dataset_with_config(
    "synthetic_two_fractional_rhs",
    data_dir=ROOT / "data",
    profile="notebook",
    noise_percent=0.1,
    seed=0,
    maxiter=0,
    popsize=2,
    progress=False,
)

summary = run_weak_pareto_discovery(
    data,
    config,
    output_dir=ROOT / "results" / "example_weak_pareto",
    verbose=True,
    test_budget="smoke",
)

print("Truth:", data.truth)
print("Selected:", summary["selected"]["equation"])
