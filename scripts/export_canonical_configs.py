#!/usr/bin/env python3
"""Export the one canonical candidate-search config for each packaged dataset.

This is a transparency helper.  It writes the publication search-space
fingerprint (including truth-agnostic uniform order grids) and truth-scoring
metadata used by the benchmark runner.  The optimizer does not use
the truth metadata; it is included here so a reader can audit the evaluation
criteria before running experiments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_configs import (  # noqa: E402
    available_dataset_names,
    config_search_space_fingerprint,
    dataset_candidate_config,
    dataset_config_philosophy,
    dataset_truth_spec,
    with_uniform_order_grids,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "configs" / "canonical_dataset_configs.json")
    args = ap.parse_args()
    payload = {
        "config_philosophy": dataset_config_philosophy(),
        "datasets": {},
    }
    for name in available_dataset_names():
        base_cfg = dataset_candidate_config(name)
        cfg = with_uniform_order_grids(base_cfg)
        truth = dataset_truth_spec(name)
        payload["datasets"][name] = {
            "search_space": config_search_space_fingerprint(cfg),
            "truth_spec_for_evaluation_only": truth.to_dict(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
