#!/usr/bin/env python3
"""Check module and public-API docstring coverage without external tools."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_API_MODULES = {
    "baselines.py",
    "dataset_configs.py",
    "denoising.py",
    "fpde_datasets.py",
    "fpde_derivatives.py",
    "fractional_weak_form.py",
    "pareto_fde_discovery.py",
    "regfracdiff.py",
    "selected_fde_io.py",
    "temporal_modes.py",
    "weak_pareto_fde_discovery.py",
    "two_dimensional/weak_pareto_2d.py",
    "real_data/frozen_soil_creep_weak.py",
}


def main() -> None:
    """Run the static documentation audit and exit nonzero on missing docs."""
    failures: list[str] = []
    python_files = sorted(
        p for p in ROOT.rglob("*.py")
        if "tests" not in p.parts and "__pycache__" not in p.parts
    )
    for path in python_files:
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not ast.get_docstring(tree):
            failures.append(f"missing module docstring: {rel}")
        if rel in PUBLIC_API_MODULES:
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name.startswith("_"):
                        continue
                    if not ast.get_docstring(node):
                        failures.append(f"missing public docstring: {rel}:{node.lineno} {node.name}")
    if failures:
        print("Documentation audit failed:")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(1)
    print(f"Documentation audit passed for {len(python_files)} non-test Python modules.")


if __name__ == "__main__":
    main()
