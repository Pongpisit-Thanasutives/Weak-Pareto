"""Pytest path setup for the integrated Pareto + PINN project."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "fractional_pinn"):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
