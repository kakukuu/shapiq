"""Resolve the same local library and shared helpers as the notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_support"))
from example_paths import configure_imports

configure_imports(ROOT)
collect_ignore = ["legacy"]
