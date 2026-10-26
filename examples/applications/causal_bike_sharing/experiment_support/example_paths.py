"""Resolve the example and an optional source checkout without machine-specific paths."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def checkout_source(root=EXAMPLE_ROOT):
    """Find the nearest ancestor containing the shapiq source package."""
    root = Path(root).resolve()
    for parent in (root, *root.parents):
        source = parent / "src"
        if (source / "shapiq" / "__init__.py").is_file():
            return source
    return None


def configure_imports(root=EXAMPLE_ROOT):
    """Prefer this checkout when available; otherwise use installed shapiq."""
    source = checkout_source(root)
    if source is not None and str(source) not in sys.path:
        sys.path.insert(0, str(source))


def library_directory(root=EXAMPLE_ROOT):
    """Return library source for reproducibility checks in either installation mode."""
    source = checkout_source(root)
    if source is not None:
        return source / "shapiq"
    import shapiq

    return Path(shapiq.__file__).resolve().parent
