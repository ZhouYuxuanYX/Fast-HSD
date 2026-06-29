"""Thin shim over the vendored EAGLE fork in the repo root.

The actual EAGLE source still lives in ``EAGLE/eagle/`` (it's pulled in here
unchanged so that minor patches stay close to upstream). This subpackage
re-exports the bits we need from the unified CLI without forcing callers to
import the old script-style entry points.
"""

from fast_hsd.eagle import runner  # noqa: F401

__all__ = ["runner"]
