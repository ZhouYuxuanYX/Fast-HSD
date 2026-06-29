"""Runtime patch installer for transformers==4.46.3.

This is the package-API wrapper around the standalone ``fast_hsd_install``
module at the repo root::

    # New idiomatic way (full branch):
    from fast_hsd.patches import install
    install()

    # Equivalent (minimal branch / any branch):
    import fast_hsd_install
    fast_hsd_install.install()

Both forms execute the same code. The duplication exists because the standalone
script needs to remain importable on ``refactor/minimal``, where the
``fast_hsd`` package itself is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

__all__ = ["install", "is_installed"]


def _load_root_installer():
    """Import the root-level ``fast_hsd_install.py`` regardless of cwd."""
    root = Path(__file__).resolve().parents[2]
    installer_path = root / "fast_hsd_install.py"
    if not installer_path.is_file():
        raise FileNotFoundError(
            f"Could not locate fast_hsd_install.py at {installer_path}. "
            f"Did you check out the repository root?"
        )
    spec = importlib.util.spec_from_file_location(
        "fast_hsd._root_installer", installer_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {installer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fast_hsd._root_installer"] = module
    spec.loader.exec_module(module)
    return module


_installer = None


def install(verbose: bool = False) -> None:
    """Install Fast-HSD's lossy-verification patches into transformers."""
    global _installer
    if _installer is None:
        _installer = _load_root_installer()
    _installer.install(verbose=verbose)


def is_installed() -> bool:
    global _installer
    if _installer is None:
        return False
    return _installer.is_installed()
