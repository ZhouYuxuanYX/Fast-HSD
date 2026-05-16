#!/usr/bin/env python
"""Standalone CLI wrapper around :mod:`fast_hsd.patches.sync`.

Adds the repo root to ``sys.path`` first so this script runs even without
``pip install -e .`` — handy on shared-storage envs where pip can't write.

Examples
--------

    # Default: target the env this python belongs to.
    python scripts/sync_transformers_patches.py

    # Target a specific conda env.
    python scripts/sync_transformers_patches.py --env /projects/.../envs/fsd

    # Inspect without changing anything.
    python scripts/sync_transformers_patches.py --check

    # Undo (restore the .fasthsd-orig backups).
    python scripts/sync_transformers_patches.py --restore
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is importable even when fast-hsd is not pip-installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fast_hsd.patches.sync import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
