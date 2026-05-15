"""Runtime monkey-patcher for Fast-HSD.

The four files under ``transformers/`` in this repository are modified copies of
``transformers==4.46.3``'s ``generation/{utils,candidate_generator,logits_process}.py``
and ``cache_utils.py``. Historically, users were instructed to manually copy these
files into their installed ``transformers`` package. That workflow is fragile
(it silently breaks on version upgrades, fails on shared environments, and
leaves no record in ``pip freeze``), so this module replaces it with a single
runtime call::

    import fast_hsd_install
    fast_hsd_install.install()
    # ...then proceed with `import transformers` as usual.

The installer overrides the offending symbols on the already-imported
``transformers`` modules. It must be invoked **before** the first call to
``model.generate(...)``; otherwise the unpatched implementation has already been
bound to the model's method resolution order.

If you prefer the original copy-files-into-site-packages workflow, it still
works — see the README "Manual install (legacy)" section. The two paths are
mutually exclusive; do not mix them.
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from pathlib import Path

REQUIRED_TRANSFORMERS_VERSION = "4.46.3"

# Resolve the directory that this file lives in so we can import the vendored
# copies regardless of the user's current working directory.
_REPO_ROOT = Path(__file__).resolve().parent
_VENDORED_DIR = _REPO_ROOT / "transformers"


def _check_transformers_version() -> None:
    import transformers

    actual = getattr(transformers, "__version__", "unknown")
    if actual != REQUIRED_TRANSFORMERS_VERSION:
        warnings.warn(
            f"Fast-HSD patches were prepared against transformers=="
            f"{REQUIRED_TRANSFORMERS_VERSION}, but the active environment has "
            f"transformers=={actual}. The patches may apply cleanly, may "
            f"silently produce wrong results, or may raise at call time. "
            f"Pin transformers=={REQUIRED_TRANSFORMERS_VERSION} for "
            f"reproducible results.",
            RuntimeWarning,
            stacklevel=2,
        )


def _load_vendored_module(relative_path: str, qualified_name: str):
    """Import a single vendored file as a module under ``qualified_name``.

    We use ``importlib.util.spec_from_file_location`` instead of placing the
    vendored ``transformers/`` directory on ``sys.path``, because doing so
    would shadow the real ``transformers`` package entirely.
    """
    import importlib.util

    src = _VENDORED_DIR / relative_path
    if not src.is_file():
        raise FileNotFoundError(
            f"Fast-HSD vendored file not found: {src}. "
            f"Did you check out the repository in full?"
        )

    spec = importlib.util.spec_from_file_location(qualified_name, src)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {src}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


_INSTALLED = False


def install(verbose: bool = False) -> None:
    """Install Fast-HSD's lossy-verification patches into transformers.

    Idempotent: calling ``install()`` more than once is a no-op after the
    first successful call.

    Parameters
    ----------
    verbose
        If True, prints a summary of which symbols were rebound.
    """
    global _INSTALLED
    if _INSTALLED:
        if verbose:
            print("[fast-hsd] patches already installed; skipping.")
        return

    _check_transformers_version()

    # Order matters: ``utils`` depends on ``candidate_generator`` and
    # ``logits_process``, so patch those first.
    import transformers.generation.candidate_generator as _cg
    import transformers.generation.logits_process as _lp
    import transformers.generation.utils as _gen_utils
    import transformers.cache_utils as _cache_utils

    patched_cg = _load_vendored_module(
        "generation/candidate_generator.py", "fast_hsd._patched_candidate_generator"
    )
    patched_lp = _load_vendored_module(
        "generation/logits_process.py", "fast_hsd._patched_logits_process"
    )
    patched_gen = _load_vendored_module(
        "generation/utils.py", "fast_hsd._patched_generation_utils"
    )
    patched_cache = _load_vendored_module(
        "cache_utils.py", "fast_hsd._patched_cache_utils"
    )

    # Rebind public symbols from the patched module onto the live transformers
    # module. We walk the patched module's ``__dict__`` rather than hard-coding
    # symbol names so that the patch covers every helper the user might import
    # transitively (e.g. ``_speculative_sampling``).
    rebound = []
    for live, patched in (
        (_cg, patched_cg),
        (_lp, patched_lp),
        (_gen_utils, patched_gen),
        (_cache_utils, patched_cache),
    ):
        for name, value in vars(patched).items():
            if name.startswith("__"):
                continue
            if hasattr(live, name) and getattr(live, name) is value:
                continue  # nothing to do — same identity
            setattr(live, name, value)
            rebound.append(f"{live.__name__}.{name}")

    _INSTALLED = True
    if verbose:
        print(f"[fast-hsd] installed {len(rebound)} patched symbols.")


def is_installed() -> bool:
    """Return True iff :func:`install` has already been called this process."""
    return _INSTALLED


if __name__ == "__main__":
    install(verbose=True)
