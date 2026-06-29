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
import re
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


_REL_IMPORT_RE = re.compile(r"^(\s*)from\s+(\.+)([\w.]*)\s+import\s+", re.MULTILINE)


def _rewrite_relative_imports(source: str, package: str) -> str:
    """Rewrite ``from .X import Y`` / ``from ..X import Y`` to absolute paths
    rooted at ``package``.

    The vendored files use relative imports because they originally lived
    inside the ``transformers`` package tree. When we load them as standalone
    modules under ``fast_hsd._patched_*`` names, those relative imports have
    no anchor and raise ``ValueError: attempted relative import beyond top-
    level package``. Rewriting them to absolute ``transformers.*`` form makes
    them resolve against the live (and, by load-order, already-patched)
    transformers package.
    """
    parts = package.split(".")

    def repl(m: "re.Match[str]") -> str:
        indent, dots, target = m.group(1), m.group(2), m.group(3)
        depth = len(dots)
        # depth=1 → same package; depth=2 → parent package; etc.
        if depth - 1 > len(parts):
            return m.group(0)
        anchor = parts[: len(parts) - (depth - 1)]
        absolute = ".".join(anchor + ([target] if target else []))
        return f"{indent}from {absolute} import "

    return _REL_IMPORT_RE.sub(repl, source)


def _load_vendored_module(relative_path: str, qualified_name: str):
    """Import a single vendored file as a module under ``qualified_name``.

    We use ``importlib.util.spec_from_file_location`` instead of placing the
    vendored ``transformers/`` directory on ``sys.path``, because doing so
    would shadow the real ``transformers`` package entirely. Relative imports
    in the vendored source are rewritten to absolute ``transformers.*`` form
    on the fly so they resolve correctly.
    """
    import importlib.util

    src = _VENDORED_DIR / relative_path
    if not src.is_file():
        raise FileNotFoundError(
            f"Fast-HSD vendored file not found: {src}. "
            f"Did you check out the repository in full?"
        )

    # Compute the original transformers package path for this vendored file
    # (e.g. ``generation/candidate_generator.py`` → ``transformers.generation``).
    rel_parts = relative_path.split("/")
    pkg = ".".join(["transformers"] + rel_parts[:-1])

    source = src.read_text()
    rewritten = _rewrite_relative_imports(source, pkg)

    spec = importlib.util.spec_from_loader(qualified_name, loader=None, origin=str(src))
    if spec is None:
        raise ImportError(f"Could not create import spec for {src}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(src)
    sys.modules[qualified_name] = module
    exec(compile(rewritten, str(src), "exec"), module.__dict__)
    return module


def _rebind(live, patched) -> list:
    """Copy public symbols from ``patched`` onto ``live``. Returns names rebound."""
    rebound = []
    for name, value in vars(patched).items():
        if name.startswith("__"):
            continue
        if hasattr(live, name) and getattr(live, name) is value:
            continue
        setattr(live, name, value)
        rebound.append(f"{live.__name__}.{name}")
    return rebound


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

    # Force the live transformers submodules to be imported before we patch
    # them, so the absolute imports inside our rewritten vendored source
    # resolve against real modules (and pick up our patches as we go).
    import transformers.cache_utils as _cache_utils
    import transformers.generation.logits_process as _lp
    import transformers.generation.candidate_generator as _cg
    import transformers.generation.utils as _gen_utils

    # Order matters: load and patch each module *before* loading the next,
    # so later loads pick up the already-patched dependencies via their
    # rewritten ``from transformers.X import Y`` imports.
    plan = [
        ("cache_utils.py", "fast_hsd._patched_cache_utils", _cache_utils),
        ("generation/logits_process.py", "fast_hsd._patched_logits_process", _lp),
        ("generation/candidate_generator.py", "fast_hsd._patched_candidate_generator", _cg),
        ("generation/utils.py", "fast_hsd._patched_generation_utils", _gen_utils),
    ]
    rebound: list = []
    for rel_path, qname, live in plan:
        patched = _load_vendored_module(rel_path, qname)
        rebound.extend(_rebind(live, patched))

    _INSTALLED = True
    if verbose:
        print(f"[fast-hsd] installed {len(rebound)} patched symbols.")


def is_installed() -> bool:
    """Return True iff :func:`install` has already been called this process."""
    return _INSTALLED


if __name__ == "__main__":
    install(verbose=True)
