"""Symlink Fast-HSD's vendored transformers patches into a conda env.

Replaces the manual `cp transformers/... <site-packages>/transformers/...`
dance from the README's "Manual install (legacy)" section. Symlinks rather
than copies so subsequent edits to the vendored files take effect in every
process — including downstream consumers (SGLang, Ray workers, SpecForge
training) that don't go through `fast_hsd.patches.install()`.

Usage::

    # Default: target the env that the running python belongs to.
    python scripts/sync_transformers_patches.py

    # Explicit target.
    python scripts/sync_transformers_patches.py --env /path/to/conda/envs/fsd

    # Inspect without changing anything.
    python scripts/sync_transformers_patches.py --check

    # Undo (replace symlinks with the .fasthsd-orig backups).
    python scripts/sync_transformers_patches.py --restore

The CLI calls ``sync()`` automatically at startup; running this script by
hand is only needed for non-CLI consumers or to verify state.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("fast_hsd.sync")

REQUIRED_TRANSFORMERS_VERSION = "4.46.3"
BACKUP_SUFFIX = ".fasthsd-orig"

# (vendored relative path inside fast-hsd-private/transformers/,
#  destination relative path inside site-packages/transformers/)
PATCH_FILES: Tuple[Tuple[str, str], ...] = (
    ("cache_utils.py", "cache_utils.py"),
    ("generation/candidate_generator.py", "generation/candidate_generator.py"),
    ("generation/logits_process.py", "generation/logits_process.py"),
    ("generation/utils.py", "generation/utils.py"),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDORED_DIR = _REPO_ROOT / "transformers"


def _find_transformers_dir(env_prefix: Path) -> Path:
    """Return ``<env_prefix>/lib/python*/site-packages/transformers``.

    Globs across python minor versions because conda envs vary.
    """
    candidates = sorted(env_prefix.glob("lib/python*/site-packages/transformers"))
    if not candidates:
        raise FileNotFoundError(
            f"No site-packages/transformers under {env_prefix}. "
            f"Is transformers installed in this env?"
        )
    if len(candidates) > 1:
        logger.warning(
            "Multiple transformers installs under %s; using %s",
            env_prefix,
            candidates[0],
        )
    return candidates[0]


def _check_version(transformers_dir: Path) -> None:
    """Read the installed transformers' __version__ and compare."""
    init = transformers_dir / "__init__.py"
    actual = None
    for line in init.read_text().splitlines():
        if line.startswith("__version__"):
            actual = line.split("=", 1)[1].strip().strip("'\"")
            break
    if actual != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Refusing to sync: transformers at {transformers_dir} is "
            f"version {actual!r}, but the vendored patches target "
            f"{REQUIRED_TRANSFORMERS_VERSION!r}. Reinstall transformers "
            f"=={REQUIRED_TRANSFORMERS_VERSION} first."
        )


def _resolve_env_prefix(env: Optional[str]) -> Path:
    if env:
        return Path(env).expanduser().resolve()
    return Path(sys.prefix).resolve()


def _link_one(src: Path, dst: Path, *, dry_run: bool) -> str:
    """Ensure ``dst`` is a symlink to ``src``. Returns one of:
    'already-linked' | 'linked' | 'would-link'.
    """
    if dst.is_symlink():
        try:
            current = dst.resolve(strict=False)
        except OSError:
            current = None
        if current == src.resolve():
            return "already-linked"

    if dry_run:
        return "would-link"

    # Back up the original (non-symlink) file once, so --restore can undo us.
    backup = dst.with_name(dst.name + BACKUP_SUFFIX)
    if dst.exists() and not dst.is_symlink() and not backup.exists():
        dst.rename(backup)
    elif dst.is_symlink() or dst.exists():
        dst.unlink()

    dst.symlink_to(src)
    return "linked"


def sync(env: Optional[str] = None, *, dry_run: bool = False, verbose: bool = True) -> int:
    """Symlink the four vendored files into the target env's transformers.

    Parameters
    ----------
    env
        Conda env prefix (e.g. ``/projects/.../envs/fsd``). If ``None``,
        uses ``sys.prefix`` — i.e. the env of the python running this code.
    dry_run
        If True, report what would happen without modifying anything.
    verbose
        Log each action.

    Returns the number of files newly linked (0 if already in sync).
    """
    env_prefix = _resolve_env_prefix(env)
    transformers_dir = _find_transformers_dir(env_prefix)
    _check_version(transformers_dir)

    n_linked = 0
    for rel_src, rel_dst in PATCH_FILES:
        src = _VENDORED_DIR / rel_src
        dst = transformers_dir / rel_dst
        if not src.is_file():
            raise FileNotFoundError(f"Vendored file missing: {src}")
        result = _link_one(src, dst, dry_run=dry_run)
        if result in ("linked", "would-link"):
            n_linked += 1
        if verbose:
            logger.info("[%s] %s -> %s", result, dst, src)

    if verbose:
        verb = "would link" if dry_run else "linked"
        logger.info(
            "fast-hsd patches: %s %d/%d files in %s",
            verb,
            n_linked,
            len(PATCH_FILES),
            transformers_dir,
        )
    return n_linked


def check(env: Optional[str] = None) -> List[Tuple[str, str]]:
    """Return the status of each patch file as a list of (path, status) tuples.

    Status is one of: 'symlinked' | 'plain-file' | 'missing'.
    """
    env_prefix = _resolve_env_prefix(env)
    transformers_dir = _find_transformers_dir(env_prefix)

    out = []
    for _, rel_dst in PATCH_FILES:
        dst = transformers_dir / rel_dst
        if not dst.exists():
            out.append((str(dst), "missing"))
        elif dst.is_symlink():
            target = os.readlink(dst)
            out.append((str(dst), f"symlinked -> {target}"))
        else:
            out.append((str(dst), "plain-file"))
    return out


def restore(env: Optional[str] = None, *, verbose: bool = True) -> int:
    """Replace symlinks with their ``.fasthsd-orig`` backups (if present).

    Returns the number of files restored.
    """
    env_prefix = _resolve_env_prefix(env)
    transformers_dir = _find_transformers_dir(env_prefix)

    n_restored = 0
    for _, rel_dst in PATCH_FILES:
        dst = transformers_dir / rel_dst
        backup = dst.with_name(dst.name + BACKUP_SUFFIX)
        if dst.is_symlink() and backup.exists():
            dst.unlink()
            backup.rename(dst)
            n_restored += 1
            if verbose:
                logger.info("restored %s from backup", dst)
        elif verbose:
            logger.info("skip %s (symlink=%s, backup=%s)", dst, dst.is_symlink(), backup.exists())
    return n_restored


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--env",
        default=None,
        help="Conda env prefix to target (default: sys.prefix of this python).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Report status; make no changes.")
    mode.add_argument("--restore", action="store_true", help="Undo: replace symlinks with .fasthsd-orig backups.")
    mode.add_argument("--dry-run", action="store_true", help="Show what would change.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    args = _build_parser().parse_args(argv)

    if args.check:
        for path, status in check(args.env):
            print(f"{status:50s}  {path}")
        return 0
    if args.restore:
        restore(args.env)
        return 0
    sync(args.env, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
