"""Acceptance rules for lossy verification in speculative decoding.

The paper unifies the prior zoo of methods into two families. This subpackage
exposes both as importable functions so they can be cross-validated against the
patched transformers implementation and reused by other projects (e.g.
SpecForge, SGLang).

Families
--------
- :mod:`fast_hsd.core.collaborative_verification` — Lenience, CoS.
  Interpolates the draft and target distributions.
- :mod:`fast_hsd.core.truncation_verification` — SpecCascade (min-p),
  Medusa typical-acceptance (eta). Accepts a draft token iff it lies in
  the allowed set induced by a truncation sampler.

Both modules share the unified :func:`acceptance.accept` entry point, which
dispatches to the right rule based on a config dict.
"""

from fast_hsd.core import (  # noqa: F401
    acceptance,
    collaborative_verification,
    truncation_verification,
)

__all__ = ["acceptance", "collaborative_verification", "truncation_verification"]
