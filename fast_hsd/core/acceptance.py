"""Unified dispatcher for lossy-verification acceptance rules.

Given a method name and a config dict, return either the acceptance probability
or a boolean accept decision. This is the entry point external projects (e.g.
SpecForge, SGLang) can call without taking a dependency on the full Fast-HSD
benchmark harness.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from fast_hsd.core import collaborative_verification as cv
from fast_hsd.core import truncation_verification as tv

__all__ = ["accept", "accept_prob"]


def accept_prob(
    p: torch.Tensor,
    q: torch.Tensor,
    x: int,
    method: str,
    params: Dict[str, Any],
) -> float:
    """Return the acceptance probability for token ``x`` under ``method``.

    Parameters
    ----------
    p, q
        Target and draft next-token distributions.
    x
        The drafted token index.
    method
        One of ``"lossless"``, ``"lenience"``, ``"cos"``.
    params
        Method-specific hyperparameters. For example, ``{"lenience": 0.4}``
        or ``{"cos_lambda": 0.6}``.
    """
    if method == "lossless":
        return float(min(1.0, p[x].item() / q[x].item()))
    if method == "lenience":
        return cv.lenience_accept_prob(p, q, x, lenience=params["lenience"])
    if method == "cos":
        return cv.cos_accept_prob(p, q, x, lam=params["cos_lambda"])
    raise ValueError(
        f"accept_prob: method must be one of "
        f"{{'lossless', 'lenience', 'cos'}}, got {method!r}"
    )


def accept(
    p: torch.Tensor,
    q: torch.Tensor,
    x: int,
    method: str,
    params: Dict[str, Any],
) -> bool:
    """Return a boolean accept decision for token ``x`` under ``method``.

    For probability-based rules (lossless, lenience, CoS) this samples a
    uniform random variable; for set-membership rules (SpecCascade, Medusa)
    it queries the allowed set directly.
    """
    if method in {"speccascade", "min_p"}:
        return tv.speccascade_accepts(p, x, p_base=params["p_base"])
    if method in {"medusa", "eta"}:
        return tv.medusa_typical_accepts(
            p, x, epsilon=params["epsilon"], delta=params.get("delta", 1.0)
        )
    # Probability-based families.
    prob = accept_prob(p, q, x, method, params)
    return bool(torch.rand(()).item() < prob)
