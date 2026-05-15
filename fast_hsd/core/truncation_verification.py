"""Truncation-based verification: SpecCascade (min-p) and Medusa (eta).

Truncation-based verification accepts a draft token iff it lies in the *allowed
set* induced by a truncation sampler over the target distribution. Tokens
outside the allowed set are rejected, with no residual resampling. See paper
§3.2.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "min_p_allowed_set",
    "eta_allowed_set",
    "speccascade_accepts",
    "medusa_typical_accepts",
    "truncation_yield_distribution",
]


def min_p_allowed_set(p: torch.Tensor, p_base: float) -> torch.Tensor:
    """Boolean mask for the min-p allowed set (paper Eq. 3).

    A token ``v`` is in the allowed set iff :math:`p(v) \\ge p_{\\text{base}}
    \\cdot \\max_v p(v)`.
    """
    if not 0.0 <= p_base <= 1.0:
        raise ValueError("p_base must be in [0, 1]")
    return p >= (p_base * p.max())


def eta_allowed_set(
    p: torch.Tensor, epsilon: float, delta: float = 1.0
) -> torch.Tensor:
    """Boolean mask for the eta-sampling allowed set (paper Eq. 4).

    A token ``v`` is in the allowed set iff :math:`p(v) \\ge
    \\min(\\varepsilon,\\, \\delta\\,e^{-H(p)})`.
    """
    if epsilon <= 0.0 or delta <= 0.0:
        raise ValueError("epsilon and delta must be positive")
    # Shannon entropy of p (in nats).
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum())
    threshold = min(epsilon, delta * math.exp(-entropy))
    return p >= threshold


def speccascade_accepts(p: torch.Tensor, x: int, p_base: float) -> bool:
    """Acceptance rule for SpecCascade (paper §3.2)."""
    mask = min_p_allowed_set(p, p_base)
    return bool(mask[x].item())


def medusa_typical_accepts(
    p: torch.Tensor, x: int, epsilon: float, delta: float = 1.0
) -> bool:
    """Acceptance rule for Medusa typical-acceptance (paper §3.2)."""
    mask = eta_allowed_set(p, epsilon, delta)
    return bool(mask[x].item())


def truncation_yield_distribution(
    p: torch.Tensor,
    q: torch.Tensor,
    allowed_mask: torch.Tensor,
) -> torch.Tensor:
    """Induced generation distribution under truncation-based verification
    (paper Eq. 9): the renormalised *draft* distribution restricted to the
    allowed set.

    .. math::
        P(\\text{generate}\\,x) = \\begin{cases}
            q(x) / Z_\\Theta, & x \\in \\mathcal A_\\Theta, \\\\
            0,                 & x \\notin \\mathcal A_\\Theta,
        \\end{cases}
        \\quad Z_\\Theta = \\sum_{v \\in \\mathcal A_\\Theta} q(v).

    This is the cornerstone observation of the paper's pitfall analysis: the
    yield is :math:`q`-shaped, not :math:`p`-shaped, even though the
    *acceptance gate* is defined by :math:`p`.
    """
    masked = q * allowed_mask.to(q.dtype)
    z = masked.sum()
    if z.item() == 0.0:
        return torch.zeros_like(q)
    return masked / z
