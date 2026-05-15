"""Collaborative verification: lenience-based relaxation and CoS.

Both methods accept a draft token with a probability that interpolates between
the draft and target distributions, in contrast with the strict ratio rule of
standard speculative decoding. See paper §3.1.
"""

from __future__ import annotations

import torch

__all__ = [
    "lenience_accept_prob",
    "cos_accept_prob",
    "lenience_yield_distribution",
    "cos_yield_distribution",
]


def lenience_accept_prob(
    p: torch.Tensor, q: torch.Tensor, x: int, lenience: float
) -> float:
    """Acceptance probability for lenience-based relaxation (paper Eq. 6).

    .. math::
        h(x) = \\min\\!\\left(1,\\,\\frac{p(x)}{\\ell\\,q(x)}\\right)

    Parameters
    ----------
    p, q
        Target and draft next-token distributions over the shared vocabulary.
    x
        The drafted token index.
    lenience
        The lenience factor :math:`\\ell \\in (0, 1]`. ``lenience=1`` recovers
        the standard lossless SD acceptance rule.

    Notes
    -----
    This is *the same formula* the patched ``transformers/generation/utils.py``
    uses inside ``_speculative_sampling`` when ``lenience<1``; the duplication
    is intentional so this function can be imported without invoking the patch.
    """
    if lenience <= 0.0:
        raise ValueError("lenience must be > 0")
    return float(min(1.0, p[x].item() / (lenience * q[x].item())))


def cos_accept_prob(
    p: torch.Tensor, q: torch.Tensor, x: int, lam: float
) -> float:
    """Acceptance probability for CoS-WE (paper §3.1, weighted-ensemble form).

    .. math::
        h(x) = \\frac{\\lambda\\,p(x) + (1-\\lambda)\\,q(x)}{q(x)}

    The acceptance probability is then clipped at 1 in the verification step.

    Parameters
    ----------
    p, q
        Target and draft next-token distributions.
    x
        The drafted token index.
    lam
        Interpolation coefficient :math:`\\lambda \\in [0, 1]`. ``lam=1``
        recovers lossless SD; ``lam=0`` always accepts (pure draft yield).
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("CoS lambda must be in [0, 1]")
    raw = (lam * p[x].item() + (1.0 - lam) * q[x].item()) / q[x].item()
    return float(min(1.0, raw))


def lenience_yield_distribution(
    p: torch.Tensor, q: torch.Tensor, lenience: float
) -> torch.Tensor:
    """Induced *generation* distribution under lenience-based relaxation
    (paper Eq. 7).

    The piecewise form depends on whether each token is in the underestimated
    (q ≤ p), moderate-overestimate (p < q ≤ p/ℓ), or severe-overestimate
    (q ≥ p/ℓ) region. This implementation is for diagnostic plots and unit
    tests; the production verification loop lives in the patched transformers.
    """
    p = p.double()
    q = q.double()
    ell = lenience

    underest = q <= p
    severe = q >= (p / ell)
    moderate = (~underest) & (~severe)

    # Total-variation distances used by the adaptive coefficient ``Delta``
    # (paper §3.1, formula below Eq. 7).
    tv_q_p_over_ell = 0.5 * (q - p / ell).abs().sum()
    tv_q_p = 0.5 * (q - p).abs().sum()
    if tv_q_p.item() == 0.0:
        # Draft equals target; the rule collapses to identity.
        return p.clone().to(p.dtype)
    delta = (tv_q_p_over_ell + 0.5 - 0.5 / ell) / tv_q_p

    yld = torch.empty_like(p)
    yld[underest] = delta * p[underest] + (1.0 - delta) * q[underest]
    yld[moderate] = q[moderate]
    yld[severe] = p[severe] / ell
    return yld


def cos_yield_distribution(
    p: torch.Tensor, q: torch.Tensor, lam: float
) -> torch.Tensor:
    """Induced generation distribution under CoS-WE: the convex mixture
    :math:`\\lambda p + (1-\\lambda) q` (paper Eq. 6)."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("CoS lambda must be in [0, 1]")
    return lam * p + (1.0 - lam) * q
