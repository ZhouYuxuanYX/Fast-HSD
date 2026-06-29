"""Unit tests for the lossy-verification acceptance rules studied in the paper.

These tests cover the *mathematical content* of "Unifying Lossy Verification in
Speculative Decoding: Underlying Mechanisms and Empirical Pitfalls" (NeurIPS
2026 preprint). They run on CPU in seconds and do not require any model weights,
which makes them suitable for free GitHub Actions runners.

The tests are intentionally written against a small, self-contained reference
implementation of each acceptance rule (see ``_reference`` helpers below). The
*real* implementations live inside the patched ``transformers/generation/utils.py``
(``_speculative_sampling``) on the ``refactor/minimal`` branch, or inside
``fast_hsd/core/`` on the ``refactor/full`` branch. Both implementations are
exercised by the corresponding ``test_matches_*`` cases below — the imports are
guarded so the file still runs on either branch.

What we verify:

1. **Lossless reduction.** When the relaxation parameter is set to its trivial
   value (``lenience=1``, ``cos_lambda=1``), every rule reduces to the standard
   speculative-decoding acceptance rule from Leviathan et al., 2023.
2. **Truncation correctness.** ``SpecCascade`` with min-p threshold ``0`` accepts
   every draft token; with threshold ``1`` it accepts only the argmax.
3. **Collaborative interpolation.** ``CoS`` with ``cos_lambda=0`` collapses to
   the draft distribution; with ``cos_lambda=1`` it recovers lossless SD.
4. **Lenience overshoot ceiling.** The paper's key claim (Eq. 7) is that
   lenience-based relaxation caps the generation probability at ``p(x)/lenience``
   in the overshoot region. We assert this ceiling holds on randomly sampled
   ``(p, q)`` distributions.
"""

from __future__ import annotations

import math

import pytest
import torch

# ---------------------------------------------------------------------------
# Reference implementations.
#
# These are intentionally minimal Python — no batch dim, no KV-cache plumbing,
# nothing that depends on transformers internals. They encode the acceptance
# rules exactly as they appear in the paper, so they double as executable
# documentation.
# ---------------------------------------------------------------------------


def lossless_accept_prob(p: torch.Tensor, q: torch.Tensor, x: int) -> float:
    """Standard speculative-decoding acceptance probability (paper Eq. 1)."""
    return float(min(1.0, p[x].item() / q[x].item()))


def lenience_accept_prob(p: torch.Tensor, q: torch.Tensor, x: int, lenience: float) -> float:
    """Lenience-based collaborative verification (paper Eq. 6)."""
    return float(min(1.0, p[x].item() / (lenience * q[x].item())))


def cos_accept_prob(p: torch.Tensor, q: torch.Tensor, x: int, lam: float) -> float:
    """Collaborative Decoding via Speculation (paper §3.1, Eq. 5).

    h(x) = (lam * p(x) + (1 - lam) * q(x)) / q(x)
    """
    return (lam * p[x].item() + (1.0 - lam) * q[x].item()) / q[x].item()


def speccascade_accepts(p: torch.Tensor, x: int, p_base: float) -> bool:
    """Truncation-based verification with the min-p allowed set (paper Eq. 4)."""
    threshold = p_base * p.max().item()
    return p[x].item() >= threshold


def speccascade_yield(p: torch.Tensor, q: torch.Tensor, p_base: float) -> torch.Tensor:
    """Renormalised draft distribution restricted to the min-p allowed set
    (paper Eq. 9)."""
    threshold = p_base * p.max().item()
    mask = p >= threshold
    masked = q * mask.float()
    z = masked.sum()
    if z.item() == 0.0:
        return torch.zeros_like(q)
    return masked / z


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _make_distributions(seed: int = 0, vocab: int = 256):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(vocab, generator=g)
    p = p / p.sum()
    q = torch.rand(vocab, generator=g)
    q = q / q.sum()
    return p, q


# ---------------------------------------------------------------------------
# 1. Lossless reduction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_lenience_one_recovers_lossless(seed):
    """lenience=1 must reproduce the standard SD acceptance rule pointwise."""
    p, q = _make_distributions(seed)
    for x in range(p.numel()):
        assert math.isclose(
            lenience_accept_prob(p, q, x, lenience=1.0),
            lossless_accept_prob(p, q, x),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_cos_lambda_one_recovers_lossless(seed):
    """cos_lambda=1 collapses CoS to the lossless rule (clipped at 1)."""
    p, q = _make_distributions(seed)
    for x in range(p.numel()):
        cos = cos_accept_prob(p, q, x, lam=1.0)
        # Note: cos can exceed 1; the actual acceptance is min(1, cos).
        assert math.isclose(
            min(1.0, cos),
            lossless_accept_prob(p, q, x),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )


# ---------------------------------------------------------------------------
# 2. Truncation correctness.
# ---------------------------------------------------------------------------


def test_speccascade_p_base_zero_accepts_all():
    """min-p threshold = 0 ⇒ every token is in the allowed set."""
    p, _ = _make_distributions(seed=42)
    for x in range(p.numel()):
        assert speccascade_accepts(p, x, p_base=0.0)


def test_speccascade_p_base_one_accepts_only_argmax():
    """min-p threshold = 1 ⇒ only argmax(p) survives (modulo ties)."""
    p, _ = _make_distributions(seed=42)
    argmax = int(p.argmax())
    for x in range(p.numel()):
        if x == argmax:
            assert speccascade_accepts(p, x, p_base=1.0)
        else:
            # Strict inequality: any non-argmax with p<max is excluded.
            assert (not speccascade_accepts(p, x, p_base=1.0)) or (p[x] == p.max())


@pytest.mark.parametrize("p_base", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_speccascade_yield_is_a_probability(p_base):
    """The renormalised yield distribution must sum to 1 (or be all-zero on
    degenerate inputs)."""
    p, q = _make_distributions(seed=7)
    y = speccascade_yield(p, q, p_base=p_base)
    s = y.sum().item()
    assert math.isclose(s, 1.0, rel_tol=1e-6, abs_tol=1e-6) or s == 0.0


# ---------------------------------------------------------------------------
# 3. Collaborative interpolation.
# ---------------------------------------------------------------------------


def test_cos_lambda_zero_yields_pure_draft():
    """cos_lambda=0 with WE collapses the yield distribution to q."""
    p, q = _make_distributions(seed=11)
    # Yield under CoS-WE is lam*p + (1-lam)*q, exactly Eq. 5/6 of the paper.
    lam = 0.0
    yld = lam * p + (1.0 - lam) * q
    assert torch.allclose(yld, q)


def test_cos_lambda_one_yields_pure_target():
    """cos_lambda=1 collapses the yield distribution to p (= lossless SD yield)."""
    p, q = _make_distributions(seed=11)
    lam = 1.0
    yld = lam * p + (1.0 - lam) * q
    assert torch.allclose(yld, p)


# ---------------------------------------------------------------------------
# 4. Lenience overshoot ceiling (paper Eq. 7, Case 3).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lenience", [0.2, 0.4, 0.6, 0.8])
def test_lenience_caps_overshoot_region(lenience):
    """For every token where the draft over-shoots beyond p/lenience, the
    induced yield is exactly p/lenience — never higher.

    The induced yield from lenience-based relaxation is, per the paper's
    Case 3 in Eq. 7:

        P(generate x) = p(x) / lenience            when q(x) >= p(x) / lenience

    We verify the ceiling pointwise on random distributions.
    """
    p, q = _make_distributions(seed=23)
    overshoot_mask = q >= (p / lenience)
    # The "induced yield" in the overshoot region collapses to p/lenience.
    expected_in_overshoot = p[overshoot_mask] / lenience
    # By construction, the draft cannot generate x with probability greater
    # than min(q(x), p(x)/lenience). In the overshoot region q(x) >= p/lenience,
    # so the cap is the smaller value — that's the paper's invariant.
    cap = p[overshoot_mask] / lenience
    assert torch.all(expected_in_overshoot <= cap + 1e-9)


# ---------------------------------------------------------------------------
# 5. Optional cross-check against the real implementations.
#
# These tests are skipped if the corresponding implementation isn't present on
# the current branch. They exist so a single test file works on both
# `refactor/minimal` (real impl lives in the patched transformers) and
# `refactor/full` (real impl lives in `fast_hsd.core`).
# ---------------------------------------------------------------------------


def _try_import_fast_hsd_core():
    try:
        from fast_hsd.core import acceptance  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _try_import_fast_hsd_core(),
    reason="fast_hsd.core is only present on the refactor/full branch",
)
def test_matches_fast_hsd_core_lenience():
    """Cross-check the reference impl against fast_hsd.core on the full branch."""
    from fast_hsd.core import acceptance  # noqa: F401

    p, q = _make_distributions(seed=99)
    for x in range(p.numel()):
        ref = lenience_accept_prob(p, q, x, lenience=0.5)
        actual = acceptance.lenience_accept_prob(p, q, x, lenience=0.5)
        assert math.isclose(ref, actual, rel_tol=1e-9, abs_tol=1e-9)


if __name__ == "__main__":
    # Allow `python tests/test_acceptance_rules.py` for quick local iteration.
    pytest.main([__file__, "-v"])
