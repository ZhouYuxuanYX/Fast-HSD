import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def tv_distance(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    return 1.0 - np.minimum(p, q).sum()


def shift_mass_to_reach_tv(p, target_tv):
    """
    Given a base distribution p (non-uniform permitted), construct q so that
    TV(p, q) = target_tv (where TV = 0.5 * L1 = 1 - sum min).
    Strategy: move total mass S = target_tv from large-probability bins to small-probability bins.
    Max feasible TV with fixed p is (1 - p.max()) (achieved by putting all q mass on argmax p).
    """
    p = np.asarray(p, dtype=float)
    assert np.isclose(p.sum(), 1.0), "p must sum to 1"
    max_tv = 1.0 - p.max()
    if target_tv > max_tv + 1e-12:
        raise ValueError(f"target_tv={target_tv} too large for this base p (max={max_tv:.6f}). "
                         f"Try a different p or smaller target.")

    if target_tv < 1e-12:
        return p.copy(), p.copy(), 0.0

    # We need to shift S = target_tv total probability mass.
    S = target_tv
    q = p.copy()

    # Indices sorted by p ascending (receivers) and descending (donors).
    recv = np.argsort(p)          # smallest first
    donor = recv[::-1]            # largest first

    i_r, i_d = 0, 0
    remaining = S
    while remaining > 1e-14 and i_r < len(recv) and i_d < len(donor):
        r_idx = recv[i_r]
        d_idx = donor[i_d]
        if r_idx == d_idx:
            break  # nothing more to do
        add_cap = 1.0 - q[r_idx]
        sub_cap = q[d_idx]
        delta = min(add_cap, sub_cap, remaining)
        q[r_idx] += delta
        q[d_idx] -= delta
        remaining -= delta
        if add_cap - delta < 1e-14:
            i_r += 1
        if sub_cap - delta < 1e-14:
            i_d += 1

    achieved_tv = 0.5 * np.abs(p - q).sum()
    # Small numerical adjust if needed
    if abs(achieved_tv - target_tv) > 5e-8:
        # Final tiny correction (distribute epsilon if any)
        diff = target_tv - achieved_tv
        if abs(diff) > 1e-6:
            raise RuntimeError(f"Could not hit target_tv precisely (diff={diff}).")
    return p, q, achieved_tv

def make_pair_with_tv(vocab_size, target_tv, base='uniform', seed=None, base_array=None):
    """
    Generate (p, q) with specified TV distance (1 - sum min = target_tv).
    base:
      - 'uniform': p is uniform
      - 'dirichlet': p ~ Dirichlet(1,...,1)
      - 'given': use base_array (must sum to 1)
    """
    rng = np.random.default_rng(seed)
    if base == 'uniform':
        p = np.ones(vocab_size) / vocab_size
    elif base == 'dirichlet':
        p = rng.dirichlet(np.ones(vocab_size))
    elif base == 'given':
        assert base_array is not None, "Provide base_array for base='given'"
        p = np.asarray(base_array, dtype=float)
        p /= p.sum()
        if len(p) != vocab_size:
            raise ValueError("base_array length mismatch vocab_size")
    else:
        raise ValueError("Unknown base type")

    p, q, tv = shift_mass_to_reach_tv(p, target_tv)
    return p, q, tv
