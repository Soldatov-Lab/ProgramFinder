"""Residual dependence between feature-ICA sources: which components form a group.

A JADE fit is a joint block-diagonaliser of the fourth-order cumulant slices
when the data follow an independent-subspace model (Theis, "Towards a general
independent subspace analysis", NIPS 2006, Thm 2.1). So the subspaces are
recovered by GROUPING the fitted components, not by a new fit. Within a
subspace the axes are not identifiable at all (Thm 1.2/1.8), which is what an
unstable axis inside a stable group looks like.

For the whitened sources ``s`` (components x features; the loadings, since the
ICA runs over features), two pairwise statistics, a != b:

``D_ab = sum_{i,j} cum(s_a, s_b, s_i, s_j)^2``
    the pair's share of the off-diagonal cumulant energy JADE could not remove;
``E_ab = corr(s_a^2, s_b^2)``
    energy dependence, the classic within-subspace signature.

Raw values are dominated by the heaviest-tailed components, so both are
referred to a null that permutes every source's features independently
(marginals kept, cross-dependence destroyed) and reported as z-scores.

Groups: any global threshold on these z-scores chains unrelated programs
through a few hub components (technical axes such as mitochondrial reads or
depth depend on many programs at once). :func:`mutual_top_groups` therefore
uses a local rule, mutual top-``k`` partners above a floor. Treat its output
as a proposal; whether a group's span comes back is a separate question,
answered by :func:`programfinder._stability.group_recovery` on refits.
"""

from __future__ import annotations

import numpy as np

from ._backend import array_module, to_host

__all__ = ["fourth_order_cumulants", "pairwise_dependence", "dependence_z",
           "ranked_partners", "mutual_top_groups"]


def fourth_order_cumulants(s, xp=np, chunk=20000):
    """Cumulant tensor ``cum(a, b, c, d)`` of the rows of ``s``, shape (K, K, K, K).

    ``s`` is (K, n), rows centred. The fourth moment is accumulated over feature
    chunks, so memory is ``K^2 x chunk`` rather than ``K^2 x n``.
    """
    K, n = s.shape
    r = (s @ s.T) / n
    moment = xp.zeros((K * K, K * K), dtype=s.dtype)
    for lo in range(0, n, chunk):
        blk = s[:, lo:lo + chunk]
        prod = (blk[:, None, :] * blk[None, :, :]).reshape(K * K, -1)
        moment += prod @ prod.T
    moment = (moment / n).reshape(K, K, K, K)
    return (moment - r[:, :, None, None] * r[None, None, :, :]
            - r[:, None, :, None] * r[None, :, None, :]
            - r[:, None, None, :] * r[None, :, :, None])


def pairwise_dependence(s, xp=np, chunk=20000):
    """Residual cross-cumulant energy ``D`` and energy correlation ``E`` (K x K)."""
    q = fourth_order_cumulants(s, xp, chunk)
    D = (q ** 2).sum(axis=(2, 3))
    idx = xp.arange(s.shape[0])
    diag4 = q[idx, idx, idx, idx]
    aabb = q[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]
    E = aabb / xp.sqrt(xp.outer(diag4 + 2, diag4 + 2))
    return D, E


def _standardise_rows(s, xp):
    s = xp.asarray(s, dtype=xp.float64)
    s = s - s.mean(1, keepdims=True)
    return s / s.std(1, keepdims=True)


def dependence_z(sources, *, reps=100, seed=0, use_gpu=True, chunk=20000):
    """D, E and their z-scores against a per-source feature-permutation null.

    Returns host arrays ``D``, ``E``, ``zD``, ``zE`` (K x K, diagonal NaN) and
    the null mean and SD of each.
    """
    xp, on_gpu = array_module(use_gpu)
    s = _standardise_rows(sources, xp)
    D, E = pairwise_dependence(s, xp, chunk)
    rng = xp.random.default_rng(seed)
    sum_d = xp.zeros_like(D); sq_d = xp.zeros_like(D)
    sum_e = xp.zeros_like(E); sq_e = xp.zeros_like(E)
    for _ in range(int(reps)):
        perm = xp.argsort(rng.random(s.shape), axis=1)
        d, e = pairwise_dependence(xp.take_along_axis(s, perm, axis=1), xp, chunk)
        sum_d += d; sq_d += d * d; sum_e += e; sq_e += e * e
    mean_d, mean_e = sum_d / reps, sum_e / reps
    sd_d = xp.sqrt(xp.maximum(sq_d / reps - mean_d ** 2, 0))      # population SD of the null
    sd_e = xp.sqrt(xp.maximum(sq_e / reps - mean_e ** 2, 0))
    sd_d = xp.where(sd_d > 0, sd_d, xp.nan)                     # the diagonal has no null spread
    sd_e = xp.where(sd_e > 0, sd_e, xp.nan)
    out = {"D": D, "E": E, "zD": (D - mean_d) / sd_d, "zE": (E - mean_e) / sd_e,
           "null_mean_D": mean_d, "null_sd_D": sd_d, "null_mean_E": mean_e, "null_sd_E": sd_e}
    out = {k: to_host(v, xp) for k, v in out.items()}
    for k in ("zD", "zE"):
        np.fill_diagonal(out[k], np.nan)
    out.update(reps=int(reps), seed=int(seed), device="gpu" if on_gpu else "cpu")
    return out


def ranked_partners(z, candidates=None):
    """For each component, the other candidates ordered by decreasing ``z``."""
    z = np.asarray(z, np.float64)
    K = z.shape[0]
    cand = np.ones(K, bool) if candidates is None else np.asarray(candidates, bool)
    ranked = []
    for a in range(K):
        others = [b for b in np.argsort(-np.nan_to_num(z[a], nan=-np.inf)) if b != a and cand[b]]
        ranked.append([int(b) for b in others])
    return ranked


def mutual_top_groups(z, candidates=None, *, k=2, z_min=5.0):
    """Connected components of the graph with an edge where a and b are in each
    other's top-``k`` partners (among ``candidates``) and ``z >= z_min``.

    Returns lists of component indices, groups of two or more only.
    """
    z = np.asarray(z, np.float64)
    K = z.shape[0]
    cand = np.ones(K, bool) if candidates is None else np.asarray(candidates, bool)
    ranked = ranked_partners(z, cand)
    top = [set(ranked[a][:k]) for a in range(K)]
    adj = np.zeros((K, K), bool)
    for a in np.flatnonzero(cand):
        for b in top[a]:
            if a in top[b] and z[a, b] >= z_min:
                adj[a, b] = adj[b, a] = True
    seen, groups = set(), []
    for a in range(K):
        if a in seen or not adj[a].any():
            continue
        stack, comp = [a], set()
        while stack:
            c = stack.pop()
            if c not in comp:
                comp.add(c)
                stack.extend(int(x) for x in np.flatnonzero(adj[c]) if x not in comp)
        seen |= comp
        groups.append(sorted(int(c) for c in comp))
    return groups
