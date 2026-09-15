"""Reliability of feature-ICA components: schedules, bootstraps, split halves.

JADE has no random initialisation, so the optimiser-restart check used with
tanh ICA is replaced by three data-side checks:

``schedule_stability``
    re-diagonalise the SAME cumulant stack under permuted Jacobi pair
    orderings. Components that move are stationary points the data cannot
    rank, not optimisation failures.
``feature_bootstrap_stability``
    refit (whitening included) on features resampled with replacement.
``split_half_stability``
    fix the whitening on all features, fit each half of a feature partition
    separately (best of several schedules by the JADE criterion), and read the
    per-axis agreement of both halves with the full fit. Alternating genomic
    blocks (``alternating_blocks``) are the appropriate partition for ATAC
    peaks; a random feature split is appropriate for genes.
``effective_support``
    inverse participation ratio of a component's cell activities: how many
    cells actually carry it. A component reproducible across schedules can
    still be a single-cell spike.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._basis import diagonal_criterion, jade_rotation, whiten_loadings
from ._backend import array_module

__all__ = ["matched_columns", "effective_support", "schedule_stability",
           "feature_bootstrap_stability", "alternating_blocks", "split_half_stability"]


def matched_columns(a, b):
    """Hungarian match of columns on |correlation|; returns ``(i, j, |r|)``."""
    a = np.asarray(a, np.float64) - np.asarray(a, np.float64).mean(0)
    b = np.asarray(b, np.float64) - np.asarray(b, np.float64).mean(0)
    a = a / np.maximum(np.linalg.norm(a, axis=0), 1e-300)
    b = b / np.maximum(np.linalg.norm(b, axis=0), 1e-300)
    corr = np.abs(a.T @ b)
    i, j = linear_sum_assignment(-corr)
    return i, j, corr[i, j]


def effective_support(activities):
    """Inverse participation ratio of each activity column, in cells."""
    weight = np.asarray(activities, np.float64) ** 2
    share = weight / np.maximum(weight.sum(0, keepdims=True), 1e-300)
    return 1.0 / (share ** 2).sum(0)


def _activities(scores_c, K_inv, W):
    return scores_c @ K_inv @ W.T


def schedule_stability(scores, components, *, seeds=(1, 2, 3, 4, 5), use_gpu=True,
                       max_sweeps=600, threshold=None):
    """Worst matched |r| of each base component over permuted Jacobi schedules.

    The cumulant stack is built once; each schedule only re-runs the Jacobi
    stage. Matching is on cell activities, which is what downstream analyses
    consume.
    """
    P = np.asarray(components, np.float64)
    Zc = np.asarray(scores, np.float64)
    Zc = Zc - Zc.mean(0)
    white = whiten_loadings(P)
    z = white["K"] @ white["x"]
    W0, base_diag, stack = jade_rotation(z, use_gpu=use_gpu, max_sweeps=max_sweeps,
                                         threshold=threshold)
    base = _activities(Zc, white["K_inv"], W0)
    r = P.shape[0]
    worst = np.full(r, np.inf)
    records = []
    for seed in seeds:
        W, diag, _ = jade_rotation(z, stack=stack, schedule_seed=seed, use_gpu=use_gpu,
                                   max_sweeps=max_sweeps, threshold=threshold)
        alt = _activities(Zc, white["K_inv"], W)
        i, _, corr = matched_columns(base, alt)
        worst[i] = np.minimum(worst[i], corr)
        records.append({"schedule_seed": int(seed), "criterion": diag["criterion"],
                        "sweeps": diag["sweeps"], "converged": diag["converged"],
                        "median_matched_abs_r": float(np.median(corr))})
    return {"worst_abs_r": worst, "base_criterion": base_diag["criterion"],
            "schedules": records, "n_unstable_0.99": int((worst <= 0.99).sum())}


def feature_bootstrap_stability(scores, components, *, n_bootstraps=10, seed=0,
                                use_gpu=True, max_sweeps=600, threshold=None):
    """Worst matched |r| of each component over feature-resampled refits."""
    P = np.asarray(components, np.float64)
    Zc = np.asarray(scores, np.float64)
    Zc = Zc - Zc.mean(0)
    white = whiten_loadings(P)
    W0, _, _ = jade_rotation(white["K"] @ white["x"], use_gpu=use_gpu,
                             max_sweeps=max_sweeps, threshold=threshold)
    base = _activities(Zc, white["K_inv"], W0)
    r, n_features = P.shape
    rng = np.random.default_rng(seed)
    worst = np.full(r, np.inf)
    medians = []
    for _ in range(n_bootstraps):
        cols = rng.integers(0, n_features, n_features)
        w = whiten_loadings(P[:, cols])
        W, _, _ = jade_rotation(w["K"] @ w["x"], use_gpu=use_gpu, max_sweeps=max_sweeps,
                                threshold=threshold)
        alt = _activities(Zc, w["K_inv"], W)
        i, _, corr = matched_columns(base, alt)
        worst[i] = np.minimum(worst[i], corr)
        medians.append(float(np.median(corr)))
    return {"worst_abs_r": worst, "bootstrap_median_abs_r": medians,
            "n_bootstraps": int(n_bootstraps), "seed": int(seed)}


def alternating_blocks(position, block_size, group=None):
    """Boolean half-mask: alternating ``block_size`` windows of ``position``.

    ``group`` (e.g. chromosome) offsets the alternation so the parity does not
    line up across groups. Adjacent features land in the same half, so a
    spatially smooth signal (copy number along the genome) is present in both
    halves while local noise is not shared.
    """
    position = np.asarray(position, np.int64)
    code = np.zeros(position.shape[0], np.int64)
    if group is not None:
        _, code = np.unique(np.asarray(group), return_inverse=True)
    return ((position // int(block_size)) + code) % 2 == 0


def _best_of_schedules(z_half, schedules, use_gpu, max_sweeps, threshold):
    """Fit one feature half under several schedules; keep the best criterion."""
    stack = None
    best_W, best_crit = None, -np.inf
    for seed in schedules:
        W, diag, stack = jade_rotation(z_half, schedule_seed=seed, use_gpu=use_gpu,
                                       max_sweeps=max_sweeps, threshold=threshold,
                                       stack=stack)
        if diag["criterion"] > best_crit:
            best_W, best_crit = W, diag["criterion"]
    return best_W, best_crit


def split_half_stability(scores, components, mask, *, schedules=(None, 1, 2), use_gpu=True,
                         max_sweeps=600, threshold=None):
    """Per-axis agreement of two half-feature fits with the full fit.

    Whitening is fixed from all features, so the halves live in the same
    coordinates as the base fit and the source correlation between two fits is
    exactly ``W_a @ W_b.T``. Each half is fitted under several schedules and
    the fit with the largest JADE criterion is kept. Returned per base axis:
    the matched |cos| against half A, against half B, and their minimum; plus
    the direct A-vs-B agreement.
    """
    mask = np.asarray(mask, bool)
    P = np.asarray(components, np.float64)
    if mask.shape != (P.shape[1],):
        raise ValueError("mask needs one boolean per feature")
    white = whiten_loadings(P)
    z = white["K"] @ white["x"]
    W0, _, _ = jade_rotation(z, use_gpu=use_gpu, max_sweeps=max_sweeps, threshold=threshold)
    out = {"n_half_a": int(mask.sum()), "n_half_b": int((~mask).sum())}
    halves = {}
    for name, half in (("a", mask), ("b", ~mask)):
        W, crit = _best_of_schedules(z[:, half], schedules, use_gpu, max_sweeps, threshold)
        Q = np.abs(W0 @ W.T)                       # base sources x half sources
        i, j = linear_sum_assignment(-Q)
        cos = np.empty(P.shape[0])
        cos[i] = Q[i, j]
        halves[name] = (W, cos)
        out[f"abs_cos_{name}"] = cos
        out[f"criterion_{name}"] = crit
    out["abs_cos_min"] = np.minimum(halves["a"][1], halves["b"][1])
    Qab = np.abs(halves["a"][0] @ halves["b"][0].T)
    i, j = linear_sum_assignment(-Qab)
    out["abs_cos_a_vs_b_matched"] = Qab[i, j]
    return out
