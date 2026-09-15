"""Feature-space ICA on a PCA span: shared whitening, contrasts and read-out.

The cached residual PCA is ``R ~= Z P`` with ``Z`` (n cells, r) scores and
``P`` (r, N features) component rows. Feature gICA rotates in a FEATURE-whitened
basis (ported from ProgramForge ``multimodal_gica_core.py``):

    x  = P - mean_features(P)                 (r, N) row-centred loadings
    K  = Lambda^{-1/2} U^T,  U Lambda U^T = x x^T / N     whitening (r, r)
    Yt = K x                                  (r, N), Yt Yt^T / N = I
    Y  = W Yt,  W in O(r)                     sources = whitened loadings
    G  = W K P                                (r, N) loadings, feature mean kept
    A  = Z_c K^{-1} W^T                       (n, r) cell activities

``A G_centred = Z_c x`` holds for EVERY orthogonal ``W``: reconstruction of the
retained span is a numeric contract, not an objective. Only the choice of ``W``
differs between contrasts:

``"jade"``
    fourth-order cumulant joint diagonalisation (Cardoso & Souloumiac), no
    random initialisation; see :mod:`programfinder._jade`.
``"picard"``
    Picard-O with the tanh log-density (planned; see ``PLAN.md``).

Display convention, shared by every contrast: activities have unit standard
deviation and positive skewness, loadings carry the corresponding scale and
sign so that ``score_to_activity @ loadings == P``.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import kurtosis, skew

from . import _jade
from ._backend import array_module, to_host

__all__ = ["whiten_loadings", "feature_ica", "jade_rotation", "diagonal_criterion",
           "display_orientation", "CONTRASTS"]

CONTRASTS = ("jade", "picard")


# --------------------------------------------------------------------------
# whitening and back-projection
# --------------------------------------------------------------------------

def whiten_loadings(P):
    """Feature whitening of the PCA loading rows.

    Returns ``K`` (r, r) with ``K x x^T K^T / N = I``, its inverse, the
    row-centred loadings ``x`` and the feature mean of each row.
    """
    P = np.asarray(P, np.float64)
    mean = P.mean(1)
    x = P - mean[:, None]
    r, N = x.shape
    C = x @ x.T / N
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if vals[-1] <= 1e-12 * vals[0]:
        raise ValueError("feature whitening needs a full-rank centred loading matrix")
    K = (vecs / np.sqrt(vals)).T
    K_inv = vecs * np.sqrt(vals)
    return {"K": K, "K_inv": K_inv, "x": x, "mean": mean, "eigenvalues": vals}


def display_orientation(A, G):
    """Unit-SD, positive-skew activities; loadings carry the scale and sign."""
    A = np.asarray(A, np.float64)
    scale = A.std(0)
    if np.any(scale <= 0):
        raise ValueError("degenerate activity column")
    sign = np.where(skew(A, axis=0) < 0, -1.0, 1.0)
    return A / scale * sign, np.asarray(G) * (scale * sign)[:, None], scale, sign


# --------------------------------------------------------------------------
# JADE contrast on a whitened source matrix
# --------------------------------------------------------------------------

def diagonal_criterion(stack, W, xp=np):
    """JADE criterion: sum of squared diagonals of the rotated cumulant slices.

    Larger is better; every stationary point of the joint diagonalisation is a
    local maximum of this quantity, so it ranks solutions found under
    different schedules or on different feature subsets.
    """
    W = xp.asarray(W)
    rotated = (W[None] @ stack) @ W.T
    return float(xp.einsum("kii->ki", rotated).__pow__(2).sum())


def jade_rotation(z, *, threshold=None, max_sweeps=600, schedule_seed=None,
                  use_gpu=True, stack=None):
    """Orthogonal ``W`` (rows = sources) diagonalising the cumulants of ``z``.

    ``z`` is the whitened (r, N) matrix. The cumulant stack is computed once
    on the selected device unless one is supplied; a supplied stack is copied,
    never mutated, so callers can re-run under several schedules cheaply.
    """
    xp, on_gpu = array_module(use_gpu)
    z = xp.asarray(z, dtype=xp.float64)
    m, n_obs = z.shape
    if threshold is None:
        threshold = 1.0 / np.sqrt(n_obs) / 100.0
    started = time.time()
    stack = _jade.cumulant_matrices(z, xp) if stack is None else xp.asarray(stack)
    rotation, diagnostics = _jade.joint_diagonalise(
        stack.copy(), m, threshold, max_sweeps, xp, schedule_seed)
    W = to_host(rotation, xp).T
    diagnostics.update(device="gpu" if on_gpu else "cpu", schedule_seed=schedule_seed,
                       n_observations=int(n_obs), n_components=int(m),
                       criterion=diagonal_criterion(stack, W, xp),
                       seconds=round(time.time() - started, 2))
    return W, diagnostics, stack


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def feature_ica(scores, components, *, contrast="jade", use_gpu=True, max_sweeps=600,
                threshold=None, schedule_seed=None, seed=0, max_iter=1000, tol=1e-7):
    """Feature-space ICA of a PCA span; returns activities, loadings and maps.

    Parameters
    ----------
    scores : (n_cells, r) PCA scores (any centring; centred internally)
    components : (r, N) PCA component rows
    contrast : ``"jade"`` (implemented) or ``"picard"`` (planned)
    use_gpu : run the cumulant and Jacobi stages on cupy when available
    max_sweeps, threshold, schedule_seed : JADE controls
    seed, max_iter, tol : reserved for the Picard contrast

    Returns a dict with ``activities`` (n, r; unit SD, positive skew),
    ``loadings`` (r, N), ``sources`` (r, N; whitened, row-centred), ``W``,
    ``K``, ``K_inv``, ``feature_mean``, ``scale``, ``sign``,
    ``score_to_activity`` (r, r; ``scores_c @ score_to_activity == activities``),
    ``feature_to_source`` (r, r; ``feature_to_source @ P == loadings``),
    ``excess_kurtosis`` per source, and ``diagnostics``.
    """
    if contrast not in CONTRASTS:
        raise ValueError(f"contrast must be one of {CONTRASTS}, got {contrast!r}")
    P = np.asarray(components, np.float64)
    Zc = np.asarray(scores, np.float64)
    Zc = Zc - Zc.mean(0)
    r = P.shape[0]
    if Zc.shape[1] != r:
        raise ValueError(f"scores have {Zc.shape[1]} columns but components have {r} rows")
    white = whiten_loadings(P)
    z = white["K"] @ white["x"]

    if contrast == "jade":
        W, diagnostics, _ = jade_rotation(z, threshold=threshold, max_sweeps=max_sweeps,
                                          schedule_seed=schedule_seed, use_gpu=use_gpu)
    else:
        raise NotImplementedError(
            "the picard contrast is the next port step; see PLAN.md in the repository "
            "root. Use contrast='jade' for now.")

    orthogonality = float(np.abs(W @ W.T - np.eye(r)).max())
    if orthogonality > 1e-8:
        raise FloatingPointError(f"contrast returned a non-orthogonal W ({orthogonality:.1e})")
    transform = W @ white["K"]                   # feature -> source, unscaled
    inverse = white["K_inv"] @ W.T               # source -> feature span
    activities_raw = Zc @ inverse
    loadings_raw = transform @ P
    activities, loadings, scale, sign = display_orientation(activities_raw, loadings_raw)
    score_to_activity = inverse / scale[None, :] * sign[None, :]
    feature_to_source = (scale * sign)[:, None] * transform
    sources = W @ z
    reconstruction = np.linalg.norm(score_to_activity @ loadings - P) / np.linalg.norm(P)
    if reconstruction > 1e-8:
        raise FloatingPointError(f"feature ICA back-projection failed ({reconstruction:.1e})")
    return {
        "contrast": contrast,
        "activities": activities,
        "loadings": loadings,
        "sources": sources,
        "W": W,
        "K": white["K"],
        "K_inv": white["K_inv"],
        "feature_mean": white["mean"],
        "whitening_eigenvalues": white["eigenvalues"],
        "scale": scale,
        "sign": sign,
        "score_to_activity": score_to_activity,
        "feature_to_source": feature_to_source,
        "excess_kurtosis": kurtosis(sources, axis=1, fisher=True),
        "reconstruction_relative_error": float(reconstruction),
        "diagnostics": diagnostics,
    }
