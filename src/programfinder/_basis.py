"""Feature-space ICA on a PCA span: shared whitening, contrasts and read-out.

The cached residual PCA is ``R ~= Z P`` with ``Z`` (n cells, r) scores and
``P`` (r, N features) component rows. Feature gICA rotates in a FEATURE-whitened
basis:

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
    Picard-O with the tanh log-density, from the optional ``python-picard``
    dependency, run on the SAME ``K`` rather than on picard's own internal
    whitening. The two whitenings differ by an orthogonal factor, so the
    feasible source set ``{W K x : W in O(r)}`` is identical either way and the
    optima coincide -- but the optimiser path does not, so a given
    ``random_state`` need not land on the same one. Reliability for this
    contrast is therefore restarts, not Jacobi schedules.

Display convention, shared by every contrast: activities have unit standard
deviation and positive skewness, loadings carry the corresponding scale and
sign so that ``score_to_activity @ loadings == P``.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.stats import kurtosis, skew

from . import _jade
from ._backend import array_module, to_host

__all__ = ["whiten_loadings", "feature_ica", "jade_rotation", "picard_rotation",
           "diagonal_criterion", "display_orientation", "CONTRASTS"]

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
# Picard-O contrast on the same whitened source matrix
# --------------------------------------------------------------------------

def logcosh_criterion(y):
    """Tanh-density log-likelihood of the sources, up to an additive constant.

    ``fun="tanh"`` is the density model ``p(y) propto 1 / cosh(y)``, so the
    likelihood Picard-O maximises is ``-sum_i E[log cosh(y_i)]``. Returned with
    that sign, so larger is better and it ranks restarts the way
    :func:`diagonal_criterion` ranks JADE schedules.
    """
    y = np.asarray(y, np.float64)
    return float(-np.logaddexp(y, -y).mean(1).sum() + y.shape[0] * np.log(2.0))


def picard_rotation(z, *, seed=0, max_iter=1000, tol=1e-7, w_init=None):
    """Orthogonal ``W`` (rows = sources) maximising the tanh log-density of ``z``.

    ``z`` is the already-whitened, row-centred (r, N) matrix, so picard is
    called with ``whiten=False, centering=False`` and its returned unmixing IS
    the rotation. CPU only -- ``python-picard`` is numpy -- which is why this
    contrast ignores the device setting.

    ``w_init`` warm-starts the optimiser from a given rotation instead of a
    random one, which is how a rotation fitted elsewhere is checked for being
    a stationary point here: pass it and the returned ``W`` should come back
    unchanged. It must be orthogonal in THIS whitening.
    """
    try:
        from picard import picard
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "contrast='picard' needs the optional python-picard dependency: "
            "pip install -e '.[picard]'") from error
    z = np.asarray(z, np.float64)
    r = z.shape[0]
    if w_init is not None:
        w_init = np.asarray(w_init, np.float64)
        error = float(np.abs(w_init @ w_init.T - np.eye(r)).max())
        # Looser than the 1e-8 the fitted W is held to: a warm start often comes
        # from a float32 artifact, and picard re-orthogonalises as it iterates.
        if error > 1e-6:
            raise ValueError(f"w_init is not orthogonal in this whitening ({error:.1e})")
    started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, unmixing, sources, n_iter = picard(
            z, n_components=r, ortho=True, fun="tanh", whiten=False,
            centering=False, max_iter=max_iter, tol=tol, random_state=seed,
            w_init=w_init, return_n_iter=True)
    W = np.asarray(unmixing, np.float64)
    messages = [str(entry.message) for entry in caught]
    diagnostics = {
        "contrast": "picard", "seed": int(seed), "n_iter": int(n_iter),
        "warm_started": w_init is not None,
        "converged": not any("did not converge" in m.lower() for m in messages),
        "warnings": messages, "device": "cpu", "max_iter": int(max_iter),
        "tol": float(tol), "criterion": logcosh_criterion(sources),
        "seconds": round(time.time() - started, 2)}
    return W, diagnostics


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def feature_ica(scores, components, *, contrast="jade", use_gpu=True, max_sweeps=600,
                threshold=None, schedule_seed=None, seed=0, max_iter=1000, tol=1e-7,
                w_init=None):
    """Feature-space ICA of a PCA span; returns activities, loadings and maps.

    Parameters
    ----------
    scores : (n_cells, r) PCA scores (any centring; centred internally)
    components : (r, N) PCA component rows
    contrast : ``"jade"`` or ``"picard"``
    use_gpu : run the cumulant and Jacobi stages on cupy when available
        (ignored by the Picard contrast, which is CPU only)
    max_sweeps, threshold, schedule_seed : JADE controls
    seed, max_iter, tol, w_init : Picard controls (``w_init`` warm-starts the
        optimiser from a rotation orthogonal in this whitening)

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
        W, diagnostics = picard_rotation(z, seed=seed, max_iter=max_iter, tol=tol,
                                         w_init=w_init)

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
