"""Matrix-free, explicitly centred randomized PCA of a residual operator.

A range finder
run against ``operator.matmat`` does not centre for free, and an uncentred
sketch spends its first direction on the mean vector. With column means ``mu``
and ``Zc = Z - 1 mu^T``,

    Zc @ R   = operator.matmat(R)  - 1 (mu^T R)
    Zc^T @ B = operator.rmatmat(B) - mu (1^T B)

``CentredOperator`` implements both, ``column_means`` computes ``mu`` through
the operator's own transpose product, and ``centring_residual`` measures the
fraction of the column mean left behind (0 correct, 1 if the correction is
dropped, 2 if applied on the wrong side).
"""

from __future__ import annotations

import time

import numpy as np

from ._backend import _backend, to_host

__all__ = ["CentredOperator", "column_means", "centring_residual", "randomized_pca",
           "subspace_agreement", "score_invariant_frobenius", "block_probe_error",
           "materialize_feature_block"]


def column_means(operator, xp, n_blocks=64):
    """``mu_j = mean_i Z_ij`` through ``rmatmat``, summed in blocks.

    The two terms of ``Z^T 1`` (observed nonzeros and binned null) are each
    O(n) while their difference is small, and ``rmatmat`` works in float32.
    Summing ``n_blocks`` contiguous groups and adding those in float64 keeps
    the significant digits of the cancellation.
    """
    n_cells = operator.shape[0]
    n_blocks = int(min(max(n_blocks, 1), n_cells))
    edges = np.linspace(0, n_cells, n_blocks + 1).astype(np.int64)
    indicator = np.zeros((n_cells, n_blocks), np.float32)
    for block in range(n_blocks):
        indicator[edges[block]:edges[block + 1], block] = 1.0
    partial = np.asarray(to_host(operator.rmatmat(indicator), xp), np.float64)
    return xp.asarray(partial.sum(axis=1) / n_cells, dtype=xp.float32)


class CentredOperator:
    """``Zc = Z - 1 mu^T`` as a matrix-free operator.

    Only ``shape``, ``device``, ``matmat`` and ``rmatmat`` are provided, which
    is all the range finder touches.
    """

    def __init__(self, operator, mean=None):
        self._operator = operator
        self.device = getattr(operator, "device", "cpu")
        self.shape = operator.shape
        xp, _ = _backend(self.device)
        self._xp = xp
        self.mean = column_means(operator, xp) if mean is None else xp.asarray(mean)

    def matmat(self, right):
        xp = self._xp
        right = xp.asarray(right, dtype=xp.float32)
        out = xp.asarray(self._operator.matmat(right))
        return out - (self.mean[None, :] @ right)

    def rmatmat(self, left):
        xp = self._xp
        left = xp.asarray(left, dtype=xp.float32)
        out = xp.asarray(self._operator.rmatmat(left))
        return out - self.mean[:, None] * left.sum(axis=0)[None, :]


def centring_residual(centred, xp):
    """``||Zc^T 1|| / ||Z^T 1||``: 0 when centred, 1 without the correction."""
    ones = np.ones((centred.shape[0], 1), np.float32)
    residual = np.asarray(to_host(centred.rmatmat(ones), xp), np.float64).ravel()
    scale = float(np.linalg.norm(to_host(centred.mean, xp))) * centred.shape[0]
    return float(np.linalg.norm(residual) / max(scale, 1e-30))


def randomized_pca(centred, n_components, oversample=20, n_power=7, seed=0, xp=np,
                   total_variance=None):
    """Randomized PCA of a mean-centred operator.

    Gaussian sketch, QR, alternating power iterations with re-orthonormalisation,
    then the SVD of the projected matrix. The SVD of ``B = Q^T Zc`` (sketch x
    n_features) is taken through a QR of ``B^T`` and an SVD of the small
    triangular factor, so no factorisation of a wide array is ever requested.
    Explained variance is ``S**2 / (n - 1)``, exact for the recovered
    directions; the ratio needs ``total_variance`` from the caller.
    """
    n_cells, n_features = centred.shape
    sketch = int(min(n_components + oversample, n_cells, n_features))
    started = time.time()
    omega = np.random.default_rng(seed).standard_normal(
        (n_features, sketch)).astype(np.float32)
    q, _ = xp.linalg.qr(centred.matmat(omega), mode="reduced")
    del omega
    for _ in range(n_power):
        z, _ = xp.linalg.qr(centred.rmatmat(q), mode="reduced")
        q, _ = xp.linalg.qr(centred.matmat(z), mode="reduced")
        del z
    projected = centred.rmatmat(q)                         # (n_features, sketch)
    qp, r_small = xp.linalg.qr(projected, mode="reduced")
    del projected
    v_r, singular, u_rt = xp.linalg.svd(r_small, full_matrices=False)
    rank = int(min(n_components, singular.size))
    scores = (q @ u_rt.T[:, :rank]) * singular[:rank][None, :]
    components = (qp @ v_r[:, :rank]).T
    explained = np.asarray(to_host(singular[:rank], xp), np.float64) ** 2 / (n_cells - 1)
    del q, qp, v_r, u_rt, r_small
    return {
        "scores": np.ascontiguousarray(to_host(scores, xp), dtype=np.float32),
        "components": np.ascontiguousarray(to_host(components, xp), dtype=np.float32),
        "singular_values": np.asarray(to_host(singular[:rank], xp), np.float64),
        "sketch_singular_values": np.asarray(to_host(singular, xp), np.float64),
        "explained_variance": explained,
        "explained_variance_ratio": (explained / total_variance
                                     if total_variance else None),
        "sketch_rank": sketch,
        "n_power_iterations": int(n_power),
        "seed": int(seed),
        "runtime_s": float(time.time() - started),
    }


def _centred_unit(matrix):
    values = np.asarray(matrix, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(values, axis=0)
    return values / np.where(norms > 0, norms, 1.0)


def subspace_agreement(a, b):
    """Principal-angle summary between two score bases.

    Both the energy statement (mean squared cosine) and the worst-rotated
    direction (smallest canonical correlation) are reported, because a sketch
    can capture 99.9% of the energy while rotating one trailing direction badly.
    """
    qa = np.linalg.qr(_centred_unit(a))[0]
    qb = np.linalg.qr(_centred_unit(b))[0]
    cosines = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)
    captured = (qb.T @ qa) ** 2
    return {
        "n": int(cosines.size),
        "mean_squared_cosine": float((cosines ** 2).mean()),
        "min_canonical_correlation": float(cosines.min()),
        "n_below_0.99": int((cosines < 0.99).sum()),
        "n_below_0.90": int((cosines < 0.90).sum()),
        "per_direction_energy_captured": captured.sum(axis=0).tolist(),
        "min_per_direction_energy_captured": float(captured.sum(axis=0).min()),
    }


def score_invariant_frobenius(operator, components, mean, scores, xp):
    """``||S - (Z - mean) P^T||_F / ||(Z - mean) P^T||_F`` through the operator.

    For an exact decomposition this is a pure provenance check. For a sketch it
    also carries the sketch's own inexactness, so it is a measured quantity.
    """
    loadings = np.ascontiguousarray(np.asarray(components, np.float32).T)
    projected = np.asarray(to_host(operator.matmat(loadings), xp), np.float64)
    expected = projected - np.asarray(mean, np.float64) @ np.asarray(
        components, np.float64).T
    got = np.asarray(scores, np.float64)
    return float(np.linalg.norm(got - expected)
                 / max(float(np.linalg.norm(expected)), 1e-30))


def materialize_feature_block(operator, start, stop, xp):
    """One feature block of ``Z`` built the way ``matmat`` defines it."""
    operator._require_bins("materialize_feature_block")
    groups = xp.asarray(operator.bin_index)
    null = operator._residual_null_block(start, stop, xp)[groups]
    observed = operator._weighted_counts_host[:, start:stop].toarray()
    return xp.asarray(observed, dtype=xp.float32) - null


def block_probe_error(operator, xp, n_blocks=4, width=2048, n_probes=8, seed=0):
    """Provenance probe on feature blocks that fit in memory.

    ``Z[:, block] @ R_block`` equals ``operator.matmat(R_full)`` when ``R_full``
    is zero outside the block, so this checks that the matrix handed to the
    decomposition is the matrix the operator defines.
    """
    rng = np.random.default_rng(seed)
    n_features = operator.shape[1]
    width = int(min(width, n_features))
    starts = sorted(rng.choice(max(n_features - width, 1),
                               size=min(n_blocks, max(n_features // width, 1)),
                               replace=False).tolist())
    errors = []
    for start in starts:
        stop = min(start + width, n_features)
        block = materialize_feature_block(operator, start, stop, xp)
        probe = rng.standard_normal((stop - start, n_probes)).astype(np.float32)
        padded = np.zeros((n_features, n_probes), np.float32)
        padded[start:stop] = probe
        reference = np.asarray(to_host(operator.matmat(padded), xp), np.float64)
        got = np.asarray(to_host(block @ xp.asarray(probe), xp), np.float64)
        scale = max(float(np.abs(reference).max()), 1e-30)
        errors.append({"start": int(start), "stop": int(stop),
                       "relative_error": float(np.abs(got - reference).max() / scale)})
        del block
    return errors
