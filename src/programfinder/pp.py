"""Preprocessing: the residual null model of an AnnData count matrix.

``pp.residual_null`` fits the feature-wise null once and records its
parameters on the AnnData, so the matrix-free residual operator can be rebuilt
deterministically by ``pp.residual_operator`` (or by ``tl.pca``) without a
second fit and without ever storing a dense cells x features matrix.

Where things land (default ``key="pf_residual"``):

- ``adata.uns[key]``: model, depth binning, block size, layer, depth source;
- ``adata.var[f"{key}_baseline"]`` / ``adata.var[f"{key}_dispersion"]`` (NB);
- ``adata.var[f"{key}_detection_rate"]`` (Bernoulli);
- ``adata.obs[f"{key}_depth"]``: the per-cell size factor (NB) or depth
  (Bernoulli) the null was fitted with.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ._backend import resolve_device
from ._residual import BernoulliResidualOperator, NBResidualOperator

__all__ = ["residual_null", "residual_operator", "MODELS"]

MODELS = ("nb", "bernoulli")


def _counts(adata, layer):
    X = adata.layers[layer] if layer is not None else adata.X
    if X is None:
        raise ValueError("adata has no matrix at the requested layer")
    if not sparse.issparse(X):
        X = sparse.csr_matrix(np.asarray(X))
    return X


def _depth(adata, depth_key, counts, model):
    if depth_key is None:
        if model == "bernoulli":
            raise ValueError(
                "the Bernoulli null needs the genome-wide per-cell depth (e.g. fragment "
                "count) from adata.obs; pass depth_key=. Row sums of a filtered peak "
                "matrix are not the depth of the modality.")
        depth = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    else:
        depth = np.asarray(adata.obs[depth_key], np.float64)
    if depth.shape != (adata.n_obs,):
        raise ValueError("depth must have one value per cell")
    return depth


def residual_null(adata, *, model="nb", layer=None, depth_key=None, depth_bins=256,
                  block_size=512, device="auto", key="pf_residual", copy=False):
    """Fit and record the feature-wise null of the count matrix.

    Parameters
    ----------
    adata : AnnData with raw counts in ``.X`` or ``.layers[layer]``
    model : ``"nb"`` (UMI counts) or ``"bernoulli"`` (detections, e.g. ATAC)
    layer : layer holding the counts; ``None`` uses ``.X``
    depth_key : ``adata.obs`` column with the per-cell depth. For NB the
        default is the row sum, turned into a size factor. For Bernoulli it is
        required and must be the depth of the whole modality.
    depth_bins : number of equal-count depth bins for the binned null, or
        ``None`` for the exact blockwise evaluation (no PCA possible then)
    block_size : features densified per block on the compute device
    device : ``"auto"``, ``"cpu"`` or ``"gpu"``
    key : name under which the null is recorded
    copy : return a modified copy instead of editing in place

    Returns the AnnData when ``copy=True``, else ``None``.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    adata = adata.copy() if copy else adata
    device = resolve_device(device)
    counts = _counts(adata, layer)
    depth = _depth(adata, depth_key, counts, model)
    if model == "nb":
        size_factors = depth / depth.mean()
        operator = NBResidualOperator(counts, size_factors=size_factors, depth_bins=depth_bins,
                                      block_size=block_size, device=device)
        adata.var[f"{key}_baseline"] = operator.baseline
        adata.var[f"{key}_dispersion"] = operator.dispersion
        adata.obs[f"{key}_depth"] = size_factors
    else:
        operator = BernoulliResidualOperator(counts, depths=depth, depth_bins=depth_bins,
                                             block_size=block_size, device=device)
        adata.var[f"{key}_detection_rate"] = operator.detection_rate
        adata.obs[f"{key}_depth"] = depth
    adata.uns[key] = {
        "model": model,
        "layer": layer,
        "depth_key": depth_key,
        "depth_bins": depth_bins,
        "block_size": int(block_size),
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "effect_link": operator.effect_link,
    }
    return adata if copy else None


def residual_operator(adata, *, key="pf_residual", device="auto"):
    """Rebuild the matrix-free residual operator recorded by ``residual_null``.

    The null parameters are read back from ``.var`` and ``.obs``, so the
    operator is identical to the one fitted, on whichever device is asked for.
    """
    if key not in adata.uns:
        raise KeyError(f"adata.uns[{key!r}] not found; run pp.residual_null first")
    spec = adata.uns[key]
    if (int(spec["n_cells"]), int(spec["n_features"])) != adata.shape:
        raise ValueError(
            f"the null was fitted on {spec['n_cells']} x {spec['n_features']} but adata is "
            f"{adata.n_obs} x {adata.n_vars}; refit with pp.residual_null after subsetting")
    device = resolve_device(device)
    counts = _counts(adata, spec["layer"])
    depth = np.asarray(adata.obs[f"{key}_depth"], np.float64)
    depth_bins = spec["depth_bins"]
    depth_bins = None if depth_bins is None else int(depth_bins)
    if spec["model"] == "nb":
        return NBResidualOperator(
            counts, size_factors=depth, depth_bins=depth_bins,
            block_size=int(spec["block_size"]), device=device,
            baseline=np.asarray(adata.var[f"{key}_baseline"], np.float64),
            dispersion=np.asarray(adata.var[f"{key}_dispersion"], np.float64))
    return BernoulliResidualOperator(
        counts, depths=depth, depth_bins=depth_bins,
        block_size=int(spec["block_size"]), device=device,
        detection_rate=np.asarray(adata.var[f"{key}_detection_rate"], np.float64))
