"""Residual PCA, feature-space ICA and its reliability, on AnnData.

    pf.pp.residual_null(adata, model="bernoulli", depth_key="n_unique")
    pf.tl.pca(adata, n_comps=100)
    pf.tl.gica(adata, n_comps=50, contrast="jade")
    pf.tl.gica_stability(adata, schedules=range(1, 6), bootstraps=10)

Results follow the scanpy layout: cell-side matrices in ``.obsm``, feature-side
matrices in ``.varm`` (features x components), parameters and diagnostics in
``.uns``.
"""

from __future__ import annotations

import time

import numpy as np

from .. import _pca, _stability
from .._backend import _backend, resolve_device, to_host
from .._basis import CONTRASTS, feature_ica
from ..pp import residual_operator

__all__ = ["pca", "gica", "gica_stability"]


# --------------------------------------------------------------------------
# residual PCA
# --------------------------------------------------------------------------

def pca(adata, n_comps=50, *, oversample=20, n_power=7, seed=0, use_key="pf_residual",
        key_added="pf_pca", device="auto", probe_blocks=4, probe_width=2048,
        check_seed=None, centring_tol=1e-3, copy=False):
    """Matrix-free centred randomized PCA of the recorded residual.

    Stores
    ------
    ``adata.obsm[f"X_{key_added}"]`` : (cells, n_comps) scores
    ``adata.varm[f"{key_added}_components"]`` : (features, n_comps) loadings
    ``adata.var[f"{key_added}_mean"]`` : column means of the residual
    ``adata.uns[key_added]`` : ``variance``, ``variance_ratio``,
        ``singular_values``, ``params`` and ``provenance`` (centring residual,
        feature-block probe errors, score invariant, and the agreement between
        two independent sketches when ``check_seed`` is given)

    ``probe_blocks=0`` skips the feature-block provenance probe (it needs the
    depth-binned null). ``check_seed`` runs a second sketch with another
    Gaussian seed and records how well the two agree, which bounds the sketch
    variance; the first sketch is the one stored.
    """
    adata = adata.copy() if copy else adata
    started = time.time()
    device = resolve_device(device)
    operator = residual_operator(adata, key=use_key, device=device)
    xp, _ = _backend(operator.device)
    n_cells = operator.shape[0]

    total_sumsq = float(np.asarray(to_host(operator.column_sumsq(), xp), np.float64).sum())
    centred = _pca.CentredOperator(operator)
    mean = np.asarray(to_host(centred.mean, xp), np.float64)
    total_variance = (total_sumsq - n_cells * float((mean ** 2).sum())) / (n_cells - 1)
    centring = _pca.centring_residual(centred, xp)
    if centring > centring_tol:
        raise FloatingPointError(
            f"explicit mean-centring left {centring:.2e} of the column mean behind; the "
            "PCA would spend a direction on the mean")
    provenance = {"centring_residual": centring, "total_centred_variance": total_variance}

    if probe_blocks and getattr(operator, "depth_bins", None) is not None:
        blocks = _pca.block_probe_error(operator, xp, n_blocks=probe_blocks,
                                        width=probe_width, seed=seed)
        provenance["block_probe_worst_relative_error"] = max(
            entry["relative_error"] for entry in blocks)
        provenance["block_probes"] = blocks

    result = _pca.randomized_pca(centred, n_comps, oversample=oversample, n_power=n_power,
                                 seed=seed, xp=xp, total_variance=total_variance)
    if check_seed is not None:
        other = _pca.randomized_pca(centred, n_comps, oversample=oversample,
                                    n_power=n_power, seed=check_seed, xp=xp,
                                    total_variance=total_variance)
        provenance["independent_sketch_agreement"] = _pca.subspace_agreement(
            result["scores"], other["scores"])
        del other
    provenance["score_invariant_frobenius_relative_error"] = _pca.score_invariant_frobenius(
        operator, result["components"], mean.astype(np.float32), result["scores"], xp)

    adata.obsm[f"X_{key_added}"] = result["scores"]
    adata.varm[f"{key_added}_components"] = np.ascontiguousarray(result["components"].T)
    adata.var[f"{key_added}_mean"] = mean
    adata.uns[key_added] = {
        "variance": result["explained_variance"],
        "variance_ratio": result["explained_variance_ratio"],
        "singular_values": result["singular_values"],
        "sketch_singular_values": result["sketch_singular_values"],
        "params": {"n_comps": int(result["scores"].shape[1]), "oversample": int(oversample),
                   "sketch_rank": result["sketch_rank"], "n_power": int(n_power),
                   "seed": int(seed), "use_key": use_key, "device": operator.device,
                   "method": "matrix-free randomized PCA, explicit mean-centring"},
        "provenance": provenance,
        "runtime_s": round(time.time() - started, 2),
    }
    if hasattr(xp, "get_default_memory_pool"):
        xp.get_default_memory_pool().free_all_blocks()
    return adata if copy else None


# --------------------------------------------------------------------------
# feature-space ICA
# --------------------------------------------------------------------------

def _span(adata, use_key, n_comps):
    scores_key = f"X_{use_key}"
    if scores_key not in adata.obsm or f"{use_key}_components" not in adata.varm:
        raise KeyError(f"no PCA under {use_key!r}; run tl.pca first")
    scores = np.asarray(adata.obsm[scores_key], np.float64)
    components = np.asarray(adata.varm[f"{use_key}_components"], np.float64).T
    available = scores.shape[1]
    n_comps = available if n_comps is None else int(n_comps)
    if n_comps > available:
        raise ValueError(f"n_comps={n_comps} exceeds the {available} stored components")
    return scores[:, :n_comps], components[:n_comps]


def gica(adata, n_comps=None, *, contrast="jade", use_key="pf_pca", key_added="pf_gica",
         device="auto", max_sweeps=600, threshold=None, schedule_seed=None, seed=0,
         copy=False):
    """Feature-space ICA of the leading ``n_comps`` residual PCs.

    For an exact ordered decomposition the leading prefix of a rank-100 PCA is
    the rank-``n_comps`` PCA, so one ``tl.pca`` at the largest rank serves a
    whole rank sweep. For the randomized path that nesting is a property of the
    sketch; check it with ``check_seed`` in ``tl.pca`` before relying on it.

    Stores
    ------
    ``adata.obsm[f"X_{key_added}"]`` : (cells, r) activities, unit SD, positive skew
    ``adata.varm[f"{key_added}_loadings"]`` : (features, r) loadings
    ``adata.uns[key_added]`` : ``contrast``, ``W``, ``K``, ``K_inv``,
        ``score_to_activity``, ``feature_to_source``, ``scale``, ``sign``,
        ``excess_kurtosis``, ``diagnostics``, ``params``
    """
    if contrast not in CONTRASTS:
        raise ValueError(f"contrast must be one of {CONTRASTS}, got {contrast!r}")
    adata = adata.copy() if copy else adata
    device = resolve_device(device)
    scores, components = _span(adata, use_key, n_comps)
    fit = feature_ica(scores, components, contrast=contrast, use_gpu=(device == "gpu"),
                      max_sweeps=max_sweeps, threshold=threshold,
                      schedule_seed=schedule_seed, seed=seed)
    adata.obsm[f"X_{key_added}"] = fit["activities"].astype(np.float32)
    adata.varm[f"{key_added}_loadings"] = np.ascontiguousarray(
        fit["loadings"].T.astype(np.float32))
    adata.uns[key_added] = {
        "contrast": contrast,
        "W": fit["W"], "K": fit["K"], "K_inv": fit["K_inv"],
        "feature_mean": fit["feature_mean"],
        "whitening_eigenvalues": fit["whitening_eigenvalues"],
        "scale": fit["scale"], "sign": fit["sign"],
        "score_to_activity": fit["score_to_activity"],
        "feature_to_source": fit["feature_to_source"],
        "excess_kurtosis": fit["excess_kurtosis"],
        "reconstruction_relative_error": fit["reconstruction_relative_error"],
        "diagnostics": fit["diagnostics"],
        "params": {"n_comps": int(components.shape[0]), "use_key": use_key,
                   "max_sweeps": int(max_sweeps), "threshold": threshold,
                   "schedule_seed": schedule_seed, "device": device},
    }
    return adata if copy else None


def gica_stability(adata, *, key="pf_gica", schedules=(1, 2, 3, 4, 5), bootstraps=0,
                   split_mask=None, split_schedules=(None, 1, 2), seed=0, device="auto",
                   min_support=100, stable_r=0.99, copy=False):
    """Reliability of the stored gICA components; results in ``uns[key]["stability"]``.

    ``schedules`` re-diagonalises the same cumulants under permuted Jacobi
    orderings; ``bootstraps`` refits on resampled features; ``split_mask``
    (one boolean per feature, e.g. from ``alternating_blocks``) runs the
    fixed-whitening split-half check. ``effective_support`` is always computed.
    Also written: ``adata.var`` is untouched; per-component arrays live in
    ``uns[key]["stability"]`` and a boolean ``stable`` mask combines the
    schedule criterion (worst |r| > ``stable_r``) with ``min_support`` cells.
    """
    adata = adata.copy() if copy else adata
    if key not in adata.uns:
        raise KeyError(f"adata.uns[{key!r}] not found; run tl.gica first")
    spec = adata.uns[key]
    if spec["contrast"] != "jade":
        raise NotImplementedError("stability checks are defined for the jade contrast")
    device = resolve_device(device)
    use_gpu = device == "gpu"
    params = spec["params"]
    scores, components = _span(adata, params["use_key"], params["n_comps"])
    common = dict(use_gpu=use_gpu, max_sweeps=int(params["max_sweeps"]),
                  threshold=params["threshold"])
    activities = np.asarray(adata.obsm[f"X_{key}"], np.float64)
    out = {"effective_support": _stability.effective_support(activities),
           "min_support": int(min_support), "stable_r": float(stable_r)}
    stable = out["effective_support"] >= min_support
    if schedules:
        sched = _stability.schedule_stability(scores, components, seeds=tuple(schedules),
                                              **common)
        out["schedule"] = sched
        stable &= sched["worst_abs_r"] > stable_r
    if bootstraps:
        out["feature_bootstrap"] = _stability.feature_bootstrap_stability(
            scores, components, n_bootstraps=int(bootstraps), seed=seed, **common)
    if split_mask is not None:
        out["split_half"] = _stability.split_half_stability(
            scores, components, split_mask, schedules=tuple(split_schedules), **common)
    out["stable"] = stable
    adata.uns[key]["stability"] = out
    return adata if copy else None
