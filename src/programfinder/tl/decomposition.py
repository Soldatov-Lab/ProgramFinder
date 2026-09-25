"""Residual PCA, feature-space ICA and its reliability, on AnnData.

    pf.pp.residual_null(adata, model="bernoulli", depth_key="n_unique")
    pf.tl.pca(adata, n_comps=100)
    pf.tl.gica(adata, n_comps=50, contrast="jade")
    pf.tl.gica_stability(adata, schedules=range(1, 6), bootstraps=10)
    pf.tl.gica_dependence(adata)                     # one fit: which components form groups
    pf.tl.gica_stability(adata, bootstraps=50, groups="partners")   # do the groups come back

Results follow the scanpy layout: cell-side matrices in ``.obsm``, feature-side
matrices in ``.varm`` (features x components), parameters and diagnostics in
``.uns``.
"""

from __future__ import annotations

import time

import numpy as np

from .. import _dependence, _pca, _stability
from .._backend import _backend, resolve_device, to_host
from .._basis import CONTRASTS, feature_ica
from ..pp import residual_operator

__all__ = ["pca", "gica", "gica_stability", "gica_dependence"]


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
         max_iter=1000, tol=1e-7, copy=False):
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
                      schedule_seed=schedule_seed, seed=seed, max_iter=max_iter, tol=tol)
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
                   "schedule_seed": schedule_seed, "seed": int(seed),
                   "max_iter": int(max_iter), "tol": float(tol),
                   "device": "cpu" if contrast == "picard" else device},
    }
    return adata if copy else None


def gica_stability(adata, *, key="pf_gica", schedules=(1, 2, 3, 4, 5), bootstraps=0,
                   split_mask=None, split_schedules=(None, 1, 2), seed=0, device="auto",
                   min_support=100, stable_r=0.99, groups=None, group_gate=0.9,
                   second_partner_margin=0.02, keep_refits=None, copy=False):
    """Reliability of the stored gICA components; results in ``uns[key]["stability"]``.

    The optimiser-side check is chosen by the stored contrast: ``schedules``
    re-diagonalises the same cumulants under permuted Jacobi orderings for
    ``"jade"``, and is read as Picard restart seeds for ``"picard"``, which has
    a random start instead. Either way the result lands under
    ``stability["schedule"]`` or ``stability["restart"]`` and feeds the same
    ``stable`` mask. ``bootstraps`` refits on resampled features and
    ``split_mask`` (one boolean per feature, e.g. from ``alternating_blocks``)
    runs the fixed-whitening split-half check; both are JADE-only for now
    (see ``PLAN.md``). ``effective_support`` is always computed.
    Also written: ``adata.var`` is untouched; per-component arrays live in
    ``uns[key]["stability"]`` and a boolean ``stable`` mask combines the
    optimiser criterion (worst |r| > ``stable_r``) with ``min_support`` cells.

    Groups (needs ``bootstraps``): whether groups of components come back as a
    span across the feature-bootstrap refits, where their axes need not. Refits
    are matched to the STORED activities. ``groups`` is

    ``"partners"``
        for every supported component whose bootstrap worst |r| is below
        ``group_gate``: the component with its top one and top two partners
        from ``tl.gica_dependence`` (one fit). The pair is kept unless the
        triple recovers more than ``second_partner_margin`` better, so the
        partner IDENTITIES come from one fit and only their number from refits.
    ``"dependence"``
        the proposed groups stored by ``tl.gica_dependence``;
    a list of index lists
        groups supplied by the caller.

    Written to ``stability["groups"]``: one record per evaluated group (members,
    worst / median smallest canonical correlation over refits, refits below
    ``group_gate``) and a per-component ``verdict``: ``"axis"`` (the component
    itself comes back), ``"group"`` (its group's span comes back, its axis does
    not: report it as the group), ``"unresolved"``, or ``"low support"``.
    ``keep_refits`` (a ``.npy`` path) keeps the matched refit activities
    (bootstraps x cells x components, float32) on disk instead of discarding them.
    """
    adata = adata.copy() if copy else adata
    if key not in adata.uns:
        raise KeyError(f"adata.uns[{key!r}] not found; run tl.gica first")
    spec = adata.uns[key]
    contrast = spec["contrast"]
    if contrast not in CONTRASTS:
        raise ValueError(f"unknown stored contrast {contrast!r}")
    if groups is not None and not bootstraps:
        raise ValueError("groups are scored on feature-bootstrap refits; pass bootstraps > 0")
    if groups == "partners" or groups == "dependence":
        if "dependence" not in spec:
            raise KeyError(f"groups={groups!r} needs tl.gica_dependence(adata, key={key!r}) first")
    device = resolve_device(device)
    use_gpu = device == "gpu"
    params = spec["params"]
    scores, components = _span(adata, params["use_key"], params["n_comps"])
    common = dict(use_gpu=use_gpu, max_sweeps=int(params["max_sweeps"]),
                  threshold=params["threshold"])
    activities = np.asarray(adata.obsm[f"X_{key}"], np.float64)
    out = {"effective_support": _stability.effective_support(activities),
           "min_support": int(min_support), "stable_r": float(stable_r),
           "contrast": contrast}
    stable = out["effective_support"] >= min_support
    if schedules:
        if contrast == "jade":
            sched = _stability.schedule_stability(scores, components,
                                                  seeds=tuple(schedules), **common)
            out["schedule"] = sched
        else:
            sched = _stability.restart_stability(
                scores, components, seeds=tuple(schedules),
                base_seed=int(params.get("seed", 0)),
                max_iter=int(params.get("max_iter", 1000)),
                tol=float(params.get("tol", 1e-7)))
            out["restart"] = sched
        stable &= sched["worst_abs_r"] > stable_r
    if (bootstraps or split_mask is not None) and contrast != "jade":
        raise NotImplementedError(
            "the feature bootstrap and split-half checks are defined for the jade "
            "contrast only; see PLAN.md. Pass bootstraps=0 and split_mask=None for "
            f"contrast={contrast!r}.")
    if bootstraps:
        boot = _stability.feature_bootstrap_stability(
            scores, components, n_bootstraps=int(bootstraps), seed=seed,
            reference=activities, return_activities=groups is not None,
            out=keep_refits, **common)
        refits = boot.pop("refit_activities", None)
        if keep_refits is not None:
            boot["refit_activities_path"] = str(keep_refits)
        out["feature_bootstrap"] = boot
        if groups is not None:
            out["groups"] = _score_groups(activities, refits, boot["worst_abs_r"],
                                          out["effective_support"] >= min_support,
                                          spec.get("dependence"), groups, group_gate,
                                          second_partner_margin, use_gpu)
    if split_mask is not None:
        out["split_half"] = _stability.split_half_stability(
            scores, components, split_mask, schedules=tuple(split_schedules), **common)
    out["stable"] = stable
    adata.uns[key]["stability"] = out
    return adata if copy else None


def _score_groups(activities, refits, axis_worst, supported, dependence, groups, gate,
                  margin, use_gpu):
    r = activities.shape[1]
    recover = lambda gs: _stability.group_recovery(activities, refits, gs, gate=gate,
                                                   use_gpu=use_gpu)
    chosen = [None] * r
    records = []
    if groups == "partners":
        ranked = dependence["partners"]
        for c in range(r):
            if not supported[c] or axis_worst[c] >= gate:
                continue
            p = [int(x) for x in ranked[c] if x >= 0][:2]
            if not p:
                continue
            cand = [[c, p[0]]] + ([[c, p[0], p[1]]] if len(p) > 1 else [])
            rec = recover(cand)
            pick = rec[1] if len(rec) > 1 and rec[1]["worst"] > rec[0]["worst"] + margin else rec[0]
            pick = dict(pick, component=c, alone=float(axis_worst[c]),
                        with_one_partner=rec[0]["worst"],
                        with_two_partners=rec[1]["worst"] if len(rec) > 1 else None)
            records.append(pick)
            chosen[c] = pick
    else:
        gs = dependence["groups"] if groups == "dependence" else [list(map(int, g)) for g in groups]
        for rec in recover(gs) if gs else []:
            records.append(rec)
            for c in rec["members"]:
                if chosen[c] is None or rec["worst"] > chosen[c]["worst"]:
                    chosen[c] = rec
    verdict = []
    for c in range(r):
        if not supported[c]:
            verdict.append("low support")
        elif axis_worst[c] >= gate:
            verdict.append("axis")
        elif chosen[c] is not None and chosen[c]["worst"] >= gate:
            verdict.append("group")
        else:
            verdict.append("unresolved")
    return {"mode": groups if isinstance(groups, str) else "given", "gate": float(gate),
            "second_partner_margin": float(margin), "records": records,
            "group_of": [None if g is None else list(g["members"]) for g in chosen],
            "group_worst": np.array([np.nan if g is None else g["worst"] for g in chosen]),
            "group_below_gate": np.array([-1 if g is None else g["below_gate"] for g in chosen]),
            "verdict": np.array(verdict, dtype=object)}


def gica_dependence(adata, *, key="pf_gica", reps=100, seed=0, statistic="energy",
                    min_support=100, top_k=2, z_min=5.0, device="auto", chunk=20000,
                    copy=False):
    """Residual dependence between the stored components, from ONE fit.

    A JADE solution is a joint block-diagonaliser under an independent-subspace
    model (Theis 2006), so which components belong together can be read off the
    dependence the fit could not remove. For every pair: the residual
    cross-cumulant energy ``D`` and the energy correlation ``E`` of the
    whitened sources (the loadings; the ICA runs over features), each as a
    z-score against a per-source feature-permutation null of ``reps``
    replicates. See :mod:`programfinder._dependence`.

    Stored in ``uns[key]["dependence"]``: ``zD``, ``zE``, ``D``, ``E`` (K x K),
    ``partners`` (K x K int, each row the supported components ordered by the
    chosen ``statistic`` z, padded with -1), ``groups`` (mutual top-``top_k``
    partners with z >= ``z_min``; a PROPOSAL -- whether a group's span comes
    back is ``tl.gica_stability(..., groups=...)``), ``supported`` and the
    parameters. ``statistic`` is ``"energy"`` (``zE``, the default) or
    ``"cumulant"`` (``zD``).
    """
    if statistic not in ("energy", "cumulant"):
        raise ValueError(f"statistic must be 'energy' or 'cumulant', got {statistic!r}")
    adata = adata.copy() if copy else adata
    if key not in adata.uns:
        raise KeyError(f"adata.uns[{key!r}] not found; run tl.gica first")
    spec = adata.uns[key]
    params = spec["params"]
    _, components = _span(adata, params["use_key"], params["n_comps"])
    x = components - np.asarray(spec["feature_mean"], np.float64)[:, None]
    sources = np.asarray(spec["W"], np.float64) @ np.asarray(spec["K"], np.float64) @ x
    device = resolve_device(device)
    res = _dependence.dependence_z(sources, reps=reps, seed=seed, use_gpu=(device == "gpu"),
                                   chunk=chunk)
    supported = _stability.effective_support(np.asarray(adata.obsm[f"X_{key}"])) >= min_support
    z = res["zE"] if statistic == "energy" else res["zD"]
    ranked = _dependence.ranked_partners(z, supported)
    K = z.shape[0]
    partners = np.full((K, K), -1, np.int64)
    for a, row in enumerate(ranked):
        partners[a, :len(row)] = row
    spec["dependence"] = {
        "zD": res["zD"], "zE": res["zE"], "D": res["D"], "E": res["E"],
        "statistic": statistic, "partners": partners, "supported": supported,
        "groups": _dependence.mutual_top_groups(z, supported, k=top_k, z_min=z_min),
        "params": {"reps": int(reps), "seed": int(seed), "min_support": int(min_support),
                   "top_k": int(top_k), "z_min": float(z_min), "device": res["device"]},
    }
    return adata if copy else None
