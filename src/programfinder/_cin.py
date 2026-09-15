"""Chromosome-scale read-outs of feature-ICA loadings.

Three tools, all functions of the loadings alone:

``genomic_autocorr``
    the six-lag genomic autocorrelation profile of each component's
    standardised loadings, and the flatness screen: a copy-number axis shifts
    a whole block, so its loadings stay correlated at megabase separation,
    while a regulatory axis decays within a few kb. Lags are in BASE PAIRS,
    not peak index, because peak density varies about tenfold.
``chromosome_effects``
    the standardised mean loading effect of each chromosome (or arm):
    ``mean(region) / sd(all loadings)``. A descriptive contrast in units of the
    component's own loading spread. It is NOT a z-test -- peaks on one
    chromosome are strongly correlated, so ``n_peaks`` is not a number of
    independent observations.
``genome_profiles``
    winsorised, block-mean genomic binning of the loadings with a batch-means
    standard error, and two piecewise-constant fits by one ADMM solver on the
    same chain graph: per-component weighted total variation (``q=1``) and a
    joint multitrack group-TV in the whitened data metric (``q=2``). The
    regularisation strength is chosen by a held-out 100 kb block split of the
    features and a one-standard-error rule (leave-one-chromosome-out jackknife).

None of this is DNA evidence and none of it carries a p value. What comes out
is a signed effect with a span, a callable support and an approximate
uncertainty.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded

from . import _genome as G
from ._backend import array_module, to_host

__all__ = ["LAG_BINS", "genomic_autocorr", "chromosome_effects", "Binned", "winsorise",
           "bin_loadings", "chain_edges", "solve_chain_tv", "lambda_max",
           "segments_for_component", "one_se_rule", "activity_whitening", "block_split",
           "select_fraction", "genome_profiles", "LAM_FRACTIONS"]

#: genomic lag bins, bp: (0-2 kb, 2-5, 5-20, 20-50, 50-200, 200-1000 kb)
LAG_BINS = ((0, 2_000), (2_000, 5_000), (5_000, 20_000), (20_000, 50_000),
            (50_000, 200_000), (200_000, 1_000_000))
#: the original screen: last-lag over first-lag autocorrelation above this is "flat"
FLAT_RATIO = 0.5

#: binning defaults (bp) and the winsorisation, in robust sd units
BLOCK_BP = 100_000
BIN_BP = 1_000_000
WINSOR_SD = 4.0
MIN_BLOCK_PEAKS = 3
MIN_BLOCK_FRACTION = 0.25
MIN_BLOCKS = 3
#: adjacent callable bins are linked across a gap no larger than this; above the
#: largest centromere gap on purpose, so an arm event places its breakpoint
#: INSIDE the gap and a whole-chromosome event stays one segment
MAX_LINK_GAP_BP = 20_000_000
#: a fitted difference below this many loading-sd units is not a breakpoint
JUMP_TOL = 0.02
#: regularisation path, as fractions of the exact lambda_max
LAM_FRACTIONS = tuple(np.geomspace(1.0, 1.0 / 400.0, 14))
SPLIT_SEED = 20260912


# ============================================================ autocorrelation

def _pair_products(Z, left, right, chunk, xp):
    Zd = xp.asarray(Z, dtype=xp.float64)
    total = xp.zeros(Z.shape[0], dtype=xp.float64)
    for start in range(0, left.size, chunk):
        i = xp.asarray(left[start:start + chunk])
        j = xp.asarray(right[start:start + chunk])
        total += (Zd[:, i] * Zd[:, j]).sum(1)
        del i, j
    return np.asarray(to_host(total, xp), np.float64)


def genomic_autocorr(loadings, chrom, mid, *, bins=LAG_BINS, flat_ratio=FLAT_RATIO,
                     use_gpu=True, chunk=40_000_000):
    """Mean product of standardised loadings over same-chromosome pairs, per lag bin.

    Each component's loadings are z-scored across features (no chromosome-wise
    demeaning, matching the original screen). Returns ``profiles`` (components
    x bins), ``n_pairs`` per bin, the ``ratio`` of the last to the first bin
    and the ``flat`` mask ``ratio > flat_ratio``.
    """
    L = np.asarray(loadings, np.float64)
    Z = L - L.mean(1, keepdims=True)
    Z /= np.maximum(Z.std(1, keepdims=True), 1e-300)
    xp, _ = array_module(use_gpu)
    out, n_pairs = [], []
    for lo, hi in bins:
        left, right = G.pairs_within_distance(chrom, mid, lo, hi)
        total = _pair_products(Z, left, right, chunk, xp)
        out.append(total / max(left.size, 1))
        n_pairs.append(int(left.size))
    profiles = np.vstack(out).T
    ratio = profiles[:, -1] / np.maximum(profiles[:, 0], 1e-12)
    return {"profiles": profiles, "bins": [list(b) for b in bins], "n_pairs": n_pairs,
            "ratio": ratio, "flat": ratio > flat_ratio, "flat_ratio": float(flat_ratio)}


# ======================================================== chromosome effects

def chromosome_effects(loadings, chrom, mid, *, level="chrom", min_peaks=50,
                       order=G.CHROM_ORDER):
    """Standardised mean loading effect per chromosome or arm.

    ``effect[k, j] = mean(loadings[k, region j]) / sd(loadings[k])``; regions
    with fewer than ``min_peaks`` features are left at zero and flagged in
    ``n_peaks``. ``dominant`` is the region of largest |effect| per component.
    """
    if level not in ("chrom", "arm"):
        raise ValueError("level must be 'chrom' or 'arm'")
    L = np.asarray(loadings, np.float64)
    chrom = np.asarray(chrom).astype(str)
    mid = np.asarray(mid, np.int64)
    if level == "chrom":
        labels = chrom
    else:
        labels = np.array([c + G.arm_of_position(c, p) for c, p in zip(chrom, mid)])
    regions = []
    for c in G.genomic_order(chrom, order):
        regions.extend([r for r in ([c] if level == "chrom" else [c + "p", c + "q"])
                        if (labels == r).any()])
    sd = np.maximum(L.std(1), 1e-300)
    effect = np.zeros((L.shape[0], len(regions)))
    n_peaks = np.zeros(len(regions), int)
    for j, r in enumerate(regions):
        m = labels == r
        n_peaks[j] = int(m.sum())
        if n_peaks[j] >= min_peaks:
            effect[:, j] = L[:, m].mean(1) / sd
    dominant = np.array([regions[i] for i in np.abs(effect).argmax(1)])
    return {"effect": effect, "regions": regions, "n_peaks": n_peaks, "level": level,
            "dominant": dominant, "dominant_effect": effect[np.arange(L.shape[0]),
                                                            np.abs(effect).argmax(1)],
            "min_peaks": int(min_peaks)}


# ================================================================== binning

@dataclass
class Binned:
    """A binned, genomically ordered loading representation with uncertainty."""
    y: np.ndarray                  # bins x K, robust signed location, sd units
    se: np.ndarray                 # bins x K, batch-means standard error
    weight: np.ndarray             # bins, scalar precision used by both fits
    bin_chrom: np.ndarray
    bin_index: np.ndarray          # bin ordinal within its chromosome
    start_bp: np.ndarray
    end_bp: np.ndarray
    arm: np.ndarray                # 'p', 'q' or 'straddles'
    n_peaks: np.ndarray
    n_blocks: np.ndarray           # callable sub-windows -- the effective n
    support_bp: np.ndarray         # callable genomic span inside the bin
    bin_bp: int
    block_bp: int
    winsor_fraction: np.ndarray    # K, share of peaks winsorised
    loading_sd: np.ndarray         # K, robust sd used to standardise
    dropped: list = field(default_factory=list)

    @property
    def n_bins(self):
        return len(self.bin_chrom)

    def subset(self, idx):
        out = copy.copy(self)
        for name in ("y", "se", "weight", "bin_chrom", "bin_index", "start_bp", "end_bp",
                     "arm", "n_peaks", "n_blocks", "support_bp"):
            setattr(out, name, getattr(self, name)[idx])
        return out

    def as_dict(self):
        return {name: getattr(self, name) for name in
                ("y", "se", "weight", "bin_chrom", "bin_index", "start_bp", "end_bp", "arm",
                 "n_peaks", "n_blocks", "support_bp", "bin_bp", "block_bp",
                 "winsor_fraction", "loading_sd")}


def _robust_sd(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def winsorise(L, n_sd=WINSOR_SD):
    """Clip each component's loadings at +-n_sd robust sd. Linear afterwards."""
    L = np.asarray(L, np.float64)
    centre = np.median(L, axis=1, keepdims=True)
    sd = np.array([[_robust_sd(row)] for row in L])
    lo, hi = centre - n_sd * sd, centre + n_sd * sd
    clipped = np.clip(L, lo, hi)
    fraction = ((L < lo) | (L > hi)).mean(1)
    return clipped, fraction, sd.ravel()


def _lengths_bp(chroms, chrom, mid, lengths):
    out = {}
    for c in chroms:
        if lengths is not None and c in lengths:
            out[c] = int(lengths[c] * 1e6)
        elif c in G.HG38_LENGTH_MB:
            out[c] = int(G.HG38_LENGTH_MB[c] * 1e6)
        else:
            out[c] = int(mid[chrom == c].max()) + 1
    return out


def bin_loadings(loadings, chrom, mid, *, bin_bp=BIN_BP, block_bp=BLOCK_BP,
                 winsor_sd=WINSOR_SD, min_block_peaks=MIN_BLOCK_PEAKS, peak_subset=None,
                 min_block_fraction=MIN_BLOCK_FRACTION, min_blocks=MIN_BLOCKS,
                 chroms=None, lengths=None, standardise=True, exclude=("chrY",)):
    """Bin the loadings genomically, with a batch-means standard error.

    * each component's loadings are winsorised at +-``winsor_sd`` robust sd;
    * features are grouped into fixed ``block_bp`` sub-windows; a sub-window
      with fewer than ``min_block_peaks`` features is not callable;
    * the BIN estimate is the unweighted mean over its callable sub-windows of
      the sub-window mean -- equal weight per unit of genome, not per feature;
    * the standard error is ``sd(block means) / sqrt(n_blocks)``: the sub-window
      is the resampling unit, adjacent features are not independent replicates.
      Correlation longer than ``block_bp`` is not captured;
    * values are divided by the component's robust loading sd, so one lambda is
      meaningful across components and consistent rescaling is a no-op.

    ``lengths`` maps chromosome -> length in Mb (hg38 by default; a chromosome
    absent from both falls back to its last feature). ``peak_subset`` selects a
    boolean subset of features for the held-out split.
    """
    L = np.asarray(loadings, np.float64)
    chrom = np.asarray(chrom).astype(str)
    mid = np.asarray(mid, np.int64)
    keep = np.ones(L.shape[1], bool) if peak_subset is None else np.asarray(peak_subset, bool)
    Lw, winsor_fraction, robust_sd = winsorise(L, winsor_sd)
    if standardise:
        Lw = Lw / np.maximum(robust_sd[:, None], 1e-300)
    else:
        robust_sd = np.ones_like(robust_sd)
    if chroms is None:
        chroms = [c for c in G.genomic_order(chrom) if c not in exclude]
    length_bp = _lengths_bp(chroms, chrom, mid, lengths)
    per_bin_blocks = max(1, bin_bp // block_bp)
    need_blocks = max(min_blocks, int(np.ceil(min_block_fraction * per_bin_blocks)))

    rows, ys, ses, dropped = [], [], [], []
    for name in chroms:
        sel = np.flatnonzero((chrom == name) & keep)
        if sel.size == 0:
            continue
        n_bins = int(np.ceil(length_bp[name] / bin_bp))
        block_id = mid[sel] // block_bp
        order = np.argsort(block_id, kind="stable")
        sel, block_id = sel[order], block_id[order]
        uniq, first, counts = np.unique(block_id, return_index=True, return_counts=True)
        means = np.add.reduceat(Lw[:, sel], first, axis=1) / counts[None, :]
        ok = counts >= min_block_peaks
        uniq, means, counts = uniq[ok], means[:, ok], counts[ok]
        cen_bin = G.centromere_window(name, bin_bp)
        for b in range(n_bins):
            lo, hi = b * bin_bp, min((b + 1) * bin_bp, length_bp[name])
            inside = (uniq >= b * bin_bp // block_bp) & (uniq < (b + 1) * bin_bp // block_bp)
            nb = int(inside.sum())
            npk = int(counts[inside].sum())
            if nb < need_blocks:
                dropped.append(dict(chrom=name, bin=b, start_bp=int(lo), end_bp=int(hi),
                                    n_blocks=nb, n_peaks=npk))
                continue
            block_means = means[:, inside]
            ys.append(block_means.mean(1))
            ses.append(block_means.std(1, ddof=1) / np.sqrt(nb))
            arm = "straddles" if (cen_bin is not None and b == cen_bin) \
                else G.arm_of_position(name, 0.5 * (lo + hi))
            rows.append(dict(chrom=name, bin=b, start_bp=int(lo), end_bp=int(hi), arm=arm,
                             n_peaks=npk, n_blocks=nb,
                             support_bp=int(min(nb * block_bp, hi - lo))))
    if not rows:
        raise ValueError("no callable bins; check bin_bp/block_bp against the feature density")
    y, se = np.vstack(ys), np.vstack(ses)
    # one scalar weight per bin for BOTH fits, so the comparison is about the
    # penalty and nothing else; per-component SEs stay in `se`
    var = np.maximum(np.mean(se ** 2, axis=1), 1e-12)
    weight = 1.0 / var
    weight = weight / np.median(weight)
    return Binned(
        y=y, se=se, weight=weight,
        bin_chrom=np.array([r["chrom"] for r in rows]),
        bin_index=np.array([r["bin"] for r in rows]),
        start_bp=np.array([r["start_bp"] for r in rows]),
        end_bp=np.array([r["end_bp"] for r in rows]),
        arm=np.array([r["arm"] for r in rows]),
        n_peaks=np.array([r["n_peaks"] for r in rows]),
        n_blocks=np.array([r["n_blocks"] for r in rows]),
        support_bp=np.array([r["support_bp"] for r in rows]),
        bin_bp=int(bin_bp), block_bp=int(block_bp),
        winsor_fraction=winsor_fraction, loading_sd=robust_sd, dropped=dropped)


def chain_edges(binned, max_gap_bp=MAX_LINK_GAP_BP):
    """Adjacent-bin pairs the penalty may act on: never across a chromosome, and
    across an uncallable gap only while the gap is under ``max_gap_bp``."""
    left, right, gap = [], [], []
    for i in range(binned.n_bins - 1):
        if binned.bin_chrom[i] != binned.bin_chrom[i + 1]:
            continue
        hole = int(binned.start_bp[i + 1] - binned.end_bp[i])
        if hole > max_gap_bp:
            continue
        left.append(i)
        right.append(i + 1)
        gap.append(hole)
    return np.array(left, int), np.array(right, int), np.array(gap, int)


# =================================================================== solver

def _linked_mask(n_bins, left, right):
    linked = np.zeros(max(n_bins - 1, 0), bool)
    for a, b in zip(np.asarray(left, int), np.asarray(right, int)):
        if b != a + 1:
            raise ValueError("chain edges must only link consecutive bins")
        linked[a] = True
    return linked


def _dual_norm_project(U, lam, q):
    if q == 1:
        return np.clip(U, -lam, lam)
    norm = np.linalg.norm(U, axis=1, keepdims=True)
    return U * np.minimum(1.0, lam / np.maximum(norm, 1e-300))


def _primal(S, Y, w, linked, lam, q):
    fit = 0.5 * float((w[:, None] * (S - Y) ** 2).sum())
    d = np.diff(S, axis=0)[linked]
    pen = float(np.abs(d).sum()) if q == 1 else float(np.linalg.norm(d, axis=1).sum())
    return fit + lam * pen


def _adjoint(V, out):
    out[...] = 0.0
    out[1:] += V
    out[:-1] -= V
    return out


def _factor(w, linked, rho):
    B = len(w)
    diag = w.astype(float).copy()
    off = -rho * linked.astype(float)
    diag[:-1] += rho * linked
    diag[1:] += rho * linked
    ab = np.zeros((2, B))
    ab[0, 1:] = off
    ab[1, :] = diag
    return cholesky_banded(ab, lower=False)


def _prox(Z, threshold, q):
    if q == 1:
        return np.sign(Z) * np.maximum(np.abs(Z) - threshold, 0.0)
    norm = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z * np.maximum(0.0, 1.0 - threshold / np.maximum(norm, 1e-300))


def solve_chain_tv(Y, w, left, right, lam, q=1, max_iter=4000, tol=1e-8, rho=None, warm=None):
    """Weighted TV (``q=1``) or group-TV (``q=2``) on a chain graph, by ADMM.

        minimise (1/2) sum_b w_b ||S_b - Y_b||^2 + lam * sum_e ||S_r(e) - S_l(e)||_q

    The ``S`` update is an exact banded Cholesky solve; ``rho`` is adapted
    from the residual balance. The returned ``relative_gap`` is the Fenchel
    duality gap of the final primal point against a projected dual point, so it
    bounds the suboptimality of the objective the fits are compared on.
    ``warm`` carries ``(V, M, rho)`` from a neighbouring lambda.
    """
    Y = np.asarray(Y, np.float64)
    w = np.asarray(w, np.float64)
    B, K = Y.shape
    linked = _linked_mask(B, left, right)
    if not linked.any():
        return Y.copy(), dict(iterations=0, relative_gap=0.0, converged=True, warm=None)
    mask = linked[:, None]
    if warm is not None:
        V, M = warm[0].copy(), warm[1].copy()
        rho = warm[2] if rho is None else rho
    else:
        V, M = np.zeros((B - 1, K)), np.zeros((B - 1, K))
        rho = float(np.median(w)) if rho is None else rho
    chol = _factor(w, linked, rho)
    scratch = np.empty((B, K))
    wy = w[:, None] * Y
    S = Y.copy()
    for it in range(max_iter):
        rhs = wy + rho * _adjoint((V - M) * mask, scratch.copy())
        S = cho_solve_banded((chol, False), rhs)
        DS = np.diff(S, axis=0) * mask
        V_old = V
        V = _prox(DS + M, lam / rho, q) * mask
        M = (M + DS - V) * mask
        if (it + 1) % 50 == 0:
            r = np.linalg.norm(DS - V)
            sdual = rho * np.linalg.norm(_adjoint((V - V_old) * mask, np.empty((B, K))))
            scale = max(np.linalg.norm(DS), np.linalg.norm(V), 1e-12)
            if r / scale < 1e-7 and sdual / max(np.linalg.norm(wy), 1e-12) < 1e-7:
                break
            if r > 10 * sdual:
                rho, chol, M = rho * 2.0, None, M * 0.5
            elif sdual > 10 * r:
                rho, chol, M = rho / 2.0, None, M * 2.0
            if chol is None:
                chol = _factor(w, linked, rho)
    U = _dual_norm_project(rho * M, lam, q) * mask
    primal = _primal(S, Y, w, linked, lam, q)
    DtU = _adjoint(U, np.empty((B, K)))
    dual = (-0.5 * float(((1.0 / w)[:, None] * DtU ** 2).sum())
            + float((U[linked] * np.diff(Y, axis=0)[linked]).sum()))
    gap = (primal - dual) / max(abs(primal), 1e-12)
    return S, dict(iterations=it + 1, relative_gap=float(gap), rho=float(rho),
                   converged=bool(gap < tol * 100 or gap < 1e-6), warm=(V, M, rho))


def lambda_max(Y, w, left, right, q=1):
    """Smallest lam whose solution is constant on every chain. Exact for a chain."""
    Y = np.asarray(Y, np.float64)
    w = np.asarray(w, np.float64)
    B = Y.shape[0]
    linked = np.zeros(max(B - 1, 0), bool)
    for a, b in zip(left, right):
        if b == a + 1:
            linked[a] = True
    best, start = 0.0, 0
    for i in range(B):
        if not ((i == B - 1) or (not linked[i])):
            continue
        sl = slice(start, i + 1)
        ww, yy = w[sl], Y[sl]
        if len(ww) > 1:
            mean = (ww[:, None] * yy).sum(0) / ww.sum()
            partial = np.cumsum(ww[:, None] * (yy - mean), axis=0)[:-1]
            norm = np.abs(partial).max() if q == 1 else np.linalg.norm(partial, axis=1).max()
            best = max(best, float(norm))
        start = i + 1
    return best


def segments_for_component(values, left, right, tol=JUMP_TOL):
    """Maximal runs of bins the fit holds at one level; a chain break ends a segment."""
    B = len(values)
    linked = np.zeros(max(B - 1, 0), bool)
    for a, b in zip(left, right):
        if b == a + 1:
            linked[a] = True
    bounds, start = [], 0
    for i in range(B):
        if (i == B - 1) or (not linked[i]) or (abs(values[i + 1] - values[i]) > tol):
            bounds.append((start, i))
            start = i + 1
    return bounds


def one_se_rule(errors, ses):
    """Largest lam (simplest model) whose held-out error is within 1 SE of the best.
    ``errors`` runs from the LARGEST lam to the smallest."""
    errors = np.asarray(errors, float)
    best = int(np.nanargmin(errors))
    threshold = errors[best] + ses[best]
    for i in range(best + 1):
        if errors[i] <= threshold:
            return i, best
    return best, best


def activity_whitening(activities, loading_sd):
    """``G`` with ``G G^T = C``, the covariance of the scale-matched activities.

    ``loading_sd`` MUST be the per-component scale ``bin_loadings`` divided the
    loadings by: the binned signal and the activities are two halves of one
    reconstruction ``A L``, and the joint metric is only invariant to component
    rescaling when the same factor is put back on the activity side.
    """
    A = np.asarray(activities, np.float64) * np.asarray(loading_sd, np.float64)[None, :]
    A = A - A.mean(0)
    C = (A.T @ A) / (len(A) - 1)
    return np.linalg.cholesky(C), C


# ======================================================== lambda selection

def block_split(chrom, mid, block_bp=BLOCK_BP, seed=SPLIT_SEED):
    """Two feature halves split by ``block_bp`` BLOCK, not by feature.

    Splitting per feature would leave both halves carrying the same local
    neighbourhood, so the held-out curve would prefer under-smoothing. This is
    a transductive feature split of a full-data ICA, not independent validation.
    """
    key = np.array([f"{c}:{m // block_bp}" for c, m in zip(np.asarray(chrom).astype(str),
                                                             np.asarray(mid, np.int64))])
    uniq, inv = np.unique(key, return_inverse=True)
    side = np.random.default_rng(seed).integers(0, 2, len(uniq))
    return side[inv] == 0, side[inv] == 1


def _align_bins(a, b):
    ka = np.array([f"{c}:{i}" for c, i in zip(a.bin_chrom, a.bin_index)])
    kb = np.array([f"{c}:{i}" for c, i in zip(b.bin_chrom, b.bin_index)])
    common = np.intersect1d(ka, kb)
    pa = {k: i for i, k in enumerate(ka)}
    pb = {k: i for i, k in enumerate(kb)}
    return (np.array([pa[k] for k in common], int), np.array([pb[k] for k in common], int))


def _path_errors(Y_tr, Y_te, w, left, right, chrom, q, Gw, fractions, **solver):
    """Regularisation path on the training half, scored on the held-out half in
    the whitened data metric for both methods."""
    Yq = Y_tr @ Gw if q == 2 else Y_tr
    lmax = lambda_max(Yq, w, left, right, q=q)
    chroms = np.unique(chrom)
    Ginv = np.linalg.inv(Gw)
    errors, dofs, per_chrom, warm = [], [], [], None
    for frac in fractions:
        S, info = solve_chain_tv(Yq, w, left, right, lmax * frac, q=q, warm=warm, **solver)
        warm = info["warm"]
        S_ic = S @ Ginv if q == 2 else S
        per_bin = (w[:, None] * (((S_ic - Y_te) @ Gw) ** 2)).sum(1)
        errors.append(float(per_bin.sum() / w.sum() / Y_te.shape[1]))
        per_chrom.append([float(per_bin[chrom == c].sum() / w[chrom == c].sum()
                                / Y_te.shape[1]) for c in chroms])
        dofs.append(int(sum(len(segments_for_component(S_ic[:, k], left, right))
                            for k in range(S_ic.shape[1]))))
    return dict(lam_max=float(lmax), errors=errors, dof=dofs,
                per_chrom=np.array(per_chrom), fractions=[float(f) for f in fractions])


def select_fraction(loadings, activities, chrom, mid, q, *, fractions=LAM_FRACTIONS,
                    seed=SPLIT_SEED, geometry=None, **solver):
    """Held-out block split + one-SE rule -> the regularisation fraction for ``q``.

    The 1-SE band is a leave-one-chromosome-out jackknife of the held-out
    error: chromosomes are the coarsest block of genomic dependence, so a
    bin-level SE would understate the spread.
    """
    geometry = dict(geometry or {})
    geometry.update(min_blocks=2, min_block_fraction=0.15)
    side_a, side_b = block_split(chrom, mid, geometry.get("block_bp", BLOCK_BP), seed)
    ba = bin_loadings(loadings, chrom, mid, peak_subset=side_a, **geometry)
    bb = bin_loadings(loadings, chrom, mid, peak_subset=side_b, **geometry)
    ia, ib = _align_bins(ba, bb)
    A, Bh = ba.subset(ia), bb.subset(ib)
    la, ra, _ = chain_edges(A)
    w = np.minimum(A.weight, Bh.weight)
    Gw, _ = activity_whitening(activities, ba.loading_sd)
    res = _path_errors(A.y, Bh.y, w, la, ra, A.bin_chrom, q, Gw, fractions, **solver)
    err = np.array(res["errors"])
    pc = res["per_chrom"]
    best0 = int(np.nanargmin(err))
    n_c = pc.shape[1]
    jack = np.array([np.nanmean(np.delete(pc[best0], j)) for j in range(n_c)])
    se = float(np.sqrt((n_c - 1) / n_c * np.sum((jack - jack.mean()) ** 2)))
    pick, best = one_se_rule(err, np.full(len(err), se))
    return float(fractions[pick]), dict(res, per_chrom=pc.tolist(), one_se=se,
                                        picked=int(pick), best=int(best),
                                        picked_fraction=float(fractions[pick]),
                                        best_fraction=float(fractions[best]),
                                        n_common_bins=int(A.n_bins), split_seed=int(seed))


# =================================================================== driver

def genome_profiles(loadings, activities, chrom, mid, *, bin_bp=BIN_BP, block_bp=BLOCK_BP,
                    winsor_sd=WINSOR_SD, lengths=None, chroms=None, joint=True,
                    fraction=None, fractions=LAM_FRACTIONS, seed=SPLIT_SEED,
                    max_gap_bp=MAX_LINK_GAP_BP, max_iter=4000, verbose=False):
    """Binned genome profiles of every component with per-IC TV and joint group-TV fits.

    ``fraction`` fixes the regularisation fraction of lambda_max (a float for
    both fits, or ``{"per_ic": f1, "joint": f2}``); ``None`` selects it by the
    held-out block split and one-SE rule, separately for each fit. Returns a
    dict with the binned representation, ``fit_per_ic``, ``fit_joint`` (bins x
    K, in component coordinates), the chain edges and the selection record.
    """
    geometry = dict(bin_bp=bin_bp, block_bp=block_bp, winsor_sd=winsor_sd,
                    lengths=lengths, chroms=chroms)
    full = bin_loadings(loadings, chrom, mid, **geometry)
    left, right, gap = chain_edges(full, max_gap_bp)
    Gw, C = activity_whitening(activities, full.loading_sd)
    Ginv = np.linalg.inv(Gw)
    methods = [(1, "per_ic")] + ([(2, "joint")] if joint else [])
    chosen, selection = {}, {}
    for q, name in methods:
        if fraction is None:
            frac, record = select_fraction(loadings, activities, chrom, mid, q,
                                           fractions=fractions, seed=seed, geometry=geometry,
                                           max_iter=max_iter)
            selection[name] = record
        else:
            frac = float(fraction[name] if isinstance(fraction, dict) else fraction)
        chosen[name] = frac
        if verbose:
            print(f"[genome_profiles] {name}: fraction {frac:.4f}", flush=True)
    fits, info = {}, {}
    for q, name in methods:
        Yq = full.y @ Gw if q == 2 else full.y
        lmax = lambda_max(Yq, full.weight, left, right, q=q)
        S, meta = solve_chain_tv(Yq, full.weight, left, right, lmax * chosen[name], q=q,
                                 max_iter=max_iter)
        fits[name] = S @ Ginv if q == 2 else S
        info[name] = dict(lam=float(lmax * chosen[name]), lam_max=float(lmax),
                          fraction=chosen[name], iterations=meta["iterations"],
                          relative_gap=meta["relative_gap"], converged=meta["converged"])
    out = full.as_dict()
    out.update(fit_per_ic=fits["per_ic"], fit_joint=fits.get("joint"),
               edges_left=left, edges_right=right, edges_gap=gap,
               n_dropped=len(full.dropped), dropped=full.dropped, fits=info,
               selection=selection, whitening_condition=float(np.linalg.cond(C)),
               params=dict(bin_bp=int(bin_bp), block_bp=int(block_bp), winsor_sd=float(winsor_sd),
                           max_gap_bp=int(max_gap_bp), jump_tol=JUMP_TOL, split_seed=int(seed)))
    return out
