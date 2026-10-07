"""Chromosome-scale read-outs of the gICA loadings, on AnnData.

    pf.pp.feature_coordinates(adata)                 # var["pf_chrom"], var["pf_mid"] from var_names
    pf.tl.genomic_autocorr(adata)                    # six-lag autocorrelation + flatness screen
    pf.tl.chromosome_effects(adata, level="arm")     # standardised mean loading effect per region
    pf.tl.copy_number_routes(adata)                  # four dosage patterns, gated, and the call
    pf.tl.genome_profiles(adata)                     # binned profiles with TV / group-TV fits

Everything lands in ``adata.uns[key]`` next to the gICA fit; the plotting
counterparts are in ``pf.pl``.
"""

from __future__ import annotations

import numpy as np

from .. import _cin
from .._backend import resolve_device

__all__ = ["genomic_autocorr", "chromosome_effects", "copy_number_routes", "genome_profiles"]


def _coordinates(adata):
    if "pf_chrom" not in adata.var or "pf_mid" not in adata.var:
        raise KeyError("no feature coordinates in adata.var; run pp.feature_coordinates first")
    return (np.asarray(adata.var["pf_chrom"]).astype(str),
            np.asarray(adata.var["pf_mid"], np.int64))


def _loadings(adata, key):
    if key not in adata.uns or f"{key}_loadings" not in adata.varm:
        raise KeyError(f"no gICA under {key!r}; run tl.gica first")
    return np.asarray(adata.varm[f"{key}_loadings"], np.float64).T      # K x features


def genomic_autocorr(adata, key="pf_gica", *, bins=_cin.LAG_BINS, flat_ratio=_cin.FLAT_RATIO,
                     device="auto", copy=False):
    """Six-lag genomic autocorrelation of each component's loadings and the flatness screen.

    Stores ``adata.uns[key]["genomic_autocorr"]`` with ``profiles`` (components
    x lag bins), ``n_pairs``, ``ratio`` (last lag / first lag) and ``flat``
    (``ratio > flat_ratio``): loadings that stay correlated at megabase
    separation behave like a dosage block, not a regulatory programme.
    """
    adata = adata.copy() if copy else adata
    chrom, mid = _coordinates(adata)
    use_gpu = resolve_device(device) == "gpu"
    adata.uns[key]["genomic_autocorr"] = _cin.genomic_autocorr(
        _loadings(adata, key), chrom, mid, bins=bins, flat_ratio=flat_ratio, use_gpu=use_gpu)
    return adata if copy else None


def chromosome_effects(adata, key="pf_gica", *, level="chrom", min_peaks=50, copy=False):
    """Standardised mean loading effect per chromosome (``level="chrom"``) or arm.

    Stores ``adata.uns[key]["chromosome_effects"]`` or ``["arm_effects"]`` with
    ``effect`` (components x regions), ``regions``, ``n_peaks`` and the
    ``dominant`` region per component. Descriptive, in units of the component's
    own loading spread; not a test statistic.
    """
    adata = adata.copy() if copy else adata
    chrom, mid = _coordinates(adata)
    name = "chromosome_effects" if level == "chrom" else "arm_effects"
    adata.uns[key][name] = _cin.chromosome_effects(_loadings(adata, key), chrom, mid,
                                                   level=level, min_peaks=min_peaks)
    return adata if copy else None


def copy_number_routes(adata, key="pf_gica", *, n_top=200, run_peaks=100, exclude=("chrY",),
                       min_arm_peaks=50, gates=None, copy=False):
    """Four loading patterns of dosage per component, each gated; copy number if any fires.

    * ``chrom_share``: largest one-chromosome share among the ``n_top`` most
      extreme features of either pole -- an arm or whole chromosome;
    * ``run_z``: largest |mean z| over ``run_peaks`` neighbouring features of
      one chromosome -- a focal amplicon too small to fill the top list;
    * ``opposite_arms_z``: the weaker of two opposite-sign arms of one
      chromosome -- p loss with q gain, which splits between the poles;
    * ``second_chrom_arm_z``: the strongest arm on a chromosome other than the
      top arm's -- one clone with changes on several chromosomes, which spreads
      the top list and keeps each per-feature loading low.

    ``exclude`` chromosomes (default chrY, donor sex) are left out of every
    route. ``gates`` overrides entries of ``_cin.COPY_NUMBER_GATES``, which were
    placed in observed gaps on one tumour ATAC basis: inspect
    ``pl.copy_number_routes`` before trusting them elsewhere. Stores
    ``adata.uns[key]["copy_number"]``; ``get.copy_number_table`` tabulates it.
    Descriptive read-outs, not tests, and no evidence of DNA copy number on
    their own.
    """
    adata = adata.copy() if copy else adata
    chrom, mid = _coordinates(adata)
    adata.uns[key]["copy_number"] = _cin.copy_number_routes(
        _loadings(adata, key), chrom, mid, n_top=n_top, run_peaks=run_peaks, exclude=exclude,
        min_arm_peaks=min_arm_peaks, gates=gates)
    return adata if copy else None


def genome_profiles(adata, key="pf_gica", *, bin_bp=_cin.BIN_BP, block_bp=_cin.BLOCK_BP,
                    winsor_sd=_cin.WINSOR_SD, lengths=None, chroms=None, joint=True,
                    fraction=None, seed=_cin.SPLIT_SEED, max_iter=4000, verbose=False,
                    copy=False):
    """Binned genome profiles of every component with piecewise-constant fits.

    Winsorised block-mean binning with a batch-means SE, then a per-component
    weighted TV fit and (``joint=True``) a joint group-TV fit in the whitened
    activity metric, both by the same solver on the same chain graph. The
    regularisation fraction is selected by a held-out block split of the
    features and a one-SE rule unless ``fraction`` is given. Stores
    ``adata.uns[key]["genome_profiles"]``; plot with ``pl.genome_profiles``.
    """
    adata = adata.copy() if copy else adata
    chrom, mid = _coordinates(adata)
    activities = np.asarray(adata.obsm[f"X_{key}"], np.float64)
    adata.uns[key]["genome_profiles"] = _cin.genome_profiles(
        _loadings(adata, key), activities, chrom, mid, bin_bp=bin_bp, block_bp=block_bp,
        winsor_sd=winsor_sd, lengths=lengths, chroms=chroms, joint=joint, fraction=fraction,
        seed=seed, max_iter=max_iter, verbose=verbose)
    return adata if copy else None
