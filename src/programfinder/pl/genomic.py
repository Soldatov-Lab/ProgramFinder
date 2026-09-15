"""Plots of the chromosome-scale read-outs (matplotlib).

    pf.pl.genomic_autocorr(adata)            # six-lag profiles, flat ones highlighted
    pf.pl.chromosome_effects(adata)          # components x chromosomes (or arms) heatmap
    pf.pl.genome_profiles(adata, pdf="genome_profiles.pdf")   # one page per 5 components

Every function reads only what the matching ``tl`` function stored in
``adata.uns[key]`` and returns the figure(s); pass ``save=`` or ``pdf=`` to
write files.
"""

from __future__ import annotations

import numpy as np

from .. import _genome as G

__all__ = ["genomic_autocorr", "chromosome_effects", "genome_profiles"]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#a8a7a1"
BLUE, RED, GREY, LIGHT = "#2a78d6", "#c0392b", "#7f8c8d", "#bdc3c7"


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _result(adata, key, name, tool):
    if key not in adata.uns or name not in adata.uns[key]:
        raise KeyError(f"no {name!r} under adata.uns[{key!r}]; run {tool} first")
    return adata.uns[key][name]


def _labels(n, components):
    idx = np.arange(n) if components is None else np.asarray(components, int)
    return idx, [f"c{k:02d}" for k in idx]


def genomic_autocorr(adata, key="pf_gica", *, components=None, highlight="flat", ax=None,
                     save=None):
    """Six-lag autocorrelation profile per component; ``highlight`` picks the red set.

    ``highlight`` is ``"flat"`` (the stored screen), an index list, or None.
    """
    plt = _plt()
    res = _result(adata, key, "genomic_autocorr", "tl.genomic_autocorr")
    profiles = np.asarray(res["profiles"])
    idx, names = _labels(profiles.shape[0], components)
    if highlight == "flat":
        red = set(np.flatnonzero(np.asarray(res["flat"], bool)).tolist())
    else:
        red = set() if highlight is None else set(np.asarray(highlight, int).tolist())
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
    else:
        fig = ax.figure
    labels = [f"{lo // 1000}-{hi // 1000}" for lo, hi in res["bins"]]
    for k, name in zip(idx, names):
        flat = k in red
        ax.plot(range(len(labels)), profiles[k], lw=1.6 if flat else 0.7,
                color=RED if flat else MUTED, alpha=0.95 if flat else 0.45,
                zorder=3 if flat else 1, label=name if flat else None)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("genomic lag (kb)")
    ax.set_ylabel("mean product of standardised loadings")
    n_flat = len(red & set(idx.tolist()))
    ax.set_title(f"genomic autocorrelation, {len(idx)} components "
                 f"(red = {n_flat} with ratio > {res['flat_ratio']:.2f})", loc="left")
    if 0 < n_flat <= 15:
        ax.legend(fontsize=6, frameon=False, ncol=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def chromosome_effects(adata, key="pf_gica", *, level="chrom", components=None, ax=None,
                       mark_dominant=True, drop=("chrY",), save=None):
    """Heatmap of the standardised mean loading effect, components x regions."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    name = "chromosome_effects" if level == "chrom" else "arm_effects"
    res = _result(adata, key, name, f"tl.chromosome_effects(level={level!r})")
    effect = np.asarray(res["effect"])
    regions = list(res["regions"])
    keep = [j for j, r in enumerate(regions) if not any(r.startswith(d) for d in drop)]
    effect, regions = effect[:, keep], [regions[j] for j in keep]
    idx, names = _labels(effect.shape[0], components)
    eff = effect[idx]
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5.5, 0.28 * len(regions)), max(2.5, 0.22 * len(idx))))
    else:
        fig = ax.figure
    cmap = LinearSegmentedColormap.from_list("bwr_pf", [BLUE, "#ffffff", RED])
    lim = max(float(np.abs(eff).max()), 1e-6)
    im = ax.imshow(eff, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto",
                   interpolation="nearest")
    if mark_dominant:
        dominant = np.asarray(res["dominant"]).astype(str)
        for i, k in enumerate(idx):
            if dominant[k] in regions:
                ax.add_patch(Rectangle((regions.index(dominant[k]) - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor=INK, lw=1.0))
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels([r.replace("chr", "") for r in regions], fontsize=6, rotation=90)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("chromosome" + (" arm" if level == "arm" else "") + " (genomic order)")
    ax.set_title("standardised mean loading effect per "
                 + ("chromosome" if level == "chrom" else "arm")
                 + (" (box = dominant region)" if mark_dominant else ""), loc="left", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("SD of own loadings", fontsize=7)
    cb.outline.set_visible(False)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def _genome_axis(bin_chrom, start_bp, order=G.CHROM_ORDER):
    """Concatenated genomic x coordinate (Mb) with chromosome offsets and ticks."""
    offset, ticks, pos = {}, [], 0.0
    for c in G.genomic_order(bin_chrom, order):
        m = bin_chrom == c
        span = (start_bp[m].max() + 1e6) / 1e6
        offset[c] = pos
        ticks.append((pos + span / 2, c))
        pos += span + 8
    x = np.array([offset[c] + s / 1e6 for c, s in zip(bin_chrom, start_bp)])
    return x, offset, ticks


def genome_profiles(adata, key="pf_gica", *, components=None, per_page=5, pdf=None,
                    highlight=None, figsize=(15, 11), show_joint=True):
    """Binned genome loading profiles with the fitted segments, one row per component.

    Grey ribbon: +-2 batch-means SE. Red: per-component TV fit. Blue dashed:
    joint group-TV fit. White bands mark chromosome breaks. ``highlight`` is an
    optional list of component indices to tag in the row title (e.g. the
    autocorrelation candidates). With ``pdf`` the pages are written to one PDF
    and closed; otherwise the list of figures is returned.
    """
    plt = _plt()
    res = _result(adata, key, "genome_profiles", "tl.genome_profiles")
    y, se = np.asarray(res["y"]), np.asarray(res["se"])
    chrom = np.asarray(res["bin_chrom"]).astype(str)
    start = np.asarray(res["start_bp"])
    fit_ic = np.asarray(res["fit_per_ic"])
    fit_joint = None if res.get("fit_joint") is None else np.asarray(res["fit_joint"])
    idx, names = _labels(y.shape[1], components)
    tagged = set() if highlight is None else set(np.asarray(highlight, int).tolist())
    x, offset, ticks = _genome_axis(chrom, start)

    pages = [idx[i:i + per_page] for i in range(0, len(idx), per_page)]
    figures = []
    writer = None
    if pdf is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        writer = PdfPages(pdf)
    for page in pages:
        fig, axes = plt.subplots(len(page), 1, figsize=figsize, sharex=True, squeeze=False)
        for ax, k in zip(axes[:, 0], page):
            ax.fill_between(x, y[:, k] - 2 * se[:, k], y[:, k] + 2 * se[:, k],
                            color=LIGHT, lw=0, alpha=0.6, step="mid")
            ax.plot(x, y[:, k], lw=0.4, color=GREY)
            for c in offset:
                m = chrom == c
                ax.plot(x[m], fit_ic[m, k], lw=1.3, color=RED)
                if show_joint and fit_joint is not None:
                    ax.plot(x[m], fit_joint[m, k], lw=1.1, color=BLUE, ls="--")
            for o in offset.values():
                ax.axvline(o - 4, color="#ecf0f1", lw=6, zorder=0)
            tag = " (candidate)" if k in tagged else ""
            ax.set_ylabel(f"c{k:02d}", rotation=0, ha="right", va="center")
            ax.set_title(f"c{k:02d}{tag}", loc="left", fontsize=8)
        axes[-1, 0].set_xticks([t for t, _ in ticks])
        axes[-1, 0].set_xticklabels([c.replace("chr", "") for _, c in ticks], fontsize=7)
        axes[-1, 0].set_xlabel(
            "genomic position (chromosomes in order; white bands are chromosome breaks; "
            "grey ribbon is +-2 batch-means SE; red = per-component TV"
            + (", blue dashed = joint group-TV)" if show_joint and fit_joint is not None
               else ")"))
        fig.tight_layout()
        if writer is not None:
            writer.savefig(fig)
            plt.close(fig)
        else:
            figures.append(fig)
    if writer is not None:
        writer.close()
        return None
    return figures
