"""Genomic coordinates of features, and the hg38 constants the CIN tools use.

Everything here is deliberately small: peak-name parsing, chromosome order,
approximate hg38 chromosome lengths and centromere MIDPOINTS (Mb), p/q arm
assignment, and same-chromosome pair enumeration by genomic distance. The
centromere table is adequate for assigning a 500 bp peak to an arm, not for
anything needing base-pair precision, and a boundary near a centromere should
be reported as an interval.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["CHROM_ORDER", "HG38_LENGTH_MB", "HG38_CENTROMERE_MB", "ACROCENTRIC",
           "SEX_CHROMS", "parse_feature_names", "genomic_order", "arm_of_position",
           "centromere_window", "pairs_within_distance"]

#: numeric genomic order; chrY last (absent in half the donors by sex)
CHROM_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
SEX_CHROMS = ("chrX", "chrY")

#: hg38 chromosome lengths, Mb (integer)
HG38_LENGTH_MB = {"chr1": 249, "chr2": 242, "chr3": 198, "chr4": 190, "chr5": 182,
                  "chr6": 171, "chr7": 159, "chr8": 145, "chr9": 138, "chr10": 134,
                  "chr11": 135, "chr12": 133, "chr13": 114, "chr14": 107, "chr15": 102,
                  "chr16": 90, "chr17": 83, "chr18": 80, "chr19": 59, "chr20": 64,
                  "chr21": 47, "chr22": 51, "chrX": 156, "chrY": 57}

#: hg38 approximate centromere MIDPOINTS, Mb. A midpoint is not a breakpoint.
HG38_CENTROMERE_MB = {
    "chr1": 123.4, "chr2": 93.9, "chr3": 90.9, "chr4": 50.0, "chr5": 48.8,
    "chr6": 59.8, "chr7": 60.1, "chr8": 45.2, "chr9": 43.0, "chr10": 39.8,
    "chr11": 53.4, "chr12": 35.5, "chr13": 17.7, "chr14": 17.2, "chr15": 19.0,
    "chr16": 36.8, "chr17": 25.1, "chr18": 18.5, "chr19": 26.2, "chr20": 28.1,
    "chr21": 12.0, "chr22": 15.0, "chrX": 60.6, "chrY": 10.4,
}
#: p arms too small to call; reported as q only
ACROCENTRIC = ("chr13", "chr14", "chr15", "chr21", "chr22")

_PEAK = re.compile(r"^(chr[^:_\-]+)[:_\-](\d+)[\-_:](\d+)$")


def parse_feature_names(names):
    """``chr1:1000-1500`` (also ``chr1-1000-1500`` / ``chr1_1000_1500``) -> chrom, start, end."""
    chrom, start, end = [], [], []
    for name in np.asarray(names).astype(str):
        m = _PEAK.match(name)
        if m is None:
            raise ValueError(f"unparsed feature name {name!r}; supply var columns instead")
        chrom.append(m.group(1))
        start.append(int(m.group(2)))
        end.append(int(m.group(3)))
    return np.asarray(chrom), np.asarray(start, np.int64), np.asarray(end, np.int64)


def genomic_order(names, order=CHROM_ORDER):
    """Chromosome names sorted into genomic order; unknown names go last, sorted."""
    pos = {c: i for i, c in enumerate(order)}
    return sorted(set(np.asarray(names).astype(str).tolist()),
                  key=lambda c: (pos.get(c, len(order)), c))


def arm_of_position(chrom, pos_bp, centromere=HG38_CENTROMERE_MB, acrocentric=ACROCENTRIC):
    """``"p"`` or ``"q"`` for a position, by the centromere midpoint; ``"?"`` if unknown."""
    if chrom in acrocentric:
        return "q"
    cen = centromere.get(chrom)
    if cen is None:
        return "?"
    return "p" if pos_bp < cen * 1e6 else "q"


def centromere_window(chrom, bin_bp, centromere=HG38_CENTROMERE_MB, acrocentric=ACROCENTRIC):
    """Index of the ``bin_bp`` window containing the centromere midpoint, or None."""
    if chrom in acrocentric or chrom not in centromere:
        return None
    return int(centromere[chrom] * 1e6 // bin_bp)


def pairs_within_distance(chrom, mid, lo, hi):
    """Index pairs on the same chromosome with genomic distance in ``(lo, hi]``.

    Each unordered pair appears once (``i`` before ``j`` in position). Built
    without a Python loop over peaks: per chromosome, a forward range per peak
    from two ``searchsorted`` calls, expanded with ``repeat``/``cumsum``.
    """
    chrom = np.asarray(chrom)
    mid = np.asarray(mid, np.int64)
    left, right = [], []
    for name in np.unique(chrom):
        where = np.flatnonzero(chrom == name)
        pos = mid[where]
        order = np.argsort(pos, kind="stable")
        where, pos = where[order], pos[order]
        start = np.searchsorted(pos, pos + lo, side="right")
        stop = np.searchsorted(pos, pos + hi, side="right")
        count = np.maximum(stop - start, 0)
        total = int(count.sum())
        if not total:
            continue
        i = np.repeat(np.arange(len(pos)), count)
        offsets = np.repeat(start, count)
        within = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
        j = offsets + within
        left.append(where[i])
        right.append(where[j])
    if not left:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(left), np.concatenate(right)
