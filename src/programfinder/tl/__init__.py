"""Tools: ``decomposition`` (pca, gica, gica_stability, gica_dependence) and ``genomic`` read-outs.

Both submodules are re-exported here, so ``pf.tl.pca`` and ``pf.tl.genomic_autocorr``
work as well as ``pf.tl.decomposition.pca`` and ``pf.tl.genomic.genomic_autocorr``.
"""

from . import decomposition, genomic
from .decomposition import gica, gica_dependence, gica_stability, pca
from .genomic import chromosome_effects, copy_number_routes, genome_profiles, genomic_autocorr

__all__ = ["decomposition", "genomic", "pca", "gica", "gica_stability", "gica_dependence",
           "genomic_autocorr", "chromosome_effects", "copy_number_routes", "genome_profiles"]
