"""ProgramFinder: residuals -> PCA -> feature-space ICA, scanpy-style.

    import programfinder as pf

    adata = pf.io.read_h5ad_rows(path, obs_query="modality == 'paired'")
    pf.pp.residual_null(adata, model="bernoulli", depth_key="n_unique")
    pf.tl.pca(adata, n_comps=100)
    pf.tl.gica(adata, n_comps=50, contrast="jade")
    pf.tl.gica_stability(adata, schedules=range(1, 6))
    pf.get.stability_table(adata)

Lower-level building blocks are importable from the private modules:
``_residual`` (matrix-free NB / Bernoulli Pearson residual operators),
``_pca`` (centred randomized PCA), ``_basis`` (feature whitening, contrasts,
read-out), ``_jade`` (JADE) and ``_stability``.
"""

from . import get, io, pp, tl
from ._basis import CONTRASTS, feature_ica, whiten_loadings
from ._pca import CentredOperator, randomized_pca
from ._residual import BernoulliResidualOperator, NBResidualOperator
from ._stability import alternating_blocks, effective_support

__all__ = [
    "pp", "tl", "io", "get",
    "NBResidualOperator", "BernoulliResidualOperator",
    "CentredOperator", "randomized_pca",
    "whiten_loadings", "feature_ica", "CONTRASTS",
    "alternating_blocks", "effective_support",
]

__version__ = "0.1.0.dev0"
