# ProgramFinder

Cell-program discovery by **residuals → PCA → feature-space ICA (gICA)**, with a
scanpy-style API on AnnData, plus chromosome-scale read-outs of the loadings
for copy-number-like components.

```text
counts (AnnData)
  -> pp.residual_null      feature-wise null: NB (UMIs) or depth-adjusted Bernoulli (ATAC)
  -> tl.pca                matrix-free, explicitly centred randomized PCA of the Pearson residual
  -> tl.gica               feature-whitened ICA of the leading PCs; contrast "jade" (Picard planned)
  -> tl.gica_stability     permuted Jacobi schedules, feature bootstraps, fixed-whitening split halves
  -> get.*                 labelled DataFrames of activities, loadings, reliability

genomic read-outs of the loadings (pf.tl.genomic / pf.pl.genomic)
  -> pp.feature_coordinates   var["pf_chrom"], var["pf_mid"] from peak names or var columns
  -> tl.genomic_autocorr      six-lag genomic autocorrelation and the flatness screen
  -> tl.chromosome_effects    standardised mean loading effect per chromosome or arm
  -> tl.genome_profiles       winsorised block-mean binning + TV / group-TV fits, held-out lambda
  -> pl.*                     the matching figures (multi-page genome profile PDF included)
```

The residual matrix is never materialised: both null models expose
`matmat` / `rmatmat` products over depth-binned nulls, so a 40k × 220k ATAC
matrix runs on one GPU in device memory measured in hundreds of MB.

## Install

```bash
python -m pip install -e '.[test,io,plot]'        # CPU
python -m pip install -e '.[test,io,plot,gpu]'    # + cupy (CUDA 12)
pytest
```

## Use

```python
import programfinder as pf

# stream a cell subset out of a large .h5ad without loading it
adata = pf.io.read_h5ad_rows(
    "multiome.h5ad",
    obs_query="modality == 'paired' and cell_states in ['Tumor likely']",
    var_mask=lambda var: var["feature_types"].eq("Peaks").to_numpy(),
)

pf.pp.residual_null(adata, model="bernoulli", depth_key="n_unique", depth_bins=256)
pf.tl.pca(adata, n_comps=100, check_seed=1)        # check_seed: second sketch, agreement recorded
pf.tl.gica(adata, n_comps=50, contrast="jade")
pf.tl.gica_stability(adata, schedules=range(1, 6), bootstraps=10,
                     split_mask=pf.alternating_blocks(peak_midpoint_bp, 200_000, group=chrom))

pf.get.stability_table(adata)        # per component: kurtosis, support, worst |r|, split-half |cos|
pf.get.activities(adata)             # cells x components
pf.get.loadings(adata)               # features x components

# chromosome-scale read-outs (ATAC peaks named chr:start-end, or var columns)
pf.pp.feature_coordinates(adata)
pf.tl.genomic_autocorr(adata)                       # uns["pf_gica"]["genomic_autocorr"]
pf.tl.chromosome_effects(adata, level="arm")        # uns["pf_gica"]["arm_effects"]
pf.tl.genome_profiles(adata)                        # uns["pf_gica"]["genome_profiles"]
pf.pl.genomic_autocorr(adata, save="autocorr.png")
pf.pl.chromosome_effects(adata, level="arm", save="arm_effects.png")
pf.pl.genome_profiles(adata, pdf="genome_profiles.pdf", highlight=candidates)
```

Where results land:

| slot | content |
| --- | --- |
| `obsm["X_pf_pca"]`, `varm["pf_pca_components"]`, `var["pf_pca_mean"]`, `uns["pf_pca"]` | PCA scores, loadings, residual column means, variance and provenance |
| `obsm["X_pf_gica"]`, `varm["pf_gica_loadings"]`, `uns["pf_gica"]` | activities (unit SD, positive skew), loadings, whitening `K`, rotation `W`, read-out maps, diagnostics |
| `uns["pf_gica"]["stability"]` | per-component reliability arrays and a combined `stable` mask |
| `var["pf_residual_*"]`, `obs["pf_residual_depth"]`, `uns["pf_residual"]` | null parameters, so `pp.residual_operator` rebuilds the operator without refitting |

## Conventions worth knowing

- **Bernoulli depth must be the modality's depth** (e.g. genome-wide fragment
  count from `obs`), not the row sum of a filtered peak matrix.
- **The PCA is randomized.** `uns["pf_pca"]["provenance"]` records the centring
  residual, feature-block probe errors, the score-invariant error and, with
  `check_seed`, the agreement of two independent sketches. Trailing directions
  can be rotated even when the captured energy is high; read the minimum
  canonical correlation, not only the mean.
- **The read-out is fixed by the contract** `score_to_activity @ loadings == P`
  for every orthogonal rotation; only the choice of rotation differs between
  contrasts. JADE and (later) Picard therefore produce different component
  numberings on the same span.
- **JADE has no random start.** Reliability is read from permuted Jacobi
  schedules on the same cumulants (stationary points the data cannot rank),
  from feature bootstraps, and from fixed-whitening split halves.
  `effective_support` catches components that are single-cell spikes.

- **Genomic read-outs are descriptive.** The autocorrelation ratio, the
  standardised chromosome effect and the fitted segments are signed effects
  with spans and approximate uncertainties. None of them is a test statistic
  or DNA evidence: the batch-means SE misses correlation longer than the
  block, and `n_peaks` is not a count of independent observations. The
  centromere table is hg38 midpoints, so an arm boundary is an interval.

## Layout

```
programfinder/
  pp.py            residual_null, residual_operator, feature_coordinates
  tl/decomposition pca, gica, gica_stability
  tl/genomic       genomic_autocorr, chromosome_effects, genome_profiles
  pl/genomic       the matching figures
  io.py, get.py
  _residual, _pca, _basis, _jade, _stability, _cin, _genome   numeric cores
```
