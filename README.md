# ProgramFinder

Cell-program discovery by **residuals → PCA → feature-space ICA (gICA)**, with a
scanpy-style API on AnnData. Extracted from the ProgramForge analysis code so
the pipeline that was spread over unversioned benchmark scripts lives in one
tested package.

```text
counts (AnnData)
  -> pp.residual_null      feature-wise null: NB (UMIs) or depth-adjusted Bernoulli (ATAC)
  -> tl.pca                matrix-free, explicitly centred randomized PCA of the Pearson residual
  -> tl.gica               feature-whitened ICA of the leading PCs; contrast "jade" (Picard planned)
  -> tl.gica_stability     permuted Jacobi schedules, feature bootstraps, fixed-whitening split halves
  -> get.*                 labelled DataFrames of activities, loadings, reliability
```

The residual matrix is never materialised: both null models expose
`matmat` / `rmatmat` products over depth-binned nulls, so a 40k × 220k ATAC
matrix runs on one GPU in device memory measured in hundreds of MB.

## Install

```bash
python -m pip install -e '.[test,io]'        # CPU
python -m pip install -e '.[test,io,gpu]'    # + cupy (CUDA 12)
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

## Provenance

`tl.gica(contrast="jade")` on the frozen ProgramForge ATAC rank-50 span
(`opt0_pca_atac.npz`) reproduces the frozen basis (`basis_cache.npz`) with the
identity permutation and minimum |cos| 1.0000 (98 sweeps, 40 s on an L40S).
The residual operators and the centred randomized PCA carry their original
tests; JADE, the basis read-out and the stability statistics gained theirs here.
