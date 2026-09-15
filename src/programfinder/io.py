"""Streaming row selection from a large ``.h5ad`` without loading it.

A multiome object can be hundreds of thousands of cells by ~750k features in
one CSR block; ``read_h5ad_rows`` reads ``obs`` first, evaluates a mask or a
pandas query on it, then gathers only the selected rows straight from the
HDF5 arrays in blocks. An optional column mask slices features, so genes and
peaks can be split into two AnnData objects from the same pass.

Requires ``h5py`` (extra ``io``) and anndata's element reader.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse

__all__ = ["read_h5ad_rows"]


def _read_elem(group):
    try:
        from anndata.io import read_elem
    except ImportError:                       # anndata < 0.11
        from anndata.experimental import read_elem
    return read_elem(group)


def _stream_csr_rows(group, rows, row_block, col_mask=None):
    """Gather ``rows`` of an h5ad CSR group, blockwise, optionally column-masked."""
    indptr = group["indptr"][...]
    data, indices = group["data"], group["indices"]
    n_cols = int(group.attrs["shape"][1])
    keep_cols = None if col_mask is None else np.flatnonzero(col_mask)
    remap = None
    if keep_cols is not None:
        remap = np.full(n_cols, -1, np.int64)
        remap[keep_cols] = np.arange(keep_cols.size)
    parts = []
    rows = np.asarray(rows, np.int64)
    for start in range(0, rows.size, row_block):
        chunk = rows[start:start + row_block]
        # read one contiguous span covering the chunk, then pick its rows
        lo, hi = int(indptr[chunk.min()]), int(indptr[chunk.max() + 1])
        span_data = data[lo:hi]
        span_indices = indices[lo:hi]
        offsets = indptr[chunk] - lo
        lengths = indptr[chunk + 1] - indptr[chunk]
        take = np.concatenate([np.arange(o, o + n) for o, n in zip(offsets, lengths)]) \
            if lengths.sum() else np.empty(0, np.int64)
        d = span_data[take]
        i = span_indices[take].astype(np.int64)
        ptr = np.concatenate([[0], np.cumsum(lengths)])
        block = sparse.csr_matrix((d, i, ptr), shape=(chunk.size, n_cols))
        if keep_cols is not None:
            block = block[:, keep_cols]
        parts.append(block)
    if not parts:
        width = n_cols if keep_cols is None else keep_cols.size
        return sparse.csr_matrix((0, width), dtype=np.float32)
    return sparse.vstack(parts).tocsr()


def read_h5ad_rows(path, *, obs_mask=None, obs_query=None, var_mask=None, layer=None,
                   row_block=2048, obs_columns=None, verbose=True):
    """Read a row (and optionally column) subset of a large ``.h5ad`` by streaming.

    Parameters
    ----------
    path : the ``.h5ad`` file
    obs_mask : callable ``obs -> bool array`` or a boolean array over all cells
    obs_query : pandas ``DataFrame.query`` string over ``obs`` (alternative)
    var_mask : callable ``var -> bool array`` or a boolean array over features
    layer : read ``layers/<layer>`` instead of ``X``
    row_block : selected rows gathered per HDF5 read
    obs_columns : keep only these ``obs`` columns (all by default)

    Returns an AnnData with a CSR ``X`` (dense sources are converted), the
    selected ``obs``/``var`` and ``uns["pf_source"]`` recording the file and
    selection sizes.
    """
    import anndata as ad
    import h5py

    started = time.time()
    with h5py.File(path, "r") as handle:
        obs = _read_elem(handle["obs"])
        var = _read_elem(handle["var"])
        if obs_mask is not None and obs_query is not None:
            raise ValueError("pass obs_mask or obs_query, not both")
        if obs_query is not None:
            rows = np.flatnonzero(obs.eval(obs_query).to_numpy())
        elif obs_mask is not None:
            mask = obs_mask(obs) if callable(obs_mask) else obs_mask
            rows = np.flatnonzero(np.asarray(mask, bool))
        else:
            rows = np.arange(obs.shape[0])
        cols = None
        if var_mask is not None:
            cols = np.asarray(var_mask(var) if callable(var_mask) else var_mask, bool)
            if cols.shape != (var.shape[0],):
                raise ValueError("var_mask needs one boolean per feature")
        group = handle["X"] if layer is None else handle["layers"][layer]
        if isinstance(group, h5py.Group):
            encoding = group.attrs.get("encoding-type", "csr_matrix")
            if encoding != "csr_matrix":
                raise NotImplementedError(f"streaming rows of a {encoding} is not supported; "
                                          "convert to CSR first")
            X = _stream_csr_rows(group, rows, row_block, cols)
        else:                                  # dense dataset
            parts = []
            for start in range(0, rows.size, row_block):
                chunk = rows[start:start + row_block]
                block = group[chunk.min():chunk.max() + 1][chunk - chunk.min()]
                parts.append(sparse.csr_matrix(block if cols is None else block[:, cols]))
            X = sparse.vstack(parts).tocsr() if parts else sparse.csr_matrix(
                (0, var.shape[0] if cols is None else int(cols.sum())))
    obs = obs.iloc[rows]
    if obs_columns is not None:
        obs = obs[list(obs_columns)]
    if cols is not None:
        var = var.iloc[np.flatnonzero(cols)]
    adata = ad.AnnData(X=X.astype(np.float32), obs=obs.copy(), var=var.copy())
    adata.uns["pf_source"] = {"path": str(path), "layer": layer,
                              "n_selected_cells": int(rows.size),
                              "n_selected_features": int(adata.n_vars),
                              "seconds": round(time.time() - started, 1)}
    if verbose:
        print(f"[programfinder.io] {adata.n_obs} x {adata.n_vars} from {path} "
              f"in {adata.uns['pf_source']['seconds']}s", flush=True)
    return adata
