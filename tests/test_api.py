"""The scanpy-style surface: pp -> tl -> get on an AnnData, and the streaming reader."""

import numpy as np
import pytest
from scipy import sparse

import anndata as ad

import programfinder as pf
from programfinder._backend import gpu_available

requires_gpu = pytest.mark.skipif(not gpu_available(), reason="no CUDA device")


def _planted_adata(seed=0, n=1500, N=1200, r=4):
    rng = np.random.default_rng(seed)
    L = np.zeros((n, r))
    for k in range(r):
        cells = rng.choice(n, n // 5, replace=False)
        L[cells, k] = rng.gamma(2, 1, cells.size)
    F = rng.laplace(size=(N, r)) * (rng.random((N, r)) < 0.15)
    depth = rng.lognormal(3, 0.3, n)
    rate = np.exp(np.clip(-3.5 + 0.6 * L @ F.T, -12, 4)) * depth[:, None]
    counts = sparse.csr_matrix(rng.poisson(rate).astype(np.float32))
    adata = ad.AnnData(X=counts)
    adata.obs["depth"] = depth
    adata.obs_names = [f"cell{i}" for i in range(n)]
    adata.var_names = [f"g{j}" for j in range(N)]
    return adata, L


def test_residual_null_records_parameters_and_rebuilds_the_same_operator():
    adata, _ = _planted_adata()
    pf.pp.residual_null(adata, model="nb", depth_bins=8, device="cpu")
    assert adata.uns["pf_residual"]["model"] == "nb"
    assert "pf_residual_baseline" in adata.var and "pf_residual_depth" in adata.obs
    first = pf.pp.residual_operator(adata, device="cpu")
    second = pf.pp.residual_operator(adata, device="cpu")
    np.testing.assert_array_equal(first.dense_binned()[0], second.dense_binned()[0])
    direct = pf.NBResidualOperator(adata.X, depth_bins=8, block_size=512, device="cpu")
    np.testing.assert_allclose(first.dense_binned()[0], direct.dense_binned()[0], atol=1e-6)


def test_bernoulli_null_requires_an_explicit_depth_and_rebuilds():
    rng = np.random.default_rng(1)
    adata = ad.AnnData(X=sparse.csr_matrix((rng.random((200, 150)) < 0.1).astype(np.float32)))
    with pytest.raises(ValueError, match="depth_key"):
        pf.pp.residual_null(adata, model="bernoulli", device="cpu")
    adata.obs["frags"] = rng.integers(500, 5000, 200).astype(float)
    pf.pp.residual_null(adata, model="bernoulli", depth_key="frags", depth_bins=6, device="cpu")
    op = pf.pp.residual_operator(adata, device="cpu")
    assert isinstance(op, pf.BernoulliResidualOperator)
    np.testing.assert_array_equal(op.detection_rate, adata.var["pf_residual_detection_rate"])


def test_residual_operator_refuses_a_subset_without_a_refit():
    adata, _ = _planted_adata()
    pf.pp.residual_null(adata, model="nb", depth_bins=8, device="cpu")
    with pytest.raises(ValueError, match="refit"):
        pf.pp.residual_operator(adata[:100].copy(), device="cpu")


def test_pipeline_end_to_end_recovers_planted_programs_cpu():
    adata, L = _planted_adata()
    pf.pp.residual_null(adata, model="nb", depth_bins=8, device="cpu")
    pf.tl.pca(adata, n_comps=8, device="cpu", check_seed=1, probe_width=64)
    assert adata.obsm["X_pf_pca"].shape == (adata.n_obs, 8)
    assert adata.varm["pf_pca_components"].shape == (adata.n_vars, 8)
    prov = adata.uns["pf_pca"]["provenance"]
    assert prov["centring_residual"] < 1e-3
    assert prov["block_probe_worst_relative_error"] < 1e-3
    assert "independent_sketch_agreement" in prov
    pf.tl.gica(adata, n_comps=6, contrast="jade", device="cpu")
    A = adata.obsm["X_pf_gica"]
    assert A.shape == (adata.n_obs, 6)
    assert adata.varm["pf_gica_loadings"].shape == (adata.n_vars, 6)
    recovered = np.abs(np.corrcoef(L.T, A.T)[: L.shape[1], L.shape[1]:]).max(1)
    assert recovered.min() > 0.7
    pf.tl.gica_stability(adata, schedules=(1, 2), bootstraps=2,
                         split_mask=pf.alternating_blocks(np.arange(adata.n_vars), 30),
                         device="cpu")
    table = pf.get.stability_table(adata)
    assert list(table.index) == [f"c{k}" for k in range(6)]
    assert {"schedule_worst_abs_r", "bootstrap_worst_abs_r", "split_half_abs_cos_min",
            "effective_support", "stable"} <= set(table.columns)
    assert pf.get.activities(adata).shape == (adata.n_obs, 6)
    assert pf.get.loadings(adata).shape == (adata.n_vars, 6)


def test_gica_needs_a_pca_and_respects_the_stored_rank():
    adata, _ = _planted_adata()
    with pytest.raises(KeyError, match="tl.pca"):
        pf.tl.gica(adata, device="cpu")
    pf.pp.residual_null(adata, model="nb", depth_bins=8, device="cpu")
    pf.tl.pca(adata, n_comps=5, device="cpu", probe_blocks=0)
    with pytest.raises(ValueError, match="exceeds"):
        pf.tl.gica(adata, n_comps=6, device="cpu")
    pf.tl.gica(adata, device="cpu")           # None -> all stored components
    assert adata.uns["pf_gica"]["params"]["n_comps"] == 5


def test_copy_returns_a_new_object_and_leaves_the_input_alone():
    adata, _ = _planted_adata()
    out = pf.pp.residual_null(adata, model="nb", depth_bins=8, device="cpu", copy=True)
    assert "pf_residual" in out.uns and "pf_residual" not in adata.uns


@requires_gpu
def test_gpu_pipeline_matches_cpu_span():
    adata, _ = _planted_adata(seed=3)
    pf.pp.residual_null(adata, model="nb", depth_bins=8, device="cpu")
    cpu = pf.tl.pca(adata, n_comps=4, device="cpu", probe_blocks=0, copy=True)
    gpu = pf.tl.pca(adata, n_comps=4, device="gpu", probe_blocks=0, copy=True)
    agreement = pf._pca.subspace_agreement(cpu.obsm["X_pf_pca"], gpu.obsm["X_pf_pca"])
    assert agreement["min_canonical_correlation"] > 0.99
    pf.tl.gica(gpu, device="gpu")
    assert gpu.uns["pf_gica"]["diagnostics"]["device"] == "gpu"


def test_read_h5ad_rows_streams_a_row_and_column_subset(tmp_path):
    pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    X = sparse.random(300, 40, density=0.2, random_state=0, format="csr", dtype=np.float32)
    adata = ad.AnnData(X=X)
    adata.obs["modality"] = rng.choice(["paired", "rna"], 300)
    adata.obs["state"] = rng.choice(["a", "b", "c"], 300)
    adata.var["is_peak"] = np.arange(40) >= 25
    path = tmp_path / "toy.h5ad"
    adata.write_h5ad(path)

    sub = pf.io.read_h5ad_rows(path, obs_query="modality == 'paired' and state != 'c'",
                               var_mask=lambda var: var["is_peak"].to_numpy(),
                               row_block=17, verbose=False)
    expected_rows = ((adata.obs["modality"] == "paired") & (adata.obs["state"] != "c")).to_numpy()
    reference = adata[expected_rows, adata.var["is_peak"].to_numpy()]
    assert sub.shape == reference.shape
    assert list(sub.obs_names) == list(reference.obs_names)
    np.testing.assert_allclose(sub.X.toarray(), reference.X.toarray())
    assert sub.uns["pf_source"]["n_selected_cells"] == int(expected_rows.sum())

    by_mask = pf.io.read_h5ad_rows(path, obs_mask=expected_rows, verbose=False)
    assert by_mask.shape == (int(expected_rows.sum()), 40)
    with pytest.raises(ValueError, match="not both"):
        pf.io.read_h5ad_rows(path, obs_mask=expected_rows, obs_query="state == 'a'")
