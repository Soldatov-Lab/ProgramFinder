"""Residual dependence (tl.gica_dependence), refit activities and group recovery."""

import anndata as ad
import numpy as np
import pytest

import programfinder as pf
from programfinder import _dependence, _stability


def _isa_sources(N=6000, seed=0):
    """Rows 0 and 1 share a heavy-tailed envelope around a spherical Gaussian pair:
    dependent, and rotation-invariant inside their plane (an independent subspace
    whose axes no fourth-order criterion can fix). Rows 2-5 are independent sparse
    Laplace sources."""
    rng = np.random.default_rng(seed)
    g = rng.exponential(size=N) ** 1.5
    pair = g * rng.standard_normal((2, N))
    rest = rng.laplace(size=(4, N)) * (rng.random((4, N)) < 0.2)
    return np.vstack([pair, rest])


def _cumulants_direct(s):
    K, n = s.shape
    r = s @ s.T / n
    m = np.einsum("an,bn,cn,dn->abcd", s, s, s, s) / n
    return (m - r[:, :, None, None] * r[None, None] - np.einsum("ac,bd->abcd", r, r)
            - np.einsum("ad,bc->abcd", r, r))


def test_fourth_order_cumulants_are_chunk_invariant_and_exact():
    s = np.random.default_rng(1).laplace(size=(4, 997))
    s -= s.mean(1, keepdims=True)
    direct = _cumulants_direct(s)
    np.testing.assert_allclose(_dependence.fourth_order_cumulants(s, chunk=13), direct, atol=1e-10)
    np.testing.assert_allclose(_dependence.fourth_order_cumulants(s, chunk=10**6), direct, atol=1e-10)


def test_dependence_z_finds_the_planted_subspace_and_nothing_else():
    res = _dependence.dependence_z(_isa_sources(), reps=30, seed=0, use_gpu=False)
    for key in ("zD", "zE", "D", "E"):
        assert res[key].shape == (6, 6)
    np.testing.assert_allclose(np.nan_to_num(res["zE"]), np.nan_to_num(res["zE"].T), atol=1e-8)
    assert res["zE"][0, 1] > 20 and res["zD"][0, 1] > 20
    off = np.abs(res["zE"][2:, 2:][~np.eye(4, dtype=bool)])
    assert off.max() < 5
    assert _dependence.mutual_top_groups(res["zE"]) == [[0, 1]]


def test_mutual_top_groups_do_not_chain_through_a_hub():
    K = 6
    z = np.zeros((K, K))
    hub = 5
    z[hub, :5] = z[:5, hub] = 30          # everyone depends on the hub
    z[0, 1] = z[1, 0] = 40                # one genuine pair
    z[2, 3] = z[3, 2] = 8
    groups = _dependence.mutual_top_groups(z, k=2, z_min=5)
    assert all(len(g) < K for g in groups)
    assert [0, 1] in groups or [0, 1, 5] in groups
    assert not any({0, 2} <= set(g) for g in groups)


def test_ranked_partners_respect_candidates():
    z = np.array([[np.nan, 3, 9], [3, np.nan, 1], [9, 1, np.nan]])
    assert _dependence.ranked_partners(z) == [[2, 1], [0, 2], [0, 1]]
    assert _dependence.ranked_partners(z, candidates=[True, True, False]) == [[1], [0], [0, 1]]


def test_group_recovery_sees_a_rotated_span_as_recovered():
    rng = np.random.default_rng(2)
    ref = rng.standard_normal((500, 3))
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    alt = ref.copy()
    alt[:, 0], alt[:, 1] = c * ref[:, 0] - s * ref[:, 1], s * ref[:, 0] + c * ref[:, 1]
    rec = _stability.group_recovery(ref, alt[None], [[0, 1], [0, 2], [2]], use_gpu=False)
    assert rec[0]["worst"] > 0.999                    # the rotated plane
    assert rec[1]["worst"] < 0.8                      # a pair that is not a group
    assert rec[2]["worst"] > 0.999 and rec[2]["below_gate"] == 0
    single = _stability.group_recovery(ref, alt[None], [[0]], use_gpu=False)[0]["worst"]
    assert abs(single - abs(np.corrcoef(ref[:, 0], alt[:, 0])[0, 1])) < 1e-9


def _span(N=6000, n=900, seed=3):
    rng = np.random.default_rng(seed)
    sources = _isa_sources(N, seed)
    mixing = np.linalg.qr(rng.standard_normal((6, 6)))[0]
    P = mixing @ sources
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    scores = rng.standard_normal((n, 6)) * np.linspace(4, 1, 6)
    return scores, P


def test_bootstrap_refit_activities_are_matched_signed_and_reproducible(tmp_path):
    scores, P = _span()
    plain = _stability.feature_bootstrap_stability(scores, P, n_bootstraps=3, use_gpu=False)
    kept = _stability.feature_bootstrap_stability(scores, P, n_bootstraps=3, use_gpu=False,
                                                  return_activities=True)
    np.testing.assert_allclose(plain["worst_abs_r"], kept["worst_abs_r"])
    A = kept["refit_activities"]
    assert A.shape == (3, scores.shape[0], 6) and A.dtype == np.float32
    np.testing.assert_allclose(kept["matched_abs_r"].min(0), kept["worst_abs_r"])
    fit = pf.feature_ica(scores, P, use_gpu=False)
    ref = fit["activities"][:, [3, 1, 0, 5, 2, 4]] * np.array([1, -1, 1, 1, -1, 1])
    onto = _stability.feature_bootstrap_stability(scores, P, n_bootstraps=2, use_gpu=False,
                                                  reference=ref, out=tmp_path / "refits.npy")
    R = np.load(tmp_path / "refits.npy", mmap_mode="r")
    r = np.array([[np.corrcoef(ref[:, k], R[b][:, k])[0, 1] for k in range(6)] for b in range(2)])
    assert (r > 0).all()
    np.testing.assert_allclose(np.abs(r), onto["matched_abs_r"], atol=1e-5)


def _adata():
    scores, P = _span()
    adata = ad.AnnData(np.zeros((scores.shape[0], P.shape[1]), np.float32))
    adata.obsm["X_pf_pca"] = scores
    adata.varm["pf_pca_components"] = P.T
    pf.tl.gica(adata, contrast="jade", device="cpu")
    return adata


def test_gica_dependence_and_group_verdicts_end_to_end():
    adata = _adata()
    with pytest.raises(KeyError, match="gica_dependence"):
        pf.tl.gica_stability(adata, schedules=(), bootstraps=2, groups="partners", device="cpu")
    with pytest.raises(ValueError, match="bootstraps"):
        pf.tl.gica_stability(adata, schedules=(), groups=[[0, 1]], device="cpu")
    pf.tl.gica_dependence(adata, reps=20, min_support=10, device="cpu")
    dep = adata.uns["pf_gica"]["dependence"]
    # the planted pair, whatever columns JADE put it in
    top = {k: int(dep["partners"][k, 0]) for k in range(6)}
    planted = [k for k in range(6) if top[top[k]] == k and dep["zE"][k, top[k]] > 20]
    assert len(planted) == 2 and dep["groups"] == [sorted(planted)]
    pf.tl.gica_stability(adata, schedules=(), bootstraps=8, groups="partners",
                         min_support=10, device="cpu")
    table = pf.get.stability_table(adata)
    assert {"top_partner", "group", "group_worst_min_cancorr", "verdict"} <= set(table.columns)
    for k in planted:                     # the plane comes back, its axes do not
        row = table.iloc[k]
        assert row["verdict"] == "group"
        assert row["bootstrap_worst_abs_r"] < 0.9 <= row["group_worst_min_cancorr"]
        assert row["group"] == "+".join(f"c{m}" for m in sorted(planted, key=lambda m: m != k))
    others = [k for k in range(6) if k not in planted]
    assert (table["verdict"].iloc[others] == "axis").all()
