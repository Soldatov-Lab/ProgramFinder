"""Feature whitening, the read-out contract, and the reliability statistics."""

import numpy as np
import pytest

from programfinder import _stability, feature_ica, whiten_loadings
from programfinder._basis import jade_rotation


def _span(seed=0, n=800, N=1500, r=6):
    """A PCA span whose loadings are a rotation of heavy-tailed feature sources."""
    rng = np.random.default_rng(seed)
    sources = rng.laplace(size=(r, N)) * (rng.random((r, N)) < 0.15)
    mixing = np.linalg.qr(rng.standard_normal((r, r)))[0]
    components = mixing @ sources
    components /= np.linalg.norm(components, axis=1, keepdims=True)
    scores = rng.standard_normal((n, r)) * np.linspace(5, 1, r)
    return scores, components, sources


def test_whitening_is_exact_and_invertible():
    _, P, _ = _span()
    white = whiten_loadings(P)
    z = white["K"] @ white["x"]
    np.testing.assert_allclose(z @ z.T / z.shape[1], np.eye(P.shape[0]), atol=1e-10)
    np.testing.assert_allclose(white["K"] @ white["K_inv"], np.eye(P.shape[0]), atol=1e-10)


def test_whitening_refuses_a_rank_deficient_span():
    _, P, _ = _span()
    P = P.copy()
    P[1] = P[0]
    with pytest.raises(ValueError, match="full-rank"):
        whiten_loadings(P)


def test_feature_ica_contract_and_display_convention():
    scores, P, sources = _span()
    fit = feature_ica(scores, P, contrast="jade", use_gpu=False)
    Zc = scores - scores.mean(0)
    # the read-out maps are exact inverses of each other on the span
    np.testing.assert_allclose(Zc @ fit["score_to_activity"], fit["activities"], atol=1e-8)
    np.testing.assert_allclose(fit["feature_to_source"] @ P, fit["loadings"], atol=1e-10)
    np.testing.assert_allclose(fit["score_to_activity"] @ fit["loadings"], P, atol=1e-10)
    # unit SD, positive skew
    np.testing.assert_allclose(fit["activities"].std(0), 1.0, atol=1e-8)
    from scipy.stats import skew
    assert (skew(fit["activities"], axis=0) >= 0).all()
    # planted feature sources recovered up to permutation and sign
    corr = np.abs(np.corrcoef(fit["sources"], sources)[: P.shape[0], P.shape[0]:])
    assert corr.max(1).min() > 0.95


def test_feature_ica_rejects_mismatched_inputs_and_unknown_contrasts():
    scores, P, _ = _span()
    with pytest.raises(ValueError, match="columns"):
        feature_ica(scores[:, :-1], P, use_gpu=False)
    with pytest.raises(ValueError, match="contrast"):
        feature_ica(scores, P, contrast="infomax", use_gpu=False)


def test_picard_contrast_obeys_the_same_contract_and_finds_the_same_sources():
    pytest.importorskip("picard")
    scores, P, sources = _span()
    fit = feature_ica(scores, P, contrast="picard", use_gpu=False)
    Zc = scores - scores.mean(0)
    np.testing.assert_allclose(Zc @ fit["score_to_activity"], fit["activities"], atol=1e-8)
    np.testing.assert_allclose(fit["score_to_activity"] @ fit["loadings"], P, atol=1e-10)
    np.testing.assert_allclose(fit["activities"].std(0), 1.0, atol=1e-8)
    corr = np.abs(np.corrcoef(fit["sources"], sources)[: P.shape[0], P.shape[0]:])
    assert corr.max(1).min() > 0.95
    assert fit["diagnostics"]["contrast"] == "picard"
    assert fit["diagnostics"]["device"] == "cpu"


def test_the_two_contrasts_agree_where_the_sources_are_identified():
    pytest.importorskip("picard")
    scores, P, _ = _span()
    jade = feature_ica(scores, P, contrast="jade", use_gpu=False)
    pic = feature_ica(scores, P, contrast="picard", use_gpu=False)
    _, _, corr = _stability.matched_columns(jade["activities"], pic["activities"])
    assert corr.min() > 0.95
    # One whitening for both contrasts, so their rotations compose orthogonally.
    np.testing.assert_allclose(jade["K"], pic["K"], atol=1e-12)
    product = jade["W"] @ pic["W"].T
    np.testing.assert_allclose(product @ product.T, np.eye(P.shape[0]), atol=1e-8)


def test_restart_stability_reports_the_criterion_of_every_restart():
    pytest.importorskip("picard")
    scores, P, _ = _span()
    out = _stability.restart_stability(scores, P, seeds=(1, 2))
    assert out["worst_abs_r"].shape == (P.shape[0],)
    assert len(out["restarts"]) == 2
    assert out["worst_abs_r"].min() > 0.95          # this span is identified
    assert isinstance(out["best_criterion_is_base"], bool)


def test_jade_rotation_reuses_a_supplied_stack_without_mutating_it():
    _, P, _ = _span()
    white = whiten_loadings(P)
    z = white["K"] @ white["x"]
    W0, d0, stack = jade_rotation(z, use_gpu=False)
    before = np.array(stack, copy=True)
    W1, d1, _ = jade_rotation(z, stack=stack, schedule_seed=3, use_gpu=False)
    np.testing.assert_array_equal(np.asarray(stack), before)
    assert d0["criterion"] == pytest.approx(d1["criterion"], rel=1e-6)
    assert np.abs(np.abs(W0 @ W1.T).max(1) - 1).max() < 1e-3


def test_effective_support_counts_cells():
    a = np.zeros((100, 2))
    a[0, 0] = 5.0                       # a single-cell spike
    a[:, 1] = 1.0                       # everyone equally
    support = _stability.effective_support(a)
    assert support[0] == pytest.approx(1.0) and support[1] == pytest.approx(100.0)


def test_matched_columns_recovers_a_permutation_with_sign_flips():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((50, 4))
    b = -a[:, [2, 0, 3, 1]]
    i, j, corr = _stability.matched_columns(a, b)
    assert corr.min() > 0.999
    assert list(j[np.argsort(i)]) == [1, 3, 0, 2]


def test_schedule_and_bootstrap_stability_report_per_component_reliability():
    scores, P, _ = _span()
    sched = _stability.schedule_stability(scores, P, seeds=(1, 2), use_gpu=False)
    assert sched["worst_abs_r"].shape == (P.shape[0],)
    assert sched["n_unstable_0.99"] == 0 and len(sched["schedules"]) == 2
    boot = _stability.feature_bootstrap_stability(scores, P, n_bootstraps=2, use_gpu=False)
    assert boot["worst_abs_r"].shape == (P.shape[0],) and boot["worst_abs_r"].min() > 0.8


def test_alternating_blocks_keep_neighbours_together_and_offset_by_group():
    pos = np.arange(0, 1000, 10)
    mask = _stability.alternating_blocks(pos, 100)
    assert mask[:10].all() and not mask[10:20].any()
    grouped = _stability.alternating_blocks(pos, 100, group=np.r_[np.zeros(50), np.ones(50)])
    assert (grouped[:50] == mask[:50]).all()          # group 0 unchanged
    assert (grouped[50:] == ~mask[50:]).all()         # group 1 parity offset by one


def test_split_half_stability_identifies_well_separated_axes():
    scores, P, _ = _span(N=4000)
    mask = _stability.alternating_blocks(np.arange(P.shape[1]), 50)
    out = _stability.split_half_stability(scores, P, mask, schedules=(None, 1), use_gpu=False)
    assert out["n_half_a"] + out["n_half_b"] == P.shape[1]
    assert out["abs_cos_min"].shape == (P.shape[0],)
    assert out["abs_cos_min"].min() > 0.9
    with pytest.raises(ValueError, match="one boolean per feature"):
        _stability.split_half_stability(scores, P, mask[:-1], use_gpu=False)
