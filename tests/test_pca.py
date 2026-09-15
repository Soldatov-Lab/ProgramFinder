"""Explicit centring and the randomized PCA (from ProgramForge's option-0 probes)."""

import numpy as np
import pytest
from scipy import sparse

from programfinder import NBResidualOperator
from programfinder import _pca


class _DenseOperator:
    """The minimum surface ``CentredOperator`` and the range finder touch."""

    def __init__(self, matrix):
        self.Z = np.asarray(matrix, np.float32)
        self.shape = self.Z.shape
        self.device = "cpu"

    def matmat(self, right):
        return self.Z @ np.asarray(right, np.float32)

    def rmatmat(self, left):
        return self.Z.T @ np.asarray(left, np.float32)


def _random_operator(seed=0, n=200, p=60):
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal((n, 4)) @ rng.standard_normal((4, p))
    offset = rng.standard_normal(p) * 3.0
    return _DenseOperator(signal + 0.3 * rng.standard_normal((n, p)) + offset)


def test_column_means_matches_the_dense_mean():
    operator = _random_operator()
    assert np.allclose(_pca.column_means(operator, np), operator.Z.mean(axis=0), atol=1e-4)


def test_centred_operator_reproduces_explicit_dense_centring():
    operator = _random_operator()
    centred = _pca.CentredOperator(operator)
    Zc = operator.Z - operator.Z.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(1)
    right = rng.standard_normal((operator.shape[1], 5)).astype(np.float32)
    left = rng.standard_normal((operator.shape[0], 3)).astype(np.float32)
    assert np.allclose(centred.matmat(right), Zc @ right, atol=1e-3)
    assert np.allclose(centred.rmatmat(left), Zc.T @ left, atol=1e-3)


def test_centring_residual_reads_zero_one_and_two():
    operator = _random_operator()
    assert _pca.centring_residual(_pca.CentredOperator(operator), np) < 1e-4

    class _Uncentred(_pca.CentredOperator):
        def rmatmat(self, left):
            return self._operator.rmatmat(left)

    class _WrongSide(_pca.CentredOperator):
        def rmatmat(self, left):
            out = self._operator.rmatmat(left)
            return out + self.mean[:, None] * left.sum(axis=0)[None, :]

    assert _pca.centring_residual(_Uncentred(operator), np) == pytest.approx(1.0, rel=1e-3)
    assert _pca.centring_residual(_WrongSide(operator), np) == pytest.approx(2.0, rel=1e-3)


def test_centring_residual_does_not_scale_with_cohort_size():
    small = _pca.centring_residual(_pca.CentredOperator(_random_operator(n=200)), np)
    large = _pca.centring_residual(_pca.CentredOperator(_random_operator(n=4000)), np)
    assert small < 1e-4 and large < 1e-4


def test_blocked_column_means_partition_is_exact():
    rng = np.random.default_rng(21)
    Z = (rng.standard_normal((997, 40)) * 1e3 + 1.0).astype(np.float32)
    operator = _DenseOperator(Z)
    exact = Z.astype(np.float64).mean(axis=0)
    for n_blocks in (1, 7, 64, 997):
        got = np.asarray(_pca.column_means(operator, np, n_blocks=n_blocks), np.float64)
        assert np.abs(got - exact).max() < 1e-2, n_blocks


def _exact_pca(matrix, rank):
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    return u[:, :rank] * s[:rank], vt[:rank], s[:rank] ** 2 / (matrix.shape[0] - 1)


def test_randomized_pca_recovers_the_exact_subspace_and_spectrum():
    operator = _random_operator(seed=3, n=300, p=80)
    result = _pca.randomized_pca(_pca.CentredOperator(operator), 6, oversample=20, n_power=7,
                                 seed=0, xp=np)
    scores, _, variance = _exact_pca(operator.Z.astype(np.float64), 6)
    agreement = _pca.subspace_agreement(scores, result["scores"])
    assert agreement["mean_squared_cosine"] > 0.999
    assert agreement["min_canonical_correlation"] > 0.99
    assert np.allclose(result["explained_variance"], variance, rtol=1e-3)


def test_randomized_pca_without_centring_would_spend_a_direction_on_the_mean():
    operator = _random_operator(seed=4, n=300, p=80)
    uncentred = _pca.randomized_pca(operator, 6, oversample=20, n_power=7, seed=0, xp=np)
    centred = _pca.randomized_pca(_pca.CentredOperator(operator), 6, oversample=20, n_power=7,
                                  seed=0, xp=np)
    exact, _, _ = _exact_pca(operator.Z.astype(np.float64), 6)
    assert _pca.subspace_agreement(exact, centred["scores"])["mean_squared_cosine"] > \
        _pca.subspace_agreement(exact, uncentred["scores"])["mean_squared_cosine"]


def test_subspace_agreement_endpoints():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((120, 4))
    assert _pca.subspace_agreement(a, a)["mean_squared_cosine"] == pytest.approx(1.0)
    q = np.linalg.qr(rng.standard_normal((120, 8)))[0]
    assert _pca.subspace_agreement(q[:, :4], q[:, 4:])["mean_squared_cosine"] < 0.2


def _small_nb_operator():
    rng = np.random.default_rng(7)
    counts = sparse.csr_matrix(rng.poisson(0.6, size=(120, 40)).astype(np.float32))
    depth = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    return NBResidualOperator(counts, size_factors=depth / depth.mean(), depth_bins=8,
                              block_size=7, device="cpu")


def test_materialized_feature_block_is_the_matrix_matmat_applies():
    operator = _small_nb_operator()
    block = _pca.materialize_feature_block(operator, 5, 20, np)
    rng = np.random.default_rng(0)
    probe = rng.standard_normal((15, 3)).astype(np.float32)
    padded = np.zeros((40, 3), np.float32)
    padded[5:20] = probe
    assert np.allclose(block @ probe, operator.matmat(padded), atol=1e-4)
    errors = _pca.block_probe_error(operator, np, n_blocks=2, width=8)
    assert max(e["relative_error"] for e in errors) < 1e-4


def test_score_invariant_is_small_for_an_exact_projection():
    operator = _small_nb_operator()
    Z = operator.dense_binned()[0].astype(np.float64)
    mean = Z.mean(0)
    scores, components, _ = _exact_pca(Z, 5)
    err = _pca.score_invariant_frobenius(operator, components.astype(np.float32),
                                         mean.astype(np.float32), scores, np)
    assert err < 1e-4
