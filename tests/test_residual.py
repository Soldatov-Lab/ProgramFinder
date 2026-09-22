"""Contracts of the residual operators."""

import numpy as np
import pytest
from scipy import sparse

from programfinder import BernoulliResidualOperator, NBResidualOperator
from programfinder._backend import gpu_available

requires_gpu = pytest.mark.skipif(not gpu_available(), reason="no CUDA device")


# --------------------------------------------------------------------------
# NB
# --------------------------------------------------------------------------

def test_nb_exact_products_match_materialized_residuals():
    rng = np.random.default_rng(5)
    counts = sparse.csr_matrix(rng.poisson(rng.uniform(0.1, 2, (35, 27))))
    operator = NBResidualOperator(counts, block_size=7, depth_bins=None, device="cpu")
    dense = operator.toarray()
    right = rng.normal(size=(27, 4))
    left = rng.normal(size=(35, 3))
    np.testing.assert_allclose(operator.matmat(right), dense @ right, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(operator.rmatmat(left), dense.T @ left, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(operator.column_sumsq(), np.square(dense).sum(0), rtol=2e-6)


def _binned_nb(seed=5, shape=(60, 41), bins=8):
    rng = np.random.default_rng(seed)
    counts = sparse.csr_matrix(rng.poisson(rng.uniform(0.1, 2, shape)))
    return NBResidualOperator(counts, block_size=7, depth_bins=bins, device="cpu")


def test_nb_binned_products_match_the_binned_residual():
    operator = _binned_nb()
    residual, _ = operator.dense_binned()
    rng = np.random.default_rng(21)
    right = rng.normal(size=(operator.shape[1], 4))
    left = rng.normal(size=(operator.shape[0], 3))
    np.testing.assert_allclose(operator.matmat(right), residual @ right, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(operator.rmatmat(left), residual.T @ left, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(operator.column_sumsq(), np.square(residual).sum(0),
                               rtol=2e-5, atol=2e-5)
    # chunks far smaller than a row: every boundary case of the nonzero pass
    np.testing.assert_allclose(operator.column_sumsq(nnz_chunk=5),
                               np.square(residual).sum(0), rtol=2e-5, atol=2e-5)


def test_nb_root_weighted_products_match_the_binned_residual():
    operator = _binned_nb()
    residual, root_weight = operator.dense_binned()
    rng = np.random.default_rng(1)
    left = rng.normal(size=(operator.shape[0], 3))
    right = rng.normal(size=(operator.shape[1], 4))
    np.testing.assert_allclose(operator.root_weighted_rmatmat(left),
                               (root_weight * residual).T @ left, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(operator.root_weighted_matmat(right),
                               (root_weight * residual) @ right, rtol=2e-5, atol=2e-5)


def test_nb_null_parameters_round_trip_through_the_constructor():
    """The recorded baseline/dispersion rebuild the identical operator."""
    rng = np.random.default_rng(3)
    counts = sparse.csr_matrix(rng.poisson(rng.uniform(0.1, 2, (50, 30))))
    first = NBResidualOperator(counts, block_size=7, depth_bins=6, device="cpu")
    second = NBResidualOperator(counts, size_factors=first.size_factors, block_size=7,
                                depth_bins=6, device="cpu", baseline=first.baseline,
                                dispersion=first.dispersion)
    np.testing.assert_array_equal(first.dense_binned()[0], second.dense_binned()[0])


# --------------------------------------------------------------------------
# Bernoulli
# --------------------------------------------------------------------------

def dense_bernoulli_reference(counts, depths, n_iter=200):
    X = (np.asarray(counts, dtype=np.float64) > 0).astype(np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    totals = X.sum(axis=0)
    q = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        if totals[j] == 0:
            continue
        low, high = 1e-12, 1.0
        for _ in range(n_iter):
            mid = np.sqrt(low * high)
            expected = (1.0 - (1.0 - mid) ** depths).sum()
            if expected > totals[j]:
                high = mid
            else:
                low = mid
        q[j] = min(np.sqrt(low * high), 1.0 - 1e-12)
    p = 1.0 - (1.0 - q[None, :]) ** depths[:, None]
    sd = np.sqrt(p * (1.0 - p))
    usable = sd > 0
    Z = np.where(usable, (X - p) / np.where(usable, sd, 1.0), 0.0)
    return q, p, Z


def _counts_and_depths(seed=0, shape=(70, 26)):
    rng = np.random.default_rng(seed)
    depths = rng.integers(3, 60, shape[0]).astype(np.float64)
    detected = rng.random(shape) < (depths[:, None] / 140.0)
    counts = detected * rng.integers(1, 6, shape)
    return sparse.csr_matrix(counts.astype(np.float64)), depths


def test_bernoulli_rate_and_residual_match_the_dense_reference():
    counts, depths = _counts_and_depths()
    q, _, Z = dense_bernoulli_reference(counts.toarray(), depths)
    operator = BernoulliResidualOperator(counts, depths, block_size=7, depth_bins=None,
                                         device="cpu")
    np.testing.assert_allclose(operator.detection_rate, q, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(operator.toarray(), Z, rtol=2e-5, atol=2e-5)


def test_bernoulli_expected_detections_match_the_observed_total():
    counts, depths = _counts_and_depths(seed=3)
    operator = BernoulliResidualOperator(counts, depths, block_size=9, depth_bins=None,
                                         device="cpu")
    p = operator._detection_probability(0, operator.shape[1])
    observed = np.asarray((counts > 0).sum(axis=0)).ravel()
    np.testing.assert_allclose(p.sum(axis=0), observed, rtol=1e-6, atol=1e-6)


def test_bernoulli_binned_products_match_the_binned_residual():
    counts, depths = _counts_and_depths(seed=2)
    operator = BernoulliResidualOperator(counts, depths, block_size=6, depth_bins=8,
                                         device="cpu")
    residual, _ = operator.dense_binned()
    rng = np.random.default_rng(5)
    right = rng.normal(size=(operator.shape[1], 3))
    left = rng.normal(size=(operator.shape[0], 2))
    np.testing.assert_allclose(operator.matmat(right), residual @ right, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(operator.rmatmat(left), residual.T @ left, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(operator.column_sumsq(), np.square(residual).sum(0),
                               rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(operator.column_sumsq(nnz_chunk=5),
                               np.square(residual).sum(0), rtol=1e-4, atol=1e-4)


def test_bernoulli_binned_probabilities_stay_calibrated():
    counts, depths = _counts_and_depths(seed=6)
    operator = BernoulliResidualOperator(counts, depths, block_size=6, depth_bins=9,
                                         device="cpu")
    expected = (operator.bin_p * operator.bin_counts[:, None]).sum(axis=0)
    np.testing.assert_allclose(expected, operator.detection_totals, rtol=1e-5, atol=1e-5)


def test_bernoulli_residual_null_is_not_the_root_fisher_weight():
    counts, depths = _counts_and_depths(seed=9)
    operator = BernoulliResidualOperator(counts, depths, block_size=6, depth_bins=8,
                                         device="cpu")
    p = operator.bin_p[:, :operator.shape[1]]
    residual_null = operator._residual_null_block(0, operator.shape[1], np)
    np.testing.assert_allclose(residual_null, np.sqrt(p / (1 - p)), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(operator.weight_block(0, operator.shape[1], np), p * (1 - p),
                               rtol=1e-5, atol=1e-6)
    assert not np.allclose(residual_null, operator._root_weight_block(0, operator.shape[1], np))


def test_bernoulli_raw_counts_are_binarized():
    depths = np.array([9.0, 11.0, 13.0, 17.0])
    raw = np.array([[4.0, 0.0], [7.0, 2.0], [0.0, 0.0], [1.0, 9.0]])
    kwargs = dict(block_size=2, depth_bins=None, device="cpu")
    np.testing.assert_allclose(
        BernoulliResidualOperator(sparse.csr_matrix(raw), depths, **kwargs).toarray(),
        BernoulliResidualOperator(sparse.csr_matrix((raw > 0) * 1.0), depths, **kwargs).toarray())


def test_bernoulli_degenerate_columns_and_zero_depths_stay_finite():
    depths = np.array([0.0, 0.0, 5.0, 7.0, 9.0, 11.0, 13.0, 20.0])
    counts = np.zeros((8, 4))
    counts[2:, 0] = 3.0
    counts[3, 2] = 1.0
    counts[5, 2] = 2.0
    counts[2, 3] = 1.0
    for bins in (None, 3):
        operator = BernoulliResidualOperator(sparse.csr_matrix(counts), depths, block_size=2,
                                             depth_bins=bins, device="cpu")
        dense = operator.toarray()
        assert np.isfinite(dense).all()
        assert np.abs(dense[:, 0]).max() == 0.0
        assert np.abs(dense[:, 1]).max() == 0.0
        if bins is not None:
            residual, root_weight = operator.dense_binned()
            assert np.abs(residual[:2]).max() == 0.0 and np.abs(root_weight[:2]).max() == 0.0


def test_bernoulli_validation():
    counts = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="zero depth"):
        BernoulliResidualOperator(counts, np.array([0.0, 5.0]), device="cpu")
    empty = sparse.csr_matrix(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        BernoulliResidualOperator(empty, np.array([-1.0, 2.0, 3.0]), device="cpu")
    with pytest.raises(ValueError, match="one value per cell"):
        BernoulliResidualOperator(empty, np.array([1.0, 2.0]), device="cpu")
    counts, depths = _counts_and_depths(seed=16, shape=(12, 3))
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="block_size"):
            BernoulliResidualOperator(counts, depths, block_size=bad, device="cpu")
        with pytest.raises(ValueError, match="block_size"):
            NBResidualOperator(counts, block_size=bad, device="cpu")


def _near_one_operator(device="cpu"):
    counts = np.zeros((4, 1))
    counts[0, 0] = 1.0
    return BernoulliResidualOperator(sparse.csr_matrix(counts), np.full(4, 26.0), block_size=1,
                                     depth_bins=1, device=device,
                                     detection_rate=np.array([0.5]))


def test_near_one_probability_is_not_rounded_away():
    operator = _near_one_operator()
    p = 1.0 - 0.5 ** 26
    assert operator.bin_p.dtype == np.float64
    residual, root_weight = operator.dense_binned()
    np.testing.assert_allclose(residual[1:, 0], -np.sqrt(p / (1 - p)), rtol=1e-6)
    np.testing.assert_allclose(root_weight[:, 0], np.sqrt(p * (1 - p)), rtol=1e-6)


@requires_gpu
def test_bernoulli_gpu_matches_cpu():
    counts, depths = _counts_and_depths(seed=21)
    kwargs = dict(block_size=6, depth_bins=8)
    host = BernoulliResidualOperator(counts, depths, device="cpu", **kwargs)
    device = BernoulliResidualOperator(counts, depths, device="gpu", **kwargs)
    np.testing.assert_allclose(host.detection_rate, device.detection_rate, rtol=1e-9, atol=1e-12)
    rng = np.random.default_rng(22)
    right = rng.normal(size=(host.shape[1], 3)).astype(np.float32)
    left = rng.normal(size=(host.shape[0], 2)).astype(np.float32)
    np.testing.assert_allclose(host.matmat(right), device.matmat(right).get(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(host.rmatmat(left), device.rmatmat(left).get(), rtol=1e-4, atol=1e-4)


@requires_gpu
def test_nb_gpu_matches_cpu():
    rng = np.random.default_rng(23)
    counts = sparse.csr_matrix(rng.poisson(rng.uniform(0.1, 2, (60, 41))))
    host = NBResidualOperator(counts, block_size=7, depth_bins=8, device="cpu")
    device = NBResidualOperator(counts, block_size=7, depth_bins=8, device="gpu")
    right = rng.normal(size=(41, 3)).astype(np.float32)
    np.testing.assert_allclose(host.matmat(right), device.matmat(right).get(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(host.column_sumsq(), device.column_sumsq().get(), rtol=1e-4, atol=1e-4)
