"""JADE: the first tests the module has ever had.

The contract under test is the one the rest of the package relies on:
``sources == unmixing @ whitening @ (p - p.mean(1)[:, None])`` exactly, a
converged joint diagonalisation, planted heavy-tailed sources recovered up to
permutation and sign, and invariance of the *solution* (not the path) to the
Jacobi pair ordering on well-separated data.
"""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from programfinder import _jade
from programfinder._backend import gpu_available

requires_gpu = pytest.mark.skipif(not gpu_available(), reason="no CUDA device")


def _planted(seed=0, m=8, T=20000):
    rng = np.random.default_rng(seed)
    S = rng.laplace(size=(m, T))
    S[: m // 2] = rng.standard_normal((m // 2, T)) ** 3
    A = rng.standard_normal((m, m))
    return A @ S, S


def _matched_abs_corr(a, b):
    corr = np.abs(np.corrcoef(a, b)[: a.shape[0], a.shape[0]:])
    i, j = linear_sum_assignment(-corr)
    return corr[i, j]


def test_round_robin_covers_every_pair_once_per_sweep():
    for m in (4, 7, 12):
        rounds = _jade.round_robin_pairs(m, schedule_seed=3)
        pairs = [tuple(sorted(p)) for left, right in rounds
                 for p in zip(left.tolist(), right.tolist())]
        assert len(pairs) == len(set(pairs)) == m * (m - 1) // 2
        for left, right in rounds:                      # disjoint within a round
            assert len(set(left.tolist()) | set(right.tolist())) == 2 * left.size


def test_whitening_produces_unit_covariance():
    X, _ = _planted()
    W, centred = _jade.whiten(X)
    z = W @ centred
    np.testing.assert_allclose(z @ z.T / z.shape[1], np.eye(z.shape[0]), atol=1e-10)


def test_contract_identity_and_convergence_cpu():
    X, S = _planted()
    whitening, unmixing, sources, diag = _jade.jade(X, use_gpu=False)
    centred = X - X.mean(1)[:, None]
    np.testing.assert_allclose(unmixing @ whitening @ centred, sources, atol=1e-12)
    np.testing.assert_allclose(unmixing @ unmixing.T, np.eye(X.shape[0]), atol=1e-10)
    assert diag["converged"] and diag["device"] == "cpu"
    assert _matched_abs_corr(sources, S).min() > 0.98


def test_solution_is_invariant_to_the_jacobi_schedule_on_separable_data():
    X, _ = _planted(seed=1)
    _, _, base, _ = _jade.jade(X, use_gpu=False)
    for seed in (1, 2):
        _, _, alt, _ = _jade.jade(X, use_gpu=False, schedule_seed=seed)
        assert _matched_abs_corr(base, alt).min() > 0.999


def test_gaussian_data_does_not_get_invented_structure():
    """No fourth-order structure: the fit must not manufacture non-Gaussian sources."""
    from scipy.stats import kurtosis

    rng = np.random.default_rng(4)
    X = rng.standard_normal((5, 5)) @ rng.standard_normal((5, 20000))
    _, _, sources, diag = _jade.jade(X, use_gpu=False)
    assert diag["converged"]
    assert np.abs(kurtosis(sources, axis=1, fisher=True)).max() < 0.2


@requires_gpu
def test_gpu_and_cpu_agree():
    X, _ = _planted(seed=2)
    _, U_cpu, src_cpu, _ = _jade.jade(X, use_gpu=False)
    _, U_gpu, src_gpu, diag = _jade.jade(X, use_gpu=True)
    assert diag["device"] == "gpu"
    assert _matched_abs_corr(src_cpu, src_gpu).min() > 0.9999
