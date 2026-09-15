"""JADE (Cardoso & Souloumiac 1993) for the feature-independent ICA arm.

Vendored unchanged from ProgramForge ``benchmarks/jade_ica.py``.

Written from the published algorithm, not ported from `jadeR.py` -- that port
is GPL-3 and vendoring it would relicense this repository.

Contract is identical to picard's
`picard(..., ortho=True, whiten=True, centering=True)`: observations are
columns of `p` (genes or peaks), the estimate is an orthogonal rotation of the
whitened data, and the returned triple satisfies

    sources == unmixing @ whitening @ (p - p.mean(1)[:, None])

exactly. Only the contrast differs: a fourth-order cumulant joint
diagonalisation instead of the tanh log-density.

GPU notes. Both stages run on the device when cupy is importable. The cumulant
stage is a batch of m(m+1)/2 GEMMs of shape (m x T)(T x m) and is bandwidth
bound; the Jacobi stage is latency bound -- each rotation touches only
4 * nbcm * m doubles -- so it is driven with the round-robin pair ordering,
which retires floor(m/2) DISJOINT pairs per launch instead of one. Rotations on
disjoint index pairs commute, and a pair's angle reads only the four entries
(p,p), (p,q), (q,p), (q,q), which the other pairs of the round never touch, so
a batched round is numerically identical to applying those same rotations one
at a time -- it is the classical cyclic Jacobi under a different, equally valid
pair ordering, not an approximation of it.

Peak device memory is one (T x m) whitened copy plus the (nbcm, m, m) cumulant
stack: 88 MB + 26 MB at m=50, T=221330.
"""
from __future__ import annotations

import time

import numpy as np

from ._backend import array_module as _array_module


def round_robin_pairs(m, schedule_seed=None):
    """Circle method: m-1 rounds of disjoint pairs covering every pair once.

    An odd m gets a bye index that is dropped from the round it appears in.

    `schedule_seed` permutes the order of the rounds. Any ordering that covers
    every pair once per sweep is a valid cyclic Jacobi sweep, so this leaves the
    algorithm intact while changing the path it takes. JADE has no random
    initialisation, so re-fitting under permuted schedules is what replaces the
    optimiser-restart check the tanh arm uses: columns that agree across
    schedules are identified by the cumulants, and columns that move are not.
    """
    order = list(range(m))
    bye = None
    if m % 2:
        bye = m
        order.append(bye)
    n = len(order)
    rounds = []
    for _ in range(n - 1):
        pairs = [(order[i], order[n - 1 - i]) for i in range(n // 2)]
        if bye is not None:
            pairs = [(a, b) for a, b in pairs if bye not in (a, b)]
        rounds.append((np.array([a for a, _ in pairs], np.intp),
                       np.array([b for _, b in pairs], np.intp)))
        order = [order[0], order[-1], *order[1:-1]]
    if schedule_seed is not None:
        np.random.default_rng(schedule_seed).shuffle(rounds)
    seen = {tuple(sorted(pair)) for left, right in rounds
            for pair in zip(left.tolist(), right.tolist())}
    if len(seen) != m * (m - 1) // 2:
        raise AssertionError("round-robin schedule does not cover every pair once")
    return rounds


def whiten(p):
    """Centre over observations and whiten to unit-variance components."""
    x = np.asarray(p, np.float64)
    x = x - x.mean(1)[:, None]
    m, n_obs = x.shape
    covariance = (x @ x.T) / n_obs
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if values[-1] <= 0:
        raise ValueError("Whitening requires a full-rank centred loading matrix")
    return (vectors / np.sqrt(values)).T, x


def cumulant_matrices(z, xp):
    """The m(m+1)/2 distinct fourth-order cumulant slices, stacked.

    Off-diagonal slices carry the sqrt(2) weight that makes the stacked
    criterion equal the one over the full, doubly-counted cumulant set.
    """
    m, n_obs = z.shape
    stack = xp.empty((m * (m + 1) // 2, m, m), dtype=z.dtype)
    # Empirical second moments rather than the identity: z is whitened only to
    # working precision, and the cumulant definition wants the moments in hand.
    r = (z @ z.T) / n_obs
    root2 = float(np.sqrt(2.0))
    k = 0
    for i in range(m):
        zi = z[i]
        for j in range(i + 1):
            moment = ((zi * z[j]) * z) @ z.T / n_obs
            slice_ = (moment - r[i, j] * r
                      - xp.outer(r[:, i], r[:, j]) - xp.outer(r[:, j], r[:, i]))
            stack[k] = slice_ if i == j else root2 * slice_
            k += 1
    return stack


def joint_diagonalise(stack, m, threshold, max_sweeps, xp, schedule_seed=None):
    """Batched cyclic Jacobi. Returns (rotation, diagnostics)."""
    rotation = xp.eye(m, dtype=stack.dtype)
    schedule = [(xp.asarray(left), xp.asarray(right))
                for left, right in round_robin_pairs(m, schedule_seed)]
    trace, sweeps, rotations = [], 0, 0
    active = True
    while active and sweeps < max_sweeps:
        active = False
        sweeps += 1
        sweep_max = 0.0
        for left, right in schedule:
            # Angle from the (p,p), (q,q), (p,q) entries only -- untouched by
            # this round's other, index-disjoint pairs.
            g0 = stack[:, left, left] - stack[:, right, right]
            g1 = stack[:, left, right] + stack[:, right, left]
            ton = (g0 * g0 - g1 * g1).sum(0)
            toff = (2.0 * g0 * g1).sum(0)
            hypot = xp.sqrt(ton * ton + toff * toff)
            theta = 0.5 * xp.arctan2(toff, ton + hypot)
            magnitude = xp.abs(theta)
            turning = magnitude > threshold
            count = int(turning.sum())
            sweep_max = max(sweep_max, float(magnitude.max()))
            if not count:
                continue
            active = True
            rotations += count
            theta = xp.where(turning, theta, 0.0)
            cos, sin = xp.cos(theta), xp.sin(theta)
            rows_p, rows_q = stack[:, left, :], stack[:, right, :]
            cr, sr = cos[None, :, None], sin[None, :, None]
            stack[:, left, :] = cr * rows_p + sr * rows_q
            stack[:, right, :] = cr * rows_q - sr * rows_p
            cols_p, cols_q = stack[:, :, left], stack[:, :, right]
            cc, sc = cos[None, None, :], sin[None, None, :]
            stack[:, :, left] = cc * cols_p + sc * cols_q
            stack[:, :, right] = cc * cols_q - sc * cols_p
            vec_p, vec_q = rotation[:, left], rotation[:, right]
            cv, sv = cos[None, :], sin[None, :]
            rotation[:, left] = cv * vec_p + sv * vec_q
            rotation[:, right] = cv * vec_q - sv * vec_p
        trace.append(sweep_max)
    return rotation, dict(sweeps=sweeps, rotations=rotations,
                          converged=not active, threshold=float(threshold),
                          max_angle_final=float(trace[-1]) if trace else None,
                          max_angle_trace=[float(v) for v in trace])


def jade(p, max_sweeps=600, threshold=None, use_gpu=True, schedule_seed=None):
    """Whitening, unmixing and sources, matching picard's return convention.

    `threshold` defaults to Cardoso's 1/sqrt(n_obs)/100 statistical floor:
    below it a rotation is not distinguishable from cumulant sampling noise.
    """
    whitening, centred = whiten(p)
    m, n_obs = centred.shape
    if threshold is None:
        threshold = 1.0 / np.sqrt(n_obs) / 100.0
    xp, on_gpu = _array_module(use_gpu)

    start = time.time()
    z = xp.asarray(whitening @ centred)
    stack = cumulant_matrices(z, xp)
    if on_gpu:
        xp.cuda.Stream.null.synchronize()
    cumulant_seconds = time.time() - start

    start = time.time()
    rotation, diagnostics = joint_diagonalise(
        stack, m, threshold, max_sweeps, xp, schedule_seed)
    if on_gpu:
        xp.cuda.Stream.null.synchronize()
    jacobi_seconds = time.time() - start

    del stack
    unmixing = xp.asnumpy(rotation).T if on_gpu else np.asarray(rotation).T
    sources = unmixing @ whitening @ centred
    diagnostics.update(device="gpu" if on_gpu else "cpu",
                       schedule_seed=schedule_seed,
                       n_observations=int(n_obs), n_components=int(m),
                       n_cumulant_matrices=int(m * (m + 1) // 2),
                       cumulant_seconds=round(cumulant_seconds, 2),
                       jacobi_seconds=round(jacobi_seconds, 2))
    if on_gpu:
        xp.get_default_memory_pool().free_all_blocks()
    return whitening, unmixing, sources, diagnostics
