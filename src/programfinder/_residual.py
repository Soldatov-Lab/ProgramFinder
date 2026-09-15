"""Matrix-free Pearson-residual operators backed by sparse count matrices.

Vendored from ProgramForge ``implicit_residual.py`` (operators only; the
fitters were left behind) together with the sparse moment null it needs. Two
nulls are provided:

``NBResidualOperator``
    feature-wise negative-binomial null, ``(X - s r) / sqrt(mu + phi mu^2)``,
    for UMI counts;
``BernoulliResidualOperator``
    depth-adjusted Bernoulli null on detections, ``(x - p) / sqrt(p (1 - p))``
    with ``p_ij = 1 - (1 - q_j)^{d_i}``, for accessibility-like data.

Both expose ``matmat`` / ``rmatmat`` / ``column_sumsq`` so a randomized PCA can
run without ever materialising the cells x features matrix.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse

from ._backend import _backend

__all__ = ["NBResidualOperator", "BernoulliResidualOperator"]


# --------------------------------------------------------------------------
# sparse method-of-moments NB null (from ProgramForge ``sparse_counts.py``)
# --------------------------------------------------------------------------

def _split_mask(rows, columns, seed, validation_fraction, xp):
    """Stateless entry split, materialized for one feature block only."""
    r = xp.arange(rows, dtype=xp.uint64)[:, None]
    c = xp.asarray(columns, dtype=xp.uint64)[None, :]
    value = (r * xp.uint64(0x9E3779B185EBCA87) + c * xp.uint64(0xC2B2AE3D27D4EB4F)
             + xp.uint64(seed + 1) * xp.uint64(0x165667B19E3779F9))
    value ^= value >> xp.uint64(29)
    threshold = int(validation_fraction * (2**32 - 1))
    return (value & xp.uint64(0xFFFFFFFF)) <= xp.uint64(threshold)


def _blocks(X, block_size):
    Xc = X.tocsc()
    for start in range(0, X.shape[1], block_size):
        stop = min(start + block_size, X.shape[1])
        yield start, stop, Xc[:, start:stop].toarray()


def _null_parameters(X, size_factors, *, seed=None, validation_fraction=0.0,
                     block_size=512):
    """Sparse method-of-moments null without a cells-by-features allocation."""
    s = np.asarray(size_factors, dtype=np.float64)
    if seed is None or validation_fraction == 0:
        sums = np.asarray(X.sum(axis=0)).ravel()
        denominator_s = np.full(X.shape[1], s.sum())
        x2 = np.asarray(X.power(2).sum(axis=0)).ravel()
        sx = np.asarray(X.T @ s).ravel()
        sum_s2 = np.full(X.shape[1], np.square(s).sum())
    else:
        sums = np.empty(X.shape[1]); denominator_s = np.empty(X.shape[1])
        x2 = np.empty(X.shape[1]); sx = np.empty(X.shape[1])
        sum_s2 = np.empty(X.shape[1])
        for start, stop, xb in _blocks(X, block_size):
            train = ~_split_mask(X.shape[0], np.arange(start, stop), seed,
                                validation_fraction, np)
            observed = xb * train
            sums[start:stop] = observed.sum(axis=0)
            denominator_s[start:stop] = (train * s[:, None]).sum(axis=0)
            x2[start:stop] = (np.square(xb) * train).sum(axis=0)
            sx[start:stop] = (observed * s[:, None]).sum(axis=0)
            sum_s2[start:stop] = (train * np.square(s)[:, None]).sum(axis=0)
    rate = np.maximum(sums / np.maximum(denominator_s, 1e-12), 1e-10)
    # sum_i mask*((x_i-s_i*r)^2-s_i*r), from blockwise sufficient statistics.
    numerator = x2 - 2 * rate * sx + rate**2 * sum_s2 - rate * denominator_s
    denominator = rate**2 * sum_s2
    phi = np.clip(numerator / np.maximum(denominator, 1e-12), 1e-4, 1e3)
    return np.log(rate), phi


def _blocks(X, block_size):
    Xc = X.tocsc()


# --------------------------------------------------------------------------
# operators (from ProgramForge ``implicit_residual.py``)
# --------------------------------------------------------------------------

def _warn_wide_feature_ratio(shape, stacklevel=3):
    if shape[1] > 2 * shape[0]:
        warnings.warn(
            f"n_features/n_cells = {shape[1] / shape[0]:.1f}. "
            "Empirically, program-count selection degrades above ~2 and "
            "fails around ~10 on planted-truth simulations, so the returned "
            "program count may be unreliable in this regime. This is an "
            "empirical regime warning, not a validity boundary.",
            RuntimeWarning, stacklevel=stacklevel,
        )


def _positive_int(value, name):
    """Coerce ``value`` to a positive int, rejecting bools and non-integral input.

    ``True`` is an ``int`` in Python, and a silently accepted ``block_size=True``
    would block over one feature at a time for the life of the operator, so it
    is rejected rather than coerced.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from None
    if coerced != value or coerced <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return coerced


def _validate_depth_bins(value):
    """``None`` for the exact blockwise mode, else a positive bin count."""
    return None if value is None else _positive_int(value, "depth_bins")


def _equal_count_bins(values, n_bins):
    """Equal-count bins over ``values``, returned as a per-cell bin index."""
    n_cells = values.shape[0]
    order = np.argsort(values, kind="stable")
    bin_index = np.empty(n_cells, dtype=np.int32)
    bin_index[order] = np.minimum(
        np.arange(n_cells) * n_bins // n_cells, n_bins - 1
    )
    return bin_index


def _validate_bin_assignment(assignment, n_cells):
    """Check an externally supplied ``(bin_index, bin_size_factors)`` pair.

    Exists so a fit on a *subset of the cells* can keep the depth-binned null of
    the full data. Re-deriving equal-count bins inside the subset moves every
    bin boundary, which perturbs ``Z`` by the per-bin step in ``log s``; passing
    the parent's assignment instead makes the subset's residual matrix exactly
    the parent's rows. Empty bins are allowed and cost nothing: ``bin_counts``
    is zero there and every bin-indexed contraction skips them.
    """
    bin_index, bin_size_factors = assignment
    bin_index = np.asarray(bin_index, dtype=np.int32)
    bin_size_factors = np.asarray(bin_size_factors, dtype=np.float64)
    if bin_index.shape != (n_cells,):
        raise ValueError("bin_assignment[0] must contain one bin per cell; got "
                         f"{bin_index.shape} for {n_cells} cells")
    if bin_size_factors.ndim != 1 or bin_size_factors.size == 0:
        raise ValueError("bin_assignment[1] must be a non-empty 1-D array of "
                         "per-bin size factors")
    if bin_index.size and (bin_index.min() < 0
                           or bin_index.max() >= bin_size_factors.size):
        raise ValueError("bin_assignment[0] indexes outside "
                         f"bin_assignment[1] ({bin_size_factors.size} bins)")
    if not np.all(bin_size_factors > 0):
        raise ValueError("bin_assignment[1] must be positive")
    return bin_index, bin_size_factors


class _ImplicitResidualOperator:
    """Matrix-free algebra shared by the feature-wise residual operators.

    A subclass supplies a sparse observation matrix and five hooks; every
    product below is written once in terms of them:

    ``_residual_block``
        the exact residual for one feature block, from densified observations;
    ``_residual_null_block``
        the dense part of ``Z`` over depth bins -- that is, ``-Z`` at a zero
        observation;
    ``weight_block``
        the Fisher weight ``w`` of the null, over depth bins;
    ``_root_weight_block``
        ``sqrt(w)``;
    ``_root_weighted_null_block``
        the dense part of ``sqrt(w) * Z`` over depth bins.

    The three null-side hooks are kept separate on purpose. Which of them
    coincide is a property of the null model, not of this algebra:

    ===========  =====================  ==============  ======================
    null         residual null          Fisher weight   root-weighted null
    ===========  =====================  ==============  ======================
    NB           ``sqrt(w)``            ``w``           ``w``
    Bernoulli    ``sqrt(p/(1-p))``      ``p(1-p)``      ``p``
    ===========  =====================  ==============  ======================

    For the NB null the first and third columns collapse onto ``sqrt(w)`` and
    ``w``; for the Bernoulli null neither collapse holds, so reusing an NB
    identity would silently fit the wrong model.
    """

    #: set by the subclass: sparse CSC observations, cells x features
    _observations = None

    def _nonzero_coordinates(self):
        csr = self._observations.tocsr()
        rows = np.repeat(np.arange(self.shape[0]), np.diff(csr.indptr))
        return csr, rows, csr.indices

    def _cache_binned_observations(self, csr, weighted_data, root_weighted_data):
        """Store the observed nonzeros of ``Z`` and of ``sqrt(w) * Z``."""
        weighted = csr.copy()
        weighted.data = np.asarray(weighted_data, dtype=np.float32)
        self._weighted_counts_host = weighted
        scaled = csr.copy()
        scaled.data = np.asarray(root_weighted_data, dtype=np.float32)
        self._root_weighted_counts_host = scaled
        self._weighted_counts = None
        self._root_weighted_counts = None

    # -- hooks ---------------------------------------------------------------

    def _residual_block(self, start, stop, observations, xp):
        raise NotImplementedError

    def _residual_null_block(self, start, stop, xp):
        raise NotImplementedError

    def weight_block(self, start, stop, xp):
        raise NotImplementedError

    def _root_weight_block(self, start, stop, xp):
        raise NotImplementedError

    def _root_weighted_null_block(self, start, stop, xp):
        raise NotImplementedError

    # -- shared machinery ----------------------------------------------------

    def _require_bins(self, what):
        if self.depth_bins is None:
            raise ValueError(
                f"{what} needs the depth-binned representation: the design "
                "weight varies with cell depth, and the binning is what keeps "
                "it matrix-free. Construct the operator with depth_bins set."
            )

    def _device_weighted_counts(self, xp):
        if self._weighted_counts is None:
            if self.device in ("gpu", "cuda"):
                from cupyx.scipy import sparse as cupy_sparse
                self._weighted_counts = cupy_sparse.csr_matrix(self._weighted_counts_host)
            else:
                self._weighted_counts = self._weighted_counts_host
        return self._weighted_counts

    def _device_root_weighted_counts(self, xp):
        if self._root_weighted_counts is None:
            if self.device in ("gpu", "cuda"):
                from cupyx.scipy import sparse as cupy_sparse
                self._root_weighted_counts = cupy_sparse.csr_matrix(
                    self._root_weighted_counts_host
                )
            else:
                self._root_weighted_counts = self._root_weighted_counts_host
        return self._root_weighted_counts

    def _bin_aggregate(self, left, xp):
        """Sum the rows of ``left`` within each depth bin."""
        out = xp.zeros((self.n_bins, left.shape[1]), dtype=xp.float32)
        xp.add.at(out, xp.asarray(self.bin_index), left)
        return out

    def _to_numpy(self, array):
        xp, _ = _backend(self.device)
        return xp.asnumpy(array) if self.device in ("gpu", "cuda") else np.asarray(array)

    def _blocks(self):
        for start in range(0, self.shape[1], self.block_size):
            stop = min(start + self.block_size, self.shape[1])
            yield start, stop, self._observations[:, start:stop].toarray()

    def _feature_blocks(self):
        for start in range(0, self.shape[1], self.block_size):
            yield start, min(start + self.block_size, self.shape[1])

    def root_weighted_rmatmat(self, left):
        """Return ``(sqrt(w) * Z).T @ left`` without constructing either matrix.

        The observed term rides on the precomputed nonzeros and the dense null
        term collapses onto the depth bins exactly as it does for ``Z`` itself.
        """
        self._require_bins("root_weighted_rmatmat")
        xp, _ = _backend(self.device)
        left = xp.asarray(left, dtype=xp.float32)
        if left.ndim != 2 or left.shape[0] != self.shape[0]:
            raise ValueError("left must have shape (n_cells, rank)")
        out = self._device_root_weighted_counts(xp).T @ left
        aggregate = self._bin_aggregate(left, xp)
        for start, stop in self._feature_blocks():
            out[start:stop] -= self._root_weighted_null_block(start, stop, xp).T @ aggregate
        return out

    def root_weighted_matmat(self, right):
        """Return ``(sqrt(w) * Z) @ right`` without constructing either matrix."""
        self._require_bins("root_weighted_matmat")
        xp, _ = _backend(self.device)
        right = xp.asarray(right, dtype=xp.float32)
        if right.ndim != 2 or right.shape[0] != self.shape[1]:
            raise ValueError("right must have shape (n_features, rank)")
        out = self._device_root_weighted_counts(xp) @ right
        null_by_bin = xp.zeros((self.n_bins, right.shape[1]), dtype=xp.float32)
        for start, stop in self._feature_blocks():
            null_by_bin += self._root_weighted_null_block(start, stop, xp) @ right[start:stop]
        return out - null_by_bin[xp.asarray(self.bin_index)]

    def dense_binned(self):
        """``(Z, sqrt(w))`` as dense arrays. Tests and small references only."""
        self._require_bins("dense_binned")
        xp, _ = _backend(self.device)
        root_weight = xp.empty(self.shape, dtype=xp.float32)
        residual_null = xp.empty(self.shape, dtype=xp.float32)
        groups = xp.asarray(self.bin_index)
        for start, stop in self._feature_blocks():
            root_weight[:, start:stop] = self._root_weight_block(start, stop, xp)[groups]
            residual_null[:, start:stop] = self._residual_null_block(start, stop, xp)[groups]
        observed = self._device_weighted_counts(xp)
        residual = xp.asarray(observed.toarray(), dtype=xp.float32) - residual_null
        return self._to_numpy(residual), self._to_numpy(root_weight)

    def matmat(self, right):
        """Return ``Z @ right`` without constructing ``Z``."""
        xp, _ = _backend(self.device)
        right = xp.asarray(right, dtype=xp.float32)
        if right.ndim != 2 or right.shape[0] != self.shape[1]:
            raise ValueError("right must have shape (n_features, rank)")
        if self.depth_bins is not None:
            out = self._device_weighted_counts(xp) @ right
            null_by_bin = xp.zeros((self.n_bins, right.shape[1]), dtype=xp.float32)
            for start, stop in self._feature_blocks():
                null_by_bin += self._residual_null_block(start, stop, xp) @ right[start:stop]
            return out - null_by_bin[xp.asarray(self.bin_index)]
        out = xp.zeros((self.shape[0], right.shape[1]), dtype=xp.float32)
        for start, stop, observations in self._blocks():
            out += self._residual_block(start, stop, observations, xp) @ right[start:stop]
        return out

    def rmatmat(self, left):
        """Return ``Z.T @ left`` without constructing ``Z``."""
        xp, _ = _backend(self.device)
        left = xp.asarray(left, dtype=xp.float32)
        if left.ndim != 2 or left.shape[0] != self.shape[0]:
            raise ValueError("left must have shape (n_cells, rank)")
        if self.depth_bins is not None:
            out = self._device_weighted_counts(xp).T @ left
            aggregate = self._bin_aggregate(left, xp)
            for start, stop in self._feature_blocks():
                out[start:stop] -= self._residual_null_block(start, stop, xp).T @ aggregate
            return out
        out = xp.empty((self.shape[1], left.shape[1]), dtype=xp.float32)
        for start, stop, observations in self._blocks():
            out[start:stop] = self._residual_block(start, stop, observations, xp).T @ left
        return out

    def column_sumsq(self):
        """Return ``sum_i Z_ij**2`` with one feature block of work memory."""
        xp, _ = _backend(self.device)
        if self.depth_bins is not None:
            out = xp.empty(self.shape[1], dtype=xp.float32)
            weighted = self._weighted_counts_host
            for start, stop in self._feature_blocks():
                null = self._residual_null_block(start, stop, xp)
                base = (xp.asarray(self.bin_counts)[:, None] * null**2).sum(axis=0)
                block = weighted[:, start:stop].tocoo()
                if block.nnz:
                    values = xp.asarray(block.data)
                    groups = xp.asarray(self.bin_index[block.row])
                    columns = xp.asarray(block.col)
                    correction = values**2 - 2 * values * null[groups, columns]
                    xp.add.at(base, columns, correction)
                out[start:stop] = base
            return out
        out = xp.empty(self.shape[1], dtype=xp.float32)
        for start, stop, observations in self._blocks():
            residual = self._residual_block(start, stop, observations, xp)
            out[start:stop] = xp.square(residual).sum(axis=0)
        return out

    def toarray(self):
        """Materialize for tests and small reference benchmarks only."""
        xp, _ = _backend(self.device)
        out = xp.empty(self.shape, dtype=xp.float32)
        for start, stop, observations in self._blocks():
            out[:, start:stop] = self._residual_block(start, stop, observations, xp)
        return self._to_numpy(out)


class NBResidualOperator(_ImplicitResidualOperator):
    """Matrix-free Pearson residuals under a feature-wise NB null model.

    The conceptual matrix is ``(X - s * rate) / sqrt(mu + phi * mu**2)``.
    Counts stay sparse on the host and only one feature block is densified on
    the selected compute device.
    """

    def __init__(self, counts, size_factors=None, *, block_size=512,
                 depth_bins=256, device="gpu", baseline=None, dispersion=None,
                 bin_assignment=None):
        self.counts = sparse.csc_matrix(counts, dtype=np.float32)
        self.counts.sum_duplicates()
        self.counts.eliminate_zeros()
        self._observations = self.counts
        self.shape = self.counts.shape
        _warn_wide_feature_ratio(self.shape)
        depth = np.asarray(self.counts.sum(axis=1)).ravel().astype(np.float64)
        self.size_factors = (depth / depth.mean() if size_factors is None
                             else np.asarray(size_factors, dtype=np.float64))
        if self.size_factors.shape != (self.shape[0],):
            raise ValueError("size_factors must contain one value per cell")
        if (baseline is None) != (dispersion is None):
            raise ValueError("supply baseline and dispersion together, or neither")
        if baseline is None:
            # Program-free method of moments: with nothing explaining the
            # programs, their structure is absorbed into the dispersion.
            # `refit_null_given_programs` is the corrected version.
            self.baseline, self.dispersion = _null_parameters(
                self.counts.tocsr(), self.size_factors
            )
        else:
            self.baseline = np.asarray(baseline, dtype=np.float64)
            self.dispersion = np.asarray(dispersion, dtype=np.float64)
            if self.baseline.shape != (self.shape[1],) or \
                    self.dispersion.shape != (self.shape[1],):
                raise ValueError("baseline and dispersion need one value per feature")
        self.rate = np.exp(self.baseline)
        self.effect_link = "log"
        self.block_size = _positive_int(block_size, "block_size")
        self.depth_bins = _validate_depth_bins(depth_bins)
        self.device = device
        self._weighted_counts = None
        if self.depth_bins is not None:
            self._prepare_binned_null(self.depth_bins, bin_assignment)
        elif bin_assignment is not None:
            raise ValueError("bin_assignment needs depth_bins; the exact "
                             "blockwise mode has no bins to assign")

    @property
    def n_bins(self):
        return len(self.bin_size_factors)

    def _prepare_binned_null(self, n_bins, assignment=None):
        """Cache sparse observed terms and a compact depth-binned null model.

        ``assignment`` overrides the equal-count binning with a supplied
        ``(bin_index, bin_size_factors)`` pair -- see
        :func:`_validate_bin_assignment` for why a cell subset wants its
        parent's bins rather than its own.
        """
        if assignment is None:
            log_s = np.log(self.size_factors)
            n_actual = min(n_bins, self.shape[0])
            self.bin_index = _equal_count_bins(log_s, n_actual)
            self.bin_size_factors = np.asarray(
                [np.exp(log_s[self.bin_index == group].mean())
                 for group in range(n_actual)]
            )
        else:
            self.bin_index, self.bin_size_factors = _validate_bin_assignment(
                assignment, self.shape[0])
            n_actual = self.bin_size_factors.size
        self.bin_counts = np.bincount(self.bin_index, minlength=n_actual).astype(np.float32)

        csr, rows, cols = self._nonzero_coordinates()
        mean = self.size_factors[rows] * self.rate[cols]
        inv_sd = 1 / np.sqrt(np.maximum(mean + self.dispersion[cols] * mean**2, 1e-8))
        weighted_data = csr.data * inv_sd

        # The Fisher weight of the same null, gathered onto the nonzeros. The
        # design-side products need `sqrt(w) * Z`, and for the NB null `sqrt(w)`
        # is also the residual null block, so this is that same quantity carried
        # onto the observed terms once rather than per iteration.
        bin_mean = self.bin_size_factors[self.bin_index[rows]] * self.rate[cols]
        root_weight = np.sqrt(
            bin_mean / np.maximum(1 + self.dispersion[cols] * bin_mean, 1e-8)
        )
        self._cache_binned_observations(csr, weighted_data,
                                        weighted_data * root_weight)

    def _residual_null_block(self, start, stop, xp):
        """``sqrt(w)`` over depth bins, which is also ``-Z`` at a zero count.

        With ``m = s_b * rate`` and Fisher weight ``w = m / (1 + phi m)``, the
        Pearson residual satisfies ``Z = sqrt(w) * t`` for the working target
        ``t = (x - m) / m``. At ``x = 0`` that is ``-sqrt(w)``, so the single
        array serves both as the null contribution to ``Z`` and as the design
        weight the weighted fit needs. This coincidence is specific to the NB
        null; see :class:`BernoulliResidualOperator`.
        """
        s = xp.asarray(self.bin_size_factors, dtype=xp.float32)[:, None]
        rate = xp.asarray(self.rate[start:stop], dtype=xp.float32)[None, :]
        phi = xp.asarray(self.dispersion[start:stop], dtype=xp.float32)[None, :]
        mean = s * rate
        return xp.sqrt(mean / xp.maximum(1 + phi * mean, 1e-8))

    #: retained name; the NB residual null and ``sqrt(w)`` are the same array
    _null_block = _residual_null_block
    _root_weight_block = _residual_null_block

    def weight_block(self, start, stop, xp):
        """Fisher weights ``w`` for one feature block, over depth bins."""
        return xp.square(self._residual_null_block(start, stop, xp))

    def _root_weighted_null_block(self, start, stop, xp):
        """``-sqrt(w) * Z`` at a zero count, which for the NB null is ``w``."""
        return self.weight_block(start, stop, xp)

    def _residual_block(self, start, stop, counts, xp):
        s = xp.asarray(self.size_factors, dtype=xp.float32)[:, None]
        rate = xp.asarray(self.rate[start:stop], dtype=xp.float32)[None, :]
        phi = xp.asarray(self.dispersion[start:stop], dtype=xp.float32)[None, :]
        mean = s * rate
        scale = xp.sqrt(xp.maximum(mean + phi * mean**2, 1e-8))
        return (xp.asarray(counts, dtype=xp.float32) - mean) / scale

def _fit_detection_rates(depths, detection_totals, block_size, xp=np, n_iter=40):
    """Per-feature detection rate ``q_j`` of the depth-adjusted Bernoulli null.

    Solves ``sum_i [1 - (1 - q_j)^d_i] = D_j`` -- the moment condition that
    makes the expected number of detections equal the observed one -- by
    bisection on the geometric midpoint of ``[1e-12, 1]``, which is the same
    bracket and the same update rule Hotspot's Bernoulli model uses. The
    left-hand side is increasing in ``q_j``, so the bracket is valid throughout.

    Probabilities are evaluated as ``-expm1(d * log1p(-q))`` rather than
    ``1 - (1 - q)**d`` so that the tiny-``q`` regime -- where ``(1 - q)**d``
    rounds to one -- keeps full relative precision. Work is blocked over
    features, so the transient is ``n_cells x block_size`` and never
    ``n_cells x n_features``.

    Features with no detections get ``q_j = 0``. Features detected in every
    positive-depth cell drive ``q_j`` to the top of the bracket, where the
    fitted probability is one and the null has no variance left; those columns
    are neutralized downstream rather than here.
    """
    depths = xp.asarray(depths, dtype=xp.float64)
    detection_totals = xp.asarray(detection_totals, dtype=xp.float64)
    n_features = int(detection_totals.shape[0])
    rate = xp.zeros(n_features, dtype=xp.float64)
    for start in range(0, n_features, block_size):
        stop = min(start + block_size, n_features)
        target = detection_totals[start:stop]
        low = xp.full(stop - start, 1e-12, dtype=xp.float64)
        high = xp.ones(stop - start, dtype=xp.float64)
        for _ in range(n_iter):
            mid = xp.sqrt(low * high)
            total = (-xp.expm1(depths[:, None] * xp.log1p(-mid)[None, :])).sum(axis=0)
            go_low = total > target
            high = xp.where(go_low, mid, high)
            low = xp.where(go_low, low, mid)
        # Clamp strictly below one so `log1p(-q)` stays finite: an always-detected
        # feature is handled by the zero-variance mask, not by an infinity.
        rate[start:stop] = xp.minimum(xp.sqrt(low * high), 1.0 - 1e-12)
    rate = xp.where(detection_totals == 0, 0.0, rate)
    return np.asarray(rate) if xp is np else xp.asnumpy(rate)


class BernoulliResidualOperator(_ImplicitResidualOperator):
    """Matrix-free Pearson residuals under a depth-adjusted Bernoulli null.

    Raw counts are binarized to detections ``x_ij in {0, 1}``. Each feature gets
    a rate ``q_j`` fitted so the expected detection total matches the observed
    one, the per-cell detection probability is ``p_ij = 1 - (1 - q_j)^{d_i}``,
    and the conceptual matrix is ``(x - p) / sqrt(p (1 - p))``. This is the
    model Hotspot uses for accessibility-like data, where what a fragment count
    carries is mostly whether the region was seen at all.

    The sparse decomposition is ``Z = x / sqrt(p(1-p)) - sqrt(p/(1-p))``. Note
    that the dense null term is **not** ``sqrt(w)`` for the Fisher weight
    ``w = p(1-p)``, unlike the NB case -- see :class:`_ImplicitResidualOperator`.
    Under root weighting the identity is much simpler: ``sqrt(w) * Z = x - p``,
    so every observed nonzero of the root-weighted residual is exactly one and
    the dense term is ``p`` itself.

    Parameters
    ----------
    counts : sparse or dense array, cells x features
        Raw counts. Anything strictly positive counts as a detection; zeros and
        non-positive entries do not.
    depths : array of shape (n_cells,), optional
        Per-cell sequencing depth of *this modality*, the ``d_i`` above. Supply
        it explicitly: for ATAC this should be the cell's fragment (or total
        insertion) depth, which is generally computed over all peaks and is not
        recoverable from a filtered peak submatrix. The default -- row sums of
        the counts passed in -- is a convenience for the case where the matrix
        already is the whole modality.
    detection_rate : array of shape (n_features,), optional
        Pre-fitted ``q_j``. Skips the bisection.
    depth_bins : int or None
        ``None`` evaluates ``p`` exactly, one feature block at a time. An
        integer caches a ``depth_bins x n_features`` table of binned
        probabilities, which is what makes the design-side products
        matrix-free.
    """

    def __init__(self, counts, depths=None, *, block_size=512, depth_bins=256,
                 device="gpu", detection_rate=None, n_bisection_iterations=40):
        raw = sparse.csc_matrix(counts, dtype=np.float32)
        raw.sum_duplicates()
        raw.eliminate_zeros()
        self.shape = raw.shape
        _warn_wide_feature_ratio(self.shape)
        detections = raw.copy()
        detections.data = (raw.data > 0).astype(np.float32)
        detections.eliminate_zeros()
        self.detections = detections
        self._observations = detections
        self.binarized = bool((raw.data > 1).any())

        if depths is None:
            depths = np.asarray(raw.sum(axis=1)).ravel()
        self.depths = np.asarray(depths, dtype=np.float64).ravel()
        if self.depths.shape != (self.shape[0],):
            raise ValueError("depths must contain one value per cell")
        if not np.all(np.isfinite(self.depths)):
            raise ValueError("depths must be finite")
        if np.any(self.depths < 0):
            raise ValueError("depths must be non-negative")
        zero_depth = self.depths == 0
        if zero_depth.any():
            # p_ij is identically zero at d_i = 0, so a detection there is not
            # merely unlikely under the null -- it has zero probability, and the
            # moment condition for q_j has no solution.
            offending = detections[zero_depth].nnz
            if offending:
                raise ValueError(
                    f"{offending} detection(s) fall in cells with zero depth. A "
                    "zero-depth cell cannot detect anything under this null; the "
                    "depths supplied do not describe the matrix passed in."
                )

        self.detection_totals = np.asarray(detections.sum(axis=0)).ravel().astype(np.float64)
        # Fisher weighting here is the logit link's, so the weighted fit's
        # effects are not log fold changes and must not be reported as such.
        self.effect_link = "logit"
        self.block_size = _positive_int(block_size, "block_size")
        self.depth_bins = _validate_depth_bins(depth_bins)
        self.device = device
        self.n_bisection_iterations = _positive_int(
            n_bisection_iterations, "n_bisection_iterations"
        )
        self._weighted_counts = None
        self._root_weighted_counts = None
        self._bin_p_device = None
        if detection_rate is None:
            xp, _ = _backend(device)
            self.detection_rate = _fit_detection_rates(
                self.depths, self.detection_totals, self.block_size, xp,
                self.n_bisection_iterations,
            )
        else:
            self.detection_rate = np.asarray(detection_rate, dtype=np.float64).ravel()
            if self.detection_rate.shape != (self.shape[1],):
                raise ValueError("detection_rate needs one value per feature")
            if not np.all(np.isfinite(self.detection_rate)):
                raise ValueError("detection_rate must be finite")
            # Order matters: NaN compares False against both bounds, so the
            # finiteness check has to come first for this one to mean anything.
            if np.any(self.detection_rate < 0) or np.any(self.detection_rate >= 1):
                raise ValueError("detection_rate must lie in [0, 1)")
            impossible = (self.detection_rate == 0) & (self.detection_totals > 0)
            if impossible.any():
                raise ValueError(
                    f"{int(impossible.sum())} feature(s) have detection_rate 0 but a "
                    "non-zero detection total. That null assigns probability zero to "
                    "observations that did occur, so the residual is undefined rather "
                    "than merely large. detection_rate 0 is only meaningful for a "
                    "feature detected nowhere."
                )
        if self.depth_bins is not None:
            self._prepare_binned_null(self.depth_bins)

    @property
    def n_bins(self):
        return self.bin_p.shape[0]

    def _detection_probability(self, start, stop, xp=np):
        """``p_ij = 1 - (1 - q_j)^{d_i}`` for one feature block, in float64."""
        log1m_q = xp.asarray(np.log1p(-self.detection_rate[start:stop]), dtype=xp.float64)
        depths = xp.asarray(self.depths, dtype=xp.float64)[:, None]
        return -xp.expm1(depths * log1m_q[None, :])

    def _prepare_binned_null(self, n_bins):
        """Cache the observed nonzeros and a depth-binned probability table."""
        n_actual = min(n_bins, self.shape[0])
        zero_depth = self.depths == 0
        if zero_depth.any() and not zero_depth.all():
            # Zero-depth cells get a bin to themselves. They detect nothing under
            # this null, and pooling them with shallow cells would hand them a
            # non-zero binned probability and so a spurious residual.
            positive = ~zero_depth
            n_positive = max(1, min(n_actual - 1, int(positive.sum())))
            self.bin_index = np.zeros(self.shape[0], dtype=np.int32)
            self.bin_index[positive] = 1 + _equal_count_bins(
                self.depths[positive], n_positive
            )
        elif zero_depth.all():
            self.bin_index = np.zeros(self.shape[0], dtype=np.int32)
        else:
            self.bin_index = _equal_count_bins(self.depths, n_actual)
        counts_per_bin = np.bincount(self.bin_index)
        self.bin_counts = counts_per_bin.astype(np.float32)
        n_actual = counts_per_bin.shape[0]
        # Average the *exact* p_ij inside each bin rather than re-evaluating the
        # null at a representative depth. p is concave in depth, so the latter
        # would bias the binned expected detection total away from D_j; this way
        # sum_b n_b * pbar_bj is still exactly D_j, which is what q_j was fitted
        # to. Every cached quantity below is then built from this same pbar, so
        # the observed and null terms of Z stay consistent with one another.
        averager = sparse.csr_matrix(
            (1.0 / counts_per_bin[self.bin_index].astype(np.float64),
             (self.bin_index, np.arange(self.shape[0]))),
            shape=(n_actual, self.shape[0]),
        )
        # float64, and not negotiable: p sits arbitrarily close to one for a
        # near-ubiquitous feature, and float32 rounds 1 - 2**-26 to exactly 1.
        # That turns a Pearson residual of -8192 at a zero observation into a
        # silent 0, because every derived quantity goes through 1 - p. Storage
        # is 8 bytes per (bin, feature); at the 256-bin default that is 2 KB per
        # feature, which is the price of the design-side products being
        # matrix-free at all.
        self.bin_p = np.empty((n_actual, self.shape[1]), dtype=np.float64)
        for start, stop in self._feature_blocks():
            self.bin_p[:, start:stop] = averager @ self._detection_probability(start, stop)
        np.clip(self.bin_p, 0.0, 1.0, out=self.bin_p)

        csr, rows, cols = self._nonzero_coordinates()
        p = self.bin_p[self.bin_index[rows], cols]
        variance = p * (1.0 - p)
        usable = variance > 0
        # Z = x / sqrt(p(1-p)) - sqrt(p/(1-p)) and sqrt(w) Z = x - p, with x = 1
        # at every stored nonzero. Degenerate columns contribute nothing, which
        # is what `cp.where(std > 0, ..., 0)` does in the dense reference.
        self._cache_binned_observations(
            csr,
            np.where(usable, csr.data / np.sqrt(np.where(usable, variance, 1.0)), 0.0),
            np.where(usable, csr.data, 0.0),
        )

    def _device_bin_p(self, xp):
        if self._bin_p_device is None:
            self._bin_p_device = xp.asarray(self.bin_p, dtype=xp.float64)
        return self._bin_p_device

    def _binned_p(self, start, stop, xp):
        """Binned ``p`` for one feature block, in float64.

        Every hook below derives from this in float64 and casts only its own
        result. The cast is safe because each result is a value rather than a
        difference of near-equal ones: ``sqrt(p/(1-p))`` and ``p(1-p)`` are the
        quantities that carry the information near ``p = 1``, and both stay
        comfortably inside float32 range once evaluated.
        """
        return self._device_bin_p(xp)[:, start:stop]

    def weight_block(self, start, stop, xp):
        """Fisher weights of the logit link, ``w = p (1 - p)``, over depth bins."""
        p = self._binned_p(start, stop, xp)
        return (p * (1 - p)).astype(xp.float32)

    def _root_weight_block(self, start, stop, xp):
        p = self._binned_p(start, stop, xp)
        return xp.sqrt(p * (1 - p)).astype(xp.float32)

    def _residual_null_block(self, start, stop, xp):
        """``-Z`` at a zero detection, which is ``sqrt(p / (1 - p))``.

        This is *not* ``sqrt(w)``. The Bernoulli Pearson residual divides by
        ``sqrt(p(1-p))`` while the Fisher weight multiplies by it, so the two
        differ by a factor of ``p(1-p)`` rather than coinciding as they do for
        the NB null.
        """
        p = self._binned_p(start, stop, xp)
        variance = p * (1 - p)
        usable = variance > 0
        null = xp.where(usable, xp.sqrt(p / xp.where(usable, 1 - p, 1)), 0)
        return null.astype(xp.float32)

    def _root_weighted_null_block(self, start, stop, xp):
        """``-sqrt(w) * Z`` at a zero detection, which is just ``p``."""
        p = self._binned_p(start, stop, xp)
        return xp.where(p * (1 - p) > 0, p, 0).astype(xp.float32)

    def _residual_block(self, start, stop, detections, xp):
        p = self._detection_probability(start, stop, xp)
        variance = p * (1 - p)
        usable = variance > 0
        x = xp.asarray(detections, dtype=xp.float64)
        residual = xp.where(
            usable, (x - p) / xp.sqrt(xp.where(usable, variance, 1.0)), 0.0
        )
        return residual.astype(xp.float32)

