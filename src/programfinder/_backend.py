"""Array-module selection and host/device helpers shared by every module."""

from __future__ import annotations

import numpy as np


def backend(device):
    """Return ``(array module, special-function module)`` for ``device``.

    ``"gpu"``/``"cuda"`` select cupy; anything else selects numpy. cupy is
    imported lazily so the CPU path has no GPU dependency.
    """
    if device in ("gpu", "cuda"):
        import cupy as xp
        from cupyx.scipy import special as xsp

        return xp, xsp
    from scipy import special as xsp

    return np, xsp


def gpu_available():
    """True when cupy imports and at least one CUDA device is visible."""
    try:
        import cupy as cp
    except ImportError:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def resolve_device(device):
    """``"auto"`` becomes ``"gpu"`` when one is available, else ``"cpu"``."""
    if device == "auto":
        return "gpu" if gpu_available() else "cpu"
    if device not in ("cpu", "gpu", "cuda"):
        raise ValueError(f"device must be 'auto', 'cpu' or 'gpu', got {device!r}")
    return "gpu" if device == "cuda" else device


def array_module(use_gpu):
    """cupy when asked for and usable, else numpy. Returns ``(module, on_gpu)``."""
    if use_gpu and gpu_available():
        import cupy as cp

        return cp, True
    return np, False


def to_host(array, xp=None):
    """numpy view of ``array`` whether it lives on the host or the device."""
    if xp is not None and hasattr(xp, "asnumpy"):
        return xp.asnumpy(array)
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


# Short alias used by the residual operator code.
_backend = backend
