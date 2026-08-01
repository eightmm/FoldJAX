"""Resolve the array library Boltz-2's vendored featurizer runs on.

Real torch when it is installed, and FoldJAX's NumPy stand-in otherwise. The
featurizer is vendored from upstream Boltz and written against torch; this keeps
those files a one-line diff from upstream while letting the torch-free JAX
environment build features.

``FOLDJAX_BOLTZ_FEATURIZER=numpy`` forces the stand-in even when torch is
present, which is how the parity test runs both and compares.
"""

from __future__ import annotations

import os


def _resolve():
    if os.environ.get("FOLDJAX_BOLTZ_FEATURIZER", "").lower() == "numpy":
        from foldjax.models.boltz2.data import _numpy_torch

        return _numpy_torch, "numpy"
    try:
        import torch as _real
    except ModuleNotFoundError:
        from foldjax.models.boltz2.data import _numpy_torch

        return _numpy_torch, "numpy"
    return _real, "torch"


torch, BACKEND = _resolve()
Tensor = torch.Tensor
from_numpy = torch.from_numpy
