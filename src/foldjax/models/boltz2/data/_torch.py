"""Bind Boltz-2's vendored featurizer to FoldJAX's NumPy array layer.

The upstream featurizer is written against a small tensor API. FoldJAX provides
that API with NumPy and deliberately has no production switch back to PyTorch:
installing PyTorch, setting an environment variable, or importing this module
from a parity environment cannot change the runtime selected by FoldJAX.

Reference tests that compare against upstream PyTorch inject a test-only module
before importing the vendored featurizer. Keeping that mechanism outside the
package makes the shipped inference path closed over NumPy and JAX.
"""

from __future__ import annotations

from foldjax.models.boltz2.data import _numpy_torch as torch

BACKEND = "numpy"
Tensor = torch.Tensor
from_numpy = torch.from_numpy
