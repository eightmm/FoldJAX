"""Standalone JAX inference port for Chai-1."""

import os

# Grow the allocator on demand while retaining enough headroom for the public
# 2048-token bucket. Explicit process-level settings always take precedence.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

# The matmul precision Chai needs is applied by `chai_precision` around its own
# entry points, not here. `jax.config.update` is process-global, so setting it
# on import re-specified the numerics of every other model sharing the process
# -- and of the many non-Chai tests collected after this module.
from foldjax.models.chai.inference import (
    InferenceConfig,
    InferenceStageUnavailableError,
    chai_precision,
    prepare_inference,
    run_inference,
)

__version__ = "0.1.0"

__all__ = [
    "InferenceConfig",
    "InferenceStageUnavailableError",
    "chai_precision",
    "prepare_inference",
    "run_inference",
]
