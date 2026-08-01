"""Standalone JAX inference port for Chai-1."""

import os

# Grow the allocator on demand while retaining enough headroom for the public
# 2048-token bucket. Explicit process-level settings always take precedence.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

import jax

# Chai's exported Torch components use full FP32 products where tensors have
# not explicitly been cast to BF16. JAX otherwise permits TF32 on NVIDIA GPUs,
# which introduces avoidable parity drift before diffusion amplifies it.
jax.config.update("jax_default_matmul_precision", "highest")

from foldjax.models.chai.inference import (  # noqa: E402
    InferenceConfig,
    InferenceStageUnavailableError,
    prepare_inference,
    run_inference,
)

__version__ = "0.1.0"

__all__ = [
    "InferenceConfig",
    "InferenceStageUnavailableError",
    "prepare_inference",
    "run_inference",
]
