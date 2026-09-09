"""Matched diagnostic override; never a native-default parity route."""

from contextlib import contextmanager
from functools import wraps
from importlib import import_module
from unittest.mock import patch


@contextmanager
def fp32_atom_aggregation(arm):
    """Accumulate BF16 atom means in FP32, then restore the output dtype.

    Both sum and count/division are widened. This deliberately changes native
    arithmetic and must be recorded as a separate experimental policy.
    """
    if arm == "native":
        import torch

        owner = import_module("protenix.model.modules.transformer")
        bf16 = torch.bfloat16
        def widen(value):
            return value.float()

        def restore(value, dtype):
            return value.to(dtype)
    elif arm == "foldjax":
        import jax.numpy as jnp

        owner = import_module("foldjax.models.protenix.models.diffusion.atom")
        bf16 = jnp.bfloat16
        def widen(value):
            return value.astype(jnp.float32)

        def restore(value, dtype):
            return value.astype(dtype)
    else:
        raise ValueError("unknown atom aggregation control arm")
    original = owner.aggregate_atom_to_token

    @wraps(original)
    def controlled(x_atom, *args, **kwargs):
        if x_atom.dtype != bf16:
            return original(x_atom, *args, **kwargs)
        return restore(original(widen(x_atom), *args, **kwargs), x_atom.dtype)

    with patch.object(owner, "aggregate_atom_to_token", controlled):
        yield
