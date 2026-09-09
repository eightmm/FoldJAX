"""Native input-encoder mixed precision, separate from trunk-wide narrowing."""

import jax
import jax.numpy as jnp

from foldjax.models.protenix.models.primitives.primitives import (
    AutocastLinearParams,
    LayerNormParams,
    LinearParams,
)


def native_input_autocast_params(params):
    """Prepare FP32 publisher parameters for the observed BF16 encoder route.

    Call before any blanket narrowing: widening rounded operands cannot recover
    the FP32 geometry projections or normalization affine values.
    """
    leaves = jax.tree.leaves(params)
    if any(
        hasattr(x, "dtype")
        and jnp.issubdtype(x.dtype, jnp.floating)
        and x.dtype != jnp.float32
        for x in leaves
    ):
        raise ValueError("native input autocast requires original FP32 parameters")

    def convert(value):
        if isinstance(value, LinearParams):
            return AutocastLinearParams(
                value.weight.astype(jnp.bfloat16),
                None if value.bias is None else value.bias.astype(jnp.bfloat16),
            )
        return value

    result = jax.tree.map(
        convert,
        params,
        is_leaf=lambda x: isinstance(x, (LinearParams, LayerNormParams)),
    )
    original = params.atom_encoder.cache
    cache = result.atom_encoder.cache._replace(
        linear_ref_pos=original.linear_ref_pos,
        linear_d=original.linear_d,
    )
    return result._replace(atom_encoder=result.atom_encoder._replace(cache=cache))
