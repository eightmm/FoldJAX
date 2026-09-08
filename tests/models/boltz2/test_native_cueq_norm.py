import jax
import jax.numpy as jnp
import pytest

from foldjax.models.boltz2.models.primitives.native_cueq_norm import (
    native_cueq_norm,
    native_cueq_output_norm,
)


@pytest.mark.parametrize(
    "shape,dtype,message",
    [
        ((), jnp.float32, "width 128"),
        ((2, 127), jnp.float32, "width 128"),
        ((2, 128), jnp.bfloat16, "FP32"),
        ((0, 128), jnp.float32, "one row"),
    ],
)
def test_rejects_invalid_kernel_boundary_before_launch(shape, dtype, message):
    x = jax.ShapeDtypeStruct(shape, dtype)
    affine = jax.ShapeDtypeStruct((128,), jnp.float32)
    with pytest.raises(ValueError, match=message):
        native_cueq_norm(x, affine, affine)


@pytest.mark.parametrize("bad_scale", [False, True])
def test_rejects_invalid_affine_shape_before_launch(bad_scale):
    x = jax.ShapeDtypeStruct((2, 128), jnp.float32)
    good = jax.ShapeDtypeStruct((128,), jnp.float32)
    bad = jax.ShapeDtypeStruct((127,), jnp.float32)
    with pytest.raises(ValueError, match="affine"):
        native_cueq_norm(x, bad if bad_scale else good, good if bad_scale else bad)


@pytest.mark.parametrize(
    "shape,dtype",
    [((), jnp.bfloat16), ((127, 1, 2, 2), jnp.bfloat16), ((128, 1, 2, 2), jnp.float32)],
)
def test_output_norm_rejects_wrong_layout_or_dtype(shape, dtype):
    affine = jax.ShapeDtypeStruct((128,), jnp.float32)
    with pytest.raises(ValueError, match="BF16"):
        native_cueq_output_norm(jax.ShapeDtypeStruct(shape, dtype), affine, affine)
