"""The fused GLU's refusal off a GPU names the kernel and the spelling that runs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from foldjax.models._glu import gated_linear_unit


@pytest.mark.skipif(jax.default_backend() == "gpu", reason="checks the off-GPU refusal")
def test_the_tokamax_glu_off_a_gpu_says_what_to_pass_instead() -> None:
    x = jnp.ones((2, 8), jnp.float32)
    w = jnp.ones((8, 4), jnp.float32)
    with pytest.raises(ValueError, match="glu_backend=xla"):
        gated_linear_unit(x, w, w, jax.nn.silu, backend="tokamax")
