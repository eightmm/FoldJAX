from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.models.diffusion.transformer import (
    ConditionedTransitionParams,
    DiffusionTransformerBlockParams,
    DiffusionTransformerStackParams,
    diffusion_transformer_stack,
)
from foldjax.models.protenix.models.primitives.attention import (
    AttentionPairBiasParams,
    AttentionParams,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LayerNormParams,
    LinearParams,
)


def test_adaptive_diffusion_transformer_scan_matches_loop() -> None:
    blocks = (_adaptive_block(0.1), _adaptive_block(-0.2))
    params = DiffusionTransformerStackParams(blocks=blocks)
    a = jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 5.0
    s = jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 7.0
    z = jnp.zeros((2, 2, 4, 1), dtype=jnp.float32)

    loop = diffusion_transformer_stack(
        a,
        s,
        z,
        params,
        num_heads=1,
        n_queries=2,
        n_keys=4,
        use_scan=False,
    )
    scanned = diffusion_transformer_stack(
        a,
        s,
        z,
        params,
        num_heads=1,
        n_queries=2,
        n_keys=4,
        use_scan=True,
    )
    compiled = diffusion_transformer_stack(
        a,
        s,
        z,
        params,
        num_heads=1,
        n_queries=2,
        n_keys=4,
        use_scan=False,
        attention_backend="xla_jit",
    )

    np.testing.assert_allclose(scanned, loop, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(compiled, loop, rtol=1e-5, atol=1e-5)


def test_global_diffusion_transformer_threads_extra_attention_bias() -> None:
    params = DiffusionTransformerStackParams(
        blocks=(_adaptive_block(0.1), _adaptive_block(-0.2))
    )
    a = jnp.asarray(
        [[[0.0, 1.0], [2.0, -1.0], [-2.0, 0.5]]],
        dtype=jnp.float32,
    )
    s = jnp.asarray(
        [[[0.5, -0.7], [-1.2, 0.3], [1.4, -0.2]]],
        dtype=jnp.float32,
    )
    z = jnp.zeros((1, 3, 3, 1), dtype=jnp.float32)
    extra_bias = jnp.asarray(
        [[0.0, 1.0, -0.5], [-0.5, 0.0, 0.8], [0.6, -0.7, 0.0]],
        dtype=jnp.float32,
    )

    without_extra = diffusion_transformer_stack(
        a,
        s,
        z,
        params,
        num_heads=1,
        use_scan=False,
    )
    loop = diffusion_transformer_stack(
        a,
        s,
        z,
        params,
        num_heads=1,
        use_scan=False,
        extra_attn_bias=extra_bias,
    )
    scanned = diffusion_transformer_stack(
        a,
        s,
        z,
        params,
        num_heads=1,
        use_scan=True,
        extra_attn_bias=extra_bias,
    )

    assert not np.allclose(loop, without_extra)
    np.testing.assert_allclose(scanned, loop, rtol=1e-5, atol=1e-5)


def _adaptive_block(offset: float) -> DiffusionTransformerBlockParams:
    layernorm = LayerNormParams(weight=jnp.ones((2,)), bias=None)
    zero = LinearParams(weight=jnp.zeros((2, 2)), bias=None)
    biased = LinearParams(
        weight=jnp.eye(2, dtype=jnp.float32) * 0.1,
        bias=jnp.full((2,), offset, dtype=jnp.float32),
    )
    adaln = AdaptiveLayerNormParams(
        layernorm_a=LayerNormParams(weight=None, bias=None),
        layernorm_s=layernorm,
        linear_s=biased,
        linear_no_bias_s=zero,
    )
    attention = AttentionPairBiasParams(
        layernorm_a=adaln,
        layernorm_kv=adaln,
        attention=AttentionParams(
            linear_q=biased,
            linear_k=zero,
            linear_v=biased,
            linear_o=biased,
            linear_g=biased,
        ),
        layernorm_z=LayerNormParams(weight=jnp.ones((1,)), bias=None),
        linear_z=LinearParams(weight=jnp.zeros((1, 1)), bias=None),
        linear_a_last=biased,
        has_s=True,
        cross_attention_mode=True,
    )
    transition = ConditionedTransitionParams(
        adaln=adaln,
        linear_a1=LinearParams(weight=jnp.ones((4, 2)) * 0.1, bias=None),
        linear_a2=LinearParams(weight=jnp.ones((4, 2)) * -0.1, bias=None),
        linear_b=LinearParams(weight=jnp.ones((2, 4)) * 0.1, bias=None),
        linear_s=biased,
    )
    return DiffusionTransformerBlockParams(
        attention_pair_bias=attention,
        conditioned_transition=transition,
    )
