"""Pure JAX Pairformer blocks for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._scan_utils import stack_layer_params
from foldjax.models.boltz2.models.primitives.attention import (
    attention_pair_bias_forward,
)
from foldjax.models.boltz2.models.primitives.transition import transition_forward
from foldjax.models.boltz2.models.triangle.triangle import (
    triangle_multiplication_forward,
)
from foldjax.models.boltz2.models.triangle.triangle_attention import (
    resolve_triangle_attention_chunk,
    resolve_triangle_attention_q_chunk,
    triangle_attention_forward,
)

PairformerLayerParams = Mapping[str, Mapping[str, jnp.ndarray]]
PairformerModuleParams = Mapping[str, list[PairformerLayerParams]]


def pairformer_module_forward(
    params: PairformerModuleParams,
    s: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    eps: float = 1e-5,
    use_scan: bool = True,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    matmul_precision: str = "highest",
    attention_backend: str = "xla",
    triangle_backend: str = "xla",
    glu_backend: str = "xla",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run a Boltz PairformerModule stack in eval mode without kernels.

    ``use_scan=False`` unrolls the layer stack in Python. ``use_scan=True``
    (default) runs the stack via ``lax.scan`` over stacked params and preserves
    the memory-stable production path.
    """

    layers = params["layers"]
    if not use_scan:
        for layer_params in layers:
            s, z = pairformer_layer_forward(
                layer_params,
                s,
                z,
                mask,
                pair_mask,
                eps=eps,
                chunk_size=chunk_size,
                triangle_attention_chunk=triangle_attention_chunk,
                triangle_attention_q_chunk=triangle_attention_q_chunk,
                transition_hidden_chunk=transition_hidden_chunk,
                matmul_precision=matmul_precision,
                attention_backend=attention_backend,
                triangle_backend=triangle_backend,
            )
        return s, z

    stacked = stack_layer_params(layers)

    def body(carry, layer_params):
        s_c, z_c = carry
        s_c, z_c = pairformer_layer_forward(
            layer_params,
            s_c,
            z_c,
            mask,
            pair_mask,
            eps=eps,
            chunk_size=chunk_size,
            triangle_attention_chunk=triangle_attention_chunk,
            triangle_attention_q_chunk=triangle_attention_q_chunk,
            transition_hidden_chunk=transition_hidden_chunk,
            matmul_precision=matmul_precision,
            attention_backend=attention_backend,
            triangle_backend=triangle_backend,
            glu_backend=glu_backend,
        )
        return (s_c, z_c), None

    (s, z), _ = jax.lax.scan(body, (s, z), stacked)
    return s, z


def pairformer_layer_forward(
    params: PairformerLayerParams,
    s: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    eps: float = 1e-5,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    matmul_precision: str = "highest",
    attention_backend: str = "xla",
    triangle_backend: str = "xla",
    glu_backend: str = "xla",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run one Boltz PairformerLayer in eval mode without dropout.

    ``triangle_backend`` selects the triangle-attention path: ``"xla"``
    (default, bit-exact) or ``"pallas"`` (opt-in GPU flash kernel).
    """

    tri_att_chunk = resolve_triangle_attention_chunk(
        z.shape[1], chunk_size, triangle_attention_chunk
    )
    tri_att_q_chunk = resolve_triangle_attention_q_chunk(
        z.shape[1], triangle_attention_q_chunk
    )
    z = z + triangle_multiplication_forward(
        params["tri_mul_out"],
        z,
        pair_mask,
        "outgoing",
        eps=eps,
        chunk_size=chunk_size,
        glu_backend=glu_backend,
    )
    z = z + triangle_multiplication_forward(
        params["tri_mul_in"],
        z,
        pair_mask,
        "incoming",
        eps=eps,
        chunk_size=chunk_size,
        glu_backend=glu_backend,
    )
    z = z + triangle_attention_forward(
        params["tri_att_start"],
        z,
        pair_mask,
        starting=True,
        eps=eps,
        chunk_size=tri_att_chunk,
        q_chunk_size=tri_att_q_chunk,
        matmul_precision=matmul_precision,
        triangle_backend=triangle_backend,
    )
    z = z + triangle_attention_forward(
        params["tri_att_end"],
        z,
        pair_mask,
        starting=False,
        eps=eps,
        chunk_size=tri_att_chunk,
        q_chunk_size=tri_att_q_chunk,
        matmul_precision=matmul_precision,
        triangle_backend=triangle_backend,
    )
    z = z + transition_forward(
        params["transition_z"],
        z,
        chunk_size=transition_hidden_chunk,
        eps=eps,
        row_chunk_size=chunk_size,
        glu_backend=glu_backend,
    )

    s_normed = _layer_norm(
        s,
        params["pre_norm_s"]["scale"],
        params["pre_norm_s"]["bias"],
        eps,
    )
    s = s + attention_pair_bias_forward(
        params["attention"],
        s=s_normed,
        # `z.float()`, as Boltz spells it at pairformer.py:107. The pair track
        # itself stays in the autocast dtype -- this widening is for the bias
        # this attention builds from it, and nothing downstream sees it. `s`
        # arrives float32 already, promoted where the stack's scan carry is
        # created so the carry types agree.
        z=z.astype(jnp.float32),
        mask=mask.astype(jnp.float32),
        k_in=s_normed,
        eps=eps,
        chunk_size=chunk_size,
        attention_backend=attention_backend,
    )
    s = s + transition_forward(
        params["transition_s"], s, eps=eps, glu_backend=glu_backend
    )
    return s, z
