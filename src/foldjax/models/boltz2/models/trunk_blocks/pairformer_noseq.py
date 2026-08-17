"""Pure JAX pair-only Pairformer (PairformerNoSeqModule) for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from foldjax.models._cp import shard_pair_rows
from foldjax.models.boltz2.models.primitives.transition import transition_forward
from foldjax.models.boltz2.models.triangle.triangle import (
    triangle_multiplication_forward,
)
from foldjax.models.boltz2.models.triangle.triangle_attention import (
    resolve_triangle_attention_chunk,
    resolve_triangle_attention_q_chunk,
    triangle_attention_forward,
)

PairformerNoSeqLayerParams = Mapping[str, Mapping[str, jnp.ndarray]]
PairformerNoSeqModuleParams = Mapping[str, list[PairformerNoSeqLayerParams]]


def pairformer_no_seq_layer_forward(
    params: PairformerNoSeqLayerParams,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    eps: float = 1e-5,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    triangle_backend: str = "xla",
) -> jnp.ndarray:
    """Run one Boltz PairformerNoSeqLayer in eval mode (no dropout)."""

    # Row-sharded under context parallelism; identity otherwise. Same seam as
    # `pairformer_layer_forward`.
    z = shard_pair_rows(z)
    tri_att_chunk = resolve_triangle_attention_chunk(
        z.shape[1], chunk_size, triangle_attention_chunk
    )
    tri_att_q_chunk = resolve_triangle_attention_q_chunk(
        z.shape[1], triangle_attention_q_chunk
    )
    z = z + triangle_multiplication_forward(
        params["tri_mul_out"], z, pair_mask, "outgoing", eps=eps, chunk_size=chunk_size
    )
    z = z + triangle_multiplication_forward(
        params["tri_mul_in"], z, pair_mask, "incoming", eps=eps, chunk_size=chunk_size
    )
    z = z + triangle_attention_forward(
        params["tri_att_start"],
        z,
        pair_mask,
        starting=True,
        eps=eps,
        chunk_size=tri_att_chunk,
        q_chunk_size=tri_att_q_chunk,
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
        triangle_backend=triangle_backend,
    )
    z = z + transition_forward(
        params["transition_z"],
        z,
        chunk_size=transition_hidden_chunk,
        eps=eps,
        row_chunk_size=chunk_size,
    )
    return z


def pairformer_no_seq_module_forward(
    params: PairformerNoSeqModuleParams,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    eps: float = 1e-5,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    triangle_backend: str = "xla",
) -> jnp.ndarray:
    """Run a Boltz PairformerNoSeqModule stack in eval mode without kernels."""

    for layer_params in params["layers"]:
        z = pairformer_no_seq_layer_forward(
            layer_params,
            z,
            pair_mask,
            eps=eps,
            chunk_size=chunk_size,
            triangle_attention_chunk=triangle_attention_chunk,
            triangle_attention_q_chunk=triangle_attention_q_chunk,
            transition_hidden_chunk=transition_hidden_chunk,
            triangle_backend=triangle_backend,
        )
    return z
