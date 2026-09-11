"""Pure JAX Pairformer blocks for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from foldjax.models._cp import shard_pair_rows
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import (
    residual_cast as _residual_cast,
)
from foldjax.models.boltz2.models.primitives._scan_utils import stack_layer_params
from foldjax.models.boltz2.models.primitives.attention import (
    attention_pair_bias_forward,
)
from foldjax.models.boltz2.models.primitives.transition import transition_forward
from foldjax.models.boltz2.models.triangle.triangle import (
    triangle_multiplication_forward,
)
from foldjax.models.boltz2.models.triangle.triangle_attention_cp import (
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
    pair_residual_dtype: jnp.dtype | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run a Boltz PairformerModule stack in eval mode without kernels.

    ``use_scan=False`` unrolls the layer stack in Python. ``use_scan=True``
    (default) runs the stack via ``lax.scan`` over stacked params and preserves
    the memory-stable production path.

    ``pair_residual_dtype`` pins the pair carry ``z``; ``None`` (default) lets
    it keep whatever width the caller handed in, which is the released
    float32. The single carry ``s`` is never pinned: upstream runs that branch
    inside ``torch.autocast(enabled=False)`` (``boltz/model/layers/
    pairformer.py:105``).
    """

    z = _residual_cast(z, pair_residual_dtype)
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
                pair_residual_dtype=pair_residual_dtype,
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
            pair_residual_dtype=pair_residual_dtype,
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
    pair_residual_dtype: jnp.dtype | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run one Boltz PairformerLayer in eval mode without dropout.

    ``triangle_backend`` selects the triangle-attention path: ``"xla"``
    (default, bit-exact) or ``"pallas"`` (opt-in GPU flash kernel).
    """

    # Under context parallelism the pair representation is sharded along its
    # rows; pinning it at every layer entry keeps the whole stack on one
    # layout without the partitioner re-deriving it per consumer. Identity
    # when no mesh is active.
    z = _residual_cast(shard_pair_rows(z), pair_residual_dtype)
    # A narrowed pair residual is still the CUDA-autocast configuration, so
    # the triangle ops are told so rather than reading it off the activation
    # width and dropping to the FP32-model program.
    native_amp = None if pair_residual_dtype is None else True
    tri_att_chunk = resolve_triangle_attention_chunk(
        z.shape[1], chunk_size, triangle_attention_chunk
    )
    tri_att_q_chunk = resolve_triangle_attention_q_chunk(
        z.shape[1], triangle_attention_q_chunk
    )
    # Every pair residual is re-pinned so the carry width is a property of the
    # stack rather than of whichever block happens to return the widest
    # operand. Under the released `None` and under a pin the sub-blocks already
    # honour, `_residual_cast` returns its argument and emits nothing.
    z = _residual_cast(
        z
        + triangle_multiplication_forward(
            params["tri_mul_out"],
            z,
            pair_mask,
            "outgoing",
            eps=eps,
            chunk_size=chunk_size,
            glu_backend=glu_backend,
            native_amp=native_amp,
        ),
        pair_residual_dtype,
    )
    z = _residual_cast(
        z
        + triangle_multiplication_forward(
            params["tri_mul_in"],
            z,
            pair_mask,
            "incoming",
            eps=eps,
            chunk_size=chunk_size,
            glu_backend=glu_backend,
            native_amp=native_amp,
        ),
        pair_residual_dtype,
    )
    z = _residual_cast(
        z
        + triangle_attention_forward(
            params["tri_att_start"],
            z,
            pair_mask,
            starting=True,
            eps=eps,
            chunk_size=tri_att_chunk,
            q_chunk_size=tri_att_q_chunk,
            matmul_precision=matmul_precision,
            triangle_backend=triangle_backend,
            native_amp=native_amp,
        ),
        pair_residual_dtype,
    )
    z = _residual_cast(
        z
        + triangle_attention_forward(
            params["tri_att_end"],
            z,
            pair_mask,
            starting=False,
            eps=eps,
            chunk_size=tri_att_chunk,
            q_chunk_size=tri_att_q_chunk,
            matmul_precision=matmul_precision,
            triangle_backend=triangle_backend,
            native_amp=native_amp,
        ),
        pair_residual_dtype,
    )
    z = _residual_cast(
        z
        + transition_forward(
            params["transition_z"],
            z,
            chunk_size=transition_hidden_chunk,
            eps=eps,
            row_chunk_size=chunk_size,
            glu_backend=glu_backend,
            native_amp_norm=(
                params["transition_z"]["fc1"]["kernel"].dtype == jnp.bfloat16
            ),
        ),
        pair_residual_dtype,
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
        # Native's single branch disables autocast, including its pair-bias
        # projection. `s` arrives FP32 to keep the layer scan carry stable.
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
