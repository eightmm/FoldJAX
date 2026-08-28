"""Pure JAX MSA module for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from foldjax.models._cp import shard_pair_rows
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import linear as _linear
from foldjax.models.boltz2.models.primitives._scan_utils import stack_layer_params
from foldjax.models.boltz2.models.primitives.transition import transition_forward
from foldjax.models.boltz2.models.triangle.triangle import (
    triangle_multiplication_forward,
)
from foldjax.models.boltz2.models.triangle.triangle_attention import (
    resolve_triangle_attention_chunk,
    resolve_triangle_attention_q_chunk,
    triangle_attention_forward,
)

Params = Mapping[str, object]


def msa_module_forward(
    params: Params,
    z: jnp.ndarray,
    emb: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    num_tokens: int = 33,
    eps: float = 1e-5,
    use_scan: bool = True,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    pair_averaging_chunk: int | None = None,
    matmul_precision: str = "highest",
    triangle_backend: str = "xla",
    glu_backend: str = "xla",
    subsample_msa: bool = False,
    num_subsampled_msa: int = 1024,
    msa_key: jnp.ndarray | None = None,
    msa_rows: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Run Boltz MSAModule.

    ``use_scan=False`` (default) unrolls the layer stack in Python (lower steady
    latency). ``use_scan=True`` runs the stack via ``lax.scan`` (faster compile).

    ``subsample_msa`` caps the MSA depth at ``num_subsampled_msa`` (default
    1024). ``MSAModuleArgs`` defaults it to False and `boltz predict` declares
    ``--subsample_msa`` as a click flag, which defaults to False whatever the
    Python signature says -- so upstream subsamples only when that flag is
    given.

    ``msa_key`` selects *which* rows. Upstream draws a fresh
    ``torch.randperm(n_msa)[:1024]`` inside every forward pass
    (``trunkv2.py``), so across a recycled trunk it sees a different slice of
    the alignment each pass and effectively ensembles over the whole thing.
    Passing a key reproduces that; passing ``None`` keeps the first
    ``num_subsampled_msa`` rows, which is deterministic and reproducible but is
    *not* what upstream computes -- with 2,372 rows and eleven passes it shows
    the model the same top 1,024 eleven times instead of most of the alignment.

    ``msa_rows`` overrides the draw with explicit row indices and takes
    precedence over ``msa_key``. It exists so a matched-tape parity harness can
    hand this module the very permutation upstream drew: without it the two
    trunks read different alignment rows and their coordinates cannot agree no
    matter how well the sampler is matched. ``None`` (default) is inert.
    """

    # Subsample MSA depth (axis 1) to num_subsampled_msa, matching Boltz CLI's
    # inference-time cap; take the first rows (deterministic, query-first order).
    msa_d = feats["msa"]
    has_del = feats["has_deletion"]
    del_val = feats["deletion_value"]
    msa_paired = feats["msa_paired"]
    msa_mask = feats["msa_mask"]
    if subsample_msa and msa_d.shape[1] > num_subsampled_msa:
        if msa_rows is not None:
            rows = jnp.asarray(msa_rows, dtype=jnp.int32)
            if rows.ndim != 1 or rows.shape[0] != num_subsampled_msa:
                raise ValueError(
                    f"msa_rows must be {num_subsampled_msa} row indices, got "
                    f"shape {rows.shape}"
                )
        elif msa_key is None:
            rows = jnp.arange(num_subsampled_msa)
        else:
            rows = jax.random.permutation(msa_key, msa_d.shape[1])[:num_subsampled_msa]
        msa_d = jnp.take(msa_d, rows, axis=1)
        has_del = jnp.take(has_del, rows, axis=1)
        del_val = jnp.take(del_val, rows, axis=1)
        msa_paired = jnp.take(msa_paired, rows, axis=1)
        msa_mask = jnp.take(msa_mask, rows, axis=1)
    m = _msa_input_embedding(
        params,
        emb,
        msa_d,
        has_del,
        del_val,
        msa_paired,
        num_tokens=num_tokens,
    )

    token_mask = feats["token_pad_mask"].astype(m.dtype)
    token_mask = token_mask[:, :, None] * token_mask[:, None, :]
    msa_mask = msa_mask.astype(m.dtype)

    layers = params["layers"]
    if not use_scan:
        for layer in layers:
            z, m = msa_layer_forward(
                layer,
                z,
                m,
                token_mask,
                msa_mask,
                eps=eps,
                chunk_size=chunk_size,
                triangle_attention_chunk=triangle_attention_chunk,
                triangle_attention_q_chunk=triangle_attention_q_chunk,
                transition_hidden_chunk=transition_hidden_chunk,
                pair_averaging_chunk=pair_averaging_chunk,
                matmul_precision=matmul_precision,
                triangle_backend=triangle_backend,
                glu_backend=glu_backend,
            )
        return z

    stacked = stack_layer_params(layers)

    def body(carry, layer):
        z_c, m_c = carry
        z_c, m_c = msa_layer_forward(
            layer,
            z_c,
            m_c,
            token_mask,
            msa_mask,
            eps=eps,
            chunk_size=chunk_size,
            triangle_attention_chunk=triangle_attention_chunk,
            triangle_attention_q_chunk=triangle_attention_q_chunk,
            transition_hidden_chunk=transition_hidden_chunk,
            pair_averaging_chunk=pair_averaging_chunk,
            matmul_precision=matmul_precision,
            triangle_backend=triangle_backend,
            glu_backend=glu_backend,
        )
        return (z_c, m_c), None

    (z, m), _ = jax.lax.scan(body, (z, m), stacked)
    return z


def _msa_input_embedding(
    params: Params,
    emb: jnp.ndarray,
    msa: jnp.ndarray,
    has_deletion: jnp.ndarray,
    deletion_value: jnp.ndarray,
    msa_paired: jnp.ndarray,
    *,
    num_tokens: int,
) -> jnp.ndarray:
    """Project one compact or publisher-native Boltz MSA feature block."""

    # Equivalent to one_hot(msa, num_tokens) @ kernel[:num_tokens] but without
    # materializing the [batch, num_seq, N, num_tokens] one-hot tensor.
    # Concatenation order in the original code is: one-hot block (rows
    # [0:num_tokens]), then has_deletion, deletion_value, msa_paired
    # (rows [num_tokens:num_tokens+3]). Split the kernel at call time; the
    # stored param pytree is left unchanged.
    kernel = params["msa_proj"]["kernel"]
    kernel_onehot = kernel[:num_tokens]  # [num_tokens, d]
    kernel_extra = kernel[num_tokens:]  # [3, d]
    msa_idx = msa.astype(jnp.int32)
    # The high-level path stores exact binary MSA flags as bool.  Cast every
    # lane to the historical projection dtype before stacking; this is also
    # robust under JAX's strict dtype-promotion mode and leaves custom callers'
    # numerical values unchanged.
    extra = jnp.stack(
        tuple(
            jnp.asarray(value, dtype=kernel_extra.dtype)
            for value in (has_deletion, deletion_value, msa_paired)
        ),
        axis=-1,
    )
    m = kernel_onehot[msa_idx] + _linear(extra, kernel_extra)
    return m + _linear(emb, params["s_proj"]["kernel"])[:, None]


def msa_layer_forward(
    params: Params,
    z: jnp.ndarray,
    m: jnp.ndarray,
    token_mask: jnp.ndarray,
    msa_mask: jnp.ndarray,
    eps: float = 1e-5,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    pair_averaging_chunk: int | None = None,
    matmul_precision: str = "highest",
    triangle_backend: str = "xla",
    glu_backend: str = "xla",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run one Boltz MSALayer in eval mode."""

    m = m + pair_weighted_averaging_forward(
        params["pair_weighted_averaging"],
        m,
        z,
        token_mask,
        eps=eps,
        row_chunk_size=pair_averaging_chunk,
    )
    m = m + transition_forward(
        params["msa_transition"], m, eps=eps, glu_backend=glu_backend
    )
    z = z + outer_product_mean_forward(
        params["outer_product_mean"], m, msa_mask, eps, chunk_size=chunk_size
    )
    z = pairformer_no_seq_layer_forward(
        params["pairformer_layer"],
        z,
        token_mask,
        eps,
        chunk_size=chunk_size,
        triangle_attention_chunk=triangle_attention_chunk,
        triangle_attention_q_chunk=triangle_attention_q_chunk,
        transition_hidden_chunk=transition_hidden_chunk,
        matmul_precision=matmul_precision,
        triangle_backend=triangle_backend,
        glu_backend=glu_backend,
    )
    return z, m


#: Byte budget for one block of PairWeightedAveraging's widened MSA tensors.
#:
#: This block widens the MSA from ``c_m`` (64) to ``num_heads * c_h`` (256)
#: channels three times over -- the projected values ``v``, the gate ``g``, and
#: the averaged output ``o`` -- and ``v`` and ``o`` are live at the same
#: instant. Read off XLA's buffer assignment for the 490-token benchmark job
#: (7,917 alignment rows), those two were ``f32[1,7917,490,256]`` and
#: ``f32[8,1,7917,32,490]``: 3.70 GiB each, 7.4 GiB of a 9,963 MiB temp arena,
#: while every other block in the trunk was already chunked down to a few
#: hundred MiB. That is the whole of this port's memory gap against upstream at
#: full alignment depth, and it appears only above 384 tokens because that is
#: where upstream turns its own version of this on (``chunk_heads_pwa`` in
#: ``trunkv2.py``, gated on ``const.chunk_size_threshold``).
#:
#: Chunking over MSA rows rather than over heads is the safer of the two: rows
#: are a pure batch axis of the einsum and of all four projections, so every
#: reduction stays whole and the accumulation order does not change. Upstream's
#: head chunk instead splits ``proj_o``'s 256-long reduction into eight partial
#: sums.
#:
#: 512 MiB rather than the transition's 256 MiB, so that the 132-token
#: benchmark job -- widened form 306 MiB over its 2,371 rows -- stays on the
#: single-op path. That job is the one the matched-tape parity harness replays,
#: and leaving it unsplit is what makes this change provably inert there: the
#: module it lowers to is byte-identical to the one that shipped (sha256 of
#: ``jax.jit(...).lower(...).as_text()``, asserted in the MSA parity tests). It
#: is also what upstream runs at that size, below its own 384-token threshold.
#:
#: The identity matters because the harness cannot settle the question by
#: measurement: six runs of unmodified code spanned 0.14522 to 0.14768 A, and
#: two with XLA autotuning disabled still differed in the fourth decimal, so a
#: repeated RMSD would not have shown a shift this small either way. A chunk is
#: exact but not bit-identical -- XLA tiles the smaller GEMMs differently -- so
#: "unchanged" has to mean the same program, not a similar number.
#:
#: Above the budget it splits: the 490-token job into eight blocks, and its
#: trunk temp arena from 9,963 MiB to 3,649 MiB.
_PWA_BUDGET_BYTES = 512 * 1024**2


def _auto_pair_averaging_chunk(m: jnp.ndarray, params: Params) -> int | None:
    """MSA rows whose widened form fits the budget, or None to do it whole."""
    if m.ndim != 4 or m.shape[1] < 2:
        return None
    wide = params["proj_m"]["kernel"].shape[-1]
    per_row = m.shape[2] * wide * m.dtype.itemsize
    if per_row <= 0 or per_row * m.shape[1] <= _PWA_BUDGET_BYTES:
        return None
    return max(1, _PWA_BUDGET_BYTES // per_row)


def pair_weighted_averaging_forward(
    params: Params,
    m: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-5,
    inf: float = 1e6,
    row_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Run Boltz PairWeightedAveraging.

    ``row_chunk_size`` splits axis 1 (MSA depth) into independent blocks, so the
    widened ``[b, n_msa, n_token, num_heads * c_h]`` intermediates are never
    resident all at once. Left as ``None`` the block size is derived from
    ``_PWA_BUDGET_BYTES``; pass ``0`` to force the single-op path.

    Below the budget this runs the original single-op body, statement for
    statement. Sharing one body with the chunked path would mean hoisting the
    pair weights above the MSA layer norm -- algebraically the same program, but
    a different HLO, which is exactly what the budget is set to avoid. Two
    bodies is the price of leaving the jobs that never needed a chunk on the
    program they already ran.
    """

    if row_chunk_size is None:
        row_chunk_size = _auto_pair_averaging_chunk(m, params)
    if row_chunk_size is not None and 0 < row_chunk_size < m.shape[1]:
        return _pair_weighted_averaging_chunked(
            params, m, z, mask, eps, inf, row_chunk_size
        )

    m = _layer_norm(m, params["norm_m"]["scale"], params["norm_m"]["bias"], eps)
    z = _layer_norm(z, params["norm_z"]["scale"], params["norm_z"]["bias"], eps)
    num_heads = params["proj_z"]["kernel"].shape[-1]
    c_h = params["proj_m"]["kernel"].shape[-1] // num_heads

    v = _linear(m, params["proj_m"]["kernel"])
    v = jnp.reshape(v, (*v.shape[:3], num_heads, c_h))
    v = jnp.transpose(v, (0, 3, 1, 2, 4))
    b = _linear(z, params["proj_z"]["kernel"])
    b = jnp.transpose(b, (0, 3, 1, 2))
    # Mask + softmax in fp32 so the `inf` constant stays finite (in fp16 it would
    # saturate to inf -> fully-masked rows give NaN). fp32 runtime is unchanged.
    mask = mask.astype(jnp.float32)
    b = b.astype(jnp.float32) + (1.0 - mask[:, None]) * (-inf)
    w = jax.nn.softmax(b, axis=-1).astype(v.dtype)
    g = jax.nn.sigmoid(_linear(m, params["proj_g"]["kernel"]))
    o = jnp.einsum("bhij,bhsjd->bhsid", w, v)
    o = jnp.transpose(o, (0, 2, 3, 1, 4))
    o = jnp.reshape(o, (*o.shape[:3], num_heads * c_h))
    return _linear(g * o, params["proj_o"]["kernel"])


def _pair_weighted_averaging_chunked(
    params: Params,
    m: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float,
    inf: float,
    row_chunk_size: int,
) -> jnp.ndarray:
    """PairWeightedAveraging over blocks of MSA rows.

    The pair weights do not depend on the MSA axis, so they are computed once
    and every block reads them. Everything that does depend on it -- the MSA
    layer norm, both widening projections, the weighted average, and the
    narrowing projection -- is done a block at a time; each reduction still runs
    over its whole axis, so no sum is split.
    """

    z = _layer_norm(z, params["norm_z"]["scale"], params["norm_z"]["bias"], eps)
    num_heads = params["proj_z"]["kernel"].shape[-1]
    c_h = params["proj_m"]["kernel"].shape[-1] // num_heads

    b = _linear(z, params["proj_z"]["kernel"])
    b = jnp.transpose(b, (0, 3, 1, 2))
    mask = mask.astype(jnp.float32)
    b = b.astype(jnp.float32) + (1.0 - mask[:, None]) * (-inf)
    w = jax.nn.softmax(b, axis=-1).astype(m.dtype)

    def block(m_block: jnp.ndarray) -> jnp.ndarray:
        # Normalized per block rather than once up front: the normalized copy is
        # as big as the MSA itself, and nothing outside this block reads it.
        m_block = _layer_norm(
            m_block, params["norm_m"]["scale"], params["norm_m"]["bias"], eps
        )
        v = _linear(m_block, params["proj_m"]["kernel"])
        v = jnp.reshape(v, (*v.shape[:3], num_heads, c_h))
        v = jnp.transpose(v, (0, 3, 1, 2, 4))
        g = jax.nn.sigmoid(_linear(m_block, params["proj_g"]["kernel"]))
        o = jnp.einsum("bhij,bhsjd->bhsid", w, v)
        o = jnp.transpose(o, (0, 2, 3, 1, 4))
        o = jnp.reshape(o, (*o.shape[:3], num_heads * c_h))
        return _linear(g * o, params["proj_o"]["kernel"])

    return jnp.concatenate(
        [
            block(m[:, start : start + row_chunk_size])
            for start in range(0, m.shape[1], row_chunk_size)
        ],
        axis=1,
    )


#: Byte budget for one block of OuterProductMean's ``[b, i, j, c, d]`` product.
#:
#: This block widens 32 channels to 32*32 before the output projection narrows
#: them to 128, and it computes that product in float32 -- ``a.float(),
#: b.float()`` is what upstream writes, and it is the largest float32 left in an
#: otherwise bfloat16 trunk. At 1,003 tokens one 128-row block is
#: ``f32[1,128,1003,32,32]``, 502 MiB; the row loop is unrolled, so several of
#: them are live at once. Narrowing the block took the trunk's temp arena from
#: 6,116 MiB to 3,860 MiB at that size -- measured with ``memory_analysis`` on
#: the trunk compiled by itself, since the full prediction program shares its
#: arena with the diffusion and confidence modules, which are float32 by design.
#:
#: 256 MiB rather than a derivation, because the response is not monotonic and
#: only measurement finds the shelf. Swept at 1,003 tokens over an 8,192-row
#: alignment, temp arena against budget: 384 MiB -> 5.93, 320 -> 3.74,
#: 256 -> 3.86, 192 -> 3.86, 128 -> 3.86, 96 -> 5.79, 64 -> 5.80 GiB. Four
#: consecutive settings share the floor and both edges fall off it, so this sits
#: in the middle of the shelf rather than on a lucky point -- but it is a
#: scheduling shelf, so re-sweep it after an XLA upgrade rather than trusting
#: the constant.
#:
#: The budget only ever *tightens* the chunk the caller asked for. Raising it
#: would move the 132-token matched-tape job off the two-block program it
#: already runs, which is the one thing the transition and pair-averaging
#: budgets above are also written to avoid. Below 513 tokens the budget does
#: not engage at all and the float32 default lowers to the program it shipped
#: with; above it the chunk narrows in both dtypes, which is exact but not
#: bit-identical -- XLA tiles the smaller GEMMs differently -- exactly as for
#: the two budgets above.
_OPM_BUDGET_BYTES = 256 * 1024**2


def _auto_outer_product_chunk(n_token: int, widened: int, chunk_size: int) -> int:
    """Token rows whose widened float32 product fits the budget."""
    per_row = n_token * widened * 4
    if per_row <= 0:
        return chunk_size
    return max(1, min(chunk_size, _OPM_BUDGET_BYTES // per_row))


def outer_product_mean_forward(
    params: Params,
    m: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-5,
    chunk_size: int = 128,
) -> jnp.ndarray:
    """Run Boltz OuterProductMean.

    Computes the result in chunks over the i (token) axis so the full
    [b, i, j, c, d] fp32 intermediate is never materialized at once. Peak
    intermediate goes from [N, N, c*d] to [chunk, N, c*d], where ``chunk`` is
    ``chunk_size`` narrowed by ``_OPM_BUDGET_BYTES``.
    """

    mask = mask.astype(m.dtype)
    m = _layer_norm(m, params["norm"]["scale"], params["norm"]["bias"], eps)
    a = _linear(m, params["proj_a"]["kernel"]) * mask[..., None]
    b = _linear(m, params["proj_b"]["kernel"]) * mask[..., None]
    # Upstream writes `torch.einsum(..., a.float(), b.float())`. Both operands
    # are already exactly representable in the compute dtype, so a float32
    # accumulator fed from the narrow operands computes the same products from
    # the same values -- without the widened copies, which are f32[8192,32096]
    # (1,003 MiB) at this job's alignment depth. Measured on an isolated dot,
    # the two forms agree to 1e-6 relative, which is reduction order and
    # nothing else. Under the float32 default the casts were no-ops and
    # `preferred_element_type` is the output dtype the dot already had, so
    # neither line changes that path at all.
    #
    # `num_mask` is the same sum written as a contraction: it counts, for each
    # token pair, the alignment rows where both are present. As a broadcast
    # product it is a [b, s, i, j] tensor -- 16.5 GiB of elementwise work at
    # 8,192 rows, fused away by XLA but still computed -- and the mask is 0/1,
    # so summing it in float32 is exact whatever the order.
    num_mask = jnp.maximum(
        jnp.einsum(
            "bsi,bsj->bij", mask, mask, preferred_element_type=jnp.float32
        ).astype(m.dtype)[..., None],
        1.0,
    )

    proj_o = params["proj_o"]
    n = a.shape[2]
    out_dtype = m.dtype
    chunk = _auto_outer_product_chunk(n, a.shape[-1] * b.shape[-1], chunk_size)
    out = jnp.zeros((a.shape[0], n, n, proj_o["kernel"].shape[-1]), dtype=out_dtype)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        a_blk = a[:, :, start:end]
        z = jnp.einsum("bsic,bsjd->bijcd", a_blk, b, preferred_element_type=jnp.float32)
        z = jnp.reshape(z, (*z.shape[:3], -1)) / num_mask[:, start:end]
        out = out.at[:, start:end].set(
            _linear(z.astype(out_dtype), proj_o["kernel"], proj_o["bias"])
        )
    return out


def pairformer_no_seq_layer_forward(
    params: Params,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    eps: float = 1e-5,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    matmul_precision: str = "highest",
    triangle_backend: str = "xla",
    glu_backend: str = "xla",
) -> jnp.ndarray:
    """Run Boltz PairformerNoSeqLayer in eval mode."""

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
    return z
