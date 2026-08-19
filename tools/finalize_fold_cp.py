#!/usr/bin/env python3
"""Wire the validated common Fold-CP ring into model-specific pair cores."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one replacement in {path}, found {count}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def insert_before_once(path: str, marker: str, content: str, sentinel: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if sentinel in text:
        return
    count = text.count(marker)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one insertion marker in {path}, found {count}"
        )
    target.write_text(text.replace(marker, content + marker, 1), encoding="utf-8")


# Boltz-2.
replace_once(
    "src/foldjax/models/boltz2/models/triangle/triangle_attention.py",
    '''from foldjax.models._cp import (
    cp_mesh,
    cp_row_shards,
    pair_row_spec,
    shard_pair_rows,
)
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
''',
    '''from foldjax.models._cp import (
    cp_layout,
    cp_mesh,
    cp_row_shards,
    pair_row_spec,
    shard_pair_rows,
)
from foldjax.models._cp_attention import ring_triangle_attention_2d
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
''',
)
replace_once(
    "src/foldjax/models/boltz2/models/triangle/triangle_attention.py",
    '''    triangle_bias = _linear(x, params["linear"]["kernel"], precision)
    triangle_bias = jnp.transpose(triangle_bias, (0, 3, 1, 2))
    triangle_bias = jnp.expand_dims(triangle_bias, axis=1)

    # `shard_map` needs the sharded axis to divide evenly; pad the rows and
''',
    '''    triangle_bias = _linear(x, params["linear"]["kernel"], precision)
    triangle_bias = jnp.transpose(triangle_bias, (0, 3, 1, 2))
    triangle_bias = jnp.expand_dims(triangle_bias, axis=1)

    if cp_layout() == "2d":
        # Keep both pair axes tiled for the whole softmax. Q stays resident;
        # K/V, mask and triangle bias rotate through the square mesh.
        out = _attention_ring_2d(
            params["mha"],
            q_x=x,
            kv_x=x,
            tri_bias=triangle_bias,
            mask_bias=mask_bias,
            precision=precision,
        )
        if not starting:
            out = jnp.swapaxes(out, -2, -3)
        return shard_pair_rows(out, row_axis=-3)

    # `shard_map` needs the sharded axis to divide evenly; pad the rows and
''',
)
insert_before_once(
    "src/foldjax/models/boltz2/models/triangle/triangle_attention.py",
    "\ndef _attention(\n",
    r'''

def _attention_ring_2d(
    params: Mapping[str, Mapping[str, jnp.ndarray]],
    q_x: jnp.ndarray,
    kv_x: jnp.ndarray,
    tri_bias: jnp.ndarray,
    mask_bias: jnp.ndarray,
    *,
    precision: jax.lax.Precision,
) -> jnp.ndarray:
    """Project and run the exact two-dimensional Fold-CP attention ring."""

    no_heads = tri_bias.shape[2]
    c_hidden = params["linear_g"]["kernel"].shape[-1] // no_heads
    qg = _linear(
        q_x,
        jnp.concatenate(
            (params["linear_q"]["kernel"], params["linear_g"]["kernel"]),
            axis=-1,
        ),
        precision,
    )
    q, gate = jnp.split(qg, 2, axis=-1)
    kv = _linear(
        kv_x,
        jnp.concatenate(
            (params["linear_k"]["kernel"], params["linear_v"]["kernel"]),
            axis=-1,
        ),
        precision,
    )
    k, v = jnp.split(kv, 2, axis=-1)
    q = jnp.swapaxes(
        q.reshape(q.shape[:-1] + (no_heads, c_hidden)),
        -2,
        -3,
    )
    k = jnp.swapaxes(
        k.reshape(k.shape[:-1] + (no_heads, c_hidden)),
        -2,
        -3,
    )
    v = jnp.swapaxes(
        v.reshape(v.shape[:-1] + (no_heads, c_hidden)),
        -2,
        -3,
    )
    q = q / jnp.sqrt(jnp.asarray(c_hidden, dtype=q.dtype))
    out = ring_triangle_attention_2d(
        q,
        k,
        v,
        tri_bias,
        mask_bias,
        precision=precision,
    )
    out = jnp.swapaxes(out, -2, -3)
    gate = jax.nn.sigmoid(gate)
    gate = gate.reshape(gate.shape[:-1] + (no_heads, c_hidden))
    out = out * gate
    out = out.reshape(out.shape[:-2] + (c_hidden * no_heads,))
    return _linear(out, params["linear_o"]["kernel"], precision)
''',
    "def _attention_ring_2d(",
)

# Protenix and OpenDDE share this core.
replace_once(
    "src/foldjax/models/protenix/models/triangle/triangle.py",
    "from foldjax.models.protenix.models.primitives.attention import AttentionParams\n",
    "from foldjax.models._cp_attention import ring_triangle_attention_2d\n"
    "from foldjax.models.protenix.models.primitives.attention import AttentionParams\n",
)
replace_once(
    "src/foldjax/models/protenix/models/triangle/triangle.py",
    '''    triangle_bias = linear(x, params.linear)
    triangle_bias = jnp.moveaxis(triangle_bias, -1, -3)
    triangle_bias = jnp.expand_dims(triangle_bias, axis=-4)

    # `shard_map` requires the sharded axis to divide evenly; pad the rows and
''',
    '''    triangle_bias = linear(x, params.linear)
    triangle_bias = jnp.moveaxis(triangle_bias, -1, -3)
    triangle_bias = jnp.expand_dims(triangle_bias, axis=-4)

    if cp_layout() == "2d":
        out = _triangle_attention_ring_2d(
            x,
            params,
            num_heads,
            mask_bias,
            triangle_bias,
        )
        if not starting:
            out = jnp.swapaxes(out, -2, -3)
        return shard_pair_rows(out, row_axis=-3)

    # `shard_map` requires the sharded axis to divide evenly; pad the rows and
''',
)
insert_before_once(
    "src/foldjax/models/protenix/models/triangle/triangle.py",
    "\ndef _triangle_attention_cp(\n",
    r'''

def _triangle_attention_ring_2d(
    x: jnp.ndarray,
    params: TriangleAttentionParams,
    num_heads: int,
    mask_bias: jnp.ndarray,
    triangle_bias: jnp.ndarray,
) -> jnp.ndarray:
    """Two-dimensional Protenix attention without a full-column gather."""

    scale = float(params.attention.linear_k.weight.shape[0] // num_heads) ** -0.5
    q = _project_heads(x, params.attention.linear_q, num_heads)
    k = _project_heads(x, params.attention.linear_k, num_heads)
    v = _project_heads(x, params.attention.linear_v, num_heads)
    q = q * jnp.asarray(scale, dtype=q.dtype)
    out = ring_triangle_attention_2d(
        q,
        k,
        v,
        triangle_bias,
        mask_bias,
    )
    out = jnp.swapaxes(out, -2, -3)
    if params.attention.linear_g is not None:
        gate = sigmoid(linear(x, params.attention.linear_g))
        gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
        out = out * gate
    out = out.reshape(out.shape[:-2] + (-1,))
    return linear(out, params.attention.linear_o)
''',
    "def _triangle_attention_ring_2d(",
)

# OpenFold3 has its own parameter containers but the same ring algorithm.
replace_once(
    "src/foldjax/models/openfold3/models/triangle_attention.py",
    "from foldjax.models._cp import cp_mesh, cp_row_shards, pair_row_spec, shard_pair_rows\n",
    '''from foldjax.models._cp import (
    cp_layout,
    cp_mesh,
    cp_row_shards,
    pair_row_spec,
    shard_pair_rows,
)
from foldjax.models._cp_attention import ring_triangle_attention_2d
''',
)
replace_once(
    "src/foldjax/models/openfold3/models/triangle_attention.py",
    '''    triangle_bias = permute_final_dims(linear(x, params.linear_z), (2, 0, 1))
    triangle_bias = jnp.expand_dims(triangle_bias, -4)

    # `shard_map` requires the sharded axis to divide evenly; pad the rows and
''',
    '''    triangle_bias = permute_final_dims(linear(x, params.linear_z), (2, 0, 1))
    triangle_bias = jnp.expand_dims(triangle_bias, -4)

    if cp_layout() == "2d":
        out = _ring_attention(
            x,
            mask_bias,
            triangle_bias,
            params.mha,
            no_heads=no_heads,
        )
        if not starting:
            out = jnp.swapaxes(out, -2, -3)
        return shard_pair_rows(out, row_axis=-3)

    # `shard_map` requires the sharded axis to divide evenly; pad the rows and
''',
)
insert_before_once(
    "src/foldjax/models/openfold3/models/triangle_attention.py",
    "\ndef _triangle_attention_cp(\n",
    r'''

def _ring_attention(
    x: jnp.ndarray,
    mask_bias: jnp.ndarray,
    triangle_bias: jnp.ndarray,
    params: AttentionParams,
    *,
    no_heads: int,
) -> jnp.ndarray:
    """Exact two-dimensional ring counterpart of dense AF3 attention."""

    query = jnp.swapaxes(split_heads(linear(x, params.linear_q), no_heads), -2, -3)
    key = jnp.swapaxes(split_heads(linear(x, params.linear_k), no_heads), -2, -3)
    value = jnp.swapaxes(split_heads(linear(x, params.linear_v), no_heads), -2, -3)
    query = query / jnp.sqrt(jnp.asarray(query.shape[-1], dtype=query.dtype))
    out = ring_triangle_attention_2d(
        query,
        key,
        value,
        triangle_bias,
        mask_bias,
    )
    out = jnp.swapaxes(out, -2, -3)
    if params.linear_g is not None:
        out = out * split_heads(jax_sigmoid(linear(x, params.linear_g)), no_heads)
    return linear(flatten_heads(out), params.linear_o)
''',
    "def _ring_attention(",
)

replace_once(
    "src/foldjax/models/protenix/cli/predict.py",
    '        "2d when the device count is a perfect square.",\n',
    '        "2d only when requested explicitly; \'auto\' currently selects 1d.",\n',
)
