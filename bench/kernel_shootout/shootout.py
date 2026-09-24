"""Pair-stack kernel shootout on one GPU: can anything replace cuEquivariance?

Three primitives at the ports' shapes (c_z 128, triangle attention 4 heads x
32, triangle multiplication hidden 128, pair transition 128 -> 2 x 512 -> 128),
bf16 operands, Boltz-2's released weights (trunk Pairformer layer 0):

* ``att``   triangle attention *core* -- q/k/v in the module layout
            ``[1, R, N, H, D]`` (what the projections produce), the shared
            ``[1, 1, H, N, N]`` triangle bias and the additive key mask
            ``[1, R, 1, 1, N]``; each arm does its own layout changes.  The
            projections, gate and output projection around it are the same
            for every arm and are not timed.
* ``mul``   the whole triangle multiplicative update ``x [1, N, N, C] -> out``.
* ``trans`` the whole pair transition ``x [1, N, N, C] -> out``.

Every (family, size, arm) cell runs in its own process (``cell``), so an OOM or
an abort in one kernel costs one cell, and ``peak_bytes_in_use`` is that
arm's own.  A cell writes one JSON line and an ``.npz`` of sampled output rows;
``report`` compares those rows against a float32 HIGHEST reference cell and
against the production arm, on the host.

Mask: the ports build the pair mask from the token pad mask, so valid keys
are a prefix of every row and pad rows are fully masked.  The inputs pad the
last ``PAD_TOKENS`` tokens, which gives both shapes (a prefix key mask and
fully masked rows); errors are read on valid rows, finiteness on both.

    python shootout.py avail --out DIR                       # availability matrix
    python shootout.py drive --out DIR --family att --sizes 1003,2096 [--arms a,b]
    python shootout.py cell --out DIR --family att --n 2096 --arm cueq_pad_highest
    python shootout.py report --out DIR

CPU smoke: ``JAX_PLATFORMS=cpu ... drive --smoke`` runs the XLA arms, cuEq's
reference body and the Pallas kernels in interpret mode at N=64.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: Pair-stack widths.  ``af3``: Boltz-2, Protenix and OpenFold3 (c_z 128,
#: triangle attention 4 x 32, multiplication hidden 128, transition 4x) --
#: read from their checkpoints.  ``opendde``: OpenDDE's trunk (c_z 384, 12 x 32,
#: hidden 384, transition 4x).  Chosen per process by ``SHOOTOUT_PROFILE``.
PROFILES = {
    "af3": {"H": 4, "C": 128, "HID": 128, "FF": 512},
    "opendde": {"H": 12, "C": 384, "HID": 384, "FF": 1536},
}
PROFILE = os.environ.get("SHOOTOUT_PROFILE", "af3")
H, C, HID, FF = (PROFILES[PROFILE][key] for key in ("H", "C", "HID", "FF"))
D = 32
PAD_TOKENS = 3
EPS = 1e-5
INF = 1e9
WEIGHTS = os.environ.get(
    "SHOOTOUT_WEIGHTS",
    "/home/jaemin/non-project/optimizing/foldjax/.foldjax/weights/boltz2/boltz2_conf.safetensors",
)
LAYER = "d:trunk/d:pairformer_module/d:layers/i:0"

ATT_ARMS = (
    "cueq_pad_highest",   # production (Boltz-2 passes HIGHEST), aligned-by-padding
    "cueq_pad_default",
    "cueq_pad_high",      # the policy precision of the ports that derive it ("high")
    "cueq_nopad_highest",  # the slow path, for the record
    "xla_boltz2",         # Boltz-2's XLA path with its own chunk policy
    "tokamax_triton",
    "jax_cudnn_seqlen",   # batch-1 bias + key mask as kv_seqlen (prefix masks only)
    "jax_cudnn_seqlen_pad8",
    "jax_cudnn_mask_r16",  # bias+mask combined: a dense per-row bias (16-row block)
    "pallas_boltz2",      # the repo's own Pallas kernel (f32 dots)
    "fj_flash_r1_64x64w4",
    "fj_flash_r2_64x64w4",
    "fj_flash_r4_64x64w4",
    "fj_flash_r2_128x64w8",
    "fj_flash_r4_64x32w4",
    "fj_flash_r4_128x32w8",
)
MUL_ARMS = tuple(
    f"{arm}_{d}"
    for d in ("out", "in")
    for arm in (
        "cueq_boltz2",        # production Boltz-2 (native AMP composition of cuEq ops)
        "cueq_fused",         # the one-call fused kernel the other ports use
        "xla_boltz2",
        "xla_boltz2_tkglu",   # XLA einsum with the tokamax GLU for the projections
        "tokamax_tm_xla",
        "tokamax_tm_triton",  # tokamax LN + GLU Triton kernels around an XLA einsum
        "xla_cmajor",         # the K1/K2 decomposition written in plain XLA
        "fj_k1k2_64w4",
        "fj_k1k2_128w8",
        "fj_k1k2_32w4",
    )
)
TRANS_ARMS = (
    "boltz2_tokamax_rc",   # production Boltz-2: tokamax GLU, row-chunked
    "boltz2_xla_rc",
    "boltz2_tokamax_full",
    "boltz2_xla_full",
    "fj_fused_64w4",
    "fj_fused_128w8",
    "fj_fused_64w4_fc64",
    "fj_fused_128w4",
    "fj_fused_32w4",
    "fj_fused_32w8",
)
ARMS = {"att": ATT_ARMS, "mul": MUL_ARMS, "trans": TRANS_ARMS}
#: The arm each table is read against.  Boltz-2's released path for ``af3``;
#: for ``opendde`` the fused cuEq calls with the policy precision and the XLA
#: GLU (its ``glu_backend`` default), the latter through Boltz-2's transition
#: code at OpenDDE's widths -- the same arithmetic, not OpenDDE's own module.
PRODUCTION = (
    {"att": "cueq_pad_highest", "mul": "cueq_boltz2", "trans": "boltz2_tokamax_rc"}
    if PROFILE == "af3"
    else {"att": "cueq_pad_high", "mul": "cueq_fused", "trans": "boltz2_xla_full"}
)


def log(record: dict, out: Path, name: str = "cells.jsonl") -> None:
    line = json.dumps(record)
    print(line, flush=True)
    with open(out / name, "a") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def load_params():
    """Boltz-2 trunk layer-0 weights: Linear kernels bf16 (the released cast),
    norm affine float32 (`trunk._cast_trunk_params` keeps it).  The ``opendde``
    profile has no Boltz-2-layout checkpoint at its widths; it gets seeded
    synthetic weights at fan-in scale, which is what timing needs and enough
    for the error columns (every arm sees the same weights)."""
    import jax.numpy as jnp
    import numpy as np

    if PROFILE != "af3":
        rng = np.random.default_rng(12)

        def lin(fan_in, fan_out):
            w = rng.normal(size=(fan_in, fan_out)) / np.sqrt(fan_in)
            return {"kernel": jnp.asarray(w, jnp.bfloat16)}

        def norm(width):
            return {"scale": jnp.asarray(1 + 0.1 * rng.normal(size=width), jnp.float32),
                    "bias": jnp.asarray(0.1 * rng.normal(size=width), jnp.float32)}

        mul = {d: {"p_in": lin(C, 2 * HID), "g_in": lin(C, 2 * HID), "p_out": lin(HID, C),
                   "g_out": lin(C, C), "norm_in": norm(C), "norm_out": norm(HID)}
               for d in ("out", "in")}
        trans = {"fc1": lin(C, FF), "fc2": lin(C, FF), "fc3": lin(FF, C), "norm": norm(C)}
        return mul, trans

    from safetensors import safe_open

    f = safe_open(WEIGHTS, "numpy")

    def get(*path):
        return jnp.asarray(f.get_tensor("/".join((LAYER, *(f"d:{p}" for p in path)))))

    def tree(module, linears, norms):
        out = {}
        for name in linears:
            out[name] = {"kernel": get(module, name, "kernel").astype(jnp.bfloat16)}
        for name in norms:
            out[name] = {"scale": get(module, name, "scale"), "bias": get(module, name, "bias")}
        return out

    mul = {d: tree(f"tri_mul_{d}", ("p_in", "g_in", "p_out", "g_out"), ("norm_in", "norm_out"))
           for d in ("out", "in")}
    trans = tree("transition_z", ("fc1", "fc2", "fc3"), ("norm",))
    return mul, trans


def token_valid(n):
    import jax.numpy as jnp

    return jnp.arange(n) < n - PAD_TOKENS


def sample_rows(n):
    """Rows whose outputs are kept: spread over the valid rows, plus one pad row."""
    rows = sorted({0, 1, n // 3, n // 2, (2 * n) // 3, n - PAD_TOKENS - 2, n - PAD_TOKENS - 1, n - 1})
    return [r for r in rows if 0 <= r < n]


#: Rows generated per random block.  Full-size inputs are this block tiled
#: along the row axis: ``jax.random.normal`` materialises a float32 bit tensor
#: before a bf16 cast, and at full size that transient (2.25 GB per operand at
#: 2,096) would set every arm's ``peak_bytes_in_use`` -- a process high-water
#: mark -- instead of the arm.  Timing is data-independent; the error columns
#: compare every arm on the same (periodic) inputs.
TILE = 64


def _tiled_normal(key, shape, dtype, axis=1):
    import jax
    import jax.numpy as jnp

    rows = shape[axis]
    block = list(shape)
    block[axis] = min(TILE, rows)
    tile = jax.random.normal(key, tuple(block), jnp.float32).astype(dtype)
    reps = [1] * len(shape)
    reps[axis] = -(-rows // block[axis])
    out = jnp.tile(tile, reps)
    return jax.lax.slice_in_dim(out, 0, rows, axis=axis)


def att_inputs(n, bias_dtype):
    import jax
    import jax.numpy as jnp

    kq, kk, kv, kb = jax.random.split(jax.random.key(n), 4)
    shape = (1, n, n, H, D)
    q = _tiled_normal(kq, shape, jnp.bfloat16)
    k = _tiled_normal(kk, shape, jnp.bfloat16)
    v = _tiled_normal(kv, shape, jnp.bfloat16)
    bias = (jax.random.normal(kb, (1, 1, H, n, n), jnp.float32) * 2.0).astype(bias_dtype)
    valid = token_valid(n)
    pair = valid[:, None] & valid[None, :]
    mask_bias = jnp.where(pair, 0.0, -INF).astype(jnp.float32)[None, :, None, None, :]
    return q, k, v, bias, mask_bias


def pair_inputs(n):
    import jax
    import jax.numpy as jnp

    x = _tiled_normal(jax.random.key(n + 7), (1, n, n, C), jnp.bfloat16)
    valid = token_valid(n)
    mask = (valid[:, None] & valid[None, :]).astype(jnp.float32)[None]
    return x, mask


# --------------------------------------------------------------------------- #
# Attention arms: fn(q, k, v, bias, mask_bias) -> [1, R, N, H, D]
# --------------------------------------------------------------------------- #


def boltz2_chunks(n):
    from foldjax.models.boltz2.models.trunk_blocks.trunk import resolve_long_sequence_chunks

    return resolve_long_sequence_chunks(
        n, chunk_size=128, triangle_attention_chunk=None,
        triangle_attention_q_chunk=None, token_attention_chunk=None,
    )


def att_arm(arm, n):
    import jax
    import jax.numpy as jnp
    from jax import lax

    scale = D**-0.5
    sw = lambda t: jnp.swapaxes(t, -2, -3)  # noqa: E731  [.., R, N, H, D] <-> [.., R, H, N, D]

    if arm.startswith("cueq_pad") or arm == "cueq_nopad_highest":
        from foldjax.models._cueq import cueq_attention_core, load_cueq

        prec = {"default": lax.Precision.DEFAULT, "high": lax.Precision.HIGH}.get(
            arm.rsplit("_", 1)[1], lax.Precision.HIGHEST)
        if arm.startswith("cueq_pad"):
            def fn(q, k, v, bias, mask_bias):
                return sw(cueq_attention_core(sw(q), sw(k), sw(v), bias, mask_bias,
                                              scale=scale, precision=prec))
        else:
            cuex = load_cueq()

            def fn(q, k, v, bias, mask_bias):
                out, _, _ = cuex.triangle_attention(
                    q=sw(q), k=sw(k), v=sw(v),
                    bias=bias, mask=mask_bias == 0, scale=scale, precision=prec)
                return sw(out)
        return fn

    if arm == "xla_boltz2":
        from foldjax.models.boltz2.models.triangle.triangle_attention import _attention_core

        chunks = boltz2_chunks(n)

        def fn(q, k, v, bias, mask_bias):
            qs = (sw(q).astype(jnp.float32) * scale).astype(q.dtype)  # native AMP scaling
            return sw(_attention_core(qs, sw(k), sw(v), bias, mask_bias,
                                      chunks["triangle_attention_chunk"],
                                      chunks["triangle_attention_q_chunk"], native_amp=True))
        return fn

    if arm.startswith("tokamax_"):
        import tokamax
        from absl import flags

        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["shootout"], known_only=True)
        impl = arm[len("tokamax_"):]

        def fn(q, k, v, bias, mask_bias):
            return tokamax.dot_product_attention(
                q, k, v, bias=bias, mask=mask_bias >= 0.0, scale=scale, implementation=impl
            ).astype(q.dtype)
        return fn

    if arm.startswith("jax_cudnn_seqlen"):
        pad8 = arm.endswith("pad8")

        def fn(q, k, v, bias, mask_bias):
            r = q.shape[1]
            qf, kf, vf = (t[0] for t in (q, k, v))  # [R, N, H, D] = BTNH
            b = bias[0]  # [1, H, N, N]
            kv_len = jnp.sum(mask_bias[0, :, 0, 0, :] == 0, axis=-1).astype(jnp.int32)
            m = n
            if pad8 and n % 8:
                extra = -n % 8
                qf, kf, vf = (jnp.pad(t, ((0, 0), (0, extra), (0, 0), (0, 0))) for t in (qf, kf, vf))
                b = jnp.pad(b, ((0, 0), (0, 0), (0, extra), (0, extra)))
                m = n + extra
            out = jax.nn.dot_product_attention(
                qf, kf, vf, bias=b, scale=scale, key_value_seq_lengths=kv_len,
                query_seq_lengths=jnp.full((r,), m, jnp.int32), implementation="cudnn")
            return out[None, :, :n]
        return fn

    if arm == "jax_cudnn_mask_r16":
        def fn(q, k, v, bias, mask_bias):
            qf, kf, vf = (t[0] for t in (q, k, v))
            mask = jnp.broadcast_to((mask_bias[0, :, 0, 0] == 0)[:, None, None, :], (q.shape[1], 1, n, n))
            b = jnp.broadcast_to(bias[0], (1, H, n, n))
            return jax.nn.dot_product_attention(
                qf, kf, vf, bias=b, mask=mask, scale=scale, implementation="cudnn")[None]
        return fn

    if arm == "pallas_boltz2":
        from foldjax.models.boltz2.models.triangle.triangle_attention_pallas import (
            pallas_attention_core,
        )

        def fn(q, k, v, bias, mask_bias):
            qs = sw(q) * jnp.asarray(scale, q.dtype)
            return sw(pallas_attention_core(qs, sw(k), sw(v), bias, mask_bias))
        return fn

    if arm.startswith("fj_flash_"):
        import fjk

        rbs, blocks = arm[len("fj_flash_"):].split("_")
        rb = int(rbs[1:])
        blocks, warps = blocks.split("w")
        bq, bk = (int(x) for x in blocks.split("x"))

        def fn(q, k, v, bias, mask_bias):
            out = fjk.flash_triangle_attention(
                q[0], k[0], v[0], bias[0, 0], mask_bias[0, :, 0, 0, :],
                scale=scale, rb=rb, bq=bq, bk=bk, num_warps=int(warps), num_stages=2)
            return out[None]
        return fn

    raise ValueError(arm)


def att_reference(q, k, v, bias, mask_bias, rows):
    import jax.numpy as jnp
    from jax import lax

    idx = jnp.asarray(rows)
    f = lambda t: t[0, idx].astype(jnp.float32)  # noqa: E731  [S, N, H, D]
    qs, ks, vs = f(q), f(k), f(v)
    s = jnp.einsum("sqhd,skhd->shqk", qs, ks, precision=lax.Precision.HIGHEST) * D**-0.5
    s = s + bias[0].astype(jnp.float32) + mask_bias[0, idx]
    p = jnp.exp(s - s.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    return jnp.einsum("shqk,skhd->sqhd", p, vs, precision=lax.Precision.HIGHEST)


# --------------------------------------------------------------------------- #
# Triangle multiplication arms: fn(params, x, mask) -> [1, N, N, C]
# --------------------------------------------------------------------------- #


def _swap_halves(w):
    import jax.numpy as jnp

    h = w.shape[-1] // 2
    return jnp.concatenate([w[..., h:], w[..., :h]], axis=-1)


def mul_arm(arm, n):
    import jax
    import jax.numpy as jnp
    from jax import lax

    base, d = arm.rsplit("_", 1)
    direction = {"out": "outgoing", "in": "incoming"}[d]

    if base in ("cueq_boltz2", "xla_boltz2", "xla_boltz2_tkglu"):
        from foldjax.models.boltz2.models.triangle.triangle import (
            triangle_multiplication_forward,
        )

        os.environ["BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND"] = (
            "cueq" if base == "cueq_boltz2" else "xla")
        glu = "tokamax" if base == "xla_boltz2_tkglu" else "xla"
        chunk = boltz2_chunks(n)["chunk_size"]

        def fn(p, x, mask):
            return triangle_multiplication_forward(
                p, x, mask, direction, chunk_size=chunk, glu_backend=glu, native_amp=True)
        return fn

    if base == "cueq_fused":
        from foldjax.models._cueq import fused_triangle_multiplication

        def fn(p, x, mask):
            return fused_triangle_multiplication(
                x, direction=direction, mask=mask,
                norm_in=(p["norm_in"]["scale"], p["norm_in"]["bias"]),
                p_in=(p["p_in"]["kernel"].T, None), g_in=(p["g_in"]["kernel"].T, None),
                norm_out=(p["norm_out"]["scale"], p["norm_out"]["bias"]),
                p_out=(p["p_out"]["kernel"].T, None), g_out=(p["g_out"]["kernel"].T, None),
                eps=EPS)
        return fn

    if base.startswith("tokamax_tm_"):
        import tokamax
        from absl import flags

        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["shootout"], known_only=True)
        impl = base[len("tokamax_tm_"):]

        def fn(p, x, mask):
            wp, wg = p["p_in"]["kernel"], p["g_in"]["kernel"]
            if direction == "incoming":
                # tokamax's incoming contracts `ckj,cki` -- a and b swap roles
                # against Boltz-2's `bkic,bkjc`; swapping the halves matches it.
                wp, wg = _swap_halves(wp), _swap_halves(wg)
            out = tokamax.triangle_multiplication(
                x[0], mask[0] > 0, wp.reshape(C, 2, HID), wg.reshape(C, 2, HID),
                p["p_out"]["kernel"], p["g_out"]["kernel"],
                p["norm_in"]["scale"], p["norm_in"]["bias"],
                p["norm_out"]["scale"], p["norm_out"]["bias"],
                direction, epsilon=EPS, implementation=impl)
            return out[None].astype(x.dtype)
        return fn

    if base == "xla_cmajor":
        def ln(t, w, b):
            t = t.astype(jnp.float32)
            m = t.mean(-1, keepdims=True)
            v = ((t - m) ** 2).mean(-1, keepdims=True)
            return (t - m) * lax.rsqrt(v + EPS) * w + b

        def fn(p, x, mask):
            xn = ln(x[0], p["norm_in"]["scale"], p["norm_in"]["bias"]).astype(jnp.bfloat16)
            proj = jnp.einsum("ijc,ch->hij", xn, p["p_in"]["kernel"],
                              preferred_element_type=jnp.float32)
            gate = jnp.einsum("ijc,ch->hij", xn, p["g_in"]["kernel"],
                              preferred_element_type=jnp.float32)
            ab = (jax.nn.sigmoid(gate) * proj * mask[0][None]).astype(jnp.bfloat16)
            a, b = ab[:HID], ab[HID:]
            dims = ((((2,), (2,)), ((0,), (0,))) if direction == "outgoing"
                    else (((1,), (1,)), ((0,), (0,))))
            e = lax.dot_general(a, b, dims)  # [H, N, N]
            e = jnp.moveaxis(e, 0, -1)
            en = ln(e, p["norm_out"]["scale"], p["norm_out"]["bias"]).astype(jnp.bfloat16)
            out = jnp.dot(en, p["p_out"]["kernel"], preferred_element_type=jnp.float32)
            g = jnp.dot(xn, p["g_out"]["kernel"], preferred_element_type=jnp.float32)
            return (out * jax.nn.sigmoid(g)).astype(x.dtype)[None]
        return fn

    if base.startswith("fj_k1k2_"):
        import fjk

        bm, warps = (int(t) for t in base[len("fj_k1k2_"):].split("w"))

        def fn(p, x, mask):
            return fjk.triangle_multiplication_k1k2(
                x[0], mask[0], p, direction=direction, eps=EPS,
                bm1=bm, bm2=bm, warps1=warps, warps2=warps)[None]
        return fn

    raise ValueError(arm)


def mul_reference(p, x, mask, rows, direction):
    import jax
    import jax.numpy as jnp
    from jax import lax

    hp = lax.Precision.HIGHEST
    f = lambda t: t.astype(jnp.float32)  # noqa: E731

    def ln(t, w, b):
        m = t.mean(-1, keepdims=True)
        v = ((t - m) ** 2).mean(-1, keepdims=True)
        return (t - m) * lax.rsqrt(v + EPS) * w + b

    def proj(t, msk):
        tn = ln(f(t), p["norm_in"]["scale"], p["norm_in"]["bias"])
        ab = jax.nn.sigmoid(jnp.dot(tn, f(p["g_in"]["kernel"]), precision=hp)) * jnp.dot(
            tn, f(p["p_in"]["kernel"]), precision=hp)
        return ab * msk[..., None], tn

    idx = jnp.asarray(rows)
    x0, m0 = x[0], mask[0]
    if direction == "outgoing":
        a_sel, xn_sel = proj(x0[idx], m0[idx])  # [S, N, 2H]
        a_sel = a_sel[..., :HID]
    else:
        a_sel, _ = proj(jnp.swapaxes(x0[:, idx], 0, 1), jnp.swapaxes(m0[:, idx], 0, 1))
        a_sel = a_sel[..., :HID]  # a[k, i_sel] laid out [S, K, H]
        xn_sel = ln(f(x0[idx]), p["norm_in"]["scale"], p["norm_in"]["bias"])
    e = jnp.zeros((len(rows), x.shape[1], HID), jnp.float32)
    step = 512
    for k0 in range(0, x.shape[1], step):
        if direction == "outgoing":
            b_blk, _ = proj(x0[:, k0:k0 + step], m0[:, k0:k0 + step])  # b[j, k]
            e = e + jnp.einsum("skc,jkc->sjc", a_sel[:, k0:k0 + step], b_blk[..., HID:], precision=hp)
        else:
            b_blk, _ = proj(x0[k0:k0 + step], m0[k0:k0 + step])  # b[k, j]
            e = e + jnp.einsum("skc,kjc->sjc", a_sel[:, k0:k0 + step], b_blk[..., HID:], precision=hp)
    en = ln(e, p["norm_out"]["scale"], p["norm_out"]["bias"])
    out = jnp.dot(en, f(p["p_out"]["kernel"]), precision=hp)
    return out * jax.nn.sigmoid(jnp.dot(xn_sel, f(p["g_out"]["kernel"]), precision=hp))


# --------------------------------------------------------------------------- #
# Transition arms: fn(params, x) -> [1, N, N, C]
# --------------------------------------------------------------------------- #


def trans_arm(arm, n):
    import jax.numpy as jnp

    if arm.startswith("boltz2_"):
        from foldjax.models.boltz2.models.primitives.transition import transition_forward

        glu = "tokamax" if "tokamax" in arm else "xla"
        rows = boltz2_chunks(n)["chunk_size"] if arm.endswith("_rc") else 0

        def fn(p, x):
            return transition_forward(p, x, row_chunk_size=rows, glu_backend=glu,
                                      native_amp_norm=True)
        return fn

    if arm.startswith("fj_fused_"):
        import fjk

        spec = arm[len("fj_fused_"):].split("_")
        bm, warps = (int(t) for t in spec[0].split("w"))
        fc = int(spec[1][2:]) if len(spec) > 1 else 128

        def fn(p, x):
            out = fjk.fused_transition(
                x.reshape(-1, C), p["norm"]["scale"], p["norm"]["bias"],
                p["fc1"]["kernel"], p["fc2"]["kernel"], p["fc3"]["kernel"],
                eps=EPS, bm=bm, fc=fc, num_warps=warps, num_stages=2)
            return out.reshape(x.shape[:-1] + (out.shape[-1],)).astype(jnp.bfloat16)
        return fn

    raise ValueError(arm)


def trans_reference(p, x, rows):
    import jax
    import jax.numpy as jnp
    from jax import lax

    hp = lax.Precision.HIGHEST
    f = lambda t: t.astype(jnp.float32)  # noqa: E731
    t = f(x[0, jnp.asarray(rows)])
    m = t.mean(-1, keepdims=True)
    v = ((t - m) ** 2).mean(-1, keepdims=True)
    tn = (t - m) * lax.rsqrt(v + EPS) * p["norm"]["scale"] + p["norm"]["bias"]
    h = jax.nn.silu(jnp.dot(tn, f(p["fc1"]["kernel"]), precision=hp)) * jnp.dot(
        tn, f(p["fc2"]["kernel"]), precision=hp)
    return jnp.dot(h, f(p["fc3"]["kernel"]), precision=hp)


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #


def custom_call_targets(text: str) -> dict[str, int]:
    import re

    counts: dict[str, int] = {}
    for m in re.finditer(r'custom_call_target="([^"]+)"', text):
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


def run_cell(args) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_default_matmul_precision", "high")  # Boltz-2's released scope
    if args.interpret:
        import fjk

        fjk.INTERPRET = True
    out_dir = Path(args.out)
    n, fam, arm = args.n, args.family, args.arm
    rec = {"profile": PROFILE, "family": fam, "n": n, "arm": arm, "mod8": n % 8,
           "bias_dtype": args.bias_dtype,
           "backend": jax.default_backend()}
    rows = sample_rows(n)
    t_start = time.perf_counter()
    try:
        if fam == "att":
            inputs = att_inputs(n, jnp.dtype(args.bias_dtype))
            if arm == "ref":
                ref = att_reference(*inputs, rows)
            else:
                fn = att_arm(arm, n)
        else:
            mul_p, trans_p = load_params()
            x, mask = pair_inputs(n)
            if fam == "mul":
                base, d = arm.rsplit("_", 1)
                p = mul_p[d]
                inputs = (p, x, mask)
                if base == "ref":
                    ref = mul_reference(p, x, mask, rows, {"out": "outgoing", "in": "incoming"}[d])
                else:
                    fn = mul_arm(arm, n)
            else:
                inputs = (trans_p, x)
                if arm == "ref":
                    ref = trans_reference(trans_p, x, rows)
                else:
                    fn = trans_arm(arm, n)
        if arm == "ref" or arm.startswith("ref_"):
            sampled = np.asarray(ref, np.float32)
            rec["ref_s"] = round(time.perf_counter() - t_start, 2)
        else:
            if fam == "att" and arm == "jax_cudnn_mask_r16":
                r16 = 16
                inputs = tuple(t[:, :r16] if i in (0, 1, 2, 4) else t for i, t in enumerate(inputs))
                rec["rows"] = r16
            jit_fn = jax.jit(fn)
            t0 = time.perf_counter()
            lowered = jit_fn.lower(*inputs)
            compiled = lowered.compile()
            rec["compile_s"] = round(time.perf_counter() - t0, 2)
            ma = compiled.memory_analysis()
            if ma is not None:
                rec["temp_mib"] = round(ma.temp_size_in_bytes / 2**20, 1)
                rec["arg_mib"] = round(ma.argument_size_in_bytes / 2**20, 1)
                rec["out_mib"] = round(ma.output_size_in_bytes / 2**20, 1)
            rec["custom_calls"] = custom_call_targets(compiled.as_text() or "")
            out = compiled(*inputs)
            out.block_until_ready()
            compiled(*inputs).block_until_ready()
            times = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                compiled(*inputs).block_until_ready()
                times.append(time.perf_counter() - t0)
            rec["ms"] = round(float(np.median(times)) * 1e3, 3)
            rec["ms_min"] = round(float(np.min(times)) * 1e3, 3)
            rec["ms_all"] = [round(t * 1e3, 3) for t in times]
            try:
                stats = jax.devices()[0].memory_stats() or {}
                if "peak_bytes_in_use" in stats:
                    rec["peak_mib"] = round(stats["peak_bytes_in_use"] / 2**20, 1)
            except Exception:  # noqa: BLE001
                pass
            kept = [r for r in rows if r < out.shape[1]]
            sampled = np.asarray(out[0, jnp.asarray(kept)], np.float32)
            rec["finite_valid_rows"] = bool(np.isfinite(sampled[[i for i, r in enumerate(kept) if r < n - PAD_TOKENS]]).all())
            pad_rows = [i for i, r in enumerate(kept) if r >= n - PAD_TOKENS]
            if pad_rows:
                rec["finite_pad_rows"] = bool(np.isfinite(sampled[pad_rows]).all())
            rows = kept
        np.savez(out_dir / f"{fam}-{n}-{arm}.npz", rows=np.asarray(rows), out=sampled)
        rec["ok"] = True
    except BaseException as error:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = repr(error)[:1500]
        if isinstance(error, BaseExceptionGroup):
            rec["sub_errors"] = [f"{type(e).__name__}: {str(e)[:800]}" for e in error.exceptions]
        rec["traceback_tail"] = traceback.format_exc()[-1500:]
    rec["cell_s"] = round(time.perf_counter() - t_start, 2)
    log(rec, out_dir)


# --------------------------------------------------------------------------- #
# Availability probes (tiny shapes, one process each)
# --------------------------------------------------------------------------- #

PROBES = (
    "tokamax_att_xla", "tokamax_att_xla_chunked", "tokamax_att_triton", "tokamax_att_cudnn",
    "tokamax_att_mosaic", "tokamax_att_mosaic_forced_sm90", "tokamax_att_mosaic_forced_sm100",
    "jax_att_xla_bias_mask", "jax_att_cudnn_bias_mask", "jax_att_cudnn_bias1_seqlen",
    "jax_att_cudnn_bias1_seqlen_bf16bias",
    "pallas_triton_boltz2_att", "pallas_triton_fj_att", "pallas_triton_fj_trimul",
    "pallas_triton_fj_transition",
    "mosaic_gpu_trivial",
    "tokamax_glu_triton", "tokamax_glu_mosaic", "tokamax_glu_mosaic_forced_sm80",
    "tokamax_glu_mosaic_forced_sm90",
    "tokamax_glu_xla", "tokamax_layer_norm_triton",
    "tokamax_trimul_xla", "tokamax_trimul_triton",
    "cueq_attention", "cueq_trimul",
)


def run_probe(args) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    name = args.probe
    rec = {"probe": name, "backend": jax.default_backend()}
    try:
        dev = jax.devices()[0]
        rec["device"] = getattr(dev, "device_kind", str(dev))
        rec["cc"] = getattr(dev, "compute_capability", None)
    except Exception:  # noqa: BLE001
        pass
    n, r = 64, 8
    t0 = time.perf_counter()
    try:
        from absl import flags

        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["shootout"], known_only=True)
        q, k, v, bias, mask_bias = (t[:, :r] if i in (0, 1, 2, 4) else t
                                    for i, t in enumerate(att_inputs(n, jnp.float32)))
        ref = att_reference(q, k, v, bias, mask_bias, list(range(r)))
        valid_rows = [i for i in range(r) if i < n - PAD_TOKENS]
        scale = D**-0.5
        result = None
        if name.startswith("tokamax_att_"):
            import tokamax

            impl = name[len("tokamax_att_"):]
            if impl.startswith("mosaic_forced"):
                from tokamax._src import gpu_utils
                from tokamax._src.ops.attention import pallas_mosaic_gpu as pmg

                arch = impl.rsplit("_", 1)[1]
                gpu_utils.is_sm90 = lambda device=None: arch == "sm90"  # noqa: E731
                gpu_utils.is_sm100 = lambda device=None: arch == "sm100"  # noqa: E731
                pmg.gpu_utils = gpu_utils
                impl = "mosaic"
            result = tokamax.dot_product_attention(
                q, k, v, bias=bias, mask=mask_bias >= 0, scale=scale, implementation=impl)
        elif name == "jax_att_xla_bias_mask" or name == "jax_att_cudnn_bias_mask":
            impl = "xla" if "xla" in name else "cudnn"
            mask = jnp.broadcast_to((mask_bias[0, :, 0, 0] == 0)[:, None, None, :], (r, 1, n, n))
            result = jax.nn.dot_product_attention(
                q[0], k[0], v[0], bias=bias[0], mask=mask, scale=scale, implementation=impl)[None]
        elif name.startswith("jax_att_cudnn_bias1_seqlen"):
            b = bias[0].astype(jnp.bfloat16) if name.endswith("bf16bias") else bias[0]
            kv_len = jnp.sum(mask_bias[0, :, 0, 0, :] == 0, axis=-1).astype(jnp.int32)
            result = jax.nn.dot_product_attention(
                q[0], k[0], v[0], bias=b, scale=scale, key_value_seq_lengths=kv_len,
                query_seq_lengths=jnp.full((r,), n, jnp.int32), implementation="cudnn")[None]
        elif name == "pallas_triton_boltz2_att":
            result = att_arm("pallas_boltz2", n)(q, k, v, bias, mask_bias)
        elif name == "pallas_triton_fj_att":
            result = att_arm("fj_flash_r2_64x64w4", n)(q, k, v, bias, mask_bias)
        elif name == "cueq_attention":
            result = att_arm("cueq_pad_highest", n)(q, k, v, bias, mask_bias)
        elif name in ("pallas_triton_fj_trimul", "cueq_trimul", "tokamax_trimul_xla",
                      "tokamax_trimul_triton"):
            mul_p, _ = load_params()
            x, mask = pair_inputs(n)
            arm = {"pallas_triton_fj_trimul": "fj_k1k2_64w4_out", "cueq_trimul": "cueq_fused_out",
                   "tokamax_trimul_xla": "tokamax_tm_xla_out",
                   "tokamax_trimul_triton": "tokamax_tm_triton_out"}[name]
            out = jax.jit(mul_arm(arm, n))(mul_p["out"], x, mask)
            ref = mul_reference(mul_p["out"], x, mask, list(range(r)), "outgoing")
            result = out[:, :r]
        elif name == "pallas_triton_fj_transition":
            _, trans_p = load_params()
            x, _ = pair_inputs(n)
            out = jax.jit(trans_arm("fj_fused_64w4", n))(trans_p, x)
            ref = trans_reference(trans_p, x, list(range(r)))
            result = out[:, :r]
        elif name.startswith("tokamax_glu_") or name == "tokamax_layer_norm_triton":
            import tokamax

            xg = jax.random.normal(jax.random.key(1), (256, 128), jnp.bfloat16)
            w = (jax.random.normal(jax.random.key(2), (128, 2, 512), jnp.float32) / 11).astype(jnp.bfloat16)
            if name == "tokamax_layer_norm_triton":
                result = tokamax.layer_norm(xg, jnp.ones(128), jnp.zeros(128), implementation="triton")
                ref = tokamax.layer_norm(xg.astype(jnp.float32), jnp.ones(128), jnp.zeros(128), implementation="xla")
            else:
                impl = name[len("tokamax_glu_"):]
                if impl.startswith("mosaic_forced_"):
                    from tokamax._src import gpu_utils
                    from tokamax._src.ops.gated_linear_unit import pallas_mosaic_gpu as gmg

                    arch = impl.rsplit("_", 1)[1]
                    gpu_utils.is_sm80 = lambda device=None: arch == "sm80"  # noqa: E731
                    gpu_utils.is_sm90 = lambda device=None: arch == "sm90"  # noqa: E731
                    gpu_utils.is_sm100 = lambda device=None: False  # noqa: E731
                    gmg.gpu_utils = gpu_utils
                    impl = "mosaic"
                result = tokamax.gated_linear_unit(xg, w, activation=jax.nn.silu, implementation=impl)
                xf, wf = xg.astype(jnp.float32), w.astype(jnp.float32)
                ref = jax.nn.silu(xf @ wf[:, 0]) * (xf @ wf[:, 1])
            valid_rows = None
        elif name == "mosaic_gpu_trivial":
            from jax.experimental import pallas as pl
            from jax.experimental.pallas import mosaic_gpu as plgpu

            def kern(x_ref, o_ref):
                o_ref[...] = x_ref[...] + 1.0

            xm = jnp.arange(128 * 128, dtype=jnp.float32).reshape(128, 128)
            result = pl.pallas_call(kern, out_shape=jax.ShapeDtypeStruct(xm.shape, xm.dtype),
                                    compiler_params=plgpu.CompilerParams())(xm)
            ref = xm + 1.0
            valid_rows = None
        else:
            raise ValueError(name)
        result = np.asarray(result, np.float32)
        ref = np.asarray(ref, np.float32)
        if valid_rows is not None:
            result, ref = result[0, valid_rows], ref[valid_rows]
        rec["max_abs_vs_f32"] = float(np.abs(result - ref).max())
        rec["ref_scale"] = float(np.abs(ref).max())
        rec["finite"] = bool(np.isfinite(result).all())
        rec["status"] = "ok"
    except BaseException as error:  # noqa: BLE001
        rec["status"] = "fail"
        rec["error_type"] = type(error).__name__
        rec["error"] = str(error)[:2000]
        if isinstance(error, BaseExceptionGroup):
            rec["sub_errors"] = [f"{type(e).__name__}: {str(e)[:800]}" for e in error.exceptions]
    rec["s"] = round(time.perf_counter() - t0, 2)
    log(rec, Path(args.out), "avail.jsonl")


# --------------------------------------------------------------------------- #
# Driver and report
# --------------------------------------------------------------------------- #


def spawn(argv: list[str], out: Path, timeout: int, label: dict, logname: str) -> None:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *argv],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            log({**label, "ok": False, "status": "fail", "returncode": proc.returncode,
                 "stderr_tail": proc.stderr[-2500:], "s": round(time.perf_counter() - t0, 1)},
                out, logname)
        else:
            sys.stdout.write(proc.stdout.splitlines()[-1] + "\n" if proc.stdout else "")
    except subprocess.TimeoutExpired as error:
        tail = (error.stderr or b"")
        tail = tail.decode(errors="replace") if isinstance(tail, bytes) else str(tail)
        log({**label, "ok": False, "status": "timeout", "timeout_s": timeout,
             "stderr_tail": tail[-2500:]}, out, logname)


def drive(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in args.sizes.split(",")]
    families = args.family.split(",")
    for n in sizes:
        for fam in families:
            arms = args.arms.split(",") if args.arms else list(ARMS[fam])
            if fam == "mul":
                refs = ["ref_out", "ref_in"]
                arms = [a for a in arms if a.rsplit("_", 1)[1] in ("out", "in")]
            else:
                refs = ["ref"]
            if fam == "att" and n % 8 == 0:
                arms = [a for a in arms if a not in ("cueq_nopad_highest", "jax_cudnn_seqlen_pad8")]
            if fam == "att" and n > 3012 and not args.arms:
                # Both are >= 4x production at 2,096 (wall_split: XLA 157 ms vs 40 ms);
                # past 3,012 they only cost job time (the XLA loop unrolls ~4k blocks).
                arms = [a for a in arms if a not in ("xla_boltz2", "pallas_boltz2")]
            for arm in refs + arms:
                if args.smoke and not smoke_ok(fam, arm):
                    continue
                argv = ["cell", "--out", str(out), "--family", fam, "--n", str(n), "--arm", arm,
                        "--reps", str(args.reps), "--bias-dtype", args.bias_dtype]
                if args.smoke:
                    argv.append("--interpret")
                spawn(argv, out, args.timeout, {"family": fam, "n": n, "arm": arm}, "cells.jsonl")


def smoke_ok(fam: str, arm: str) -> bool:
    """Arms that can run on the CPU: XLA ones, cuEq's reference body, Pallas in
    interpret mode.  tokamax Triton, cuDNN and the repo's Pallas kernel cannot."""
    del fam
    if "tkglu" in arm or "tokamax" in arm and "tokamax_tm_xla" not in arm:
        return False
    if arm.startswith(("ref", "fj_", "xla_", "boltz2_xla", "tokamax_tm_xla")):
        return True
    return arm.startswith("cueq") and arm != "cueq_nopad_highest"


def avail(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    probes = args.probes.split(",") if args.probes else list(PROBES)
    for probe in probes:
        spawn(["probe", "--out", str(out), "--probe", probe], out, args.timeout,
              {"probe": probe}, "avail.jsonl")


def report(args) -> None:
    import numpy as np

    out = Path(args.out)
    cells_path = out / "cells.jsonl"
    cells = [json.loads(line) for line in open(cells_path)] if cells_path.exists() else []
    latest = {}
    for c in cells:
        latest[(c["family"], c["n"], c["arm"])] = c
    lines = [
        f"# pair-kernel shootout, profile {PROFILE} (H={H}, C={C}, HID={HID}, FF={FF})",
        "",
        "- att times are the attention *core* only (q/k/v/bias/mask -> out, each arm's own layout",
        "  changes); the like-for-like cuEq anchor is align.py (24.5 ms at 2,096, f32 bias, 32 rows",
        "  scaled), not wall_split's 39-41 ms, which includes projections, gate and out-projection.",
        "- mul and trans are whole modules (x -> out).",
        "- tokamax rows run heuristic tile configs: there is no sm_120 tuning cache.",
        "- jax_cudnn_mask_r16 runs 16 rows (its bias+mask is materialised per row); its ms is",
        "  scaled to all rows for the ratio, and its temp column is the point.",
        "- custom calls are HLO targets, not CUDA kernel names.",
        f"- production arms: {PRODUCTION}",
    ]
    for fam in ("att", "mul", "trans"):
        sizes = sorted({n for f, n, _ in latest if f == fam})
        for n in sizes:
            lines.append(f"\n## {fam}  N={n}  (N mod 8 = {n % 8})")
            lines.append("| arm | ms (median of 7) | vs production | temp MiB | peak MiB | max-abs vs f32 ref (ref max) | max-abs vs production | finite pad rows | custom calls |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for (f, nn, arm), c in sorted(latest.items(), key=lambda kv: kv[0][2]):
                if f != fam or nn != n or arm.startswith("ref"):
                    continue
                if fam == "mul":
                    d = arm.rsplit("_", 1)[1]
                    prod_arm = f"{PRODUCTION['mul']}_{d}"
                    ref_arm = f"ref_{d}"
                else:
                    prod_arm, ref_arm = PRODUCTION[fam], "ref"
                prod = latest.get((fam, n, prod_arm), {})
                err_ref = err_prod = "-"
                ref_npz = out / f"{fam}-{n}-{ref_arm}.npz"
                scale = f"{float(np.abs(np.load(ref_npz)['out']).max()):.3g}" if ref_npz.exists() else "-"
                npz = out / f"{fam}-{n}-{arm}.npz"
                if c.get("ok") and npz.exists():
                    mine = np.load(npz)
                    valid = mine["rows"] < n - PAD_TOKENS
                    for tag, other_arm in (("ref", ref_arm), ("prod", prod_arm)):
                        other = out / f"{fam}-{n}-{other_arm}.npz"
                        if not other.exists() or other_arm == arm:
                            continue
                        o = np.load(other)
                        common = [i for i, r in enumerate(mine["rows"]) if r in set(o["rows"].tolist()) and valid[i]]
                        oi = [list(o["rows"]).index(mine["rows"][i]) for i in common]
                        e = float(np.abs(mine["out"][common] - o["out"][oi]).max()) if common else float("nan")
                        if tag == "ref":
                            err_ref = f"{e:.4g}"
                        else:
                            err_prod = f"{e:.4g}"
                ms_full = c.get("ms")
                if ms_full and c.get("rows"):
                    ms_full = ms_full * n / c["rows"]  # a row-block arm, scaled to all rows
                ratio = (f"{ms_full / prod['ms']:.3f}x" if ms_full and prod.get("ms") else "-")
                ms = c.get("ms", "FAIL" if not c.get("ok") else "-")
                if c.get("rows") and c.get("ms"):
                    ms = f"{c['ms']} ({c['rows']} rows; x{n / c['rows']:.0f} = {ms_full:.1f})"
                calls = ",".join(f"{k.split('$')[-1][:28]}:{v}" for k, v in (c.get("custom_calls") or {}).items())
                err_txt = "" if c.get("ok") else " ERR " + (c.get("error") or c.get("status", ""))[:120].replace("|", "/")
                lines.append(
                    f"| {arm}{err_txt} | {ms} | {ratio} | {c.get('temp_mib', '-')} | {c.get('peak_mib', '-')} "
                    f"| {err_ref} ({scale}) | {err_prod} | {c.get('finite_pad_rows', '-')} | {calls} |")
    text = "\n".join(lines)
    (out / "report.md").write_text(text + "\n")
    print(text)
    if (out / "avail.jsonl").exists():
        print("\n## availability")
        for line in open(out / "avail.jsonl"):
            a = json.loads(line)
            detail = (f"max-abs {a.get('max_abs_vs_f32', float('nan')):.3g} (ref scale {a.get('ref_scale', 0):.3g})"
                      if a.get("status") == "ok" else f"{a.get('error_type', '')}: {(a.get('error') or a.get('stderr_tail', ''))[:300]} {a.get('sub_errors', '')}")
            print(f"- {a['probe']}: {a.get('status')} -- {detail}".replace("\n", " "))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--out", required=True)
    c.add_argument("--family", required=True, choices=("att", "mul", "trans"))
    c.add_argument("--n", type=int, required=True)
    c.add_argument("--arm", required=True)
    c.add_argument("--reps", type=int, default=7)
    c.add_argument("--bias-dtype", default="float32")
    c.add_argument("--interpret", action="store_true")
    d = sub.add_parser("drive")
    d.add_argument("--out", required=True)
    d.add_argument("--family", default="att,mul,trans")
    d.add_argument("--sizes", default="1003,1354,2096,2100,3012,4100")
    d.add_argument("--arms", default="")
    d.add_argument("--reps", type=int, default=7)
    d.add_argument("--timeout", type=int, default=1500)
    d.add_argument("--bias-dtype", default="float32")
    d.add_argument("--smoke", action="store_true")
    p = sub.add_parser("probe")
    p.add_argument("--out", required=True)
    p.add_argument("--probe", required=True)
    a = sub.add_parser("avail")
    a.add_argument("--out", required=True)
    a.add_argument("--probes", default="")
    a.add_argument("--timeout", type=int, default=600)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"cell": run_cell, "drive": drive, "probe": run_probe, "avail": avail, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
