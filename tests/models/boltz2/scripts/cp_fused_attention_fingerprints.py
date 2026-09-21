"""Compiled-program fingerprints for every route `cp_fused_attention` touches.

Run it against two source trees with the same interpreter, devices and flags
and compare the output line by line: the serial diffusion programs and both
context-parallel routes with the option off must be identical across the two,
and so must the one fused tile a CPU can lower.

    JAX_PLATFORMS=cpu \
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    PYTHONPATH=<tree>/src python <this file>

The hash is over the HLO text with source metadata stripped, the way
`tests/models/protenix/scripts/atom_cp_invariance.py` does it: `metadata={...}`
and `stack_frame_id=N` carry file paths and line numbers, so a comment added
above a function would otherwise read as a changed program.
`temp_size_in_bytes` is printed beside it because two programs can hash alike
and allocate differently only if the hash is not over the whole program -- so
a drift in either is a drift.
"""

from __future__ import annotations

import hashlib
import re

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models._cp import context_parallel
from foldjax.models._cp_atom import pair_bias_attention_2d, place_atoms
from foldjax.models._cp_attention import tile_attention_tokamax
from foldjax.models.boltz2.models.diffusion.atom import (
    atom_transformer_forward,
    get_indexing_matrix,
    single_to_keys,
)
from foldjax.models.boltz2.models.diffusion.diffusion_transformer import (
    _attention_pair_bias_no_proj_z_forward,
)

HEADS = 2
DIM = COND = 8
W, HK = 32, 128
# 256 rather than 128 so the 1-D mesh's four shards each keep 64 atoms, which
# is two query windows: `single_to_keys_local` needs at least the halo radius
# of half-windows per shard.
BATCH, ATOMS = 1, 256
WINDOWS = ATOMS // W


def _fingerprint(text: str) -> str:
    text = re.sub(r",?\s*metadata=\{[^}]*\}", "", text)
    text = re.sub(r",?\s*stack_frame_id=\d+", "", text)
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def _report(name: str, compiled, *args) -> None:
    lowered = compiled.lower(*args)
    text = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    temp = lowered.compile().memory_analysis().temp_size_in_bytes
    print(f"{name} {_fingerprint(text)[:32]} temp={temp}")


def _params(rng):
    def weight(rows, cols, scale=0.2):
        return jnp.asarray(rng.normal(scale=scale, size=(rows, cols)), jnp.float32)

    def bvec(size, scale=0.1):
        return jnp.asarray(rng.normal(scale=scale, size=(size,)), jnp.float32)

    def norm(size, with_bias=True):
        out = {"scale": jnp.ones((size,), jnp.float32)}
        if with_bias:
            out["bias"] = jnp.zeros((size,), jnp.float32)
        return out

    def adaln(cond, act):
        return {
            "s_norm": norm(cond, with_bias=False),
            "s_scale": {"kernel": weight(cond, act), "bias": bvec(act)},
            "s_bias": {"kernel": weight(cond, act)},
        }

    hidden = 2 * DIM
    layer = {
        "adaln": adaln(COND, DIM),
        "pair_bias_attn": {
            "num_heads": HEADS,
            "proj_q": {"kernel": weight(DIM, DIM), "bias": bvec(DIM)},
            "proj_k": {"kernel": weight(DIM, DIM)},
            "proj_v": {"kernel": weight(DIM, DIM)},
            "proj_g": {"kernel": weight(DIM, DIM)},
            "proj_o": {"kernel": weight(DIM, DIM)},
        },
        "output_projection": {"kernel": weight(COND, DIM), "bias": bvec(DIM)},
        "transition": {
            "adaln": adaln(COND, DIM),
            "swish_gate": {"kernel": weight(DIM, 2 * hidden)},
            "a_to_b": {"kernel": weight(DIM, hidden)},
            "b_to_a": {"kernel": weight(hidden, DIM)},
            "output_projection": {"kernel": weight(COND, DIM), "bias": bvec(DIM)},
        },
    }
    return {"diffusion_transformer": {"layers": [layer]}}


def main() -> int:
    rng = np.random.default_rng(20260921)
    params = _params(rng)
    q = jnp.asarray(rng.normal(size=(BATCH, ATOMS, DIM)), jnp.float32)
    c = jnp.asarray(rng.normal(size=(BATCH, ATOMS, COND)), jnp.float32)
    bias = jnp.asarray(rng.normal(size=(BATCH, WINDOWS, W, HK, HEADS)), jnp.float32)
    mask = jnp.asarray(rng.random((BATCH, ATOMS)) > 0.1, jnp.float32)

    indexing = get_indexing_matrix(ATOMS // W, W, HK)

    def to_keys(x):
        keys = single_to_keys(x, indexing, W, HK)
        return jnp.reshape(keys, (x.shape[0], -1, x.shape[-1]))

    def atom(distributed: bool):
        def run(qv, cv, bv, mv):
            return atom_transformer_forward(
                params,
                q=qv,
                c=cv,
                bias=bv,
                to_keys=to_keys,
                mask=mv,
                attn_window_queries=W,
                attn_window_keys=HK,
                multiplicity=1,
                attention_backend="xla",
                atom_context_parallel=distributed,
            )

        return jax.jit(run)

    # 1. The serial atom-window transformer: the program the port ships off a
    #    mesh, which this change must not move at all.
    _report("serial_atom_xla", atom(False), q, c, bias, mask)

    # 2. The serial token attention, in both backends. The fused one is the
    #    released `diffusion_attention_backend`, and it is the call the atom
    #    site borrows under a mesh.
    tokens, channels = 16, DIM
    eye = jnp.eye(channels, dtype=jnp.float32)
    attention_params = {
        "proj_q": {"kernel": eye, "bias": jnp.zeros((channels,), jnp.float32)},
        "proj_g": {"kernel": jnp.zeros_like(eye)},
        "proj_k": {"kernel": eye},
        "proj_v": {"kernel": eye},
        "proj_o": {"kernel": eye},
    }
    single = jnp.asarray(rng.normal(size=(1, tokens, channels)), jnp.float32)
    pair = jnp.asarray(rng.normal(size=(1, tokens, tokens, HEADS)), jnp.float32)
    token_mask = jnp.asarray(rng.random((1, tokens)) > 0.2, jnp.float32)

    def token(backend: str):
        def run(sv, bv, mv):
            return _attention_pair_bias_no_proj_z_forward(
                attention_params,
                s=sv,
                bias=bv,
                mask=mv,
                k_in=sv,
                multiplicity=1,
                inf=1e6,
                attention_backend=backend,
            )

        return jax.jit(run)

    _report("serial_token_xla", token("xla"), single, pair, token_mask)
    _report("serial_token_tokamax", token("tokamax"), single, pair, token_mask)

    # 3. The one fused tile a CPU can lower. It is the ring's, not this
    #    option's, and it is here because the `logits_scale` seam is on it.
    rows, heads, queries, keys, channels_tile = 2, 2, 4, 4, 4
    tile = tuple(
        jnp.asarray(rng.normal(size=shape), jnp.float32)
        for shape in (
            (rows, heads, queries, channels_tile),
            (rows, heads, keys, channels_tile),
            (rows, heads, keys, channels_tile),
            (1, heads, queries, keys),
        )
    )
    tile_mask = jnp.where(
        jnp.asarray(rng.random((rows, 1, 1, keys)) > 0.25),
        jnp.asarray(0.0, jnp.float32),
        jnp.asarray(-jnp.inf, jnp.float32),
    )
    _report(
        "ring_tile_tokamax_xla",
        jax.jit(
            lambda *operands: tile_attention_tokamax(
                *operands, implementation="xla"
            )
        ),
        *tile,
        tile_mask,
    )

    if jax.device_count() < 4:
        print("mesh routes skipped: need four devices")
        return 0

    # 4. Both context-parallel routes with the option off. `cp_fused_attention`
    #    is never entered here, so these are the programs a distributed run
    #    compiles today.
    for layout in ("1d", "2d"):
        with context_parallel(4, layout=layout):
            qd = place_atoms(q, atom_axis=1)
            cd = place_atoms(c, atom_axis=1)
            md = place_atoms(mask, atom_axis=1)
            _report(f"cp_{layout}_atom_off", atom(True), qd, cd, bias, md)

    heads_2d, dim_2d, tokens_2d = 2, 3, 8
    q2 = jnp.asarray(rng.normal(size=(1, tokens_2d, heads_2d, dim_2d)), jnp.float32)
    k2 = jnp.asarray(rng.normal(size=(1, tokens_2d, heads_2d, dim_2d)), jnp.float32)
    v2 = jnp.asarray(rng.normal(size=(1, tokens_2d, heads_2d, dim_2d)), jnp.float32)
    b2 = jnp.asarray(
        rng.normal(size=(1, heads_2d, tokens_2d, tokens_2d)), jnp.float32
    )
    m2 = jnp.asarray(rng.random((1, tokens_2d)) > 0.2, jnp.float32)
    with context_parallel(4, layout="2d"):
        _report(
            "cp_2d_token_off",
            jax.jit(
                lambda *operands: pair_bias_attention_2d(
                    *operands, scale=float(dim_2d) ** -0.5
                )
            ),
            q2,
            k2,
            v2,
            b2,
            m2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
