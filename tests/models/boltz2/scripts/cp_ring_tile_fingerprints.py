"""Compiled-program fingerprints for Boltz-2's two 2-D ring entry points.

This port reaches the gather-free Fold-CP ring through two different modules --
the trunk's CP dispatcher (`triangle/triangle_attention_cp.py`, which the
Pairformer and the pair-only Pairformer import) and the serial module's own
context-parallel branch (`triangle/triangle_attention.py`, which the MSA
stack's private `pairformer_no_seq_layer_forward` imports). Both are here,
together with the MSA stack that drives the second one, because
`triangle_attention_ring_kernel` has to be read the same way at both and its
default has to compile the program a distributed run compiled before.

Run it against two source trees with the same interpreter, devices and flags
and compare the output line by line: with the option at its default `xla`
every line must be identical across the two.

    JAX_PLATFORMS=cpu \
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    PYTHONPATH=<tree>/src python <this file>

The hash is over the HLO text with source metadata stripped, the way
`cp_fused_attention_fingerprints.py` does it: `metadata={...}` and
`stack_frame_id=N` carry file paths and line numbers, so a comment added above
a function would otherwise read as a changed program. `temp_size_in_bytes` is
printed beside it because two programs can hash alike and allocate differently
only if the hash is not over the whole program -- so a drift in either is a
drift.
"""

from __future__ import annotations

import hashlib
import re

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models._cp import context_parallel
from foldjax.models.boltz2.models.triangle import (
    triangle_attention as serial_triangle,
)
from foldjax.models.boltz2.models.triangle import (
    triangle_attention_cp as trunk_triangle,
)
from foldjax.models.boltz2.models.trunk_blocks import msa as msa_module

DEVICES = 4
# Divisible by the 2x2 grid's side on both pair axes and on the alignment
# depth, so no arm is measuring a padded remainder.
TOKENS, DEPTH = 12, 36
CS, CM, CZ, HEADS, NUM_TOKENS, LAYERS = 5, 10, 12, 2, 7, 2


def _fingerprint(text: str) -> str:
    text = re.sub(r",?\s*metadata=\{[^}]*\}", "", text)
    text = re.sub(r",?\s*stack_frame_id=\d+", "", text)
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def _report(name: str, jitted, *args) -> None:
    lowered = jitted.lower(*args)
    text = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    temp = lowered.compile().memory_analysis().temp_size_in_bytes
    print(f"{name} {_fingerprint(text)[:32]} temp={temp}")


def _arrays(rng):
    def arr(*shape, scale=0.5):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)

    def norm(size):
        return {"scale": arr(size) * 0.1 + 1.0, "bias": arr(size) * 0.1}

    def weight(fan_in, fan_out):
        return jnp.asarray(
            rng.normal(size=(fan_in, fan_out), scale=1.0 / np.sqrt(fan_in)),
            dtype=jnp.float32,
        )

    return arr, norm, weight


def main() -> int:
    rng = np.random.default_rng(20260922)
    arr, norm, weight = _arrays(rng)

    def tri_att():
        return {
            "layer_norm": norm(CZ),
            "linear": {"kernel": weight(CZ, HEADS)},
            "mha": {
                name: {"kernel": weight(CZ, CZ)}
                for name in (
                    "linear_q",
                    "linear_k",
                    "linear_v",
                    "linear_g",
                    "linear_o",
                )
            },
        }

    def transition(c):
        return {
            "norm": norm(c),
            "fc1": {"kernel": weight(c, 4 * c)},
            "fc2": {"kernel": weight(c, 4 * c)},
            "fc3": {"kernel": weight(4 * c, c)},
        }

    def tri_mult():
        return {
            "norm_in": norm(CZ),
            "norm_out": norm(CZ),
            "g_in": {"kernel": weight(CZ, 2 * CZ)},
            "p_in": {"kernel": weight(CZ, 2 * CZ)},
            "p_out": {"kernel": weight(CZ, CZ)},
            "g_out": {"kernel": weight(CZ, CZ)},
        }

    def msa_layer():
        return {
            "pair_weighted_averaging": {
                "norm_m": norm(CM),
                "norm_z": norm(CZ),
                "proj_m": {"kernel": weight(CM, HEADS * 3)},
                "proj_z": {"kernel": weight(CZ, HEADS)},
                "proj_g": {"kernel": weight(CM, HEADS * 3)},
                "proj_o": {"kernel": weight(HEADS * 3, CM)},
            },
            "msa_transition": transition(CM),
            "outer_product_mean": {
                "norm": norm(CM),
                "proj_a": {"kernel": weight(CM, 8)},
                "proj_b": {"kernel": weight(CM, 8)},
                "proj_o": {"kernel": weight(8 * 8, CZ), "bias": arr(CZ)},
            },
            "pairformer_layer": {
                "tri_mul_out": tri_mult(),
                "tri_mul_in": tri_mult(),
                "tri_att_start": tri_att(),
                "tri_att_end": tri_att(),
                "transition_z": transition(CZ),
            },
        }

    attention = tri_att()
    z = arr(1, TOKENS, TOKENS, CZ)
    keep = rng.random(TOKENS) > 0.2
    keep[0] = True
    pair_mask = jnp.asarray((keep[:, None] & keep[None, :])[None]).astype(jnp.float32)

    module_params = {
        "msa_proj": {"kernel": weight(NUM_TOKENS + 3, CM)},
        "s_proj": {"kernel": weight(CS, CM)},
        "layers": [msa_layer() for _ in range(LAYERS)],
    }
    emb = arr(1, TOKENS, CS)
    feats = {
        "msa": jnp.asarray(
            rng.integers(0, NUM_TOKENS, size=(1, DEPTH, TOKENS)), dtype=jnp.int32
        ),
        "has_deletion": jnp.asarray(
            (rng.random((1, DEPTH, TOKENS)) > 0.7).astype(np.float32)
        ),
        "deletion_value": jnp.asarray(
            rng.random((1, DEPTH, TOKENS)), dtype=jnp.float32
        ),
        "msa_paired": jnp.asarray(
            (rng.random((1, DEPTH, TOKENS)) > 0.5).astype(np.float32)
        ),
        "msa_mask": jnp.asarray(
            ((rng.random((DEPTH, TOKENS)) > 0.15) & keep[None, :])[None].astype(
                np.float32
            )
        ),
        "token_pad_mask": jnp.asarray(keep[None].astype(np.float32)),
    }

    assert jax.device_count() == DEVICES, jax.devices()
    with context_parallel(DEVICES, layout="2d"):
        # The two ring entries, at both triangle directions. A fresh closure
        # per arm: `jax.jit` keys its cache on the callable, so a reused one
        # would replay the first arm's program.
        for name, entry in (
            ("msa_ring", serial_triangle.triangle_attention_forward),
            ("trunk_ring", trunk_triangle.triangle_attention_forward),
        ):
            for direction in (True, False):
                def one(params, x, mask, entry=entry, direction=direction):
                    return entry(params, x, mask, starting=direction)

                _report(
                    f"{name}_{'start' if direction else 'end'}",
                    jax.jit(one),
                    attention,
                    z,
                    pair_mask,
                )

        # The stack that drives the second entry, in both of its layer
        # spellings: the scan traces one ring per direction for every layer,
        # the unrolled loop traces one per layer and direction.
        for use_scan in (True, False):
            def stack(params, pair, embedding, features, use_scan=use_scan):
                return msa_module.msa_module_forward(
                    params,
                    pair,
                    embedding,
                    features,
                    num_tokens=NUM_TOKENS,
                    use_scan=use_scan,
                    chunk_size=3,
                    pair_averaging_chunk=2,
                )

            _report(
                f"msa_module_scan={use_scan}",
                jax.jit(stack),
                module_params,
                z,
                emb,
                feats,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
