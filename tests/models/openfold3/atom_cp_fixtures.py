"""Shapes and synthetic parameters for OpenFold3's distributed atom graph.

Two properties of this fixture are load-bearing, and both are the reason the
mesh arms below can tell a correct implementation from a plausible one.

**The atom axis is padded.** ``ATOM_MASK`` has 90 real atoms out of 96, so
``atom_blocks.block_indices`` exercises its *shift*: block 22 and block 23 both
read atoms 82..89 rather than a window centred on themselves, and with
``FOLDJAX_OF3_CP_REAL_ATOMS=60`` nine blocks read atoms 52..59. That shift is
derived from ``jnp.sum(atom_mask)``, so it is a traced value with no static
bound -- which is what makes the halo exchange Boltz-2 and Protenix use wrong
here and the ring gather right. A fixture whose atoms were all real would pass
against either.

**The token count and the block count differ.** 12 tokens against 24 query
blocks, so a per-device shape assertion naming 12 or 24 is unambiguous about
which axis it is reading. Both divide 4, a 2x2 grid and a 3x3 grid, which is
what the three mesh arms need.

No upstream import: the parameters are random arrays at the layouts the port's
own NamedTuples declare (PyTorch's ``[out, in]`` for every weight). Upstream
agreement is gated elsewhere (``test_torch_parity_atom_encoder`` and friends);
what these arms compare is a sharded program against a serial one, and for
that the numbers only have to be the same numbers on both sides.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

C_ATOM = 8
C_ATOM_PAIR = 6
C_TOKEN = 8
C_S = 8
C_Z = 4
C_ELEMENT = 12
C_CHARS = 5
C_HIDDEN = 4
ATOM_HEADS = 2
TOKEN_HEADS = 2
TRANSITION_N = 2
NO_BLOCKS = 2

N_QUERY = 4
N_KEY = 8
N_ATOM = 96
N_TOKEN = 12
N_BLOCKS = N_ATOM // N_QUERY
N_SAMPLE = 2
SIGMA_DATA = 16.0

#: Real atoms out of ``N_ATOM``. 90 leaves the last two blocks shifted; the
#: environment override exists so one arm can push the shift much further.
REAL_ATOMS = int(os.environ.get("FOLDJAX_OF3_CP_REAL_ATOMS", "90"))


class AtomCase(NamedTuple):
    """The batch features the diffusion encoder and decoder read."""

    batch: dict[str, jnp.ndarray]
    xl_noisy: jnp.ndarray
    t: jnp.ndarray
    si: jnp.ndarray
    si_trunk: jnp.ndarray
    zij: jnp.ndarray


def _rng() -> np.random.Generator:
    return np.random.default_rng(20260915)


def _atom_counts(real_atoms: int) -> np.ndarray:
    """Split ``real_atoms`` over ``N_TOKEN`` tokens with an uneven tail."""

    base, extra = divmod(real_atoms, N_TOKEN)
    counts = np.full(N_TOKEN, base, dtype=np.int64)
    counts[:extra] += 1
    return counts


def build_case(real_atoms: int | None = None) -> AtomCase:
    """Features whose atom axis is padded and whose token axis is not."""

    real = REAL_ATOMS if real_atoms is None else real_atoms
    rng = _rng()
    counts = _atom_counts(real)
    owners = np.repeat(np.arange(N_TOKEN, dtype=np.int32), counts)
    # Padded atoms keep the last token as their owner, which is what the
    # featurizer's `np.repeat` prefix leaves behind. Nothing reads the value --
    # the atom mask routes them to the aggregate's overflow bin and the block
    # pair mask zeroes their attention -- but both arms must see the same one.
    owners = np.concatenate(
        [owners, np.full(N_ATOM - real, N_TOKEN - 1, dtype=np.int32)]
    )
    atom_mask = np.concatenate(
        [np.ones(real, dtype=np.float32), np.zeros(N_ATOM - real, dtype=np.float32)]
    )

    def sample(*shape: int) -> jnp.ndarray:
        return jnp.asarray(
            np.broadcast_to(
                rng.standard_normal(shape).astype(np.float32),
                (N_SAMPLE, *shape),
            ).copy()
        )

    batch = {
        "ref_pos": sample(N_ATOM, 3),
        "ref_charge": sample(N_ATOM),
        "ref_mask": jnp.broadcast_to(jnp.asarray(atom_mask), (N_SAMPLE, N_ATOM)).copy(),
        "ref_element": sample(N_ATOM, C_ELEMENT),
        "ref_atom_name_chars": sample(N_ATOM, 4, C_CHARS),
        # Reference conformers in runs of eight atoms, so the `vlm` term keeps
        # some key slots and drops others inside the same block.
        "ref_space_uid": jnp.broadcast_to(
            jnp.asarray((np.arange(N_ATOM) // 8).astype(np.float32)),
            (N_SAMPLE, N_ATOM),
        ).copy(),
        "atom_mask": jnp.broadcast_to(
            jnp.asarray(atom_mask), (N_SAMPLE, N_ATOM)
        ).copy(),
        "atom_to_token_index": jnp.broadcast_to(
            jnp.asarray(owners.astype(np.float32)), (N_SAMPLE, N_ATOM)
        ).copy(),
        "token_mask": jnp.ones((N_SAMPLE, N_TOKEN), dtype=jnp.float32),
        "num_atoms_per_token": jnp.broadcast_to(
            jnp.asarray(counts.astype(np.float32)), (N_SAMPLE, N_TOKEN)
        ).copy(),
    }
    return AtomCase(
        batch=batch,
        xl_noisy=sample(N_ATOM, 3) * 10.0,
        t=jnp.asarray([4.0], dtype=jnp.float32),
        si=sample(N_TOKEN, C_S),
        si_trunk=sample(N_TOKEN, C_S),
        zij=sample(N_TOKEN, N_TOKEN, C_Z),
    )


# --- synthetic parameters --------------------------------------------------


def _linear(rng, out_features: int, in_features: int, *, bias: bool):
    from foldjax.models.openfold3.models.primitives import LinearParams

    scale = 1.0 / np.sqrt(in_features)
    return LinearParams(
        weight=jnp.asarray(
            (rng.standard_normal((out_features, in_features)) * scale).astype(
                np.float32
            )
        ),
        bias=(
            jnp.asarray(rng.standard_normal(out_features).astype(np.float32))
            if bias
            else None
        ),
    )


def _layer_norm(rng, channels: int, *, offset: bool = True):
    from foldjax.models.openfold3.models.primitives import LayerNormParams

    return LayerNormParams(
        weight=jnp.asarray(
            (1.0 + 0.1 * rng.standard_normal(channels)).astype(np.float32)
        ),
        bias=(
            jnp.asarray((0.1 * rng.standard_normal(channels)).astype(np.float32))
            if offset
            else None
        ),
    )


def _adaln(rng, c_a: int, c_s: int):
    from foldjax.models.openfold3.models.primitives import (
        AdaLNParams,
        LayerNormParams,
    )

    return AdaLNParams(
        layer_norm_a=LayerNormParams(),
        layer_norm_s=_layer_norm(rng, c_s, offset=False),
        linear_g=_linear(rng, c_a, c_s, bias=True),
        linear_s=_linear(rng, c_a, c_s, bias=False),
    )


def _attention(rng, channels: int, *, heads: int, gating: bool):
    from foldjax.models.openfold3.models.attention import AttentionParams

    hidden = heads * C_HIDDEN
    return AttentionParams(
        # This layer's inner attention is the one place upstream gives
        # `linear_q` a bias (`att_pair_bias_mha_init`).
        linear_q=_linear(rng, hidden, channels, bias=True),
        linear_k=_linear(rng, hidden, channels, bias=False),
        linear_v=_linear(rng, hidden, channels, bias=False),
        linear_o=_linear(rng, channels, hidden, bias=False),
        linear_g=(_linear(rng, hidden, channels, bias=False) if gating else None),
    )


def _transition(rng, c_a: int, c_s: int):
    from foldjax.models.openfold3.models.primitives import (
        ConditionedTransitionBlockParams,
        SwiGLUParams,
    )

    hidden = TRANSITION_N * c_a
    return ConditionedTransitionBlockParams(
        layer_norm=_adaln(rng, c_a, c_s),
        swiglu=SwiGLUParams(
            linear_a=_linear(rng, hidden, c_a, bias=False),
            linear_b=_linear(rng, hidden, c_a, bias=False),
        ),
        linear_g=_linear(rng, c_a, c_s, bias=True),
        linear_out=_linear(rng, c_a, hidden, bias=False),
    )


def _atom_transformer(rng):
    from foldjax.models.openfold3.models.attention_pair_bias import (
        CrossAttentionPairBiasParams,
    )
    from foldjax.models.openfold3.models.diffusion_transformer import (
        AtomTransformerBlockParams,
        AtomTransformerParams,
    )

    blocks = tuple(
        AtomTransformerBlockParams(
            attention_pair_bias=CrossAttentionPairBiasParams(
                layer_norm_a_q=_adaln(rng, C_ATOM, C_ATOM),
                layer_norm_a_k=_adaln(rng, C_ATOM, C_ATOM),
                linear_z=_linear(rng, ATOM_HEADS, C_ATOM_PAIR, bias=False),
                mha=_attention(rng, C_ATOM, heads=ATOM_HEADS, gating=True),
                linear_ada_out=_linear(rng, C_ATOM, C_ATOM, bias=True),
            ),
            conditioned_transition=_transition(rng, C_ATOM, C_ATOM),
        )
        for _ in range(NO_BLOCKS)
    )
    return AtomTransformerParams(
        blocks=blocks, layer_norm_z=_layer_norm(rng, C_ATOM_PAIR)
    )


def build_params():
    """A ``DenoiserParams`` whose every leaf is a fixed random draw."""

    from foldjax.models.openfold3.models.atom_features import (
        AtomAttentionDecoderParams,
        AtomAttentionEncoderParams,
        AtomPairConditioningParams,
        NoisyPositionEmbedderParams,
        RefAtomFeatureEmbedderParams,
    )
    from foldjax.models.openfold3.models.attention_pair_bias import (
        AdaAttentionPairBiasParams,
    )
    from foldjax.models.openfold3.models.denoiser import DenoiserParams
    from foldjax.models.openfold3.models.diffusion_transformer import (
        DiffusionTransformerBlockParams,
        DiffusionTransformerParams,
    )

    rng = _rng()
    encoder = AtomAttentionEncoderParams(
        ref_atom_feature_embedder=RefAtomFeatureEmbedderParams(
            linear_ref_pos=_linear(rng, C_ATOM, 3, bias=False),
            linear_ref_charge=_linear(rng, C_ATOM, 1, bias=False),
            linear_ref_mask=_linear(rng, C_ATOM, 1, bias=False),
            linear_ref_element=_linear(rng, C_ATOM, C_ELEMENT, bias=False),
            linear_ref_atom_chars=_linear(rng, C_ATOM, 4 * C_CHARS, bias=False),
            linear_ref_offset=_linear(rng, C_ATOM_PAIR, 3, bias=False),
            linear_inv_sq_dists=_linear(rng, C_ATOM_PAIR, 1, bias=False),
            linear_valid_mask=_linear(rng, C_ATOM_PAIR, 1, bias=False),
        ),
        pair_conditioning=AtomPairConditioningParams(
            linear_l=_linear(rng, C_ATOM_PAIR, C_ATOM, bias=False),
            linear_m=_linear(rng, C_ATOM_PAIR, C_ATOM, bias=False),
            pair_mlp_1=_linear(rng, C_ATOM_PAIR, C_ATOM_PAIR, bias=False),
            pair_mlp_2=_linear(rng, C_ATOM_PAIR, C_ATOM_PAIR, bias=False),
            pair_mlp_3=_linear(rng, C_ATOM_PAIR, C_ATOM_PAIR, bias=False),
        ),
        atom_transformer=_atom_transformer(rng),
        linear_q=_linear(rng, C_TOKEN, C_ATOM, bias=False),
        noisy_position_embedder=NoisyPositionEmbedderParams(
            layer_norm_s=_layer_norm(rng, C_S, offset=False),
            linear_s=_linear(rng, C_ATOM, C_S, bias=False),
            layer_norm_z=_layer_norm(rng, C_Z, offset=False),
            linear_z=_linear(rng, C_ATOM_PAIR, C_Z, bias=False),
            linear_r=_linear(rng, C_ATOM, 3, bias=False),
        ),
    )
    decoder = AtomAttentionDecoderParams(
        linear_q_in=_linear(rng, C_ATOM, C_TOKEN, bias=False),
        atom_transformer=_atom_transformer(rng),
        layer_norm=_layer_norm(rng, C_ATOM, offset=False),
        linear_q_out=_linear(rng, 3, C_ATOM, bias=False),
    )
    token_blocks = tuple(
        DiffusionTransformerBlockParams(
            attention_pair_bias=AdaAttentionPairBiasParams(
                layer_norm_a=_adaln(rng, C_TOKEN, C_S),
                linear_ada_out=_linear(rng, C_TOKEN, C_S, bias=True),
                linear_z=_linear(rng, TOKEN_HEADS, C_Z, bias=False),
                mha=_attention(rng, C_TOKEN, heads=TOKEN_HEADS, gating=True),
            ),
            conditioned_transition=_transition(rng, C_TOKEN, C_S),
        )
        for _ in range(NO_BLOCKS)
    )
    return DenoiserParams(
        atom_attn_enc=encoder,
        layer_norm_s=_layer_norm(rng, C_S),
        linear_s=_linear(rng, C_TOKEN, C_S, bias=False),
        diffusion_transformer=DiffusionTransformerParams(
            blocks=token_blocks, layer_norm_z=_layer_norm(rng, C_Z)
        ),
        layer_norm_a=_layer_norm(rng, C_TOKEN),
        atom_attn_dec=decoder,
    )
