"""Tiny but non-degenerate Protenix diffusion atom-graph parameters.

The context-parallel gates run in subprocesses (a forced host device count has
to be set before JAX initialises), and both the serial reference and every
mesh variant need exactly the same weights.  Building them here rather than
inside each probe string is what keeps "the same model" from being three
similar ones, and every array is drawn from one seeded generator so a variant
that silently rebuilt its parameters would disagree everywhere rather than
subtly.

Deliberately scaled windows: ``n_queries=4`` / ``n_keys=8`` has the same
half-window geometry as the released ``32``/``128`` -- key padding of
``(n_keys - n_queries) // 2`` on each side, an even number of half windows per
key window -- at a size a four- or nine-device CPU mesh can actually split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.models.diffusion.atom import (
    AtomAttentionDecoderParams,
    AtomAttentionEncoderCacheParams,
    AtomAttentionEncoderParams,
    AtomPairMlpParams,
)
from foldjax.models.protenix.models.diffusion.transformer import (
    ConditionedTransitionParams,
    DiffusionTransformerBlockParams,
    DiffusionTransformerStackParams,
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

#: One shape that a 2x2 and a 3x3 mesh can both split: 96 atoms is a multiple
#: of ``n_queries * rows`` for rows in {1, 2, 3}, 24 atom windows divide both
#: row counts, and 24 tokens divide both the rows and the columns.
N_TOKEN = 24
ATOMS_PER_TOKEN = 4
N_ATOM = N_TOKEN * ATOMS_PER_TOKEN
N_QUERIES = 4
N_KEYS = 8
N_WINDOWS = N_ATOM // N_QUERIES
N_HEADS = 2
C_ATOM = 8
C_ATOMPAIR = 6
C_TOKEN = 8
C_TRUNK = 8
C_PAIR = 6
N_SAMPLE = 2
#: Width of the raw relative-position feature the pair conditioning reads.
#: Unused whenever the pair cache is supplied, which every case here does.
N_PAIR_FEATURES = 2


@dataclass(frozen=True)
class AtomGraphCase:
    """Everything one denoiser-step comparison needs, weights included."""

    encoder: AtomAttentionEncoderParams
    decoder: AtomAttentionDecoderParams
    atom_to_token_idx: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_charge: jnp.ndarray
    ref_mask: jnp.ndarray
    ref_element: jnp.ndarray
    ref_atom_name_chars: jnp.ndarray
    d_lm: jnp.ndarray
    v_lm: jnp.ndarray
    pad_info: dict[str, jnp.ndarray]
    r_l: jnp.ndarray
    s_trunk: jnp.ndarray
    z_pair: jnp.ndarray
    a_token: jnp.ndarray
    atom_mask: jnp.ndarray


class _Draw:
    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def array(self, *shape: int, scale: float = 0.35) -> jnp.ndarray:
        return jnp.asarray(self._rng.normal(size=shape, scale=scale), dtype=jnp.float32)

    def linear(self, out_width: int, in_width: int, *, bias: bool) -> LinearParams:
        return LinearParams(
            weight=self.array(out_width, in_width),
            bias=self.array(out_width) if bias else None,
        )

    def norm(self, width: int) -> LayerNormParams:
        return LayerNormParams(
            weight=self.array(width, scale=0.1) + 1.0,
            bias=self.array(width, scale=0.1),
        )

    def adaln(self, width: int, conditioning: int) -> AdaptiveLayerNormParams:
        return AdaptiveLayerNormParams(
            layernorm_a=LayerNormParams(weight=None, bias=None),
            layernorm_s=self.norm(conditioning),
            linear_s=self.linear(width, conditioning, bias=True),
            linear_no_bias_s=self.linear(width, conditioning, bias=False),
        )

    def block(self, width: int, pair_width: int) -> DiffusionTransformerBlockParams:
        adaln = self.adaln(width, width)
        return DiffusionTransformerBlockParams(
            attention_pair_bias=AttentionPairBiasParams(
                layernorm_a=adaln,
                layernorm_kv=self.adaln(width, width),
                attention=AttentionParams(
                    linear_q=self.linear(width, width, bias=True),
                    linear_k=self.linear(width, width, bias=False),
                    linear_v=self.linear(width, width, bias=False),
                    linear_o=self.linear(width, width, bias=False),
                    linear_g=self.linear(width, width, bias=False),
                ),
                layernorm_z=self.norm(pair_width),
                linear_z=self.linear(N_HEADS, pair_width, bias=False),
                linear_a_last=self.linear(width, width, bias=True),
                has_s=True,
                cross_attention_mode=True,
            ),
            conditioned_transition=ConditionedTransitionParams(
                adaln=self.adaln(width, width),
                linear_a1=self.linear(2 * width, width, bias=False),
                linear_a2=self.linear(2 * width, width, bias=False),
                linear_b=self.linear(width, 2 * width, bias=False),
                linear_s=self.linear(width, width, bias=True),
            ),
        )


def build_case(*, seed: int = 20260915, n_blocks: int = 2) -> AtomGraphCase:
    """One reproducible diffusion atom graph, weights and activations."""

    draw = _Draw(seed)
    encoder = AtomAttentionEncoderParams(
        cache=AtomAttentionEncoderCacheParams(
            linear_ref_pos=draw.linear(C_ATOM, 3, bias=False),
            linear_ref_charge=draw.linear(C_ATOM, 1, bias=False),
            linear_f=draw.linear(C_ATOM, 385, bias=False),
            linear_d=draw.linear(C_ATOMPAIR, 3, bias=False),
            linear_invd=draw.linear(C_ATOMPAIR, 1, bias=False),
            linear_v=draw.linear(C_ATOMPAIR, 1, bias=False),
        ),
        linear_cl=draw.linear(C_ATOMPAIR, C_ATOM, bias=False),
        linear_cm=draw.linear(C_ATOMPAIR, C_ATOM, bias=False),
        small_mlp=AtomPairMlpParams(
            linear_1=draw.linear(C_ATOMPAIR, C_ATOMPAIR, bias=False),
            linear_2=draw.linear(C_ATOMPAIR, C_ATOMPAIR, bias=False),
            linear_3=draw.linear(C_ATOMPAIR, C_ATOMPAIR, bias=False),
        ),
        atom_transformer=DiffusionTransformerStackParams(
            blocks=tuple(draw.block(C_ATOM, C_ATOMPAIR) for _ in range(n_blocks))
        ),
        linear_q=draw.linear(C_TOKEN, C_ATOM, bias=False),
        layernorm_s=draw.norm(C_TRUNK),
        linear_s=draw.linear(C_ATOM, C_TRUNK, bias=False),
        layernorm_z=draw.norm(C_PAIR),
        linear_z=draw.linear(C_ATOMPAIR, C_PAIR, bias=False),
        linear_r=draw.linear(C_ATOM, 3, bias=False),
    )
    decoder = AtomAttentionDecoderParams(
        linear_a=draw.linear(C_ATOM, C_TOKEN, bias=False),
        layernorm_q=draw.norm(C_ATOM),
        linear_out=draw.linear(3, C_ATOM, bias=False),
        atom_transformer=DiffusionTransformerStackParams(
            blocks=tuple(draw.block(C_ATOM, C_ATOMPAIR) for _ in range(n_blocks))
        ),
    )

    rng = np.random.default_rng(seed + 1)
    # Not `repeat(arange)`: a uniform map would hide a routing error that
    # crosses a shard boundary, and repeated indices are what makes the
    # scatter-mean accumulate rather than assign.
    assignment = np.sort(rng.integers(0, N_TOKEN, size=N_ATOM - N_TOKEN))
    assignment = np.concatenate([np.arange(N_TOKEN), assignment])
    assignment.sort()
    atom_to_token_idx = jnp.asarray(assignment, dtype=jnp.int32)

    mask = np.ones(N_ATOM, dtype=bool)
    # A padded tail, so the masks are exercised rather than merely present.
    mask[-5:] = False
    atom_mask = jnp.asarray(mask)

    pad_left = (N_KEYS - N_QUERIES) // 2
    q_abs = np.arange(N_WINDOWS * N_QUERIES).reshape(N_WINDOWS, N_QUERIES)
    k_abs = (
        np.arange(N_KEYS)[None, :]
        + np.arange(N_WINDOWS)[:, None] * N_QUERIES
        - pad_left
    )
    mask_trunked = (
        (q_abs[..., None] < N_ATOM)
        & (k_abs[:, None, :] >= 0)
        & (k_abs[:, None, :] < N_ATOM)
    )

    return AtomGraphCase(
        encoder=encoder,
        decoder=decoder,
        atom_to_token_idx=atom_to_token_idx,
        ref_pos=draw.array(N_ATOM, 3, scale=4.0),
        ref_charge=draw.array(N_ATOM, scale=0.5),
        ref_mask=jnp.asarray(mask, dtype=jnp.float32),
        ref_element=jnp.asarray(
            np.eye(128, dtype=np.float32)[
                np.random.default_rng(seed + 2).integers(0, 128, size=N_ATOM)
            ]
        ),
        ref_atom_name_chars=jnp.asarray(
            np.eye(64, dtype=np.float32)[
                np.random.default_rng(seed + 3).integers(0, 64, size=(N_ATOM, 4))
            ]
        ),
        d_lm=draw.array(N_WINDOWS, N_QUERIES, N_KEYS, 3, scale=2.0),
        v_lm=jnp.asarray(mask_trunked[..., None], dtype=jnp.float32),
        pad_info={"mask_trunked": jnp.asarray(mask_trunked)},
        r_l=draw.array(N_SAMPLE, N_ATOM, 3, scale=3.0),
        s_trunk=draw.array(1, N_TOKEN, C_TRUNK),
        z_pair=draw.array(1, N_TOKEN, N_TOKEN, C_PAIR),
        a_token=draw.array(N_SAMPLE, N_TOKEN, C_TOKEN),
        atom_mask=atom_mask,
    )


#: Widths of the token stage, kept beside the atom widths so one place says
#: what the tiny model is.
C_SINGLE = 8
C_SINGLE_INPUTS = 4
C_NOISE = 4
TOKEN_HEADS = 2
SIGMA_DATA = 16.0


def build_module_case(
    *,
    seed: int = 20260916,
    n_blocks: int = 2,
) -> tuple[Any, AtomGraphCase, dict[str, jnp.ndarray]]:
    """A whole tiny ``DiffusionModuleParams`` plus its static features.

    Enough of the module to run ``diffusion_module_forward`` and the sampler:
    the pair conditioning cache is supplied, so the relative-position and pair
    transition parameters are never read and are present only to complete the
    NamedTuple.
    """

    from foldjax.models.protenix.models.diffusion.diffusion import (
        DiffusionConditioningParams,
        DiffusionModuleParams,
    )
    from foldjax.models.protenix.models.primitives.primitives import (
        TransitionParams,
    )
    from foldjax.models.protenix.models.trunk_blocks.embedders import (
        FourierParams,
        RelativePositionParams,
    )

    case = build_case(seed=seed, n_blocks=n_blocks)
    draw = _Draw(seed + 17)

    def transition(width: int) -> TransitionParams:
        return TransitionParams(
            layer_norm=draw.norm(width),
            linear_a=draw.linear(2 * width, width, bias=False),
            linear_b=draw.linear(2 * width, width, bias=False),
            linear_out=draw.linear(width, 2 * width, bias=False),
        )

    conditioning = DiffusionConditioningParams(
        relpe=RelativePositionParams(linear_no_bias=draw.linear(C_PAIR, 2, bias=False)),
        layernorm_z=draw.norm(2 * C_PAIR),
        linear_z=draw.linear(C_PAIR, 2 * C_PAIR, bias=False),
        transition_z1=transition(C_PAIR),
        transition_z2=transition(C_PAIR),
        layernorm_s=draw.norm(C_TRUNK + C_SINGLE_INPUTS),
        linear_s=draw.linear(C_SINGLE, C_TRUNK + C_SINGLE_INPUTS, bias=False),
        fourier=FourierParams(
            w=draw.array(C_NOISE, scale=0.3),
            b=draw.array(C_NOISE, scale=0.3),
        ),
        layernorm_n=draw.norm(C_NOISE),
        linear_n=draw.linear(C_SINGLE, C_NOISE, bias=False),
        transition_s1=transition(C_SINGLE),
        transition_s2=transition(C_SINGLE),
    )
    params = DiffusionModuleParams(
        conditioning=conditioning,
        atom_encoder=case.encoder,
        layernorm_s=draw.norm(C_SINGLE),
        linear_s=draw.linear(C_TOKEN, C_SINGLE, bias=False),
        diffusion_transformer=DiffusionTransformerStackParams(
            blocks=tuple(
                _token_block(draw, C_TOKEN, C_SINGLE, C_PAIR) for _ in range(n_blocks)
            )
        ),
        layernorm_a=draw.norm(C_TOKEN),
        atom_decoder=case.decoder,
    )
    features = {
        "s_inputs": draw.array(N_TOKEN, C_SINGLE_INPUTS),
        "s_trunk": draw.array(N_TOKEN, C_TRUNK),
        "pair_z": draw.array(N_TOKEN, N_TOKEN, C_PAIR),
        "x_noisy": draw.array(N_SAMPLE, N_ATOM, 3, scale=8.0),
        "t_hat": jnp.asarray([12.0, 3.0], dtype=jnp.float32),
    }
    return params, case, features


def _token_block(
    draw: _Draw,
    width: int,
    conditioning: int,
    pair_width: int,
) -> DiffusionTransformerBlockParams:
    """The global token-attention block: adaptive, not cross-attention."""

    return DiffusionTransformerBlockParams(
        attention_pair_bias=AttentionPairBiasParams(
            layernorm_a=draw.adaln(width, conditioning),
            layernorm_kv=None,
            attention=AttentionParams(
                linear_q=draw.linear(width, width, bias=True),
                linear_k=draw.linear(width, width, bias=False),
                linear_v=draw.linear(width, width, bias=False),
                linear_o=draw.linear(width, width, bias=False),
                linear_g=draw.linear(width, width, bias=False),
            ),
            layernorm_z=draw.norm(pair_width),
            linear_z=draw.linear(TOKEN_HEADS, pair_width, bias=False),
            linear_a_last=draw.linear(width, conditioning, bias=True),
            has_s=True,
            cross_attention_mode=False,
        ),
        conditioned_transition=ConditionedTransitionParams(
            adaln=draw.adaln(width, conditioning),
            linear_a1=draw.linear(2 * width, width, bias=False),
            linear_a2=draw.linear(2 * width, width, bias=False),
            linear_b=draw.linear(width, 2 * width, bias=False),
            linear_s=draw.linear(width, conditioning, bias=True),
        ),
    )


__all__ = [
    "ATOMS_PER_TOKEN",
    "AtomGraphCase",
    "C_ATOM",
    "C_ATOMPAIR",
    "C_PAIR",
    "C_TOKEN",
    "C_TRUNK",
    "N_ATOM",
    "N_HEADS",
    "N_KEYS",
    "N_PAIR_FEATURES",
    "N_QUERIES",
    "N_SAMPLE",
    "N_TOKEN",
    "N_WINDOWS",
    "SIGMA_DATA",
    "TOKEN_HEADS",
    "build_case",
    "build_module_case",
]
