"""OpenDDE's tiny diffusion module over the shared atom-graph fixtures.

The atom graph itself is Protenix': OpenDDE's diffusion module calls straight
through ``foldjax.models.protenix.models.diffusion.diffusion`` and reuses the
same ``AtomAttentionEncoderParams`` / ``AtomAttentionDecoderParams`` leaves. So
the weights, the atom<->token map, the masks and the window geometry come from
``tests.models.protenix.atom_cp_fixtures`` unchanged -- building a second
"similar" model here is the failure mode that file was written to avoid, and a
divergence between the two ports' fixtures would be invisible in the residuals.

What is OpenDDE's own is the conditioning: its
``DiffusionConditioningParams`` has two fields Protenix' does not
(``layernorm_z_trunk`` / ``linear_z_trunk``, which compress the trunk pair
before the relative-position concatenation), and its denoiser conditions the
single stream itself and hands the result to the shared network as
``conditioned_single_s``. That is the path this file builds, because it is the
path OpenDDE's model runs and the one the context-parallel keyword now travels
along.

``N_TOKEN`` is a *structural*-token count here, deliberately: OpenDDE diffuses
over its expanded structural tokens, so the axis that has to divide the mesh is
that one, and the padding axis a caller pins is ``structural_tokens``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

# `_Draw` and `_token_block` are imported private on purpose: they are the
# seeded generator and the global-attention block shape that make "the same
# model" one object rather than two similar ones. Re-implementing either here
# would reintroduce the divergence the shared fixture prevents.
from tests.models.protenix.atom_cp_fixtures import (
    C_NOISE,
    C_PAIR,
    C_SINGLE,
    C_SINGLE_INPUTS,
    C_TOKEN,
    C_TRUNK,
    N_ATOM,
    N_SAMPLE,
    N_TOKEN,
    AtomGraphCase,
    _Draw,
    _token_block,
    build_case,
)

#: Width of the raw relative-position feature OpenDDE's ``relpe`` reads. Unused
#: whenever the pair conditioning cache is supplied, which every case here does
#: -- and which the released OpenDDE graph does too, from
#: ``diffusion_conditioning_prepare_cache`` on the structural branch.
N_RELP_FEATURES = 2


def build_module_case(
    *,
    seed: int = 20260917,
    n_blocks: int = 2,
) -> tuple[Any, AtomGraphCase, dict[str, jnp.ndarray]]:
    """A whole tiny OpenDDE ``DiffusionModuleParams`` plus its static features.

    Enough to run OpenDDE's ``diffusion_module_forward`` and its sampler. The
    pair conditioning cache is supplied, so ``relpe``, ``layernorm_z_trunk``,
    ``linear_z_trunk``, ``layernorm_z``, ``linear_z`` and both pair transitions
    are never read and exist only to complete the NamedTuple.

    ``extra_attn_bias`` is drawn rather than zeroed. It is the one operand
    OpenDDE always supplies and Protenix never does -- a replicated
    ``[N_token, N_token]`` bias added to token-attention logits whose queries
    the distributed atom graph now delivers on CP rows -- so a fixture that
    zeroed it would certify the absence of a difference it cannot see.
    """

    from foldjax.models.opendde.models.diffusion_conditioning import (
        DiffusionConditioningParams,
    )
    from foldjax.models.protenix.models.diffusion.diffusion import (
        DiffusionModuleParams,
    )
    from foldjax.models.protenix.models.diffusion.transformer import (
        DiffusionTransformerStackParams,
    )
    from foldjax.models.protenix.models.primitives.primitives import (
        TransitionParams,
    )
    from foldjax.models.protenix.models.trunk_blocks.embedders import (
        FourierParams,
        RelativePositionParams,
    )

    case = build_case(seed=seed, n_blocks=n_blocks)
    draw = _Draw(seed + 23)

    def transition(width: int) -> TransitionParams:
        return TransitionParams(
            layer_norm=draw.norm(width),
            linear_a=draw.linear(2 * width, width, bias=False),
            linear_b=draw.linear(2 * width, width, bias=False),
            linear_out=draw.linear(width, 2 * width, bias=False),
        )

    conditioning = DiffusionConditioningParams(
        relpe=RelativePositionParams(
            linear_no_bias=draw.linear(C_PAIR, N_RELP_FEATURES, bias=False)
        ),
        layernorm_z_trunk=draw.norm(C_PAIR),
        linear_z_trunk=draw.linear(C_PAIR, C_PAIR, bias=False),
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
        "extra_attn_bias": draw.array(N_TOKEN, N_TOKEN, scale=1.5),
    }
    return params, case, features


__all__ = ["N_RELP_FEATURES", "build_module_case"]
