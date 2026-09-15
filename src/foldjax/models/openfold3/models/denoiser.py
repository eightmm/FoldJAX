"""The diffusion denoiser body (AF3 Algorithm 20, lines 2-9).

Composes the pieces that are already gated individually: EDM input scaling, the
atom attention encoder, a token-level diffusion transformer, the atom attention
decoder, and EDM output blending.

Two ordering details matter and are gated:

* The trunk single representation enters **twice** — once inside the encoder as
  ``si_trunk``, and again as an additive ``linear_s(layer_norm_s(si))`` on the
  token representation afterwards. They are different tensors upstream
  (``si_trunk`` vs the conditioned ``si``).
* ``layer_norm_a`` is applied to the token representation *after* the diffusion
  transformer and *before* the decoder, not at either end.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models._cp import cp_mesh, shard_pair_rows, shard_single
from foldjax.models.openfold3.models.atom_features import (
    AtomAttentionDecoderParams,
    AtomAttentionEncoderParams,
    atom_attention_decoder,
    atom_attention_encoder,
)
from foldjax.models.openfold3.models.diffusion_schedule import (
    combine_denoiser_output,
    scale_noisy_positions,
)
from foldjax.models.openfold3.models.diffusion_transformer import (
    DiffusionTransformerParams,
    diffusion_transformer,
)
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
)


class DenoiserParams(NamedTuple):
    """Parameters for the denoiser body."""

    atom_attn_enc: AtomAttentionEncoderParams
    layer_norm_s: LayerNormParams
    linear_s: LinearParams
    diffusion_transformer: DiffusionTransformerParams
    layer_norm_a: LayerNormParams
    atom_attn_dec: AtomAttentionDecoderParams


def denoise(
    batch: Mapping[str, jnp.ndarray],
    xl_noisy: jnp.ndarray,
    t: jnp.ndarray,
    si: jnp.ndarray,
    si_trunk: jnp.ndarray,
    zij: jnp.ndarray,
    params: DenoiserParams,
    *,
    n_query: int,
    n_key: int,
    atom_heads: int,
    token_heads: int,
    n_token: int,
    sigma_data: float,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
    glu_backend: str = "xla",
    cp_atom_windows: bool = False,
) -> jnp.ndarray:
    """Denoise one set of noisy coordinates.

    Args:
        batch: feature mapping for the atom encoder/decoder, plus ``token_mask``.
        xl_noisy: ``[..., N_atom, 3]`` noisy coordinates.
        t: ``[...]`` noise level.
        si: ``[..., N_token, C_s]`` conditioned single representation.
        si_trunk: ``[..., N_token, C_s]`` trunk single representation.
        zij: ``[..., N_token, N_token, C_z]`` conditioned pair representation.
        params: mapped parameters.
        n_query: atom query block height.
        n_key: atom key window width.
        atom_heads: atom transformer head count.
        token_heads: token diffusion transformer head count.
        n_token: static token count.
        sigma_data: EDM data standard deviation.
        inf: masking constant.
        mask_transition: upstream's ``_mask_trans``.
        eps: layer norm epsilon.
        cp_atom_windows: split the diffusion atom graph over the
            context-parallel rows. Ignored without a mesh; the caller resolves
            the request against the shapes once, outside the rollout (see
            ``models/openfold3/models/atom_cp.py:resolve_atom_windows``).

    Returns:
        ``[..., N_atom, 3]`` denoised coordinates.
    """
    distributed = bool(cp_atom_windows) and cp_mesh() is not None
    atom_mask = batch["atom_mask"]
    xl_noisy = xl_noisy * atom_mask[..., None]
    rl_noisy = scale_noisy_positions(xl_noisy, t, sigma_data=sigma_data)

    if distributed:
        # The token transformer's per-block pair bias is a projection of this
        # tensor and the encoder's atom-pair gather rotates its row tiles, so
        # constraining it here is what keeps both reading a pair-sharded
        # operand rather than a collected one.
        zij = shard_pair_rows(zij)
    ai, ql, cl, plm = atom_attention_encoder(
        batch,
        params.atom_attn_enc,
        n_query=n_query,
        n_key=n_key,
        no_heads=atom_heads,
        n_token=n_token,
        rl=rl_noisy,
        si_trunk=si_trunk,
        zij_trunk=zij,
        inf=inf,
        eps=eps,
        glu_backend=glu_backend,
        cp_atom_windows=cp_atom_windows,
    )

    # The conditioned single representation is added on top of the encoder output.
    ai = ai + linear(layer_norm(si, params.layer_norm_s, eps=eps), params.linear_s)

    if distributed:
        # The reduce-scatter already delivered this on CP rows; saying so keeps
        # the token attention's queries on the same axis its pair-bias rows are
        # split along.
        ai = shard_single(ai)

    ai = diffusion_transformer(
        ai,
        si,
        zij,
        params.diffusion_transformer,
        no_heads=token_heads,
        mask=batch["token_mask"],
        inf=inf,
        mask_transition=mask_transition,
        eps=eps,
        glu_backend=glu_backend,
    )
    ai = layer_norm(ai, params.layer_norm_a, eps=eps)
    if distributed:
        ai = shard_single(ai)

    rl_update = atom_attention_decoder(
        batch,
        ai,
        ql,
        cl,
        plm,
        params.atom_attn_dec,
        n_query=n_query,
        n_key=n_key,
        no_heads=atom_heads,
        inf=inf,
        eps=eps,
        glu_backend=glu_backend,
        cp_atom_windows=cp_atom_windows,
    )

    xl_out = combine_denoiser_output(xl_noisy, rl_update, t, sigma_data=sigma_data)
    return xl_out * atom_mask[..., None]
