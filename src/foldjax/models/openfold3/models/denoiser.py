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
    atom_pair_conditioning,
    noisy_position_trunk_conditioning,
    ref_atom_feature_embedder,
)
from foldjax.models.openfold3.models.attention import validate_attention_backend
from foldjax.models.openfold3.models.diffusion_schedule import (
    combine_denoiser_output,
    scale_noisy_positions,
)
from foldjax.models.openfold3.models.diffusion_transformer import (
    DiffusionTransformerParams,
    atom_transformer_pair_biases,
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


class AtomDiffusionCache(NamedTuple):
    """Step-invariant atom conditioning hoisted out of the diffusion scan.

    Built once per prediction at the conditioning's natural leading width.
    Width 1 serves every sample only when the samples share all of its
    inputs: the trunk ``si``/``zij``, the reference atom features, and every
    atom mask and token/atom mapping in ``batch``. A caller whose samples
    differ in any of these (per-sample padding, say) builds the cache at the
    sample width from the same per-sample ``batch`` it passes to
    :func:`denoise`, or passes no cache. The cache is also tied to the
    parameters, ``n_query``/``n_key`` and ``eps`` it was prepared with; a
    :func:`denoise` call with other values must not reuse it. Nothing here can
    check the values, so the caller owns that match.

    * ``cl``: ``[..., N_atom, c_atom]`` reference + trunk single atom
      conditioning (before adding ``linear_r(rl)``).
    * ``enc_pair_bias``: ``[enc_blocks, ..., N_blocks, H_atom, N_query, N_key]``
      encoder atom-transformer pair biases.
    * ``dec_pair_bias``: ``[dec_blocks, ..., N_blocks, H_atom, N_query, N_key]``
      decoder atom-transformer pair biases.

    ``plm`` is consumed when projecting both bias stacks and not retained, so
    it is freed before the rollout's token transformer runs.
    """

    cl: jnp.ndarray
    enc_pair_bias: jnp.ndarray
    dec_pair_bias: jnp.ndarray


def prepare_atom_diffusion_cache(
    batch: Mapping[str, jnp.ndarray],
    si_trunk: jnp.ndarray,
    zij_trunk: jnp.ndarray,
    params: DenoiserParams,
    *,
    n_query: int,
    n_key: int,
    eps: float = 1e-5,
) -> AtomDiffusionCache:
    """Precompute step-invariant atom single conditioning and pair biases.

    Evaluates ``ref_atom_feature_embedder``, the ``si_trunk`` and ``zij_trunk``
    terms of ``noisy_position_embedder``, ``atom_pair_conditioning``, and both
    the encoder and decoder ``atom_transformer`` pair-bias projections once,
    with the same layer norms, projections and add order as an uncached
    :func:`denoise` step.
    """
    if cp_mesh() is not None:
        raise ValueError(
            "prepare_atom_diffusion_cache is only supported outside a "
            "context-parallel mesh"
        )
    enc_params = params.atom_attn_enc
    if enc_params.noisy_position_embedder is None:
        raise ValueError(
            "prepare_atom_diffusion_cache requires a diffusion encoder with "
            "noisy_position_embedder"
        )
    cl, plm = ref_atom_feature_embedder(
        batch, enc_params.ref_atom_feature_embedder, n_query=n_query, n_key=n_key
    )
    cl, plm = noisy_position_trunk_conditioning(
        batch,
        cl,
        plm,
        si_trunk,
        zij_trunk,
        enc_params.noisy_position_embedder,
        n_query=n_query,
        n_key=n_key,
        eps=eps,
    )
    plm = atom_pair_conditioning(
        cl,
        plm,
        batch["atom_mask"],
        enc_params.pair_conditioning,
        n_query=n_query,
        n_key=n_key,
    )
    enc_pair_bias = atom_transformer_pair_biases(
        plm, enc_params.atom_transformer, eps=eps
    )
    dec_pair_bias = atom_transformer_pair_biases(
        plm, params.atom_attn_dec.atom_transformer, eps=eps
    )
    return AtomDiffusionCache(
        cl=cl,
        enc_pair_bias=enc_pair_bias,
        dec_pair_bias=dec_pair_bias,
    )


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
    attention_backend: str = "xla",
    cp_atom_windows: bool = False,
    atom_cache: AtomDiffusionCache | None = None,
) -> jnp.ndarray:
    """Denoise one set of noisy coordinates.

    Args:
        batch: feature mapping for the atom encoder/decoder, plus ``token_mask``.
        xl_noisy: ``[..., N_atom, 3]`` noisy coordinates.
        t: ``[...]`` noise level.
        si: ``[..., N_token, C_s]`` conditioned single representation. Its
            leading sample axis may be 1 and broadcast against per-sample
            activations.
        si_trunk: ``[..., N_token, C_s]`` trunk single representation. Its
            leading sample axis may be 1 and broadcast against per-sample
            activations.
        zij: ``[..., N_token, N_token, C_z]`` conditioned pair representation.
            Its leading sample axis may be 1 and broadcast against per-sample
            activations.
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
        glu_backend: SwiGLU transition backend.
        attention_backend: ``"xla"`` (default) or ``"tokamax"`` for the
            diffusion token transformer's self-attention only. Context-parallel
            meshes and the atom encoder/decoder always keep ``"xla"``.
        cp_atom_windows: split the diffusion atom graph over the
            context-parallel rows. Ignored without a mesh; the caller resolves
            the request against the shapes once, outside the rollout (see
            ``models/openfold3/models/atom_cp.py:resolve_atom_windows``).
        atom_cache: optional precomputed step-invariant atom conditioning from
            :func:`prepare_atom_diffusion_cache`. When supplied, the encoder
            and decoder reuse ``cl`` and the per-block pair biases instead of
            rebuilding them per step, so it must come from this ``batch`` (or
            one whose samples it broadcasts to) and these ``params``,
            ``n_query``, ``n_key`` and ``eps``; see :class:`AtomDiffusionCache`.

    Returns:
        ``[..., N_atom, 3]`` denoised coordinates.
    """
    validate_attention_backend(attention_backend)
    mesh = cp_mesh()
    distributed = bool(cp_atom_windows) and mesh is not None
    attn_backend = "xla" if mesh is not None else attention_backend
    if atom_cache is not None and distributed:
        raise ValueError(
            "atom_cache is not supported with distributed cp_atom_windows"
        )
    atom_mask = batch["atom_mask"]
    xl_noisy = xl_noisy * atom_mask[..., None]
    rl_noisy = scale_noisy_positions(xl_noisy, t, sigma_data=sigma_data)

    if distributed:
        # The token transformer's per-block pair bias is a projection of this
        # tensor and the encoder's atom-pair gather rotates its row tiles, so
        # constraining it here is what keeps both reading a pair-sharded
        # operand rather than a collected one.
        zij = shard_pair_rows(zij)
    if atom_cache is None:
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
    else:
        ai, ql, cl, plm = atom_attention_encoder(
            batch,
            params.atom_attn_enc,
            n_query=n_query,
            n_key=n_key,
            no_heads=atom_heads,
            n_token=n_token,
            rl=rl_noisy,
            inf=inf,
            eps=eps,
            glu_backend=glu_backend,
            cp_atom_windows=cp_atom_windows,
            cached_cl=atom_cache.cl,
            cached_pair_bias=atom_cache.enc_pair_bias,
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
        attention_backend=attn_backend,
    )
    ai = layer_norm(ai, params.layer_norm_a, eps=eps)
    if distributed:
        ai = shard_single(ai)

    if atom_cache is None:
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
    else:
        rl_update = atom_attention_decoder(
            batch,
            ai,
            ql,
            cl,
            None,
            params.atom_attn_dec,
            n_query=n_query,
            n_key=n_key,
            no_heads=atom_heads,
            inf=inf,
            eps=eps,
            glu_backend=glu_backend,
            cp_atom_windows=cp_atom_windows,
            cached_pair_bias=atom_cache.dec_pair_bias,
        )

    xl_out = combine_denoiser_output(xl_noisy, rl_update, t, sigma_data=sigma_data)
    return xl_out * atom_mask[..., None]
