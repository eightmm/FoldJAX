"""Pure JAX port of Boltz-2 ``TemplateV2Module`` (trunkv2.py).

Inference-only, weight-compatible. Mirrors the torch forward op-for-op:
distogram binning (own bins), pseudo-beta CB pair distances, per-frame unit
vectors, visibility pair mask, a PairformerNoSeq stack per template, averaging
over templates, and a ``u_proj`` output added into ``z``.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import linear as _linear
from foldjax.models.boltz2.models.trunk_blocks.pairformer_noseq import (
    pairformer_no_seq_module_forward,
)

Params = Mapping[str, object]


def has_template_feats(feats: Mapping[str, jnp.ndarray]) -> bool:
    """Return True when ``feats`` carries a non-dummy template signal.

    The featurizer always emits template_* arrays (dummy zeros when no template
    is provided). ``template_mask`` is all-zero for the dummy case, so any
    nonzero entry means a real template is present. Used to gate the template
    path so no-template inputs stay byte-identical to the no-template trunk.
    """

    mask = feats.get("template_mask")
    if mask is None:
        return False
    return bool(jnp.any(mask != 0))


def template_module_forward(
    params: Params,
    z: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    pair_mask: jnp.ndarray,
    *,
    min_dist: float = 3.25,
    max_dist: float = 50.75,
    num_bins: int = 38,
    eps: float = 1e-5,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    triangle_backend: str = "xla",
) -> jnp.ndarray:
    """Run the Boltz-2 TemplateV2Module forward, returning the z-contribution.

    Mirrors ``TemplateV2Module.forward`` in ``trunkv2.py``. ``z`` is [B, N, N,
    token_z]; ``pair_mask`` is [B, N, N]; template feats carry a template axis T.
    Returns the additive update u of shape [B, N, N, token_z] (caller does
    ``z = z + u``).
    """

    res_type = feats["template_restype"]
    frame_rot = feats["template_frame_rot"]
    frame_t = feats["template_frame_t"]
    frame_mask = feats["template_mask_frame"].astype(z.dtype)
    cb_coords = feats["template_cb"].astype(jnp.float32)
    ca_coords = feats["template_ca"].astype(jnp.float32)
    cb_mask = feats["template_mask_cb"].astype(z.dtype)
    visibility_ids = feats["visibility_ids"]
    template_mask = jnp.any(feats["template_mask"] != 0, axis=2).astype(jnp.float32)
    num_templates = jnp.clip(template_mask.sum(axis=1), 1.0)

    # Pairwise masks [B, T, N, N, 1]
    b_cb_mask = (cb_mask[:, :, :, None] * cb_mask[:, :, None, :])[..., None]
    b_frame_mask = (frame_mask[:, :, :, None] * frame_mask[:, :, None, :])[..., None]

    b_size, t_size = res_type.shape[:2]
    n_tokens = res_type.shape[2]
    tmlp_pair_mask = (
        visibility_ids[:, :, :, None] == visibility_ids[:, :, None, :]
    ).astype(z.dtype)

    # Distogram with the template's OWN bins (fp32).
    cb_dists = jnp.sqrt(
        jnp.sum(
            (cb_coords[:, :, :, None, :] - cb_coords[:, :, None, :, :]) ** 2,
            axis=-1,
        )
    )
    boundaries = jnp.linspace(min_dist, max_dist, num_bins - 1, dtype=jnp.float32)
    distogram_idx = jnp.sum(cb_dists[..., None] > boundaries, axis=-1)
    distogram = jnp.eye(num_bins, dtype=z.dtype)[distogram_idx]

    # Per-frame unit vectors (fp32). Match torch axis placement exactly:
    # frame on the j-axis (unsqueeze(2)), ca on the i-axis (unsqueeze(3)).
    fr = jnp.swapaxes(frame_rot[:, :, None], -1, -2)  # [B, T, 1, N, 3, 3]
    ft = frame_t[:, :, None, :, :, None]  # [B, T, 1, N, 3, 1]
    ca = ca_coords[:, :, :, None, :, None]  # [B, T, N, 1, 3, 1]
    vector = jnp.matmul(fr.astype(jnp.float32), ca - ft.astype(jnp.float32))
    norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    unit_vector = jnp.where(norm > 0, vector / norm, jnp.zeros_like(vector))
    unit_vector = jnp.squeeze(unit_vector, axis=-1).astype(z.dtype)

    a_tij = jnp.concatenate([distogram, b_cb_mask, unit_vector, b_frame_mask], axis=-1)
    a_tij = a_tij * tmlp_pair_mask[..., None]

    res_type = res_type.astype(z.dtype)
    res_type_i = jnp.broadcast_to(
        res_type[:, :, :, None, :],
        (b_size, t_size, n_tokens, n_tokens, res_type.shape[-1]),
    )
    res_type_j = jnp.broadcast_to(
        res_type[:, :, None, :, :],
        (b_size, t_size, n_tokens, n_tokens, res_type.shape[-1]),
    )
    a_tij = jnp.concatenate([a_tij, res_type_i, res_type_j], axis=-1)
    a_tij = _linear(a_tij, params["a_proj"]["kernel"])

    # Per-template pairformer over v.
    z_norm = _layer_norm(
        z[:, None],
        params["z_norm"]["scale"],
        params["z_norm"]["bias"],
        eps,
    )
    v = _linear(z_norm, params["z_proj"]["kernel"]) + a_tij
    v = v.reshape((b_size * t_size, n_tokens, n_tokens, v.shape[-1]))

    pm = jnp.broadcast_to(
        pair_mask[:, None], (b_size, t_size, n_tokens, n_tokens)
    ).reshape((b_size * t_size, n_tokens, n_tokens))

    v = v + pairformer_no_seq_module_forward(
        params["pairformer"],
        v,
        pm,
        eps=eps,
        chunk_size=chunk_size,
        triangle_attention_chunk=triangle_attention_chunk,
        triangle_attention_q_chunk=triangle_attention_q_chunk,
        transition_hidden_chunk=transition_hidden_chunk,
        triangle_backend=triangle_backend,
    )
    v = _layer_norm(
        v,
        params["v_norm"]["scale"],
        params["v_norm"]["bias"],
        eps,
    )
    v = v.reshape((b_size, t_size, n_tokens, n_tokens, v.shape[-1]))

    # Aggregate over templates.
    tm = template_mask[:, :, None, None, None].astype(v.dtype)
    u = (v * tm).sum(axis=1) / num_templates[:, None, None, None].astype(v.dtype)

    # Output projection.
    u = _linear(jnp.maximum(u, 0.0), params["u_proj"]["kernel"])
    return u
