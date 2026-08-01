"""Chai-1 feature_embedding.pt port (JAX).

The exported embedder takes the per-modality precomputed features (produced by
the chai_lab feature generators), applies in-graph transforms (one-hot, reshape,
RBF restraint encoding, residue-type embedding), concatenates them per modality,
and applies a single bf16 ``Linear`` (``input_projs.<MODALITY>.0``). Output dict
keys: ATOM, ATOM_PAIR, TOKEN, TOKEN_PAIR, MSA, TEMPLATES (all bf16 at runtime).

Math recovered from the TorchScript ``forward_256.code`` and per-leaf
``.forward.code``. ``embed_features`` returns the full modality dict; each
modality also has a standalone ``embed_<modality>`` for isolated parity gating.

RBF restraint widths (c0 for TokenDistanceRestraint, c2 for
TokenPairPocketRestraint) and the TemplateResType offset stride (c3) are
TorchScript constants not present in the upstream Python; their values are
established by the real-component parity gate (the JAX branch is compared to the
live component output). ``RESTRAINT_WIDTH`` / ``RESTYPE_OFFSET_STRIDE`` capture
the verified values.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from foldjax.models.chai.models.primitives import (
    embedding,
    linear_bf16,
    one_hot,
    rbf_restraint_encoding,
)

# Verified against the live feature_embedding.pt component (see parity tests).
TDR_WIDTH = 4.8  # TokenDistanceRestraint radii step (6.0 -> 30.0, 6 radii)
TPP_WIDTH = 2.8  # TokenPairPocketRestraint radii step (6.0 -> 20.0, 6 radii)
RESTYPE_OFFSET_STRIDE = 33  # c3: residue-type embedding offset per template position


def _proj(state, modality, x):
    w = jnp.asarray(state[f"input_projs.{modality}.0.weight"])
    b = jnp.asarray(state[f"input_projs.{modality}.0.bias"])
    return linear_bf16(x, w, b)


def map_input_proj(
    state: Mapping[str, Any], modality: str
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract (weight, bias) for ``input_projs.<modality>.0`` as JAX arrays."""
    w = jnp.asarray(state[f"input_projs.{modality}.0.weight"])
    b = jnp.asarray(state[f"input_projs.{modality}.0.bias"])
    return w, b


def _msa_cat(f: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.concatenate(
        [
            f["IsPairedMSA"][..., None],
            one_hot(f["MSADataSource"].astype(jnp.int32), 6),
            f["MSADeletionValue"][..., None],
            f["MSAHasDeletion"][..., None],
            one_hot(f["MSAOneHot"].astype(jnp.int32), 33),
        ],
        axis=-1,
    )  # -> 42


def embed_msa(
    features: Mapping[str, jnp.ndarray], weight: jnp.ndarray, bias: jnp.ndarray
) -> jnp.ndarray:
    """MSA branch: cat (42) -> bf16 Linear -> 64."""
    return linear_bf16(_msa_cat(features), weight, bias)


def _atom_cat(f: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    # AtomNameOneHot (b, n, 4) int codes -> one_hot(65) -> flatten last two = 260.
    name = one_hot(f["AtomNameOneHot"].astype(jnp.int32), 65)
    name = name.reshape(name.shape[:-2] + (name.shape[-2] * name.shape[-1],))  # 260
    return jnp.concatenate(
        [
            name,
            f["AtomRefCharge"][..., None],
            one_hot(f["AtomRefElement"].astype(jnp.int32), 130),
            f["AtomRefMask"][..., None],
            f["AtomRefPos"],
        ],
        axis=-1,
    )  # -> 395


def _atom_pair_cat(f: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.concatenate(
        [
            one_hot(f["BlockedAtomPairDistogram"].astype(jnp.int32), 12),
            f["InverseSquaredBlockedAtomPairDistances"],
        ],
        axis=-1,
    )  # -> 14


def _token_cat(f: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.concatenate(
        [
            f["ChainIsCropped"][..., None],
            f["ESMEmbeddings"],
            one_hot(f["IsDistillation"].astype(jnp.int32), 2),
            f["MSADeletionMean"][..., None],
            f["MSAProfile"],
            f["MissingChainContact"][..., None],
            one_hot(f["ResidueType"].astype(jnp.int32), 33),
            one_hot(f["TokenBFactor"].astype(jnp.int32), 3),
            one_hot(f["TokenPLDDT"].astype(jnp.int32), 4),
        ],
        axis=-1,
    )  # -> 2638


def _token_pair_cat(f: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    tdr = rbf_restraint_encoding(
        f["TokenDistanceRestraint"][..., 0], _TDR_RADII, width=TDR_WIDTH
    )
    tpp = rbf_restraint_encoding(
        f["TokenPairPocketRestraint"][..., 0], _TPP_RADII, width=TPP_WIDTH
    )
    return jnp.concatenate(
        [
            one_hot(f["DockingConstraintGenerator"].astype(jnp.int32), 6),
            one_hot(f["RelativeChain"].astype(jnp.int32), 6),
            one_hot(f["RelativeEntity"].astype(jnp.int32), 3),
            one_hot(f["RelativeSequenceSeparation"].astype(jnp.int32), 67),
            one_hot(f["RelativeTokenSeparation"].astype(jnp.int32), 67),
            tdr,
            tpp,
        ],
        axis=-1,
    )  # -> 163


def _templates_cat(
    f: Mapping[str, jnp.ndarray], restype_embed_weight: jnp.ndarray
) -> jnp.ndarray:
    dist = one_hot(f["TemplateDistogram"].astype(jnp.int32), 39)
    mask = f["TemplateMask"]
    # TemplateResType (b, t, n, 1): EncodingType.OUTERSUM. The graph adds a
    # per-trailing-position offset (arange(s) * stride), embeds, then forms the
    # token-pair outer sum emb[...,:,None,:] + emb[...,None,:,:] -> (b, t, n, n, 32).
    res = f["TemplateResType"].astype(jnp.int32)
    s = res.shape[-1]
    offsets = jnp.arange(s, dtype=jnp.int32) * RESTYPE_OFFSET_STRIDE
    emb = embedding(restype_embed_weight, res + offsets)  # (b, t, n, s, 32)
    emb = jnp.sum(emb, axis=-2)  # collapse trailing offset axis -> (b, t, n, 32)
    restype = emb[..., :, None, :] + emb[..., None, :, :]  # outer sum -> (b,t,n,n,32)
    uvec = f["TemplateUnitVector"]
    return jnp.concatenate([dist, mask, restype, uvec], axis=-1)  # -> 76


# Restraint radii buffers (from the component state_dict).
_TDR_RADII = jnp.asarray([6.0, 10.8, 15.6, 20.4, 25.2, 30.0], dtype=jnp.float32)
_TPP_RADII = jnp.asarray([6.0, 8.8, 11.6, 14.4, 17.2, 20.0], dtype=jnp.float32)


def embed_features(
    features: Mapping[str, jnp.ndarray], state: Mapping[str, Any]
) -> dict[str, jnp.ndarray]:
    """Full feature embedder: returns the bf16 modality dict matching the component."""
    restype_w = jnp.asarray(
        state["feature_embeddings.TEMPLATES.TemplateResType.embedding.weight"]
    )
    return {
        "ATOM": _proj(state, "ATOM", _atom_cat(features)),
        "ATOM_PAIR": _proj(state, "ATOM_PAIR", _atom_pair_cat(features)),
        "TOKEN": _proj(state, "TOKEN", _token_cat(features)),
        "TOKEN_PAIR": _proj(state, "TOKEN_PAIR", _token_pair_cat(features)),
        "MSA": _proj(state, "MSA", _msa_cat(features)),
        "TEMPLATES": _proj(state, "TEMPLATES", _templates_cat(features, restype_w)),
    }


def embed_non_msa_features(
    features: Mapping[str, jnp.ndarray], state: Mapping[str, Any]
) -> dict[str, jnp.ndarray]:
    """Embed modalities whose shapes do not depend on MSA row depth."""

    restype_w = jnp.asarray(
        state["feature_embeddings.TEMPLATES.TemplateResType.embedding.weight"]
    )
    return {
        "ATOM": _proj(state, "ATOM", _atom_cat(features)),
        "ATOM_PAIR": _proj(state, "ATOM_PAIR", _atom_pair_cat(features)),
        "TOKEN": _proj(state, "TOKEN", _token_cat(features)),
        "TOKEN_PAIR": _proj(state, "TOKEN_PAIR", _token_pair_cat(features)),
        "TEMPLATES": _proj(state, "TEMPLATES", _templates_cat(features, restype_w)),
    }
