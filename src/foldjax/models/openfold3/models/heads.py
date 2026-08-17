"""Pair-level prediction heads (AF3 subsection 4.3.2 and section 4.4).

Three heads that look nearly identical and are not. Every one projects the pair
representation to per-bin logits with a bias-free linear, but:

* PAE normalizes and does **not** symmetrize — aligned error is directional.
* PDE normalizes and **does** symmetrize — distance error is not directional.
* Distogram symmetrizes but has **no layer norm** at all.

Getting any of those three wrong produces plausible logits of the right shape, so
each difference is pinned by the parity gate.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models._cp import shard_pair_rows
from foldjax.models.openfold3.models.atomize import max_atom_per_token_masked_select
from foldjax.models.openfold3.models.pairformer import (
    PairformerStackParams,
    pairformer_stack,
)
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
)


class PairHeadParams(NamedTuple):
    """Parameters for a pair-level logit head.

    ``layer_norm`` is ``None`` for the distogram head, which has none.
    """

    linear: LinearParams
    layer_norm: LayerNormParams | None = None


def _pair_logits(
    z: jnp.ndarray,
    params: PairHeadParams,
    *,
    symmetrize: bool,
    eps: float,
) -> jnp.ndarray:
    hidden = z if params.layer_norm is None else layer_norm(
        z, params.layer_norm, eps=eps
    )
    logits = linear(hidden, params.linear)
    if symmetrize:
        logits = logits + jnp.swapaxes(logits, -2, -3)
    return logits


def predicted_aligned_error_head(
    z: jnp.ndarray, params: PairHeadParams, *, eps: float = 1e-5
) -> jnp.ndarray:
    """Return PAE logits ``[..., N, N, C_out]``. Not symmetrized."""
    return _pair_logits(z, params, symmetrize=False, eps=eps)


def predicted_distance_error_head(
    z: jnp.ndarray, params: PairHeadParams, *, eps: float = 1e-5
) -> jnp.ndarray:
    """Return PDE logits ``[..., N, N, C_out]``. Symmetrized."""
    return _pair_logits(z, params, symmetrize=True, eps=eps)


def distogram_head(
    z: jnp.ndarray, params: PairHeadParams, *, eps: float = 1e-5
) -> jnp.ndarray:
    """Return distogram logits ``[..., N, N, C_out]``.

    Symmetrized and, unlike the error heads, applied straight to ``z``.
    """
    if params.layer_norm is not None:
        raise ValueError("the distogram head has no layer norm")
    return _pair_logits(z, params, symmetrize=True, eps=eps)


class AtomHeadParams(NamedTuple):
    """Parameters for a per-atom logit head (pLDDT, experimentally resolved).

    The projection emits ``max_atoms_per_token * c_out`` channels per token,
    which are then reshaped to per-atom slots and compacted by the mask.
    """

    layer_norm: LayerNormParams
    linear: LinearParams


def atom_logit_head(
    s: jnp.ndarray,
    params: AtomHeadParams,
    mask: jnp.ndarray,
    *,
    max_atoms_per_token: int,
    c_out: int,
    n_atom: int,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Project a token representation to per-atom logits.

    Shared by ``PerResidueLDDTAllAtom`` and
    ``ExperimentallyResolvedHeadAllAtom``, which are identical in structure.

    Args:
        s: ``[..., N_token, C_s]`` single representation.
        params: mapped parameters.
        mask: ``[..., N_token * max_atoms_per_token]`` valid-atom mask.
        max_atoms_per_token: padding width per token.
        c_out: bins per atom.
        n_atom: static output atom count.
        eps: layer norm epsilon.

    Returns:
        ``[..., n_atom, c_out]`` logits.
    """
    logits = linear(layer_norm(s, params.layer_norm, eps=eps), params.linear)
    n_token = s.shape[-2]
    logits = logits.reshape(
        (*s.shape[:-2], n_token * max_atoms_per_token, c_out)
    )
    return max_atom_per_token_masked_select(logits, mask, n_atom=n_atom)


class PairformerEmbeddingParams(NamedTuple):
    """Parameters for the confidence heads' ``PairformerEmbedding``.

    Upstream runs this before pLDDT, PAE, PDE and the experimentally-resolved
    head, so those heads do *not* see the trunk representations directly. Only the
    distogram head reads the trunk pair embedding.
    """

    linear_i: LinearParams
    linear_j: LinearParams
    linear_distance: LinearParams
    pairformer_stack: PairformerStackParams


def pairformer_embedding(
    si_input: jnp.ndarray,
    si: jnp.ndarray,
    zij: jnp.ndarray,
    x_pred: jnp.ndarray,
    params: PairformerEmbeddingParams,
    *,
    single_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    no_heads_pair: int,
    no_heads_pair_bias: int,
    min_bin: float,
    max_bin: float,
    no_bin: int,
    inf: float = 1e9,
    eps: float = 1e-5,
    chunk_size: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Embed the predicted geometry into the confidence representations.

    ``x_pred`` carries a sample axis and nothing here mixes samples, so this works
    either batched across samples or one sample at a time. Which one it gets is
    :func:`~foldjax.models.openfold3.inference.predict`'s decision, taken on
    ``per_sample_token_cutoff``: batched is faster, per-sample is what fits, and
    over ~750 tokens the pair representations are large enough that holding one per
    sample decides whether the prediction runs.

    Returns:
        ``(si, zij)`` for the confidence heads.
    """
    zij = (
        zij
        + linear(si_input[..., :, None, :], params.linear_i)
        + linear(si_input[..., None, :, :], params.linear_j)
    )

    # A squared-distance one-hot: bin k is on when the squared distance falls
    # between bins[k]**2 and bins[k+1]**2, with the last bin open-ended.
    bins = jnp.linspace(min_bin, max_bin, no_bin, dtype=zij.dtype)
    squared = bins**2
    upper = jnp.concatenate([squared[1:], jnp.asarray([inf], dtype=zij.dtype)])
    dij = jnp.sum(
        (x_pred[..., :, None, :] - x_pred[..., None, :, :]) ** 2,
        axis=-1,
        keepdims=True,
    )
    dij = ((dij > squared) & (dij < upper)).astype(zij.dtype)
    # Born sharded under context parallelism: the distance embedding and the
    # outer-sum init otherwise materialize whole on every device before the
    # stack's own constraints take over.
    zij = shard_pair_rows(zij + linear(dij, params.linear_distance))

    return pairformer_stack(
        si,
        zij,
        params.pairformer_stack,
        single_mask=single_mask,
        pair_mask=pair_mask,
        no_heads_pair=no_heads_pair,
        no_heads_pair_bias=no_heads_pair_bias,
        inf=inf,
        eps=eps,
        chunk_size=chunk_size,
    )
