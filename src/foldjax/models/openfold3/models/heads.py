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


def _project_distance_bins(
    squared_distance: jnp.ndarray,
    params: LinearParams,
    *,
    dtype: jnp.dtype,
    min_bin: float,
    max_bin: float,
    no_bin: int,
    inf: float,
) -> jnp.ndarray:
    """Project the strict distance-bin one-hot without materializing it.

    The historical expression built ``[..., N, N, no_bin]`` zero/one values and
    passed them through a bias-free linear.  A one-hot dot selects one row of the
    transposed weight, except that a value exactly on either bin edge selects no
    row.  Locate that row directly so confidence embedding does not retain the
    released 39-channel pair temporary.

    Two IEEE details are deliberate.  The released multi-term dot canonicalizes
    a selected negative zero to positive zero, and every *unselected* NaN or
    infinity still participates as ``0 * weight``.  The small reduction over the
    weight matrix reproduces those cases: the sole non-finite value may survive
    only when it is the selected infinity; every other non-finite combination
    produces NaN.

    Non-monotonic custom bin ranges, single-bin dots, and incompatible parameter
    layouts retain the dense expression.  This keeps the private helper an
    optimization rather than a new low-level input contract.
    """

    weight = params.weight
    monotonic_squared_bins = min_bin >= 0.0 and max_bin >= min_bin
    # A length-one dot is lowered as a multiply and can therefore retain -0.
    # Keep that degenerate layout on the dense expression; reductions with at
    # least two terms canonicalize zero exactly as the indexed path does.
    indexed_layout = no_bin > 1 and weight.ndim == 2 and weight.shape[-1] == no_bin
    if not monotonic_squared_bins or not indexed_layout:
        bins = jnp.linspace(min_bin, max_bin, no_bin, dtype=dtype)
        squared = bins**2
        upper = jnp.concatenate(
            [squared[1:], jnp.asarray([inf], dtype=dtype)]
        )
        one_hot = (
            (squared_distance[..., None] > squared)
            & (squared_distance[..., None] < upper)
        ).astype(dtype)
        return linear(one_hot, params)

    bins = jnp.linspace(min_bin, max_bin, no_bin, dtype=dtype)
    squared = bins**2
    upper = jnp.concatenate([squared[1:], jnp.asarray([inf], dtype=dtype)])

    # ``side="left"`` makes an exact internal edge point at the preceding bin;
    # the strict comparisons below then reject it, matching the two comparisons
    # in the dense one-hot expression.  NaN, infinity, and the open final bound
    # are rejected by those same comparisons.
    index = jnp.searchsorted(squared, squared_distance, side="left") - 1
    safe_index = jnp.clip(index, 0, no_bin - 1)
    valid = (
        (index >= 0)
        & (squared_distance > squared[safe_index])
        & (squared_distance < upper[safe_index])
    )

    table = jnp.swapaxes(weight, -1, -2)
    selected = jnp.take(table, safe_index, axis=0)
    output_dtype = jnp.result_type(dtype, weight.dtype)
    selected = selected.astype(output_dtype)
    zero = jnp.zeros((), dtype=output_dtype)
    # The multi-term dot starts from +0, so selecting -0 also yields +0.
    selected = jnp.where(selected == 0, zero, selected)

    nonfinite_count = jnp.sum(~jnp.isfinite(table), axis=0)
    no_nonfinite = nonfinite_count == 0
    sole_selected_infinity = (nonfinite_count == 1) & jnp.isinf(selected)
    preserves_nonfinite_semantics = jnp.where(
        valid[..., None],
        no_nonfinite | sole_selected_infinity,
        no_nonfinite,
    )
    projected = jnp.where(valid[..., None], selected, zero)
    projected = jnp.where(
        preserves_nonfinite_semantics,
        projected,
        jnp.asarray(jnp.nan, dtype=output_dtype),
    )
    if params.bias is not None:
        projected = projected + params.bias
    return projected


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
    ``per_sample_token_cutoff`` and the released five-sample width: batching can
    be faster, per-sample is what fits, and either long targets or high sample
    counts make holding one pair representation per sample the dominant cost.

    Returns:
        ``(si, zij)`` for the confidence heads.
    """
    zij = (
        zij
        + linear(si_input[..., :, None, :], params.linear_i)
        + linear(si_input[..., None, :, :], params.linear_j)
    )

    # Project the squared-distance one-hot by selecting its one active weight row.
    # The helper preserves strict-edge, non-finite, and signed-zero dot semantics
    # without materializing the released ``[..., N, N, 39]`` temporary.
    dij = jnp.sum(
        (x_pred[..., :, None, :] - x_pred[..., None, :, :]) ** 2,
        axis=-1,
    )
    # Born sharded under context parallelism: the distance embedding and the
    # outer-sum init otherwise materialize whole on every device before the
    # stack's own constraints take over.
    zij = shard_pair_rows(
        zij
        + _project_distance_bins(
            dij,
            params.linear_distance,
            dtype=zij.dtype,
            min_bin=min_bin,
            max_bin=max_bin,
            no_bin=no_bin,
            inf=inf,
        )
    )

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
