"""MSA input embedding (AF3 Algorithm 8, lines 1-4).

Concatenates the one-hot MSA with its two deletion features, projects to the MSA
channel width, and adds the single-representation projection broadcast across
sequences.

MSA **subsampling is not performed here.** The released config sets
``subsample_all_msa=True`` with
``min_subsampled_all_msa == max_subsampled_all_msa == 1024``, and upstream's
``MSAModuleEmbedder.forward`` applies it with no ``self.training`` guard -- so it
runs at inference too. An earlier version of this note claimed the opposite.

The count is deterministic (1024), but the row *selection* uses ``torch.randperm``,
which JAX cannot reproduce. What that costs depends on the MSA:

* **at most 1024 valid rows**: upstream keeps every valid row in ascending order
  and pads with all-masked rows. Appending all-masked rows is bit-identical to
  omitting them (``test_masked_msa_rows_are_a_no_op``), so this port agrees
  exactly.
* **more than 1024 valid rows**: upstream keeps a random 1024 of them. The host
  cycle planner stores their compact union and supplies each recycle's selection.

Selecting rows is a data-pipeline decision, not a network one, so
:func:`subsample_msa` performs the selection outside the embedder: it reproduces
upstream's ordering and explicit selected-index tapes remain available for
low-level parity tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.openfold3.data.featurize import (
    _COMPACT_MSA_INDICES,
    _COMPACT_MSA_MARKER,
)
from foldjax.models.openfold3.models.atom_features import (
    AtomAttentionEncoderParams,
    atom_attention_encoder,
)
from foldjax.models.openfold3.models.primitives import LinearParams, linear
from foldjax.models.openfold3.models.relpos import relpos_complex


class MSAEmbedderParams(NamedTuple):
    """Parameters for ``MSAModuleEmbedder`` without the subsampling path."""

    linear_m: LinearParams
    linear_s_input: LinearParams


def _released_msa_one_hot(batch: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """Return the historical dense MSA feature, reconstructing private storage.

    Direct callers keep precedence: whenever public ``msa`` is present it is
    returned untouched, even if a stale private marker accompanies it. The compact
    branch requires both private leaves and validates their static contract while
    tracing. Category 32 is outside the requested class count, so JAX reconstructs
    serving-padding cells as the historical all-zero vector.
    """

    dense = batch.get("msa")
    if dense is not None:
        return dense

    if _COMPACT_MSA_MARKER not in batch or _COMPACT_MSA_INDICES not in batch:
        raise KeyError(
            "OpenFold3 MSA input needs public 'msa' or the complete private "
            "compact representation"
        )
    marker = jnp.asarray(batch[_COMPACT_MSA_MARKER])
    indices = jnp.asarray(batch[_COMPACT_MSA_INDICES])
    if marker.shape != () or marker.dtype != jnp.dtype(jnp.float32):
        raise ValueError("OpenFold3 compact MSA marker must be scalar float32")
    if indices.dtype != jnp.dtype(jnp.uint8):
        raise ValueError("OpenFold3 compact MSA indices must have dtype uint8")
    mask = batch.get("msa_mask")
    if mask is not None and indices.ndim != mask.ndim:
        raise ValueError(
            "OpenFold3 compact MSA indices and msa_mask must have the same rank"
        )

    # ``marker`` is provenance, while ``indices`` is the dynamic data. The key
    # pattern and shapes already give the compact mapping a distinct JIT PyTree;
    # using marker as an arithmetic gate would change forged/direct-call semantics.
    return jax.nn.one_hot(
        indices.astype(jnp.int32), 32, dtype=jnp.int32
    )


def msa_embedder(
    batch: Mapping[str, jnp.ndarray],
    s_input: jnp.ndarray,
    params: MSAEmbedderParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Embed the MSA and add the single representation.

    Args:
        batch: needs ``msa`` ``[..., N_msa, N_token, C]``, ``has_deletion`` and
            ``deletion_value`` ``[..., N_msa, N_token]``, and ``msa_mask``.
        s_input: ``[..., N_token, C_s_input]`` single representation.
        params: mapped parameters.

    Returns:
        ``(m, msa_mask)``: ``[..., N_msa, N_token, C_m]`` MSA embedding and the
        mask, returned unchanged since subsampling is not applied.
    """
    msa_feat = jnp.concatenate(
        [
            _released_msa_one_hot(batch),
            batch["has_deletion"][..., None],
            batch["deletion_value"][..., None],
        ],
        axis=-1,
    )
    m = linear(msa_feat, params.linear_m)
    # The single representation is shared across every MSA row.
    m = m + linear(s_input, params.linear_s_input)[..., None, :, :]
    return m, batch["msa_mask"]


class InputEmbedderParams(NamedTuple):
    """Parameters for ``InputEmbedderAllAtom`` (AF3 Algorithm 2)."""

    atom_attn_enc: AtomAttentionEncoderParams
    linear_s: LinearParams
    linear_z_i: LinearParams
    linear_z_j: LinearParams
    linear_relpos: LinearParams
    linear_token_bonds: LinearParams


def input_embedder(
    batch: Mapping[str, jnp.ndarray],
    params: InputEmbedderParams,
    *,
    n_query: int,
    n_key: int,
    atom_heads: int,
    n_token: int,
    max_relative_idx: int,
    max_relative_chain: int,
    inf: float = 1e9,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build the input single and pair representations.

    The pair representation is an outer sum of two *different* projections of
    ``s_input`` (``linear_z_i`` for the row, ``linear_z_j`` for the column), so it
    is not symmetric; using one projection for both would be wrong.

    Args:
        batch: needs the atom-encoder features plus ``restype``, ``profile``,
            ``deletion_mean``, ``token_bonds`` and the relpos features.
        params: mapped parameters.
        n_query: atom query block height.
        n_key: atom key window width.
        atom_heads: atom transformer head count.
        n_token: static token count.
        max_relative_idx: relpos clip for residue/token offsets.
        max_relative_chain: relpos clip for chain offsets.
        inf: masking constant.
        eps: layer norm epsilon.

    Returns:
        ``(s_input, s, z)``.
    """
    a, _ql, _cl, _plm = atom_attention_encoder(
        batch,
        params.atom_attn_enc,
        n_query=n_query,
        n_key=n_key,
        no_heads=atom_heads,
        n_token=n_token,
        inf=inf,
        eps=eps,
    )

    s_input = jnp.concatenate(
        [
            a,
            batch["restype"],
            batch["profile"],
            batch["deletion_mean"][..., None],
        ],
        axis=-1,
    )
    s = linear(s_input, params.linear_s)

    # Row and column use different projections, so z is asymmetric by design.
    z = (
        linear(s_input, params.linear_z_i)[..., :, None, :]
        + linear(s_input, params.linear_z_j)[..., None, :, :]
    )

    relpos_feats = relpos_complex(
        batch,
        max_relative_idx=max_relative_idx,
        max_relative_chain=max_relative_chain,
    ).astype(z.dtype)
    z = z + linear(relpos_feats, params.linear_relpos)
    z = z + linear(
        batch["token_bonds"][..., None].astype(z.dtype), params.linear_token_bonds
    )
    return s_input, s, z


def subsample_msa(
    batch: Mapping[str, jnp.ndarray],
    no_subsampled: int,
    *,
    invalid_order: jnp.ndarray | None = None,
    valid_order: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Subsample the MSA features to ``no_subsampled`` rows.

    Mirrors upstream's ``_subsample_all_msa``: rows whose mask is entirely zero are
    "invalid" and are only used as filler. With at least ``no_subsampled`` valid
    rows, upstream keeps a random subset of them; otherwise it keeps all of them in
    ascending order and fills with random invalid rows.

    Args:
        batch: needs ``msa``, ``has_deletion``, ``deletion_value`` and ``msa_mask``.
        no_subsampled: rows to keep. Returned unchanged if the MSA is smaller.
        invalid_order: permutation of the invalid row indices, if upstream's
            ``randperm`` for them is known. Ascending order otherwise, which
            differs from upstream only in *which* all-masked rows fill the tail --
            and those contribute nothing, so the result is unaffected.
        valid_order: permutation of the valid row indices. Required to match
            upstream when there are more valid rows than ``no_subsampled``,
            because the choice is random there and cannot be reproduced.

    Returns:
        A new mapping with the four MSA features subsampled; other keys are copied.
    """
    mask = batch["msa_mask"]
    n_msa = mask.shape[-2]
    if n_msa <= no_subsampled:
        return dict(batch)

    valid = jnp.sum(mask, axis=-1) > 0
    # Rank valid rows before invalid ones, preserving ascending order within each
    # group; a stable sort of the negated flag does exactly that.
    if valid_order is None and invalid_order is None:
        order = jnp.argsort(~valid, axis=-1, stable=True)
    else:
        rank = jnp.arange(n_msa)
        if valid_order is not None:
            rank = rank.at[valid_order].set(jnp.arange(n_msa))
        keys = jnp.where(valid, rank, n_msa + rank)
        if invalid_order is not None:
            keys = keys.at[invalid_order].set(
                n_msa + jnp.arange(invalid_order.shape[-1])
            )
            keys = jnp.where(valid, rank, keys)
        order = jnp.argsort(keys, axis=-1, stable=True)

    selected = order[..., :no_subsampled]
    out = dict(batch)
    out["msa"] = jnp.take(batch["msa"], selected, axis=-3)
    out["has_deletion"] = jnp.take(batch["has_deletion"], selected, axis=-2)
    out["deletion_value"] = jnp.take(batch["deletion_value"], selected, axis=-2)
    out["msa_mask"] = jnp.take(mask, selected, axis=-2)
    return out
