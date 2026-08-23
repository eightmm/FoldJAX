"""The confidence head: pLDDT, PAE, PDE, resolved atoms, pTM and ipTM.

Four things in here read wrongly on a first pass, and all four are reproduced
rather than corrected:

* the pair representation is **added to its own trunk output**. Upstream's
  `FoldingTrunk` is residual throughout, so `pair + folding_trunk(pair)` is
  `2 * pair + updates`. It looks like a missing assignment and it is what the
  released weights were trained against.
* pLDDT bins span **0 to 1**, not 0 to 100. A head that scales them to 100
  produces confidences that look right and rank right and are a hundred times
  too large.
* pTM and ipTM are read off the **PAE logits**; there is no separate TM head,
  and `d0` is computed per structure from the token count.
* `s_norm`, `s_inputs_to_single` and `s_input_to_s` are in the checkpoint and
  are never used. They are not loaded here, and their absence is not a gap.

The per-chain ipTM matrix is the one place upstream reads a chain count off
the data with `.item()`; here it is an argument, because a traced maximum
cannot size a loop.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from foldjax.models._cp import shard_pair_rows
from foldjax.models.esmfold2.models.embedders import row_attention_pooling
from foldjax.models.esmfold2.models.primitives import layer_norm, linear
from foldjax.models.esmfold2.models.segments import (
    MAX_ATOMS_PER_TOKEN,
    sum_by_token,
)
from foldjax.models.esmfold2.models.trunk import folding_trunk

Params = Mapping[str, jnp.ndarray]

#: `mol_type`'s non-polymer code. Ligand tokens take a fixed interface weight
#: rather than an inferred one.
NONPOLYMER_ID = 4
EPS = 1e-6
def gather_token_to_atom(
    token_features: jnp.ndarray, atom_to_token: jnp.ndarray
) -> jnp.ndarray:
    """Broadcast per-token features out to the atoms that belong to them."""
    return jnp.take_along_axis(token_features, atom_to_token[..., None], axis=1)


def categorical_mean(
    logits: jnp.ndarray, start: float, end: float
) -> jnp.ndarray:
    """Expected value over evenly spaced bins, using bin *centres*.

    The edges are `n_bins + 1` points from `start` to `end`; using the edges
    themselves rather than their midpoints biases every estimate by half a bin.
    """
    n_bins = logits.shape[-1]
    edges = jnp.linspace(start, end, n_bins + 1, dtype=jnp.float32)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return jnp.sum(jax.nn.softmax(logits.astype(jnp.float32), axis=-1) * centres, -1)


def intra_token_index(atom_to_token: jnp.ndarray) -> jnp.ndarray:
    """Each atom's rank inside its own token.

    Atoms of one token are contiguous, so this is a running count reset at
    every boundary -- written as a cumulative maximum rather than a segment
    scan because that is what upstream does and the two disagree on
    non-contiguous input.
    """
    same_as_previous = jnp.concatenate(
        [
            jnp.zeros_like(atom_to_token[:, :1], dtype=bool),
            atom_to_token[:, 1:] == atom_to_token[:, :-1],
        ],
        axis=1,
    )
    running = jnp.cumsum(jnp.ones_like(atom_to_token), axis=-1)
    group_start = jnp.where(same_as_previous, 0, running)
    group_start = jax.lax.cummax(group_start, axis=1)
    return running - group_start


def _scatter_sum(
    values: jnp.ndarray,
    atom_to_token: jnp.ndarray,
    n_tokens: int,
    atom_mask: jnp.ndarray,
    *,
    contiguous_atom_groups: bool = False,
) -> jnp.ndarray:
    if contiguous_atom_groups:
        reduced = sum_by_token(values, atom_to_token, n_tokens, atom_mask)
        if values.shape[1] == 0 or not jnp.issubdtype(values.dtype, jnp.floating):
            return reduced

        # The historical dense product evaluates ``0 * nonfinite`` for every
        # atom that a token does not own. Preserve that invalid-value behavior
        # without retaining the owner matrix in the normal finite case.
        has_nan = jnp.any(jnp.isnan(values), axis=1)
        positive_inf = jnp.any(jnp.isposinf(values), axis=1)
        negative_inf = jnp.any(jnp.isneginf(values), axis=1)
        is_inf = jnp.isinf(values)
        first_inf_owner = jnp.min(
            jnp.where(
                is_inf,
                atom_to_token,
                jnp.full_like(atom_to_token, n_tokens),
            ),
            axis=1,
        )
        last_inf_owner = jnp.max(
            jnp.where(is_inf, atom_to_token, jnp.zeros_like(atom_to_token)),
            axis=1,
        )
        one_inf_owner = first_inf_owner == last_inf_owner
        token = jnp.arange(n_tokens, dtype=atom_to_token.dtype)[None]
        owns_every_inf = one_inf_owner[:, None] & (token == first_inf_owner[:, None])
        nan = jnp.asarray(jnp.nan, dtype=values.dtype)
        infinity = jnp.where(
            positive_inf & negative_inf,
            nan,
            jnp.where(
                positive_inf,
                jnp.asarray(jnp.inf, dtype=values.dtype),
                jnp.asarray(-jnp.inf, dtype=values.dtype),
            ),
        )
        dense_nonfinite = jnp.where(owns_every_inf, infinity[:, None], nan)
        return jnp.where(
            has_nan[:, None],
            nan,
            jnp.where((positive_inf | negative_inf)[:, None], dense_nonfinite, reduced),
        )
    one_hot = jax.nn.one_hot(atom_to_token, n_tokens, dtype=values.dtype)
    return jnp.einsum("ba,bat->bt", values, one_hot)


def confidence_head(
    s_inputs: jnp.ndarray,
    z: jnp.ndarray,
    x_pred: jnp.ndarray,
    params: Params,
    prefix: str = "",
    *,
    distogram_atom_idx: jnp.ndarray,
    token_mask: jnp.ndarray,
    atom_to_token: jnp.ndarray,
    atom_mask: jnp.ndarray,
    asym_id: jnp.ndarray,
    mol_type: jnp.ndarray,
    n_layers: int,
    n_chains: int,
    num_samples: int = 1,
    trunk_dtype: object = jnp.float32,
    relative_position_encoding: jnp.ndarray | None = None,
    token_bonds_encoding: jnp.ndarray | None = None,
    contiguous_atom_groups: bool = False,
) -> dict[str, jnp.ndarray]:
    """`ConfidenceHead`, returning upstream's dictionary key for key.

    Everything except the pair representation is expanded to
    `batch * num_samples` first: the head genuinely reruns per structure,
    which is why its cost scales with the sample count rather than amortising.
    """
    dot = f"{prefix}." if prefix else ""

    def spread(x: jnp.ndarray) -> jnp.ndarray:
        return x if num_samples == 1 else jnp.repeat(x, num_samples, axis=0)

    s_inputs_normed = layer_norm(
        s_inputs,
        params[f"{dot}s_inputs_norm.weight"],
        params[f"{dot}s_inputs_norm.bias"],
    )
    pair = layer_norm(
        z, params[f"{dot}z_norm.weight"], params[f"{dot}z_norm.bias"]
    )
    if relative_position_encoding is not None:
        pair = pair + relative_position_encoding
    if token_bonds_encoding is not None:
        pair = pair + token_bonds_encoding
    pair = pair + linear(s_inputs_normed, params, f"{dot}s_to_z")[:, :, None, :]
    pair = pair + linear(s_inputs_normed, params, f"{dot}s_to_z_transpose")[:, None]
    pair = pair + linear(
        linear(s_inputs_normed, params, f"{dot}s_to_z_prod_in1")[:, :, None, :]
        * linear(s_inputs_normed, params, f"{dot}s_to_z_prod_in2")[:, None, :, :],
        params,
        f"{dot}s_to_z_prod_out",
    )

    pair = spread(pair)
    atom_to_token = spread(atom_to_token)
    atom_mask = spread(atom_mask).astype(jnp.float32)
    rep_idx = spread(distogram_atom_idx)
    mask = spread(token_mask).astype(jnp.float32)
    asym_id = spread(asym_id)
    mol_type = spread(mol_type)
    batch, n_tokens = mask.shape

    rep_coords = jnp.take_along_axis(x_pred, rep_idx[..., None], axis=1)
    offsets = rep_coords[:, :, None, :] - rep_coords[:, None, :, :]
    rep_distances = jnp.sqrt(jnp.sum(offsets * offsets, axis=-1) + 1e-12)
    boundaries = params[f"{dot}boundaries"]
    bins = jnp.sum(rep_distances[..., None] > boundaries, axis=-1)
    # Born sharded under context parallelism, before the head's own trunk:
    # `spread` just repeated it once per sample.
    pair = shard_pair_rows(
        pair + params[f"{dot}dist_bin_pairwise_embed.weight"][bins]
    )

    pair_mask = mask[:, :, None] * mask[:, None, :]
    # The trunk's own output already contains `pair`; adding it again is
    # upstream's arithmetic, not a missing assignment. The trunk itself runs
    # under its own bfloat16 autocast there, and the delta comes back to
    # float32 before the add -- `pair.add_(pair_delta.float())`.
    trunk_params = {
        name: (
            value.astype(trunk_dtype)
            if name.startswith(f"{dot}folding_trunk") and value.dtype == jnp.float32
            else value
        )
        for name, value in params.items()
    }
    pair = pair + folding_trunk(
        pair.astype(trunk_dtype),
        trunk_params,
        f"{dot}folding_trunk",
        n_layers=n_layers,
        mask=pair_mask,
    ).astype(jnp.float32)
    single = row_attention_pooling(
        pair, mask, params, f"{dot}row_attention_pooling"
    )

    at_atoms = gather_token_to_atom(single, atom_to_token)
    intra = jnp.clip(intra_token_index(atom_to_token), max=MAX_ATOMS_PER_TOKEN - 1)

    plddt_logits = jnp.einsum(
        "bac,bacn->ban",
        layer_norm(
            at_atoms,
            params[f"{dot}plddt_ln.weight"],
            params[f"{dot}plddt_ln.bias"],
        ),
        params[f"{dot}plddt_weight"][intra],
    )
    # Bins run 0 to 1. Upstream never scales them to 100 and neither does this.
    plddt_per_atom = categorical_mean(plddt_logits, 0.0, 1.0)

    weighted = plddt_per_atom * atom_mask
    plddt = _scatter_sum(
        weighted,
        atom_to_token,
        n_tokens,
        atom_mask,
        contiguous_atom_groups=contiguous_atom_groups,
    ) / jnp.clip(
        _scatter_sum(
            atom_mask,
            atom_to_token,
            n_tokens,
            atom_mask,
            contiguous_atom_groups=contiguous_atom_groups,
        ),
        min=1e-6,
    )
    complex_plddt = jnp.sum(weighted, axis=-1) / (jnp.sum(atom_mask, axis=-1) + EPS)

    is_ligand = (mol_type == NONPOLYMER_ID).astype(jnp.float32)
    inter_chain = (asym_id[:, :, None] != asym_id[:, None, :]).astype(jnp.float32)
    near_contact = (rep_distances < 8).astype(jnp.float32)
    interface = jnp.max(
        near_contact * inter_chain * (1.0 - is_ligand)[..., None], axis=-1
    )
    interface_weight = jnp.where(is_ligand > 0, 2.0, interface)
    atom_interface = atom_mask * gather_token_to_atom(
        interface_weight[..., None], atom_to_token
    )[..., 0]
    complex_iplddt = jnp.sum(plddt_per_atom * atom_interface, axis=-1) / (
        jnp.sum(atom_interface, axis=-1) + EPS
    )
    plddt_ca = jnp.take_along_axis(plddt_per_atom, rep_idx, axis=1)

    pae_logits = linear(
        layer_norm(
            pair, params[f"{dot}pae_ln.weight"], params[f"{dot}pae_ln.bias"]
        ),
        params,
        f"{dot}pae_head",
    )
    pde_logits = linear(
        layer_norm(
            pair, params[f"{dot}pde_ln.weight"], params[f"{dot}pde_ln.bias"]
        ),
        params,
        f"{dot}pde_head",
    )
    resolved_logits = jnp.einsum(
        "bac,bacn->ban",
        layer_norm(
            at_atoms,
            params[f"{dot}resolved_ln.weight"],
            params[f"{dot}resolved_ln.bias"],
        ),
        params[f"{dot}resolved_weight"][intra],
    )

    # pTM and ipTM come out of the PAE distribution; there is no TM head.
    n_bins = pae_logits.shape[-1]
    bin_width = 32.0 / n_bins
    centres = jnp.arange(0.5 * bin_width, 32.0, bin_width, dtype=jnp.float32)
    n_res = jnp.sum(mask, axis=-1, keepdims=True)
    d0 = 1.24 * (jnp.clip(n_res, min=19.0) - 15) ** (1 / 3) - 1.8
    tm_per_bin = 1.0 / (1.0 + (centres / d0) ** 2)
    tm_expected = jnp.sum(
        jax.nn.softmax(pae_logits, axis=-1) * tm_per_bin[:, None, None, :], axis=-1
    )

    ptm = jnp.max(
        jnp.sum(tm_expected * pair_mask, axis=-1)
        / (jnp.sum(pair_mask, axis=-1) + EPS),
        axis=-1,
    )
    inter_chain_mask = inter_chain * pair_mask
    iptm = jnp.max(
        jnp.sum(tm_expected * inter_chain_mask, axis=-1)
        / (jnp.sum(inter_chain_mask, axis=-1) + EPS),
        axis=-1,
    )

    chains = jax.nn.one_hot(asym_id, n_chains, dtype=jnp.float32) * mask[..., None]
    chains = jnp.swapaxes(chains, 1, 2)
    numerator = jnp.einsum("bij,bci,bdj->bcd", tm_expected, chains, chains)
    denominator = jnp.einsum("bci,bdj->bcd", chains, chains) + EPS
    pair_chains_iptm = numerator / denominator

    return {
        "plddt_logits": plddt_logits,
        "plddt": plddt,
        "plddt_per_atom": plddt_per_atom,
        "plddt_ca": plddt_ca,
        "complex_plddt": complex_plddt,
        "complex_iplddt": complex_iplddt,
        "pae_logits": pae_logits,
        "pae": categorical_mean(pae_logits, 0.0, 32.0),
        "pde_logits": pde_logits,
        "pde": categorical_mean(pde_logits, 0.0, 32.0),
        "resolved_logits": resolved_logits,
        "ptm": ptm,
        "iptm": iptm,
        "pair_chains_iptm": pair_chains_iptm,
    }


__all__ = [
    "categorical_mean",
    "confidence_head",
    "gather_token_to_atom",
    "intra_token_index",
]
