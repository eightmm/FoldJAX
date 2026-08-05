"""The trunk with recycling (AF3 Algorithm 1, lines 1-10).

Runs the input embedder once, then loops: recycle the pair and single
representations through their own projections, fold in the MSA (and optionally
templates), and run the Pairformer.

The recycling projections are the part worth stating explicitly. Each cycle does
``z = z_init + linear_z(layer_norm_z(z))`` — the *initial* embedding plus a
projection of the previous cycle's output, not an accumulation. Starting from
zeros makes the first cycle equal to ``z_init`` plus a projection of zeros, which
is not the same as skipping the projection.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.openfold3.models.input_embedders import (
    InputEmbedderParams,
    MSAEmbedderParams,
    input_embedder,
    msa_embedder,
)
from foldjax.models.openfold3.models.msa_module import (
    MSAModuleStackParams,
    msa_module_stack,
)
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
from foldjax.models.openfold3.models.template_module import (
    TemplateEmbedderParams,
    template_embedder,
)


class TrunkParams(NamedTuple):
    """Parameters for the trunk.

    ``layer_norm_z``/``linear_z`` and ``layer_norm_s``/``linear_s`` are the
    recycling projections; they are applied once per cycle, not per block.
    """

    input_embedder: InputEmbedderParams
    msa_module_embedder: MSAEmbedderParams
    msa_module: MSAModuleStackParams
    pairformer_stack: PairformerStackParams
    layer_norm_z: LayerNormParams
    linear_z: LinearParams
    layer_norm_s: LayerNormParams
    linear_s: LinearParams
    # Absent when the checkpoint has no template tower; the trunk then skips it.
    template_embedder: TemplateEmbedderParams | None = None


def trunk(
    batch: Mapping[str, jnp.ndarray],
    params: TrunkParams,
    *,
    num_cycles: int,
    n_query: int,
    n_key: int,
    atom_heads: int,
    n_token: int,
    max_relative_idx: int,
    max_relative_chain: int,
    no_heads_msa: int,
    no_heads_pair: int,
    no_heads_pair_bias: int,
    opm_first: bool = True,
    inf: float = 1e9,
    eps: float = 1e-5,
    chunk_size: int | None = None,
    scan_blocks: bool = True,
    scan_cycles: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the trunk for ``num_cycles`` recycling iterations.

    Args:
        batch: features for the input embedder, MSA embedder and masks.
        params: mapped trunk parameters.
        num_cycles: recycling iterations; must be at least one.
        n_query: atom query block height.
        n_key: atom key window width.
        atom_heads: atom transformer head count.
        n_token: static token count.
        max_relative_idx: relpos clip for residue/token offsets.
        max_relative_chain: relpos clip for chain offsets.
        no_heads_msa: MSA pair-weighted-averaging head count.
        no_heads_pair: triangle attention head count.
        no_heads_pair_bias: Pairformer single-attention head count.
        opm_first: MSA module outer-product-mean ordering.
        inf: masking constant.
        eps: layer norm epsilon.
        chunk_size: rows of the pair representation to process at a time. Caps
            triangle attention's ``[rows, heads, N, N]`` scores and the pair
            transition's widened activation, which together are the trunk's peak
            (models/row_chunking.py).
        scan_blocks: run the Pairformer blocks as a scan rather than unrolling.
        scan_cycles: run recycling as one loop rather than emitting the body once
            per cycle. Same arithmetic; a quarter of the graph at four cycles.

    Returns:
        ``(s_input, s, z)``.
    """
    if num_cycles < 1:
        raise ValueError("num_cycles must be at least 1")

    s_input, s_init, z_init = input_embedder(
        batch,
        params.input_embedder,
        n_query=n_query,
        n_key=n_key,
        atom_heads=atom_heads,
        n_token=n_token,
        max_relative_idx=max_relative_idx,
        max_relative_chain=max_relative_chain,
        inf=inf,
        eps=eps,
    )

    s = jnp.zeros_like(s_init)
    z = jnp.zeros_like(z_init)

    token_mask = batch["token_mask"]
    pair_mask = token_mask[..., :, None] * token_mask[..., None, :]

    # The MSA embedding reads only the features and the input embedding, neither of
    # which a cycle changes, so it is the same tensor every cycle. It used to be
    # rebuilt inside the loop, which at the released four cycles meant three
    # redundant evaluations. (A port that resampled the alignment per cycle -- some
    # do -- could not hoist this.)
    m, msa_mask = msa_embedder(batch, s_input, params.msa_module_embedder)

    def cycle(
        carry: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        s, z = carry
        # Initial embedding plus a projection of the previous cycle, not a sum.
        z = z_init + linear(
            layer_norm(z, params.layer_norm_z, eps=eps), params.linear_z
        )

        # Templates fold into z between the recycling projection and the MSA
        # module, matching upstream's run_trunk ordering.
        if params.template_embedder is not None:
            z = z + template_embedder(
                batch,
                z,
                params.template_embedder,
                pair_mask=pair_mask,
                no_heads=no_heads_pair,
                inf=inf,
                eps=eps,
                chunk_size=chunk_size,
            )

        z = msa_module_stack(
            m,
            z,
            params.msa_module,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            opm_first=opm_first,
            inf=inf,
            eps=eps,
            chunk_size=chunk_size,
        )

        s = s_init + linear(
            layer_norm(s, params.layer_norm_s, eps=eps), params.linear_s
        )
        return pairformer_stack(
            s,
            z,
            params.pairformer_stack,
            single_mask=token_mask,
            pair_mask=pair_mask,
            no_heads_pair=no_heads_pair,
            no_heads_pair_bias=no_heads_pair_bias,
            inf=inf,
            eps=eps,
            chunk_size=chunk_size,
            scan_blocks=scan_blocks,
        )

    # Every cycle is the same computation on a different carry -- same weights, same
    # features -- so a loop is exact and emits the body once instead of
    # ``num_cycles`` times. That matters here because the body contains the 48-block
    # Pairformer and the template stack: unrolled, the released four cycles produce
    # a graph four times larger, and compile time follows it.
    if scan_cycles and num_cycles > 1:
        s, z = jax.lax.fori_loop(0, num_cycles, lambda _i, carry: cycle(carry), (s, z))
    else:
        for _cycle in range(num_cycles):
            s, z = cycle((s, z))

    return s_input, s, z
