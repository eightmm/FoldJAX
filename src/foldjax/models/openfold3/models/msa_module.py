"""MSA module block and stack (AF3 Algorithm 8, lines 5-15).

Two upstream details drive the shape of this port:

* ``opm_first`` moves the outer product mean from after the MSA update to before
  it, which changes which ``m`` the pair update sees.
* The last block sets ``skip_msa_update = last_block and opm_first``, and when it
  does, ``msa_att_row`` and ``msa_transition`` are ``None`` and absent from the
  checkpoint. Its MSA output is therefore passed through untouched.

The AF2-only MSA attention variants (row attention with pair bias, column
attention, column global attention) are not part of this path: AF3 replaces them
with ``MSAPairWeightedAveraging``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.openfold3.models.msa import (
    MSAPairWeightedAveragingParams,
    OuterProductMeanParams,
    msa_pair_weighted_averaging,
    outer_product_mean,
)
from foldjax.models.openfold3.models.pair_block import PairBlockParams, pair_block
from foldjax.models.openfold3.models.primitives import (
    SwiGLUTransitionParams,
    swiglu_transition,
)
from foldjax.models.openfold3.models.stacking import can_scan, scan_stack


class MSAModuleBlockParams(NamedTuple):
    """Parameters for one ``MSAModuleBlock``.

    ``msa_att_row`` and ``msa_transition`` are ``None`` for a block that skips
    the MSA update, mirroring the absent checkpoint entries.
    """

    outer_product_mean: OuterProductMeanParams
    pair_stack: PairBlockParams
    msa_att_row: MSAPairWeightedAveragingParams | None = None
    msa_transition: SwiGLUTransitionParams | None = None


class MSAModuleStackParams(NamedTuple):
    """Parameters for an ``MSAModuleStack``: one entry per block, in order."""

    blocks: tuple[MSAModuleBlockParams, ...]


def msa_module_block(
    m: jnp.ndarray,
    z: jnp.ndarray,
    params: MSAModuleBlockParams,
    *,
    msa_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    no_heads_msa: int,
    no_heads_pair: int,
    opm_first: bool = True,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
    chunk_size: int | None = None,
    glu_backend: str = "xla",
    opm_eps: float = 1e-3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply one MSA module block.

    Args:
        m: ``[..., N_seq, N_token, C_m]`` MSA embedding.
        z: ``[..., N_token, N_token, C_z]`` pair embedding.
        params: mapped block parameters.
        msa_mask: ``[..., N_seq, N_token]`` MSA mask.
        pair_mask: ``[..., N_token, N_token]`` pair mask.
        no_heads_msa: head count for pair-weighted averaging.
        no_heads_pair: head count for triangle attention in the pair stack.
        opm_first: run the outer product mean before the MSA update.
        inf: masking constant.
        mask_transition: upstream's ``_mask_trans``.
        eps: layer norm epsilon.
        opm_eps: outer product mean normalization epsilon.

    Returns:
        ``(m, z)`` updated representations.
    """
    if opm_first:
        z = z + outer_product_mean(
            m,
            params.outer_product_mean,
            mask=msa_mask,
            eps=opm_eps,
            layer_norm_eps=eps,
            chunk_size=chunk_size,
        )

    if params.msa_att_row is not None:
        if params.msa_transition is None:
            raise ValueError(
                "a block with msa_att_row must also carry msa_transition"
            )
        m = m + msa_pair_weighted_averaging(
            m,
            z,
            params.msa_att_row,
            no_heads=no_heads_msa,
            mask=pair_mask,
            inf=inf,
            eps=eps,
        )
        m = m + swiglu_transition(
            m,
            params.msa_transition,
            mask=msa_mask if mask_transition else None,
            eps=eps,
            glu_backend=glu_backend,
        )
        if not opm_first:
            z = z + outer_product_mean(
                m,
                params.outer_product_mean,
                mask=msa_mask,
                eps=opm_eps,
                layer_norm_eps=eps,
                chunk_size=chunk_size,
            )

    z = pair_block(
        z,
        params.pair_stack,
        pair_mask=pair_mask,
        no_heads_pair=no_heads_pair,
        inf=inf,
        mask_transition=mask_transition,
        eps=eps,
        chunk_size=chunk_size,
        glu_backend=glu_backend,
    )
    return m, z


def _scannable_prefix(body, carry, blocks) -> int:
    """How many leading ``blocks`` one scan can run with the unrolled loop's
    arithmetic: every parameter leaf a strongly typed array with the first
    block's dtype, and the carry keeping its shape, dtype and weak type through
    each block. ``can_scan`` checks only structure and leaf shapes; stacking
    would promote mixed or weak dtypes, and a scan carry cannot widen or gain an
    axis as the unrolled loop lets ``m`` and ``z`` do. Stops at the first block
    that fails, so later blocks are never probed with a carry they would not
    have received."""

    def strong_array(x):
        return isinstance(x, (jax.Array, np.ndarray)) and not getattr(
            x, "weak_type", False
        )

    def spec(x):
        return (jnp.shape(x), jnp.result_type(x), getattr(x, "weak_type", False))

    if not all(strong_array(c) for c in carry):
        return 0
    first = None
    count = 0
    for block in blocks:
        leaves = jax.tree.leaves(block)
        if not all(strong_array(leaf) for leaf in leaves):
            break
        dtypes = [leaf.dtype for leaf in leaves]
        if first is None:
            first = dtypes
        elif dtypes != first:
            break
        out = jax.eval_shape(body, carry, block)
        if any(
            (o.shape, o.dtype, o.weak_type) != spec(c)
            for o, c in zip(out, carry, strict=True)
        ):
            break
        count += 1
    return count


def msa_module_stack(
    m: jnp.ndarray,
    z: jnp.ndarray,
    params: MSAModuleStackParams,
    *,
    msa_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    no_heads_msa: int,
    no_heads_pair: int,
    opm_first: bool = True,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
    chunk_size: int | None = None,
    glu_backend: str = "xla",
    opm_eps: float = 1e-3,
    scan_prefix: bool = False,
) -> jnp.ndarray:
    """Run every MSA module block and return the pair representation.

    Upstream's ``_wrap_up`` discards the MSA embedding, so only ``z`` escapes the
    stack. ``scan_prefix`` lets a stack that cannot be scanned whole scan its
    longest homogeneous prefix; :func:`trunk_cycle` sets it, the path whose
    memory and outputs were measured with it.
    """
    settings = dict(
        msa_mask=msa_mask,
        pair_mask=pair_mask,
        no_heads_msa=no_heads_msa,
        no_heads_pair=no_heads_pair,
        opm_first=opm_first,
        inf=inf,
        mask_transition=mask_transition,
        eps=eps,
        chunk_size=chunk_size,
        glu_backend=glu_backend,
        opm_eps=opm_eps,
    )

    def body(carry, block):
        return msa_module_block(carry[0], carry[1], block, **settings)

    # A block configured with ``skip_msa_update`` has no msa_att_row at all, so a
    # stack that mixes the two cannot be scanned; can_scan detects exactly that.
    if can_scan(params.blocks):
        _, z = scan_stack(body, (m, z), params.blocks)
        return z
    # With scan_prefix, scan the longest stackable prefix and unroll only the
    # rest. Fully unrolled at the top level of the padded route's standalone
    # trunk cycle, the blocks' row-chunk output buffers were allocated together
    # at program start and the later ones held idle (about 22 GiB of a 61 GiB
    # arena at 4,352 tokens).
    scanned = 0
    if scan_prefix:
        count = len(params.blocks) - 1
        while count > 1 and not can_scan(params.blocks[:count]):
            count -= 1
        if count > 1:
            count = _scannable_prefix(body, (m, z), params.blocks[:count])
        scanned = count if count > 1 else 0
    if scanned:
        m, z = scan_stack(body, (m, z), params.blocks[:scanned])
    for block in params.blocks[scanned:]:
        m, z = msa_module_block(m, z, block, **settings)
    return z
