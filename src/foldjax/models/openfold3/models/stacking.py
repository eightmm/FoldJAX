"""Run a stack of identically-shaped blocks as one scanned body.

The released trunk is 48 Pairformer blocks, 24 diffusion transformer blocks, 4 MSA
blocks and 2 template blocks. Unrolling them puts that many copies of each body in
one HLO module, and XLA then plans buffers across the whole thing and compiles it
piece by piece. Measured on a 574-token target (4HHB), for the trunk alone:

* temporary memory 46.46 GiB unrolled, 24.37 GiB scanned -- and the unrolled
  version could not run at all on a 96 GiB card;
* compile time 456 s unrolled, 108 s scanned.

The arithmetic is identical; only the schedule changes.

This is also why chunking the triangle attention did nothing (see
``triangle_attention``): the peak was set by the unrolled graph, not by the
attention logits; those are what ``chunk_size`` bounds instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from foldjax.models._stacking import StackedLayers, stacked_or_stack


def can_scan(blocks: Sequence[Any]) -> bool:
    """Whether ``blocks`` can be stacked into one scanned body.

    Two blocks are stackable only if their parameter trees have the same structure
    *and* matching leaf shapes. Both fail in practice: an MSA block configured with
    ``skip_msa_update`` has no ``msa_att_row`` at all, so its tree differs, and
    stacking mismatched shapes would raise deep inside ``jnp.stack`` with no
    indication of which block was the odd one.
    """
    if len(blocks) < 2:
        return False
    if isinstance(blocks, StackedLayers):
        # The representation already owns one pytree with a leading layer
        # axis. Avoid rebuilding 48 layer views merely to inspect it again at
        # trace time.
        return True
    reference = jax.tree.structure(blocks[0])
    shapes = [jnp.shape(leaf) for leaf in jax.tree.leaves(blocks[0])]
    for block in blocks[1:]:
        if jax.tree.structure(block) != reference:
            return False
        if [jnp.shape(leaf) for leaf in jax.tree.leaves(block)] != shapes:
            return False
    return True


def scan_stack[Carry](
    body: Callable[[Carry, Any], Carry],
    carry: Carry,
    blocks: Sequence[Any],
) -> Carry:
    """Apply ``body`` once per block, as a scan over stacked parameters.

    Args:
        body: ``(carry, block_params) -> carry``.
        carry: the initial carry; its structure must be preserved by ``body``.
        blocks: per-block parameters, all of the same structure and shapes.

    Returns:
        The final carry.

    Raises:
        ValueError: the blocks are not stackable. Callers check :func:`can_scan`
            first and fall back to a loop, so reaching this means the check was
            skipped.
    """
    if not can_scan(blocks):
        raise ValueError(
            "blocks are not stackable: their parameter trees or leaf shapes differ"
        )
    stacked = stacked_or_stack(blocks)

    def scanned(state: Carry, block: Any) -> tuple[Carry, None]:
        return body(state, block), None

    result, _ = jax.lax.scan(scanned, carry, stacked)
    return result
