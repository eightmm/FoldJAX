"""The scanned block stacks match the unrolled ones.

Three stacks here run their blocks as a ``lax.scan`` over stacked parameters rather
than emitting one copy of the block body per block, matching what the trunk's
Pairformer stack already did. The transformation is a scheduling change, not a
mathematical one, so what has to be checked is that the two paths agree -- and that
the scan is actually taken, since a guard that never fires is dead code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai.models.diffusion_transformer import (
    HEAD_DIM,
    NUM_HEADS,
    DiffusionTransformerBlockParams,
    DiffusionTransformerParams,
    diffusion_transformer_stack,
)

# Head count and width are fixed by the architecture, so the channel width follows
# from them rather than being picked here.
TOKENS = 12
SAMPLES = 2
CHANNELS = NUM_HEADS * HEAD_DIM
PAIR = 8


def _block(rng: np.random.Generator) -> DiffusionTransformerBlockParams:
    # Small enough that six blocks keep activations O(1): at scale 0.1 they reach
    # ~800, where one float32 ulp is ~1e-4 and a tight comparison stops being a
    # statement about the scan.
    def array(*shape, scale: float = 0.01):
        return jnp.asarray(rng.normal(size=shape).astype(np.float32) * scale)

    # The conditioned norms project cond -> 2*C and split it into scale and shift,
    # so those weights are matrices rather than vectors. The transition widens to
    # 2*HIDDEN and splits into value and gate.
    hidden = CHANNELS * 2
    return DiffusionTransformerBlockParams(
        q_bias=array(NUM_HEADS * HEAD_DIM),
        transition_norm_weight=array(CHANNELS * 2, CHANNELS),
        transition_a_weight=array(hidden * 2, CHANNELS),
        transition_b_weight=array(CHANNELS, hidden),
        transition_gate_weight=array(CHANNELS, CHANNELS),
        transition_gate_bias=array(CHANNELS),
        attention_norm_weight=array(CHANNELS * 2, CHANNELS),
        qkv_weight=array(NUM_HEADS * HEAD_DIM * 3, CHANNELS),
        attention_gate_weight=array(CHANNELS, CHANNELS),
        attention_gate_bias=array(CHANNELS),
        pair_norm_weight=array(PAIR),
        pair_norm_bias=array(PAIR),
        pair_linear_weight=array(NUM_HEADS, PAIR),
        output_weight=array(CHANNELS, NUM_HEADS * HEAD_DIM),
    )


@pytest.fixture
def case():
    rng = np.random.default_rng(0)
    params = DiffusionTransformerParams(blocks=tuple(_block(rng) for _ in range(6)))
    # single is [batch, samples, tokens, channels] here.
    single = jnp.asarray(
        rng.normal(size=(1, SAMPLES, TOKENS, CHANNELS)).astype(np.float32)
    )
    cond = jnp.asarray(
        rng.normal(size=(1, SAMPLES, TOKENS, CHANNELS)).astype(np.float32)
    )
    pair = jnp.asarray(rng.normal(size=(1, TOKENS, TOKENS, PAIR)).astype(np.float32))
    mask = jnp.ones((1, TOKENS), dtype=bool)
    return params, single, cond, pair, mask


def test_scan_matches_the_loop(case) -> None:
    params, single, cond, pair, mask = case
    scanned = diffusion_transformer_stack(
        single, cond, pair, mask, params, use_scan=True
    )
    looped = diffusion_transformer_stack(
        single, cond, pair, mask, params, use_scan=False
    )
    assert scanned.shape == looped.shape
    # A scan and a loop reduce in a different order, so this is float32 noise.
    np.testing.assert_allclose(
        np.asarray(scanned, dtype=np.float64),
        np.asarray(looped, dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_the_stack_is_not_the_identity(case) -> None:
    """Guard: if the blocks did nothing, the comparison above would prove nothing."""
    params, single, cond, pair, mask = case
    out = diffusion_transformer_stack(single, cond, pair, mask, params)
    assert not np.allclose(np.asarray(out), np.asarray(single), atol=1e-4)


def test_a_single_block_stack_still_runs(case) -> None:
    """The scan needs at least two blocks to stack, so one block takes the loop."""
    params, single, cond, pair, mask = case
    one = DiffusionTransformerParams(blocks=params.blocks[:1])
    scanned = diffusion_transformer_stack(single, cond, pair, mask, one, use_scan=True)
    looped = diffusion_transformer_stack(single, cond, pair, mask, one, use_scan=False)
    np.testing.assert_allclose(np.asarray(scanned), np.asarray(looped))


def test_scan_is_jit_compatible(case) -> None:
    params, single, cond, pair, mask = case
    compiled = jax.jit(
        lambda s, c, p, m: diffusion_transformer_stack(s, c, p, m, params)
    )
    np.testing.assert_allclose(
        np.asarray(compiled(single, cond, pair, mask), dtype=np.float64),
        np.asarray(
            diffusion_transformer_stack(single, cond, pair, mask, params),
            dtype=np.float64,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_an_empty_stack_is_refused(case) -> None:
    _params, single, cond, pair, mask = case
    with pytest.raises(ValueError, match="at least one block"):
        diffusion_transformer_stack(
            single, cond, pair, mask, DiffusionTransformerParams(blocks=())
        )
