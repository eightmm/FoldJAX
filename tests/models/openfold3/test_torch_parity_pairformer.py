"""Torch-vs-JAX parity for AttentionPairBias, the Pairformer block, and the stack."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import (
    map_attention_pair_bias,
    map_pairformer_block,
    map_pairformer_stack,
)
from foldjax.models.openfold3.models.attention_pair_bias import attention_pair_bias
from foldjax.models.openfold3.models.pairformer import (
    pairformer_block,
    pairformer_stack,
)

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_S, C_Z, N = 10, 8, 5
HEADS_PAIR_BIAS, C_HIDDEN_PAIR_BIAS = 2, 4
HEADS_PAIR, C_HIDDEN_MUL, C_HIDDEN_PAIR_ATT = 2, 6, 4


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _assert_close(actual: jnp.ndarray, expected, name: str) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected.detach().numpy(), dtype=np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{name} diverged from the OpenFold3 reference",
    )


def _apb():
    from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias

    return AttentionPairBias(
        c_q=C_S,
        c_k=C_S,
        c_v=C_S,
        c_s=C_S,
        c_z=C_Z,
        c_hidden=C_HIDDEN_PAIR_BIAS,
        no_heads=HEADS_PAIR_BIAS,
        gating=True,
        inf=1e9,
    )


def _block_kwargs() -> dict:
    return {
        "c_s": C_S,
        "c_z": C_Z,
        "c_hidden_pair_bias": C_HIDDEN_PAIR_BIAS,
        "no_heads_pair_bias": HEADS_PAIR_BIAS,
        "c_hidden_mul": C_HIDDEN_MUL,
        "c_hidden_pair_att": C_HIDDEN_PAIR_ATT,
        "no_heads_pair": HEADS_PAIR,
        "transition_type": "swiglu",
        "transition_n": 2,
        "pair_dropout": 0.0,
        "fuse_projection_weights": False,
        "inf": 1e9,
    }


def _block():
    from openfold3.core.model.latent.pairformer import PairFormerBlock

    return PairFormerBlock(**_block_kwargs())


def _stack(no_blocks: int):
    from openfold3.core.model.latent.pairformer import PairFormerStack

    return PairFormerStack(
        no_blocks=no_blocks, blocks_per_ckpt=None, **_block_kwargs()
    )


def _masks(torch):
    single = torch.ones(1, N)
    single[:, 4:] = 0.0
    pair = single[..., None] * single[..., None, :]
    return single, pair


def _jax_call(fn, s, z, single, pair, params, **extra):
    return fn(
        jnp.asarray(s.numpy()),
        jnp.asarray(z.numpy()),
        params,
        single_mask=jnp.asarray(single.numpy()),
        pair_mask=jnp.asarray(pair.numpy()),
        no_heads_pair=HEADS_PAIR,
        no_heads_pair_bias=HEADS_PAIR_BIAS,
        **extra,
    )


def test_attention_pair_bias_matches_torch(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_apb())
    a = torch.randn(1, N, C_S)
    z = torch.randn(1, N, N, C_Z)
    single, _pair = _masks(torch)
    with torch.no_grad():
        expected = module(a=a, z=z, mask=single)
    actual = attention_pair_bias(
        jnp.asarray(a.numpy()),
        jnp.asarray(z.numpy()),
        map_attention_pair_bias(dict(module.state_dict())),
        no_heads=HEADS_PAIR_BIAS,
        mask=jnp.asarray(single.numpy()),
    )
    _assert_close(actual, expected, "AttentionPairBias")


def test_attention_pair_bias_query_projection_has_a_bias(
    openfold3_source: Path,
) -> None:
    """Unlike every other Attention in the model, this one's linear_q has a bias."""
    _torch()
    state = dict(_apb().state_dict())
    assert "mha.linear_q.bias" in state
    assert "mha.linear_k.bias" not in state
    # The strict mapper must reject the bias-free spelling for this layer.
    from foldjax.models.openfold3.bridge.torch_mapping import map_attention

    with pytest.raises(KeyError, match="unexpected bias"):
        map_attention(state, "mha", q_bias=False)


def test_pairformer_block_matches_torch(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    module = randomized(_block())
    s = torch.randn(1, N, C_S)
    z = torch.randn(1, N, N, C_Z)
    single, pair = _masks(torch)
    with torch.no_grad():
        expected_s, expected_z = module(
            s=s, z=z, single_mask=single, pair_mask=pair
        )
    actual_s, actual_z = _jax_call(
        pairformer_block, s, z, single, pair, map_pairformer_block(
            dict(module.state_dict())
        )
    )
    _assert_close(actual_s, expected_s, "PairFormerBlock(s)")
    _assert_close(actual_z, expected_z, "PairFormerBlock(z)")


def test_pairformer_block_single_update_sees_the_updated_pair(
    openfold3_source: Path, randomized
) -> None:
    """The pair stack runs before the single update; order is observable."""
    torch = _torch()
    module = randomized(_block())
    s = torch.randn(1, N, C_S)
    z = torch.randn(1, N, N, C_Z)
    single, pair = _masks(torch)
    params = map_pairformer_block(dict(module.state_dict()))
    actual_s, actual_z = _jax_call(pairformer_block, s, z, single, pair, params)

    # Feeding the ORIGINAL z to the single update must give a different s.
    from foldjax.models.openfold3.models.attention_pair_bias import (
        attention_pair_bias as apb,
    )
    from foldjax.models.openfold3.models.primitives import swiglu_transition

    wrong = jnp.asarray(s.numpy()) + apb(
        jnp.asarray(s.numpy()),
        jnp.asarray(z.numpy()),
        params.attn_pair_bias,
        no_heads=HEADS_PAIR_BIAS,
        mask=jnp.asarray(single.numpy()),
    )
    wrong = wrong + swiglu_transition(
        wrong, params.single_transition, mask=jnp.asarray(single.numpy())
    )
    assert not np.allclose(np.asarray(actual_s), np.asarray(wrong), rtol=1e-3)
    assert actual_z.shape == (1, N, N, C_Z)


@pytest.mark.parametrize("no_blocks", [1, 3])
def test_pairformer_stack_matches_torch(
    openfold3_source: Path, randomized, no_blocks: int
) -> None:
    torch = _torch()
    module = randomized(_stack(no_blocks))
    s = torch.randn(1, N, C_S)
    z = torch.randn(1, N, N, C_Z)
    single, pair = _masks(torch)
    with torch.no_grad():
        expected_s, expected_z = module(
            s=s, z=z, single_mask=single, pair_mask=pair, chunk_size=None
        )
    params = map_pairformer_stack(dict(module.state_dict()))
    assert len(params.blocks) == no_blocks
    actual_s, actual_z = _jax_call(pairformer_stack, s, z, single, pair, params)
    _assert_close(actual_s, expected_s, f"PairFormerStack({no_blocks})(s)")
    _assert_close(actual_z, expected_z, f"PairFormerStack({no_blocks})(z)")


def test_stack_mapper_rejects_a_missing_block(openfold3_source: Path) -> None:
    """A dropped block must fail, not silently shorten the stack."""
    _torch()
    state = {
        key: value
        for key, value in _stack(3).state_dict().items()
        if not key.startswith("blocks.1.")
    }
    with pytest.raises(KeyError, match="non-contiguous"):
        map_pairformer_stack(state)


def test_stack_mapper_reports_an_empty_stack(openfold3_source: Path) -> None:
    with pytest.raises(KeyError, match="no pairformer blocks"):
        map_pairformer_stack({})


def test_pairformer_block_state_dict_layout(openfold3_source: Path) -> None:
    _torch()
    prefixes = {key.split(".")[0] for key in _block().state_dict()}
    assert prefixes == {"pair_stack", "attn_pair_bias", "single_transition"}
