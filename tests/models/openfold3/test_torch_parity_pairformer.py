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


def test_attention_pair_bias_and_pairformer_bf16_high_precision_match_upstream(
    openfold3_source: Path, monkeypatch
) -> None:
    """Under bf16-mixed, upstream runs PairFormer ``AttentionPairBias`` core in fp32."""
    from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias
    from openfold3.core.model.primitives import attention as of3_attn

    from foldjax.models.openfold3.dtype import narrow_floats

    torch = _torch()

    # On CPU PyTorch disables `torch.amp.autocast("cuda", dtype=torch.float32)`;
    # reproduce CUDA autocast's exact fp32 q/k/v/bias promotion inside `_attention`.
    def cuda_equiv_attention(query, key, value, biases, use_high_precision=True):
        in_dtype = query.dtype
        if use_high_precision:
            q, k, v = query.float(), key.float(), value.float()
            bs = [b.float() for b in biases]
        else:
            q, k, v, bs = query, key, value, biases
        scores = torch.einsum("...qc,...kc->...qk", q, k)
        for bias in bs:
            scores = scores + bias
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=scores.dtype)
        return torch.einsum("...qk,...kc->...qc", probs, v).to(dtype=in_dtype)

    monkeypatch.setattr(of3_attn, "_attention", cuda_equiv_attention)

    apb = AttentionPairBias(
        c_q=8, c_k=8, c_v=8, c_s=8, c_z=4, c_hidden=4, no_heads=2, gating=True, inf=1e9
    )
    with torch.no_grad():
        for param in apb.parameters():
            param.zero_()
        apb.layer_norm_a.weight.fill_(1.0)
        apb.layer_norm_z.weight.fill_(1.0)
        apb.linear_z.weight[:, 0] = 128.0
        apb.mha.linear_q.weight.copy_(torch.eye(8) * 0.8)
        apb.mha.linear_k.weight.copy_(torch.eye(8) * 0.8)
        apb.mha.linear_v.weight.copy_(torch.eye(8))
        apb.mha.linear_o.weight.copy_(torch.eye(8))

    state_fp32 = {k: v.clone() for k, v in apb.state_dict().items()}
    for mod in apb.modules():
        if not isinstance(mod, torch.nn.LayerNorm):
            for _, param in mod.named_parameters(recurse=False):
                param.data = param.data.to(torch.bfloat16)

    a = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ],
        dtype=torch.bfloat16,
    )
    z = torch.tensor(
        [
            [
                [[2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.bfloat16,
    )
    mask = torch.ones(1, 2, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_hi = apb(a=a, z=z, mask=mask, use_high_precision_attention=True)
        ref_lo = apb(a=a, z=z, mask=mask, use_high_precision_attention=False)

    jax_params = narrow_floats(map_attention_pair_bias(state_fp32), jnp.bfloat16)
    a_j = jnp.asarray(a.float().numpy(), jnp.bfloat16)
    z_j = jnp.asarray(z.float().numpy(), jnp.bfloat16)
    m_j = jnp.asarray(mask.float().numpy(), jnp.bfloat16)
    jax_hi = attention_pair_bias(
        a_j, z_j, jax_params, no_heads=2, mask=m_j, high_precision=True
    )
    jax_lo = attention_pair_bias(
        a_j, z_j, jax_params, no_heads=2, mask=m_j, high_precision=False
    )

    assert jax_hi.dtype == jax_lo.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(jax_hi, np.float32), ref_hi.float().numpy(), rtol=0, atol=1e-2
    )
    np.testing.assert_allclose(
        np.asarray(jax_lo, np.float32), ref_lo.float().numpy(), rtol=0, atol=1e-2
    )
    assert (
        np.max(np.abs(np.asarray(jax_lo, np.float32) - ref_hi.float().numpy())) > 0.04
    )

    # Also check a randomized bf16 PairFormerBlock end-to-end.
    blk = _block()
    with torch.no_grad():
        for param in blk.parameters():
            torch.nn.init.normal_(param, std=0.2)
    blk_state = {k: v.clone() for k, v in blk.state_dict().items()}
    for mod in blk.modules():
        if not isinstance(mod, torch.nn.LayerNorm):
            for _, param in mod.named_parameters(recurse=False):
                param.data = param.data.to(torch.bfloat16)
    s_t = torch.randn(1, N, C_S, dtype=torch.bfloat16)
    z_t = torch.randn(1, N, N, C_Z, dtype=torch.bfloat16)
    single_t, pair_t = _masks(torch)
    single_t, pair_t = single_t.to(torch.bfloat16), pair_t.to(torch.bfloat16)
    with torch.no_grad():
        exp_s, exp_z = blk(s=s_t, z=z_t, single_mask=single_t, pair_mask=pair_t)
    blk_params = narrow_floats(map_pairformer_block(blk_state), jnp.bfloat16)
    act_s, act_z = pairformer_block(
        jnp.asarray(s_t.float().numpy(), jnp.bfloat16),
        jnp.asarray(z_t.float().numpy(), jnp.bfloat16),
        blk_params,
        single_mask=jnp.asarray(single_t.float().numpy(), jnp.bfloat16),
        pair_mask=jnp.asarray(pair_t.float().numpy(), jnp.bfloat16),
        no_heads_pair=HEADS_PAIR,
        no_heads_pair_bias=HEADS_PAIR_BIAS,
        high_precision_attention=True,
    )
    assert act_s.dtype == act_z.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(act_s, np.float32), exp_s.float().numpy(), rtol=2e-2, atol=3e-2
    )
    np.testing.assert_allclose(
        np.asarray(act_z, np.float32), exp_z.float().numpy(), rtol=2e-2, atol=3e-2
    )
