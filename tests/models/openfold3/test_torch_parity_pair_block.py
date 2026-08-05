"""Torch-vs-JAX parity for triangle attention and the pair stack block."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import (
    map_pair_block,
    map_triangle_attention,
)
from foldjax.models.openfold3.models.pair_block import pair_block
from foldjax.models.openfold3.models.triangle_attention import triangle_attention

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_Z, C_HIDDEN_MUL, C_HIDDEN_ATT, HEADS, N = 8, 6, 4, 2, 5


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


def _tri_att(starting: bool):
    from openfold3.core.model.layers.triangular_attention import (
        TriangleAttentionEndingNode,
        TriangleAttentionStartingNode,
    )

    cls = TriangleAttentionStartingNode if starting else TriangleAttentionEndingNode
    return cls(c_in=C_Z, c_hidden=C_HIDDEN_ATT, no_heads=HEADS)


def _pair_block():
    from openfold3.core.model.latent.base_blocks import PairBlock

    return PairBlock(
        c_z=C_Z,
        c_hidden_mul=C_HIDDEN_MUL,
        c_hidden_pair_att=C_HIDDEN_ATT,
        no_heads_pair=HEADS,
        transition_type="swiglu",
        transition_n=2,
        pair_dropout=0.0,
        fuse_projection_weights=False,
        inf=1e9,
    )


def _pair_mask(torch):
    mask = torch.ones(1, N, N)
    mask[:, 4:, :] = 0.0
    mask[:, :, 4:] = 0.0
    return mask


@pytest.mark.parametrize("starting", [True, False])
def test_triangle_attention_matches_torch(
    openfold3_source: Path, randomized, starting: bool
) -> None:
    torch = _torch()
    module = randomized(_tri_att(starting))
    x = torch.randn(1, N, N, C_Z)
    with torch.no_grad():
        expected = module(x)
    actual = triangle_attention(
        jnp.asarray(x.numpy()),
        map_triangle_attention(dict(module.state_dict())),
        no_heads=HEADS,
        starting=starting,
    )
    _assert_close(actual, expected, f"TriangleAttention(starting={starting})")


@pytest.mark.parametrize("starting", [True, False])
def test_triangle_attention_matches_torch_with_mask(
    openfold3_source: Path, randomized, starting: bool
) -> None:
    torch = _torch()
    module = randomized(_tri_att(starting))
    x = torch.randn(1, N, N, C_Z)
    mask = _pair_mask(torch)
    with torch.no_grad():
        expected = module(x, mask=mask)
    actual = triangle_attention(
        jnp.asarray(x.numpy()),
        map_triangle_attention(dict(module.state_dict())),
        no_heads=HEADS,
        mask=jnp.asarray(mask.numpy()),
        starting=starting,
    )
    _assert_close(actual, expected, f"TriangleAttention(masked,{starting})")


def test_starting_and_ending_node_actually_differ(
    openfold3_source: Path, randomized
) -> None:
    """A dropped transpose would let the starting-node test cover both."""
    torch = _torch()
    module = randomized(_tri_att(starting=True))
    params = map_triangle_attention(dict(module.state_dict()))
    x = jnp.asarray(torch.randn(1, N, N, C_Z).numpy())
    start = triangle_attention(x, params, no_heads=HEADS, starting=True)
    end = triangle_attention(x, params, no_heads=HEADS, starting=False)
    assert not np.allclose(np.asarray(start), np.asarray(end), rtol=1e-3, atol=1e-3)


def test_pair_block_matches_torch(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    module = randomized(_pair_block())
    z = torch.randn(1, N, N, C_Z)
    mask = _pair_mask(torch)
    with torch.no_grad():
        expected = module(z, pair_mask=mask)
    actual = pair_block(
        jnp.asarray(z.numpy()),
        map_pair_block(dict(module.state_dict())),
        pair_mask=jnp.asarray(mask.numpy()),
        no_heads_pair=HEADS,
    )
    _assert_close(actual, expected, "PairBlock")


def test_pair_block_matches_torch_unmasked(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    module = randomized(_pair_block())
    z = torch.randn(1, N, N, C_Z)
    mask = torch.ones(1, N, N)
    with torch.no_grad():
        expected = module(z, pair_mask=mask)
    actual = pair_block(
        jnp.asarray(z.numpy()),
        map_pair_block(dict(module.state_dict())),
        pair_mask=jnp.asarray(mask.numpy()),
        no_heads_pair=HEADS,
    )
    _assert_close(actual, expected, "PairBlock(unmasked)")


def test_pair_block_respects_mask_transition_flag(
    openfold3_source: Path, randomized
) -> None:
    """``_mask_trans=False`` is a real upstream option; it must change the result."""
    torch = _torch()
    module = randomized(_pair_block())
    z = torch.randn(1, N, N, C_Z)
    mask = _pair_mask(torch)
    with torch.no_grad():
        expected = module(z, pair_mask=mask, _mask_trans=False)
    actual = pair_block(
        jnp.asarray(z.numpy()),
        map_pair_block(dict(module.state_dict())),
        pair_mask=jnp.asarray(mask.numpy()),
        no_heads_pair=HEADS,
        mask_transition=False,
    )
    _assert_close(actual, expected, "PairBlock(_mask_trans=False)")


def test_pair_block_state_dict_layout(openfold3_source: Path) -> None:
    """Pin the sub-module prefixes the block mapper walks."""
    _torch()
    prefixes = {key.split(".")[0] for key in _pair_block().state_dict()}
    assert prefixes == {
        "tri_mul_out",
        "tri_mul_in",
        "tri_att_start",
        "tri_att_end",
        "pair_transition",
    }


def test_pair_block_end_attention_is_a_starting_node(openfold3_source: Path) -> None:
    """Upstream builds tri_att_end with starting=True and transposes around it."""
    _torch()
    block = _pair_block()
    assert block.tri_att_start.starting is True
    assert block.tri_att_end.starting is True


def test_mapper_reports_a_missing_nested_key(openfold3_source: Path) -> None:
    _torch()
    state = dict(_pair_block().state_dict())
    del state["tri_att_start.mha.linear_q.weight"]
    with pytest.raises(KeyError, match="tri_att_start.mha.linear_q.weight"):
        map_pair_block(state)
