"""Torch-vs-JAX parity for the template pair stack (AF2 Algorithm 16)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import (
    map_pair_block,
    map_template_pair_stack,
)
from foldjax.models.openfold3.models.pair_block import pair_block
from foldjax.models.openfold3.models.template_module import template_pair_stack

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_T, N, HEADS = 8, 5, 2


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


def _kwargs(tri_mul_first: bool) -> dict:
    return {
        "c_t": C_T,
        "c_hidden_tri_mul": 6,
        "c_hidden_tri_att": 4,
        "no_heads": HEADS,
        "transition_type": "swiglu",
        "pair_transition_n": 2,
        "dropout_rate": 0.0,
        "tri_mul_first": tri_mul_first,
        "fuse_projection_weights": False,
        "ckpt_per_template": False,
        "inf": 1e9,
    }


def _block(tri_mul_first: bool):
    from openfold3.core.model.latent.template_module import TemplatePairBlock

    return TemplatePairBlock(**_kwargs(tri_mul_first))


def _stack(no_blocks: int, tri_mul_first: bool):
    from openfold3.core.model.latent.template_module import TemplatePairStack

    return TemplatePairStack(
        no_blocks=no_blocks, blocks_per_ckpt=None, **_kwargs(tri_mul_first)
    )


def _mask(torch):
    token = torch.ones(1, N)
    token[:, 4:] = 0.0
    return token[..., None] * token[..., None, :]


@pytest.mark.parametrize("tri_mul_first", [True, False])
def test_template_pair_block_matches_torch(
    openfold3_source: Path, randomized, tri_mul_first: bool
) -> None:
    torch = _torch()
    module = randomized(_block(tri_mul_first))
    # The block takes a template axis and loops over it internally.
    t = torch.randn(1, 2, N, N, C_T)
    mask = _mask(torch).unsqueeze(1).expand(1, 2, N, N)
    with torch.no_grad():
        expected = module(t=t, mask=mask)
    actual = pair_block(
        jnp.asarray(t.numpy()),
        map_pair_block(dict(module.state_dict())),
        pair_mask=jnp.asarray(mask.numpy()),
        no_heads_pair=HEADS,
        tri_mul_first=tri_mul_first,
    )
    _assert_close(actual, expected, f"TemplatePairBlock({tri_mul_first})")


def test_tri_mul_first_changes_the_result(openfold3_source: Path, randomized) -> None:
    """The two orderings share a layout, so only the numbers distinguish them."""
    torch = _torch()
    params = map_pair_block(dict(randomized(_block(True)).state_dict()))
    t = jnp.asarray(torch.randn(1, N, N, C_T).numpy())
    mask = jnp.asarray(_mask(torch).numpy())
    first = pair_block(
        t, params, pair_mask=mask, no_heads_pair=HEADS, tri_mul_first=True
    )
    second = pair_block(
        t, params, pair_mask=mask, no_heads_pair=HEADS, tri_mul_first=False
    )
    assert not np.allclose(np.asarray(first), np.asarray(second), rtol=1e-3)


@pytest.mark.parametrize("no_blocks", [1, 2])
def test_template_pair_stack_matches_torch(
    openfold3_source: Path, randomized, no_blocks: int
) -> None:
    torch = _torch()
    module = randomized(_stack(no_blocks, tri_mul_first=True))
    # Upstream expects a template axis: [*, N_templ, N, N, C_t].
    t = torch.randn(1, 2, N, N, C_T)
    mask = _mask(torch).unsqueeze(1).expand(1, 2, N, N)
    with torch.no_grad():
        expected = module(t=t, mask=mask, chunk_size=None)
    params = map_template_pair_stack(dict(module.state_dict()))
    assert len(params.blocks) == no_blocks
    actual = template_pair_stack(
        jnp.asarray(t.numpy()),
        params,
        mask=jnp.asarray(mask.numpy()),
        no_heads=HEADS,
        tri_mul_first=True,
    )
    _assert_close(actual, expected, f"TemplatePairStack({no_blocks})")


def test_stack_applies_a_final_layer_norm(openfold3_source: Path, randomized) -> None:
    """A plain pair stack has no trailing norm; the template stack does."""
    torch = _torch()
    module = randomized(_stack(1, tri_mul_first=True))
    state = dict(module.state_dict())
    assert "layer_norm.weight" in state
    params = map_template_pair_stack(state)
    assert params.layer_norm.weight is not None

    t = jnp.asarray(torch.randn(1, 2, N, N, C_T).numpy())
    mask = jnp.asarray(_mask(torch).unsqueeze(1).expand(1, 2, N, N).numpy())
    with_norm = template_pair_stack(
        t, params, mask=mask, no_heads=HEADS, tri_mul_first=True
    )
    without_norm = pair_block(
        t, params.blocks[0], pair_mask=mask, no_heads_pair=HEADS
    )
    assert not np.allclose(
        np.asarray(with_norm), np.asarray(without_norm), rtol=1e-3
    )


def test_template_stack_state_dict_layout(openfold3_source: Path) -> None:
    _torch()
    prefixes = {key.split(".")[0] for key in _stack(2, True).state_dict()}
    assert prefixes == {"blocks", "layer_norm"}


def test_template_stack_mapper_rejects_a_missing_block(
    openfold3_source: Path,
) -> None:
    _torch()
    state = {
        key: value
        for key, value in _stack(3, True).state_dict().items()
        if not key.startswith("blocks.1.")
    }
    with pytest.raises(KeyError, match="non-contiguous"):
        map_template_pair_stack(state)
