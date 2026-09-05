"""Torch-vs-JAX parity for the diffusion transformer block and stack."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import (
    map_ada_attention_pair_bias,
    map_diffusion_transformer,
    map_diffusion_transformer_block,
)
from foldjax.models.openfold3.models.attention_pair_bias import ada_attention_pair_bias
from foldjax.models.openfold3.models.diffusion_transformer import (
    diffusion_transformer,
    diffusion_transformer_block,
)

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_TOKEN, C_S, C_Z, N = 8, 6, 4, 5
HEADS, C_HIDDEN = 2, 4


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
    from openfold3.core.model.layers.attention_pair_bias import (
        DiffusionAttentionPairBias,
    )

    return DiffusionAttentionPairBias(
        c_q=C_TOKEN,
        c_k=C_TOKEN,
        c_v=C_TOKEN,
        c_s=C_S,
        c_z=C_Z,
        c_hidden=C_HIDDEN,
        no_heads=HEADS,
        gating=True,
        inf=1e9,
    )


def _kwargs() -> dict:
    # n_query/n_key are None for the self-attention configuration; they select
    # the cross-attention (neighbourhood) variant when set.
    return {
        "c_a": C_TOKEN,
        "c_s": C_S,
        "c_z": C_Z,
        "c_hidden": C_HIDDEN,
        "no_heads": HEADS,
        "n_transition": 2,
        "n_query": None,
        "n_key": None,
        "inf": 1e9,
    }


def _block():
    from openfold3.core.model.layers.diffusion_transformer import (
        DiffusionTransformerBlock,
    )

    return DiffusionTransformerBlock(**_kwargs())


def _stack(no_blocks: int):
    from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer

    return DiffusionTransformer(no_blocks=no_blocks, blocks_per_ckpt=None, **_kwargs())


def _inputs(torch):
    a = torch.randn(1, N, C_TOKEN)
    s = torch.randn(1, N, C_S)
    z = torch.randn(1, N, N, C_Z)
    mask = torch.ones(1, N)
    mask[:, 4:] = 0.0
    return a, s, z, mask


def test_ada_attention_pair_bias_matches_torch(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_apb())
    a, s, z, mask = _inputs(torch)
    with torch.no_grad():
        expected = module(a=a, z=z, s=s, mask=mask)
    actual = ada_attention_pair_bias(
        jnp.asarray(a.numpy()),
        jnp.asarray(s.numpy()),
        jnp.asarray(z.numpy()),
        map_ada_attention_pair_bias(dict(module.state_dict())),
        no_heads=HEADS,
        mask=jnp.asarray(mask.numpy()),
    )
    _assert_close(actual, expected, "AdaAttentionPairBias")


def test_ada_variant_has_its_own_layout(openfold3_source: Path) -> None:
    """The AdaLN variant swaps layer_norm_a for an AdaLN and adds an output gate."""
    _torch()
    state = set(_apb().state_dict())
    assert "layer_norm_a.linear_g.weight" in state  # AdaLN, not a plain norm
    assert "linear_ada_out.weight" in state
    assert "linear_ada_out.bias" in state
    # OpenBind normalizes z once in the enclosing transformer stack.
    assert "layer_norm_z.weight" not in state


def test_diffusion_transformer_block_matches_torch(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_block())
    a, s, z, mask = _inputs(torch)
    with torch.no_grad():
        expected = module(a=a, s=s, z=z, mask=mask)
    actual = diffusion_transformer_block(
        jnp.asarray(a.numpy()),
        jnp.asarray(s.numpy()),
        jnp.asarray(z.numpy()),
        map_diffusion_transformer_block(dict(module.state_dict())),
        no_heads=HEADS,
        mask=jnp.asarray(mask.numpy()),
    )
    _assert_close(actual, expected, "DiffusionTransformerBlock")


def test_transition_consumes_the_updated_activation(
    openfold3_source: Path, randomized
) -> None:
    """Upstream deviates from the SI: the transition sees post-attention `a`."""
    torch = _torch()
    module = randomized(_block())
    a, s, z, mask = _inputs(torch)
    params = map_diffusion_transformer_block(dict(module.state_dict()))
    ja, js, jz = (jnp.asarray(x.numpy()) for x in (a, s, z))
    jmask = jnp.asarray(mask.numpy())
    actual = diffusion_transformer_block(ja, js, jz, params, no_heads=HEADS, mask=jmask)

    from foldjax.models.openfold3.models.primitives import conditioned_transition_block

    attended = ja + ada_attention_pair_bias(
        ja, js, jz, params.attention_pair_bias, no_heads=HEADS, mask=jmask
    )
    # Feeding the ORIGINAL a to the transition must give a different answer.
    wrong = attended + conditioned_transition_block(
        ja, js, params.conditioned_transition, mask=jmask
    )
    assert not np.allclose(np.asarray(actual), np.asarray(wrong), rtol=1e-3)


@pytest.mark.parametrize("no_blocks", [1, 3])
def test_diffusion_transformer_matches_torch(
    openfold3_source: Path, randomized, no_blocks: int
) -> None:
    torch = _torch()
    module = randomized(_stack(no_blocks))
    a, s, z, mask = _inputs(torch)
    with torch.no_grad():
        expected = module(a=a, s=s, z=z, mask=mask)
    params = map_diffusion_transformer(dict(module.state_dict()))
    assert len(params.blocks) == no_blocks
    actual = diffusion_transformer(
        jnp.asarray(a.numpy()),
        jnp.asarray(s.numpy()),
        jnp.asarray(z.numpy()),
        params,
        no_heads=HEADS,
        mask=jnp.asarray(mask.numpy()),
    )
    _assert_close(actual, expected, f"DiffusionTransformer({no_blocks})")


def test_openbind_stack_owns_the_only_pair_norm(
    openfold3_source: Path, randomized
) -> None:
    """OpenBind's pair norm is mapped once, before all transformer blocks."""
    torch = _torch()
    module = randomized(_stack(3), seed=23)
    a, s, z, mask = _inputs(torch)
    with torch.no_grad():
        expected = module(a=a, s=s, z=z, mask=mask)

    state = dict(module.state_dict())
    assert "layer_norm_z.weight" in state
    assert not any(
        key.endswith("attention_pair_bias.layer_norm_z.weight") for key in state
    )
    params = map_diffusion_transformer(state)
    assert all(
        not hasattr(block.attention_pair_bias, "layer_norm_z")
        for block in params.blocks
    )

    actual = diffusion_transformer(
        jnp.asarray(a.numpy()),
        jnp.asarray(s.numpy()),
        jnp.asarray(z.numpy()),
        params,
        no_heads=HEADS,
        mask=jnp.asarray(mask.numpy()),
    )
    _assert_close(actual, expected, "OpenBind stack-level pair norm")


def test_mapper_rejects_a_legacy_stack_without_the_openbind_norm(
    openfold3_source: Path,
) -> None:
    _torch()
    state = dict(_stack(1).state_dict())
    del state["layer_norm_z.weight"]

    with pytest.raises(KeyError, match="layer_norm_z.weight"):
        map_diffusion_transformer(state)


def test_block_state_dict_layout(openfold3_source: Path) -> None:
    _torch()
    prefixes = {key.split(".")[0] for key in _block().state_dict()}
    assert prefixes == {"attention_pair_bias", "conditioned_transition"}


def test_mapper_rejects_a_missing_block(openfold3_source: Path) -> None:
    _torch()
    state = {
        key: value
        for key, value in _stack(3).state_dict().items()
        if not key.startswith("blocks.1.")
    }
    with pytest.raises(KeyError, match="non-contiguous"):
        map_diffusion_transformer(state)
