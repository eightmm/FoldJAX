"""Torch-vs-JAX parity for the MSA module block and stack (AF3 Algorithm 8)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import (
    map_msa_module_block,
    map_msa_module_stack,
)
from foldjax.models.openfold3.models.msa_module import (
    msa_module_block,
    msa_module_stack,
)

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_M, C_Z, N_SEQ, N_TOKEN = 8, 6, 3, 5
HEADS_MSA, HEADS_PAIR = 2, 2


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


def _kwargs(opm_first: bool) -> dict:
    return {
        "c_m": C_M,
        "c_z": C_Z,
        "c_hidden_msa_att": 4,
        "c_hidden_opm": 4,
        "c_hidden_mul": 6,
        "c_hidden_pair_att": 4,
        "no_heads_msa": HEADS_MSA,
        "no_heads_pair": HEADS_PAIR,
        "transition_type": "swiglu",
        "transition_n": 2,
        "msa_dropout": 0.0,
        "pair_dropout": 0.0,
        "opm_first": opm_first,
        "fuse_projection_weights": False,
        "inf": 1e9,
        "eps": 1e-3,
    }


def _block(opm_first: bool, last_block: bool = False):
    from openfold3.core.model.latent.msa_module import MSAModuleBlock

    return MSAModuleBlock(last_block=last_block, **_kwargs(opm_first))


def _stack(no_blocks: int, opm_first: bool):
    from openfold3.core.model.latent.msa_module import MSAModuleStack

    return MSAModuleStack(
        no_blocks=no_blocks, blocks_per_ckpt=None, **_kwargs(opm_first)
    )


def _inputs(torch):
    m = torch.randn(1, N_SEQ, N_TOKEN, C_M)
    z = torch.randn(1, N_TOKEN, N_TOKEN, C_Z)
    msa_mask = torch.ones(1, N_SEQ, N_TOKEN)
    msa_mask[:, 2, :] = 0.0
    token = torch.ones(1, N_TOKEN)
    token[:, 4:] = 0.0
    pair_mask = token[..., None] * token[..., None, :]
    return m, z, msa_mask, pair_mask


def _jax_args(m, z, msa_mask, pair_mask, opm_first: bool) -> dict:
    return {
        "msa_mask": jnp.asarray(msa_mask.numpy()),
        "pair_mask": jnp.asarray(pair_mask.numpy()),
        "no_heads_msa": HEADS_MSA,
        "no_heads_pair": HEADS_PAIR,
        "opm_first": opm_first,
        "opm_eps": 1e-3,
    }


@pytest.mark.parametrize("opm_first", [True, False])
def test_msa_module_block_matches_torch(
    openfold3_source: Path, randomized, opm_first: bool
) -> None:
    torch = _torch()
    module = randomized(_block(opm_first))
    m, z, msa_mask, pair_mask = _inputs(torch)
    with torch.no_grad():
        expected_m, expected_z = module(
            m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask
        )
    actual_m, actual_z = msa_module_block(
        jnp.asarray(m.numpy()),
        jnp.asarray(z.numpy()),
        map_msa_module_block(dict(module.state_dict())),
        **_jax_args(m, z, msa_mask, pair_mask, opm_first),
    )
    _assert_close(actual_m, expected_m, f"MSAModuleBlock(opm_first={opm_first})(m)")
    _assert_close(actual_z, expected_z, f"MSAModuleBlock(opm_first={opm_first})(z)")


def test_opm_first_actually_changes_the_result(
    openfold3_source: Path, randomized
) -> None:
    """opm_first reorders which m the pair update sees; it must be observable."""
    torch = _torch()
    module = randomized(_block(opm_first=True))
    m, z, msa_mask, pair_mask = _inputs(torch)
    params = map_msa_module_block(dict(module.state_dict()))
    _first_m, first_z = msa_module_block(
        jnp.asarray(m.numpy()),
        jnp.asarray(z.numpy()),
        params,
        **_jax_args(m, z, msa_mask, pair_mask, opm_first=True),
    )
    _last_m, last_z = msa_module_block(
        jnp.asarray(m.numpy()),
        jnp.asarray(z.numpy()),
        params,
        **_jax_args(m, z, msa_mask, pair_mask, opm_first=False),
    )
    assert not np.allclose(np.asarray(first_z), np.asarray(last_z), rtol=1e-3)


def test_last_block_skips_the_msa_update(openfold3_source: Path, randomized) -> None:
    """With opm_first, the final block has no msa_att_row/msa_transition at all."""
    torch = _torch()
    module = randomized(_block(opm_first=True, last_block=True))
    assert module.skip_msa_update is True
    state = dict(module.state_dict())
    assert not any(key.startswith("msa_att_row.") for key in state)
    assert not any(key.startswith("msa_transition.") for key in state)

    params = map_msa_module_block(state)
    assert params.msa_att_row is None
    assert params.msa_transition is None

    m, z, msa_mask, pair_mask = _inputs(torch)
    with torch.no_grad():
        expected_m, expected_z = module(
            m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask
        )
    actual_m, actual_z = msa_module_block(
        jnp.asarray(m.numpy()),
        jnp.asarray(z.numpy()),
        params,
        **_jax_args(m, z, msa_mask, pair_mask, opm_first=True),
    )
    _assert_close(actual_m, expected_m, "MSAModuleBlock(last)(m)")
    _assert_close(actual_z, expected_z, "MSAModuleBlock(last)(z)")
    # The MSA embedding must pass through untouched.
    np.testing.assert_allclose(
        np.asarray(actual_m), m.numpy(), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("no_blocks", [1, 3])
def test_msa_module_stack_matches_torch(
    openfold3_source: Path, randomized, no_blocks: int
) -> None:
    torch = _torch()
    module = randomized(_stack(no_blocks, opm_first=True))
    m, z, msa_mask, pair_mask = _inputs(torch)
    with torch.no_grad():
        expected = module(m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask)
    params = map_msa_module_stack(dict(module.state_dict()))
    assert len(params.blocks) == no_blocks
    # Only the final block skips the MSA update.
    assert params.blocks[-1].msa_att_row is None
    if no_blocks > 1:
        assert params.blocks[0].msa_att_row is not None
    actual = msa_module_stack(
        jnp.asarray(m.numpy()),
        jnp.asarray(z.numpy()),
        params,
        **_jax_args(m, z, msa_mask, pair_mask, opm_first=True),
    )
    _assert_close(actual, expected, f"MSAModuleStack({no_blocks})")


def test_stack_mapper_rejects_a_missing_block(openfold3_source: Path) -> None:
    _torch()
    state = {
        key: value
        for key, value in _stack(3, opm_first=True).state_dict().items()
        if not key.startswith("blocks.1.")
    }
    with pytest.raises(KeyError, match="non-contiguous"):
        map_msa_module_stack(state)


def test_block_mapper_rejects_a_half_present_msa_update(
    openfold3_source: Path,
) -> None:
    """Dropping only the transition must fail, not silently skip the update."""
    _torch()
    state = {
        key: value
        for key, value in _block(opm_first=True).state_dict().items()
        if not key.startswith("msa_transition.")
    }
    with pytest.raises(KeyError, match="both be present or"):
        map_msa_module_block(state)


def test_msa_module_block_state_dict_layout(openfold3_source: Path) -> None:
    _torch()
    prefixes = {key.split(".")[0] for key in _block(opm_first=True).state_dict()}
    assert prefixes == {
        "outer_product_mean",
        "pair_stack",
        "msa_att_row",
        "msa_transition",
    }
