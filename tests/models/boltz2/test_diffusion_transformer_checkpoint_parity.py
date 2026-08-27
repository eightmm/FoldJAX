import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.bridge.torch_mapping import (
    map_diffusion_transformer_layer_state_dict,
)
from foldjax.models.boltz2.models.diffusion.diffusion_transformer import (
    diffusion_transformer_layer_forward,
)
from tests.models.boltz2.torch_checkpoint import load_torch_checkpoint_state_dict

CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)
BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
PREFIX = "structure_module.score_model.token_transformer.layers.0"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_torch_checkpoint_state_dict(CHECKPOINT)


def test_checkpoint_diffusion_transformer_layer_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    torch_module = _load_torch_layer(checkpoint_state)
    params = map_diffusion_transformer_layer_state_dict(
        checkpoint_state,
        PREFIX,
        num_heads=8,
    )
    a, s, bias, mask = _layer_inputs()

    with torch.no_grad():
        expected = torch_module(
            a,
            s,
            bias=bias,
            mask=mask,
            multiplicity=1,
        )
    actual = diffusion_transformer_layer_forward(
        params,
        jnp.asarray(a.numpy()),
        jnp.asarray(s.numpy()),
        jnp.asarray(bias.numpy()),
        jnp.asarray(mask.numpy()),
        multiplicity=1,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )


def test_checkpoint_diffusion_transformer_layer_accepts_flash_backend(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    params = map_diffusion_transformer_layer_state_dict(
        checkpoint_state,
        PREFIX,
        num_heads=8,
    )
    a, s, bias, mask = _layer_inputs()
    a_j = jnp.asarray(a.numpy())
    s_j = jnp.asarray(s.numpy())
    bias_j = jnp.asarray(bias.numpy())
    mask_j = jnp.asarray(mask.numpy())

    expected = diffusion_transformer_layer_forward(
        params,
        a_j,
        s_j,
        bias_j,
        mask_j,
        multiplicity=1,
    )
    compiled = jax.jit(
        lambda p, a_, s_, bias_, mask_: diffusion_transformer_layer_forward(
            p,
            a_,
            s_,
            bias_,
            mask_,
            multiplicity=1,
            attention_backend="tokamax",
        )
    )
    actual = compiled(params, a_j, s_j, bias_j, mask_j)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=2e-3,
        atol=2e-3,
    )


def test_checkpoint_diffusion_transformer_layer_chunk_matches_unchunked(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    """Query-axis chunking is a memory optimisation, not a change of answer.

    ``diffusion.py`` passes a live ``token_attention_chunk`` here, so this
    guards a production path: each query block attends over every key, which
    makes chunking exact in arithmetic.

    It is not exact in floating point, and this test used to demand that it
    was. Slicing the query axis changes the ``bihd,bjhd->bhij`` extent, XLA
    picks a different contraction schedule for the smaller operand, and the
    reassociated sum lands on different bits. That is a property of the
    backend rather than of this port: the same divergence reproduces in a
    twenty-line script against the bare primitive, with and without torch
    imported, in both the shipped and the parity environment.

    Measured here on CPU with the released checkpoint: 1.7e-5 absolute and
    1.1e-4 relative over 3,072 elements. The tolerance below sits about
    sixty times above that, which leaves room for another machine's schedule
    while still failing if chunking ever changes the answer for real.
    """

    params = map_diffusion_transformer_layer_state_dict(
        checkpoint_state,
        PREFIX,
        num_heads=8,
    )
    a, s, bias, mask = _layer_inputs()
    a_j = jnp.asarray(a.numpy())
    s_j = jnp.asarray(s.numpy())
    bias_j = jnp.asarray(bias.numpy())
    mask_j = jnp.asarray(mask.numpy())

    expected = diffusion_transformer_layer_forward(
        params,
        a_j,
        s_j,
        bias_j,
        mask_j,
        multiplicity=1,
    )
    actual = diffusion_transformer_layer_forward(
        params,
        a_j,
        s_j,
        bias_j,
        mask_j,
        multiplicity=1,
        chunk_size=2,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-3,
        atol=1e-3,
    )


def _load_torch_layer(state: dict[str, torch.Tensor]) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.transformersv2 import DiffusionTransformerLayer

    module = DiffusionTransformerLayer(
        heads=8,
        dim=768,
        dim_single_cond=768,
        post_layer_norm=False,
    ).eval()
    module_state = {
        key.removeprefix(f"{PREFIX}."): value
        for key, value in state.items()
        if key.startswith(f"{PREFIX}.")
    }
    module.load_state_dict(module_state)
    return module


def _layer_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residues = 4
    a_values = torch.linspace(-0.25, 0.25, steps=residues * 768)
    s_values = torch.linspace(0.3, -0.3, steps=residues * 768)
    bias_values = torch.linspace(-0.1, 0.1, steps=residues * residues * 8)
    a = a_values.reshape(1, residues, 768)
    s = s_values.reshape(1, residues, 768)
    bias = bias_values.reshape(1, residues, residues, 8)
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    return a, s, bias, mask
