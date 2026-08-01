from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from foldjax.models.boltz2.bridge.torch_checkpoint import load_checkpoint_state_dict
from foldjax.models.boltz2.bridge.torch_mapping import (
    map_triangle_multiplication_state_dict,
)
from foldjax.models.boltz2.models.triangle.triangle import (
    triangle_multiplication_forward,
)

CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


@pytest.mark.parametrize(
    ("prefix", "direction"),
    [
        ("pairformer_module.layers.0.tri_mul_out", "outgoing"),
        ("pairformer_module.layers.0.tri_mul_in", "incoming"),
    ],
)
def test_checkpoint_triangle_multiplication_matches_torch(
    checkpoint_state: dict[str, torch.Tensor],
    prefix: str,
    direction: str,
) -> None:
    x, mask = _triangle_inputs(checkpoint_state, prefix)
    params = map_triangle_multiplication_state_dict(checkpoint_state, prefix)

    expected = _torch_triangle_multiplication_forward(
        checkpoint_state,
        prefix,
        x,
        mask,
        direction,
    )
    actual = triangle_multiplication_forward(
        params,
        jnp.asarray(x.numpy()),
        jnp.asarray(mask.numpy()),
        direction=direction,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-4,
        atol=2e-4,
    )


def test_triangle_mapping_reports_missing_key(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    prefix = "pairformer_module.layers.0.tri_mul_out"
    state = dict(checkpoint_state)
    del state[f"{prefix}.p_out.weight"]

    with pytest.raises(KeyError, match="Missing required TriangleMultiplication"):
        map_triangle_multiplication_state_dict(state, prefix)


def test_triangle_multiplication_preserves_bfloat16_activation_dtype(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
    params = _random_triangle_params(dim=8)
    x = jnp.linspace(-0.5, 0.5, num=1 * 6 * 6 * 8, dtype=jnp.bfloat16).reshape(
        1, 6, 6, 8
    )
    mask = jnp.ones((1, 6, 6), dtype=jnp.bfloat16)

    out = triangle_multiplication_forward(
        params,
        x,
        mask,
        direction="outgoing",
        chunk_size=3,
    )

    assert out.dtype == jnp.bfloat16


def _random_triangle_params(dim: int):
    rng = np.random.default_rng(0)

    def w(*shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.1, dtype=jnp.bfloat16)

    return {
        "norm_in": {"scale": w(dim), "bias": w(dim)},
        "p_in": {"kernel": w(dim, dim * 2)},
        "g_in": {"kernel": w(dim, dim * 2)},
        "norm_out": {"scale": w(dim), "bias": w(dim)},
        "p_out": {"kernel": w(dim, dim)},
        "g_out": {"kernel": w(dim, dim)},
    }


def _triangle_inputs(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = state[f"{prefix}.norm_in.weight"].shape[0]
    values = torch.linspace(-0.5, 0.5, steps=1 * 4 * 4 * dim, dtype=torch.float32)
    x = values.reshape(1, 4, 4, dim)
    mask = torch.tensor(
        [[
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ]],
        dtype=torch.float32,
    )
    return x, mask


def _torch_triangle_multiplication_forward(
    state: dict[str, torch.Tensor],
    prefix: str,
    x: torch.Tensor,
    mask: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    x = functional.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=state[f"{prefix}.norm_in.weight"],
        bias=state[f"{prefix}.norm_in.bias"],
        eps=1e-5,
    )
    x_in = x
    projected = functional.linear(x, state[f"{prefix}.p_in.weight"]) * torch.sigmoid(
        functional.linear(x, state[f"{prefix}.g_in.weight"])
    )
    projected = projected * mask.unsqueeze(-1)
    a, b = torch.chunk(projected.float(), 2, dim=-1)

    if direction == "outgoing":
        out = torch.einsum("bikd,bjkd->bijd", a, b)
    elif direction == "incoming":
        out = torch.einsum("bkid,bkjd->bijd", a, b)
    else:
        raise AssertionError(direction)

    out = functional.layer_norm(
        out,
        normalized_shape=(out.shape[-1],),
        weight=state[f"{prefix}.norm_out.weight"],
        bias=state[f"{prefix}.norm_out.bias"],
        eps=1e-5,
    )
    out = functional.linear(out, state[f"{prefix}.p_out.weight"])
    gate = torch.sigmoid(functional.linear(x_in, state[f"{prefix}.g_out.weight"]))
    return out * gate
