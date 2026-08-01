from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from foldjax.models.boltz2.bridge.torch_checkpoint import load_checkpoint_state_dict
from foldjax.models.boltz2.bridge.torch_mapping import map_transition_state_dict
from foldjax.models.boltz2.models.primitives.transition import transition_forward

PREFIX = "msa_module.layers.0.msa_transition"
CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)


@pytest.fixture(scope="module")
def transition_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


def test_checkpoint_transition_matches_torch_no_chunk(
    transition_state: dict[str, torch.Tensor],
) -> None:
    x = _input_from_checkpoint_dim(transition_state)
    params = map_transition_state_dict(transition_state, PREFIX)

    expected = _torch_transition_forward(transition_state, x)
    actual = transition_forward(params, np.asarray(x))

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_checkpoint_transition_matches_torch_chunked(
    transition_state: dict[str, torch.Tensor],
) -> None:
    x = _input_from_checkpoint_dim(transition_state)
    params = map_transition_state_dict(transition_state, PREFIX)

    expected = _torch_transition_forward(transition_state, x, chunk_size=64)
    actual = transition_forward(params, np.asarray(x), chunk_size=64)

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


def _input_from_checkpoint_dim(state: dict[str, torch.Tensor]) -> torch.Tensor:
    dim = state[f"{PREFIX}.norm.weight"].shape[0]
    values = torch.linspace(-1.0, 1.0, steps=2 * 3 * dim, dtype=torch.float32)
    return values.reshape(2, 3, dim)


def _torch_transition_forward(
    state: dict[str, torch.Tensor],
    x: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    x = functional.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=state[f"{PREFIX}.norm.weight"],
        bias=state[f"{PREFIX}.norm.bias"],
        eps=1e-5,
    )
    if chunk_size is None:
        hidden = functional.silu(
            functional.linear(x, state[f"{PREFIX}.fc1.weight"])
        ) * functional.linear(x, state[f"{PREFIX}.fc2.weight"])
        return functional.linear(hidden, state[f"{PREFIX}.fc3.weight"])

    out = x.new_zeros((*x.shape[:-1], state[f"{PREFIX}.fc3.weight"].shape[0]))
    hidden_dim = state[f"{PREFIX}.fc3.weight"].shape[1]
    for start in range(0, hidden_dim, chunk_size):
        stop = min(start + chunk_size, hidden_dim)
        hidden = functional.silu(
            functional.linear(x, state[f"{PREFIX}.fc1.weight"][start:stop])
        ) * functional.linear(x, state[f"{PREFIX}.fc2.weight"][start:stop])
        out = out + functional.linear(
            hidden,
            state[f"{PREFIX}.fc3.weight"][:, start:stop],
        )
    return out
