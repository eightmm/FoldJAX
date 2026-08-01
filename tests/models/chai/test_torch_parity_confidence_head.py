"""Official-component parity gates for the Chai confidence head."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai.inference import _run_staged_confidence_block
from foldjax.models.chai.models.confidence import (
    NUM_BLOCKS,
    confidence_head_forward,
    confidence_head_initialize,
    confidence_head_project,
    map_confidence_head,
)

pytestmark = pytest.mark.official_parity

torch = pytest.importorskip("torch")


def _load_component(official_asset_path):
    return torch.jit.load(
        official_asset_path("confidence_head.pt"), map_location="cpu"
    ).eval()


def _inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260714)
    batch, tokens, atoms = 1, 4, 8
    return {
        "token_single_input_repr": rng.normal(
            scale=0.25, size=(batch, tokens, 384)
        ).astype(np.float32),
        "token_single_trunk_repr": rng.normal(
            scale=0.25, size=(batch, tokens, 384)
        ).astype(np.float32),
        "token_pair_trunk_repr": rng.normal(
            scale=0.25, size=(batch, tokens, tokens, 256)
        ).astype(np.float32),
        "token_single_mask": np.asarray([[True, True, True, False]]),
        "atom_single_mask": np.asarray(
            [[True, True, True, True, True, True, False, False]]
        ),
        "atom_coords": rng.normal(size=(batch, atoms, 3)).astype(np.float32),
        "token_reference_atom_index": np.asarray([[0, 2, 4, 6]], dtype=np.int64),
        "atom_token_index": np.asarray([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=np.int64),
        "atom_within_token_index": np.asarray(
            [[0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.int64
        ),
    }


def _torch_inputs(inputs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in inputs.items():
        tensor = torch.from_numpy(value)
        if key in {
            "token_single_input_repr",
            "token_single_trunk_repr",
            "token_pair_trunk_repr",
        }:
            tensor = tensor.to(torch.bfloat16)
        result[key] = tensor
    return result


def test_confidence_mapping_is_exhaustive(official_asset_path) -> None:
    component = _load_component(official_asset_path)
    state = component.state_dict()
    assert len(state) == 106
    params = map_confidence_head(state)
    assert len(params.blocks) == NUM_BLOCKS
    assert params.atom_distance_bins.shape == (15,)
    assert params.plddt_projection_weight.shape == (1850, 384)

    missing = dict(state)
    missing.pop("pae_projection.weight")
    with pytest.raises(KeyError, match="pae_projection.weight"):
        map_confidence_head(missing)
    extra = dict(state)
    extra["unexpected"] = np.zeros(1, dtype=np.float32)
    with pytest.raises(ValueError, match="unexpected confidence tensors"):
        map_confidence_head(extra)


def test_full_confidence_head_matches_official_component(official_asset_path) -> None:
    component = _load_component(official_asset_path)
    inputs = _inputs()
    with torch.no_grad():
        expected = component.forward_256(**_torch_inputs(inputs))
    params = map_confidence_head(component.state_dict())
    jax_inputs = {key: jnp.asarray(value) for key, value in inputs.items()}
    actual = confidence_head_forward(**jax_inputs, params=params)

    assert [tuple(value.shape) for value in actual] == [
        (1, 4, 4, 64),
        (1, 4, 4, 64),
        (1, 8, 50),
    ]
    absolute_tolerances = {"pae": 0.07, "pde": 0.10, "plddt": 0.18}
    for name, jax_value, torch_value in zip(
        ("pae", "pde", "plddt"), actual, expected, strict=True
    ):
        torch_array = torch_value.float().numpy()
        jax_array = np.asarray(jax_value, dtype=np.float32)
        np.testing.assert_allclose(
            jax_array,
            torch_array,
            rtol=1e-2,
            atol=absolute_tolerances[name],
            err_msg=name,
        )


def test_staged_confidence_matches_official_component(official_asset_path) -> None:
    component = _load_component(official_asset_path)
    inputs = _inputs()
    with torch.no_grad():
        expected = component.forward_256(**_torch_inputs(inputs))
    params = map_confidence_head(component.state_dict())
    values = {key: jnp.asarray(value) for key, value in inputs.items()}

    single, pair = confidence_head_initialize(
        values["token_single_input_repr"],
        values["token_single_trunk_repr"],
        values["token_pair_trunk_repr"],
        values["atom_single_mask"],
        values["atom_coords"],
        values["token_reference_atom_index"],
        params,
    )
    for block in params.blocks:
        single, pair = _run_staged_confidence_block(
            single, pair, values["token_single_mask"], block
        )
    actual = confidence_head_project(
        single,
        pair,
        values["atom_token_index"],
        values["atom_within_token_index"],
        params,
    )

    absolute_tolerances = {"pae": 0.07, "pde": 0.10, "plddt": 0.18}
    for name, staged, torch_value in zip(
        ("pae", "pde", "plddt"), actual, expected, strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(staged, dtype=np.float32),
            torch_value.float().numpy(),
            rtol=1e-2,
            atol=absolute_tolerances[name],
            err_msg=name,
        )
