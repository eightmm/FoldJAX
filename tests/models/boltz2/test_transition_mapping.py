import jax.numpy as jnp
import pytest
import torch

from foldjax.models.boltz2.bridge.torch_mapping import map_transition_state_dict

PREFIX = "msa_module.layers.0.msa_transition"


def _state_dict() -> dict[str, torch.Tensor]:
    return {
        f"{PREFIX}.norm.weight": torch.tensor([1.0, 2.0, 3.0]),
        f"{PREFIX}.norm.bias": torch.tensor([4.0, 5.0, 6.0]),
        f"{PREFIX}.fc1.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        f"{PREFIX}.fc2.weight": torch.arange(20, dtype=torch.float32).reshape(5, 4),
        f"{PREFIX}.fc3.weight": torch.arange(15, dtype=torch.float32).reshape(3, 5),
    }


def test_maps_transition_weights_to_jax_layout() -> None:
    params = map_transition_state_dict(_state_dict(), PREFIX)

    assert set(params) == {"norm", "fc1", "fc2", "fc3"}
    assert set(params["norm"]) == {"scale", "bias"}
    assert set(params["fc1"]) == {"kernel"}
    assert set(params["fc2"]) == {"kernel"}
    assert set(params["fc3"]) == {"kernel"}

    assert params["norm"]["scale"].shape == (3,)
    assert params["norm"]["bias"].shape == (3,)
    assert params["fc1"]["kernel"].shape == (3, 4)
    assert params["fc2"]["kernel"].shape == (4, 5)
    assert params["fc3"]["kernel"].shape == (5, 3)

    assert jnp.array_equal(params["norm"]["scale"], jnp.array([1.0, 2.0, 3.0]))
    assert jnp.array_equal(params["norm"]["bias"], jnp.array([4.0, 5.0, 6.0]))
    assert jnp.array_equal(
        params["fc1"]["kernel"],
        jnp.asarray(_state_dict()[f"{PREFIX}.fc1.weight"]).T,
    )
    assert jnp.array_equal(
        params["fc2"]["kernel"],
        jnp.asarray(_state_dict()[f"{PREFIX}.fc2.weight"]).T,
    )
    assert jnp.array_equal(
        params["fc3"]["kernel"],
        jnp.asarray(_state_dict()[f"{PREFIX}.fc3.weight"]).T,
    )


def test_transition_mapping_reports_missing_keys() -> None:
    state = _state_dict()
    missing_key = f"{PREFIX}.fc2.weight"
    del state[missing_key]

    with pytest.raises(KeyError, match="Missing required Transition state_dict keys"):
        map_transition_state_dict(state, PREFIX)
