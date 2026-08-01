from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.opendde.bridge.torch_mapping import (
    map_distogram_state_dict,
    map_released_distogram_state_dict,
)
from foldjax.models.opendde.models.heads import distogram_head


def _state(c_z: int, bins: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "distogram_head.linear.weight": rng.normal(size=(bins, c_z)).astype(np.float32),
        "distogram_head.linear.bias": rng.normal(size=(bins,)).astype(np.float32),
    }


def test_distogram_matches_opendde_reference_formula() -> None:
    rng = np.random.default_rng(11)
    z = rng.normal(size=(2, 3, 3, 4)).astype(np.float32)
    state = _state(c_z=4, bins=5)
    params = map_distogram_state_dict(state)

    actual = np.asarray(distogram_head(jnp.asarray(z), params))
    logits = z @ state["distogram_head.linear.weight"].T
    logits = logits + state["distogram_head.linear.bias"]
    expected = logits + np.swapaxes(logits, -2, -3)

    assert actual.shape == (2, 3, 3, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual, np.swapaxes(actual, -2, -3))


def test_released_distogram_contract_is_384_channels_and_96_bins() -> None:
    params = map_released_distogram_state_dict(_state(c_z=384, bins=96))

    assert params.linear.weight.shape == (96, 384)
    assert params.linear.bias.shape == (96,)


def test_released_distogram_contract_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"expected \(96, 384\)"):
        map_released_distogram_state_dict(_state(c_z=128, bins=64))


def test_distogram_mapper_reports_missing_checkpoint_key() -> None:
    with pytest.raises(KeyError, match="distogram_head.linear.bias"):
        map_distogram_state_dict(
            {"distogram_head.linear.weight": np.ones((96, 384), dtype=np.float32)}
        )
