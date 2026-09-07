import jax.numpy as jnp
import numpy as np
import pytest

from bench.boltz_transition_probe import transition_params


def test_transition_probe_mapping_retains_original_affine_and_narrows_kernels():
    weights = {
        "norm.weight": np.ones(2, np.float32),
        "norm.bias": np.zeros(2, np.float32),
        "fc1.weight": np.ones((4, 2), np.float32),
        "fc2.weight": np.ones((4, 2), np.float32),
        "fc3.weight": np.ones((2, 4), np.float32),
    }
    params = transition_params(weights)
    assert params["norm"]["scale"].dtype == jnp.float32
    assert params["norm"]["bias"].dtype == jnp.float32
    assert params["fc1"]["kernel"].shape == (2, 4)
    assert params["fc3"]["kernel"].shape == (4, 2)
    for name in ("fc1", "fc2", "fc3"):
        assert params[name]["kernel"].dtype == jnp.bfloat16
    assert all(value.dtype == np.float32 for value in weights.values())


def test_transition_probe_mapping_fails_on_missing_native_weight():
    with pytest.raises(KeyError, match="Missing required"):
        transition_params({})
