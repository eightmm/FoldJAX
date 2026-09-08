from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.esmfold2_pair_init_probe import PARTS, candidate_prefix
from foldjax.models.esmfold2.models import model
from tests.models.esmfold2.test_distogram_output_flag import _cheap_features


@pytest.mark.parametrize("compiled", [False, True])
def test_actual_pair_prefix_stops_before_missing_trunk_parameters(compiled):
    originals = {
        name: getattr(model, name)
        for name in (
            "inputs_embedding",
            "linear",
            "relative_position_encoding",
            "shard_pair_rows",
        )
    }
    settings = replace(model.ModelSettings(), d_pair=4, trunk_dtype="bfloat16")
    params = {
        "z_init_1.weight": jnp.ones((4, 451), jnp.float32) * 0.01,
        "z_init_2.weight": jnp.ones((4, 451), jnp.float32) * -0.01,
        "rel_pos.embed.weight": jnp.ones((4, 139), jnp.float32) * 0.01,
        "token_bonds.weight": jnp.ones((4, 1), jnp.float32),
    }
    x = jnp.ones((1, 2, 451), jnp.float32)

    def run(f, p, a):
        return candidate_prefix(model, f, p, a, settings)

    result = (jax.jit(run) if compiled else run)(_cheap_features(), params, x)
    assert set(result) == set(PARTS) | {"z_init"}
    assert result["z_init"].shape == (1, 2, 2, 4)
    assert all(v.dtype == jnp.bfloat16 for v in result.values())
    expected = result["z_init_1"][:, :, None] + result["z_init_2"][:, None]
    expected = expected + result["rel_pos"] + result["token_bonds"]
    np.testing.assert_array_equal(result["z_init"], expected)
    for key, value in originals.items():
        assert getattr(model, key) is value


@pytest.mark.parametrize("error", [False, True])
def test_prefix_failure_restores_every_hook(error):
    def original(*args, **kwargs):
        return None

    def predict(*args, **kwargs):
        if error:
            raise RuntimeError("prediction failed")

    module = SimpleNamespace(
        inputs_embedding=original,
        linear=original,
        relative_position_encoding=original,
        shard_pair_rows=original,
        predict=predict,
    )
    with pytest.raises(RuntimeError if error else ValueError):
        candidate_prefix(module, {}, {}, None, None)
    for name in (
        "inputs_embedding",
        "linear",
        "relative_position_encoding",
        "shard_pair_rows",
    ):
        assert getattr(module, name) is original
