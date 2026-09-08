"""Confidence's non-fused autocast must retain the FP32 residual stream."""

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import heads


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("cp", [False, True])
def test_confidence_trunk_preserves_native_residual_and_affine(monkeypatch, dtype, cp):
    weight = jnp.array([1.0035, -0.1037], jnp.float32)
    pair = jnp.full((1, 2, 2, 2), 1.001, jnp.float32)
    params = {
        "s_inputs_norm.weight": weight,
        "s_inputs_norm.bias": weight,
        "z_norm.weight": weight,
        "z_norm.bias": weight,
        "boundaries": jnp.array([1.0]),
        "dist_bin_pairwise_embed.weight": jnp.zeros((2, 2)),
        "folding_trunk.blocks.0.norm.weight": weight,
    }
    monkeypatch.setattr(heads, "cp_mesh", lambda: object() if cp else None)
    monkeypatch.setattr(heads, "shard_pair_rows", lambda x: x)
    monkeypatch.setattr(heads, "layer_norm", lambda x, *args: x)
    monkeypatch.setattr(heads, "linear", lambda x, *args: jnp.zeros_like(x))

    class BoundaryReachedError(Exception):
        pass

    def trunk(x, selected, prefix, **kwargs):
        native = dtype == jnp.bfloat16 and not cp
        assert kwargs["native_autocast"] is native
        assert prefix == "folding_trunk"
        expected = pair if native else pair.astype(dtype)
        assert x.dtype == expected.dtype
        np.testing.assert_array_equal(x, expected)
        actual_weight = selected["folding_trunk.blocks.0.norm.weight"]
        expected_weight = weight if native else weight.astype(dtype)
        assert actual_weight.dtype == expected_weight.dtype
        np.testing.assert_array_equal(actual_weight, expected_weight)
        if native:
            assert selected is params
        raise BoundaryReachedError

    monkeypatch.setattr(heads, "folding_trunk", trunk)
    with pytest.raises(BoundaryReachedError):
        heads.confidence_head(
            jnp.zeros((1, 2, 2)),
            pair,
            jnp.zeros((1, 2, 3)),
            params,
            distogram_atom_idx=jnp.array([[0, 1]]),
            token_mask=jnp.ones((1, 2)),
            atom_to_token=jnp.array([[0, 1]]),
            atom_mask=jnp.ones((1, 2)),
            asym_id=jnp.zeros((1, 2), jnp.int32),
            mol_type=jnp.zeros((1, 2), jnp.int32),
            n_layers=1,
            n_chains=1,
            trunk_dtype=dtype,
        )
    assert params["folding_trunk.blocks.0.norm.weight"] is weight
