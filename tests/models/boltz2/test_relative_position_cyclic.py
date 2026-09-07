"""Native cyclic wrapping is enabled only when any input period is positive."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.boltz_relpos_probe import dense_relative_features
from foldjax.models.boltz2.models.trunk_blocks.trunk import relative_position_forward


def _features(case):
    residues = np.asarray([[0, 5, 7000]], dtype=np.int32)
    periods = np.zeros((1, 3), dtype=np.float32)
    if case == "cyclic":
        residues = np.asarray([[0, 5, 1]], dtype=np.int32)
        periods[:] = 6
    elif case == "mixed":
        periods[0, 0] = 6
    elif case == "ordinary":
        residues[0, -1] = 437
    batch, tokens = residues.shape
    return {
        "residue_index": residues,
        "asym_id": np.zeros((batch, tokens), dtype=np.int32),
        "entity_id": np.zeros((batch, tokens), dtype=np.int32),
        "sym_id": np.zeros((batch, tokens), dtype=np.int32),
        "token_index": np.broadcast_to(
            np.arange(tokens, dtype=np.int32), residues.shape
        ),
        "cyclic_period": periods,
    }


@pytest.mark.parametrize("case", ["noncyclic", "cyclic", "mixed", "ordinary"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("jit", [False, True])
def test_production_categories_match_independent_native_dense(case, dtype, jit):
    features = _features(case)
    expected = dense_relative_features(features)
    # Identity projection exposes every category directly, without learned
    # weights or GEMM reduction obscuring an incorrect residue bin.
    params = {"linear_layer": {"kernel": jnp.eye(139, dtype=dtype)}}
    run = jax.jit(relative_position_forward) if jit else relative_position_forward
    actual = run(params, jax.tree.map(jnp.asarray, features))
    np.testing.assert_array_equal(actual.astype(jnp.float32), expected)
    assert actual.dtype == dtype
    if case == "noncyclic":
        assert actual[0, 0, 2, 0] == 1
        assert actual[0, 2, 0, 64] == 1
    elif case == "mixed":
        # Native's any() is global: a noncyclic token still uses its sentinel
        # period when another token contains a positive period.
        assert actual[0, 0, 2, 64] == 1


@pytest.mark.parametrize("jit", [False, True])
def test_explicitly_disabled_cyclic_encoding_does_not_wrap(jit):
    features = _features("mixed")
    reference_features = {**features, "cyclic_period": np.zeros((1, 3), np.float32)}
    expected = dense_relative_features(reference_features)
    params = {"linear_layer": {"kernel": jnp.eye(139)}}

    def run(features):
        return relative_position_forward(params, features, cyclic_pos_enc=False)

    actual = (jax.jit(run) if jit else run)(jax.tree.map(jnp.asarray, features))
    np.testing.assert_array_equal(actual, expected)
