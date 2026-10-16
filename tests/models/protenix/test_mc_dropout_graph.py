"""A managed MC-dropout coin is input data, preserving native draws/arithmetic."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix import runner
from foldjax.models.protenix.models import model as model_module
from foldjax.models.protenix.models.predict import protenix_predict_static
from foldjax.models.protenix.models.trunk_blocks import trunk
from tests.models.protenix.test_model import _toy_features, _toy_params
from tests.models.protenix.test_trunk import _pairformer_output_params


def _assert_bits_equal(actual, expected):
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(
            np.asarray(left).reshape(-1).view(np.uint8),
            np.asarray(right).reshape(-1).view(np.uint8),
        )


def _inputs(dtype, *, tokens=3):
    params = _pairformer_output_params()
    params = params._replace(
        pairformer_stack=params.pairformer_stack._replace(blocks=())
    )
    params = jax.tree.map(
        lambda value: value.astype(dtype) if hasattr(value, "dtype") else value,
        params,
    )
    features = {
        "relp": jnp.zeros((tokens, tokens, 2), dtype=dtype),
        "token_bonds": jnp.zeros((tokens, tokens), dtype=dtype),
    }
    inputs = jnp.arange(tokens * 2, dtype=jnp.float32).reshape(tokens, 2) / 7
    return features, inputs, params


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("prng_impl", ["threefry2x32", "rbg"])
def test_coin_outcomes_share_abstract_stablehlo_and_native_trunk_outputs(
    dtype, scan, prng_impl
):
    with jax.default_prng_impl(prng_impl):
        _check_coin_outcomes(dtype, scan)


def _check_coin_outcomes(dtype, scan):
    features, inputs, params = _inputs(dtype)
    keys_off, off = runner._mc_dropout_inputs(
        19, num_recycles=3, apply_rate=0.0, rate=0.4, input_digest=71
    )
    keys_on, on = runner._mc_dropout_inputs(
        19, num_recycles=3, apply_rate=1.0, rate=0.4, input_digest=71
    )

    def abstract(tree):
        return jax.tree.map(
            lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), tree
        )

    assert jax.tree.structure((keys_off, off)) == jax.tree.structure((keys_on, on))
    assert abstract((keys_off, off)) == abstract((keys_on, on))
    assert keys_off.shape == (3, *jax.random.key_data(jax.random.key(0)).shape)

    def run(features, inputs, params, keys, enabled):
        return trunk.pairformer_output_from_s_inputs(
            features,
            inputs,
            params,
            num_recycles=3,
            use_cycle_scan=scan,
            cycle_pair_dropout_keys=keys,
            pair_dropout_enabled=enabled,
        )

    program = jax.jit(run)
    args_off = (features, inputs, params, keys_off, off)
    args_on = (features, inputs, params, keys_on, on)
    hlo_off = str(program.lower(*abstract(args_off)).compiler_ir("stablehlo"))
    hlo_on = str(program.lower(*abstract(args_on)).compiler_ir("stablehlo"))
    assert hlo_off == hlo_on
    assert "stablehlo.case" in hlo_on
    actual_off, actual_on = program(*args_off), program(*args_on)
    assert program._cache_size() == 1

    def native(features, inputs, params, keys=None):
        return trunk.pairformer_output_from_s_inputs(
            features,
            inputs,
            params,
            num_recycles=3,
            use_cycle_scan=scan,
            cycle_pair_dropout_keys=keys,
        )

    with jax.numpy_dtype_promotion("strict"):
        expected_off = jax.jit(native)(features, inputs, params)
        expected_on = jax.jit(native)(features, inputs, params, keys_on)
    _assert_bits_equal(actual_off, expected_off)
    _assert_bits_equal(actual_on, expected_on)
    # Construct each cycle's mask independently: RBG's vmap semantics differ
    # from its sequential draws and must not become this replay reference.
    masks = jnp.stack(
        [jax.random.bernoulli(key, p=0.6, shape=actual_on[-1].shape) for key in keys_on]
    )
    expected_replay = jax.jit(
        lambda features, inputs, params, masks: trunk.pairformer_output_from_s_inputs(
            features,
            inputs,
            params,
            num_recycles=3,
            use_cycle_scan=scan,
            cycle_pair_dropout_keep_masks=masks,
        )
    )(features, inputs, params, masks)
    _assert_bits_equal(actual_on, expected_replay)
    assert actual_off[-1].dtype == actual_on[-1].dtype == dtype


@pytest.mark.parametrize("prng_impl,width", [("threefry2x32", 2), ("rbg", 4)])
def test_disabled_host_inputs_do_not_draw_random_keys(monkeypatch, prng_impl, width):
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled MC dropout drew a key")

    monkeypatch.setattr(jax.random, "fold_in", unexpected)
    monkeypatch.setattr(jax.random, "split", unexpected)
    with jax.default_prng_impl(prng_impl):
        keys, enabled = runner._mc_dropout_inputs(
            19, num_recycles=3, apply_rate=0.0, rate=0.4, input_digest=71
        )
    assert keys.shape == (3, width) and keys.dtype == jnp.uint32
    assert not bool(enabled)
    np.testing.assert_array_equal(keys, 0)


def test_disabled_runtime_branch_does_not_generate_masks(monkeypatch):
    features, inputs, params = _inputs(jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(27), 3)
    draws = []
    original = jax.random.bernoulli

    def recording(*args, **kwargs):
        jax.debug.callback(lambda: draws.append(True), ordered=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(jax.random, "bernoulli", recording)
    program = jax.jit(
        lambda enabled: trunk.pairformer_output_from_s_inputs(
            features,
            inputs,
            params,
            num_recycles=3,
            cycle_pair_dropout_keys=keys,
            pair_dropout_enabled=enabled,
        )
    )
    jax.block_until_ready(program(jnp.asarray(False)))
    jax.effects_barrier()
    assert not draws
    jax.block_until_ready(program(jnp.asarray(True)))
    jax.effects_barrier()
    assert len(draws) == 3


@pytest.mark.parametrize("prng_impl", ["threefry2x32", "rbg"])
def test_disabled_dropout_preserves_the_legacy_key_rejection_policy(prng_impl):
    with jax.default_prng_impl(prng_impl), jax.legacy_prng_key("error"):
        features, inputs, params = _inputs(jnp.float32)
        keys, enabled = runner._mc_dropout_inputs(
            19, num_recycles=3, apply_rate=0.0, rate=0.4, input_digest=71
        )

        def run(features, inputs, params, keys=None, enabled=None):
            return trunk.pairformer_output_from_s_inputs(
                features,
                inputs,
                params,
                num_recycles=3,
                cycle_pair_dropout_keys=keys,
                pair_dropout_enabled=enabled,
            )

        program = jax.jit(run)
        expected = program(features, inputs, params)
        actual = program(features, inputs, params, keys, enabled)
        _assert_bits_equal(actual, expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_enabled_padded_dropout_keeps_the_native_real_prefix(dtype):
    features, inputs, params = _inputs(dtype)
    keys = jax.random.split(jax.random.PRNGKey(27), 3)
    expected = jax.jit(
        lambda features, inputs, params, keys: trunk.pairformer_output_from_s_inputs(
            features,
            inputs,
            params,
            num_recycles=3,
            cycle_pair_dropout_keys=keys,
            use_cycle_scan=False,
        )
    )(features, inputs, params, keys)
    padded, _, _ = _inputs(dtype, tokens=6)
    padded_inputs = jnp.pad(inputs, ((0, 3), (0, 0)))
    valid = jnp.arange(6) < 3
    actual = jax.jit(
        lambda features, inputs, params, keys, enabled: (
            trunk.pairformer_output_from_s_inputs(
                features,
                inputs,
                params,
                num_recycles=3,
                cycle_pair_dropout_keys=keys,
                pair_dropout_enabled=enabled,
                pair_dropout_valid_mask=valid[:, None] & valid[None, :],
                use_cycle_scan=False,
            )
        )
    )(padded, padded_inputs, params, keys, jnp.asarray(True))
    _assert_bits_equal(actual[1][:3], expected[1])
    _assert_bits_equal(actual[2][:3, :3], expected[2])


@pytest.mark.parametrize("prng_impl", ["threefry2x32", "rbg"])
def test_managed_coin_outcomes_reuse_one_whole_model_executable(prng_impl):
    with jax.default_prng_impl(prng_impl):
        _check_whole_model_executable()


def _check_whole_model_executable():
    params, features = _toy_params(), _toy_features()
    model_module._compiled_protenix_infer.clear_cache()
    try:
        for apply_rate in (0.0, 1.0):
            keys, enabled = runner._mc_dropout_inputs(
                19, num_recycles=2, apply_rate=apply_rate, rate=0.4, input_digest=71
            )
            output = protenix_predict_static(
                params,
                features,
                key=jax.random.PRNGKey(19),
                num_samples=1,
                num_sampling_steps=1,
                recycling_steps=2,
                input_atom_heads=1,
                atom_encoder_heads=1,
                token_heads=1,
                atom_decoder_heads=1,
                n_queries=2,
                n_keys=4,
                stop_after_trunk=True,
                cycle_pair_dropout_keys=keys,
                pair_dropout_enabled=enabled,
                diffusion_attention_backend="xla_jit",
            )
            jax.block_until_ready(output)
        assert model_module._compiled_protenix_infer._cache_size() == 1
    finally:
        model_module._compiled_protenix_infer.clear_cache()
