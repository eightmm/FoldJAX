"""ESMC's token-layout storage changes ownership, never its layer arithmetic."""

from __future__ import annotations

import dataclasses
import gc
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import esmc
from foldjax.models.esmfold2.models import model as structure_model
from tests.models.esmfold2.test_deterministic_ops import (
    _toy_parameters,
    _toy_settings,
)
from tests.models.esmfold2.test_lm_embedding_cache import _parameters


def _features():
    return (
        np.array(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18],
                [20, 21, 22, 23, 24, 25, 26, 27, 28],
            ],
            np.int32,
        ),
        np.array([[0, 0, 0, 1, 1, 2, 2, 2, 2], [0, 0, 0, 0, 1, 1, 1, 2, 2]], np.int32),
        np.array([[0, 1, 1, 0, 1, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 0, 1]], np.int32),
        np.array([[0, 0, 0, 0, 0, 1, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 2, 0]], np.int32),
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]], bool),
    )


def _checkpoint(dtype, *, mixed=False):
    parameters = {
        name: value.astype(dtype) for name, value in _toy_parameters().items()
    }
    if mixed:
        name = "transformer.blocks.0.attn.out_proj.weight"
        parameters[name] = parameters[name].astype(jnp.float32)
    return parameters


def _same_bits(left, right):
    a, b = np.asarray(left), np.asarray(right)
    assert a.shape == b.shape and a.dtype == b.dtype
    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("packed_length", [None, 24])
@pytest.mark.parametrize(
    "dtype,autocast,mixed",
    [
        (jnp.float32, False, False),
        (jnp.bfloat16, False, False),
        (jnp.bfloat16, True, False),
        (jnp.float32, True, False),
        (jnp.bfloat16, False, True),
    ],
)
def test_real_tiny_encoder_keeps_every_layer_and_the_original_embedding(
    dtype, autocast, mixed, deterministic, packed_length
):
    settings = dataclasses.replace(_toy_settings(), autocast_bfloat16=autocast)
    params, features = _checkpoint(dtype, mixed=mixed), _features()
    kwargs = dict(
        settings=settings,
        packed_length=packed_length,
        deterministic=deterministic,
        lm_mask_pct=0.4,
        mask_key=jax.random.key(73),
    )
    expected = esmc.lm_hidden_states(*features, params, **kwargs)
    actual = esmc.lm_hidden_states(*features, params, **kwargs, state_storage="buffer")
    _same_bits(actual, expected)
    assert actual.shape == (2, 9, settings.n_layers + 1, settings.d_model)
    # One modified residue occupies two structure tokens, and neither ligands
    # nor padding acquires an ESMC hidden state.
    _same_bits(actual[0, 1], actual[0, 2])
    np.testing.assert_array_equal(np.asarray(actual[0, [5, 6, 8]]), 0)
    np.testing.assert_array_equal(np.asarray(actual[1, 7]), 0)

    parameters = _parameters()
    parameters["language_model.base_z_combine"] = parameters[
        "language_model.base_z_combine"
    ][: settings.n_layers + 1]
    for compute in (jnp.float32, jnp.bfloat16):
        combine = jax.jit(
            lambda states: structure_model.language_model_embedding(
                states, parameters, compute_dtype=compute
            )
        )
        _same_bits(combine(actual), combine(expected))


@pytest.mark.parametrize("deterministic", [False, True])
def test_sink_observes_the_same_states_without_enabling_compiled_blocks(
    monkeypatch, deterministic
):
    settings, parameters = _toy_settings(), _checkpoint(jnp.float32)
    ids = jnp.array([[0, 5, 7, 2]], jnp.int32)
    sequences = jnp.zeros_like(ids)
    calls = []
    original = esmc._block_pool

    def pool(policy):
        calls.append(policy)
        return original(policy)

    monkeypatch.setattr(esmc, "_block_pool", pool)
    expected = esmc.encode(
        ids, sequences, parameters, settings=settings, deterministic=deterministic
    )
    states = []
    result = esmc.encode(
        ids,
        sequences,
        parameters,
        settings=settings,
        deterministic=deterministic,
        state_sink=lambda index, state: states.append((index, np.asarray(state))),
    )
    assert result is None
    assert [index for index, _state in states] == list(range(settings.n_layers + 1))
    for index, state in states:
        _same_bits(state, expected[index])
    assert calls == ([True, True] if deterministic else [])


def test_buffer_path_avoids_the_full_stack_and_post_stack_gather(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("buffer storage entered the full-stack path")

    monkeypatch.setattr(esmc, "hidden_states_for_tokens", forbidden)
    monkeypatch.setattr(esmc.jnp, "stack", forbidden)
    actual = esmc.lm_hidden_states(
        *_features(),
        _checkpoint(jnp.bfloat16),
        settings=dataclasses.replace(_toy_settings(), autocast_bfloat16=True),
        state_storage="buffer",
    )
    assert actual.shape == (2, 9, 4, 8) and actual.dtype == jnp.float32


def test_layer_updates_reuse_one_destination_and_release_previous_states(monkeypatch):
    original, destinations, states = esmc._store_token_state, [], []

    def observe(destination, state, *args):
        pointer = destination.unsafe_buffer_pointer()
        updated = original(destination, state, *args)
        updated.block_until_ready()
        assert destination.is_deleted()
        assert updated.unsafe_buffer_pointer() == pointer
        destinations.append(pointer)
        states.append(weakref.ref(state))
        return updated

    monkeypatch.setattr(esmc, "_store_token_state", observe)
    actual = esmc.lm_hidden_states(
        *_features(),
        _checkpoint(jnp.float32),
        settings=_toy_settings(),
        state_storage="buffer",
    )
    actual.block_until_ready()
    gc.collect()
    assert len(destinations) == 4 and len(set(destinations)) == 1
    assert all(reference() is None for reference in states)


def test_lowering_aliases_the_destination_and_gathers_only_one_layer():
    shape = (2, 9, 4, 8)
    signature = (
        jax.ShapeDtypeStruct(shape, jnp.float32),
        jax.ShapeDtypeStruct((2, 24, 8), jnp.bfloat16),
        jax.ShapeDtypeStruct((2, 9), jnp.int32),
        jax.ShapeDtypeStruct((2, 9), jnp.bool),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    lowered = esmc._store_token_state.lower(*signature)
    hlo = lowered.compiler_ir("hlo").as_hlo_text()
    assert "input_output_alias={ {}: (0, {}, may-alias) }" in hlo
    stablehlo = lowered.as_text()
    gathers = [line for line in stablehlo.splitlines() if "stablehlo.gather" in line]
    assert gathers and all("tensor<2x9x8xbf16>" in line for line in gathers)
    assert "stablehlo.dynamic_update_slice" in stablehlo
    assert "stablehlo.transpose" not in stablehlo
    stats = lowered.compile().memory_analysis()
    assert stats.alias_size_in_bytes == np.prod(shape) * 4


def test_mixed_dtype_preflight_neither_dispatches_nor_retains_weights(monkeypatch):
    from jax._src import compiler, dispatch

    parameters = _checkpoint(jnp.bfloat16, mixed=True)
    references = [weakref.ref(value) for value in parameters.values()]
    ids = jnp.array([[0, 5, 2]], jnp.int32)
    sequences = jnp.zeros_like(ids)

    def forbidden(*args, **kwargs):
        raise AssertionError("a dtype-only preflight reached device execution")

    with monkeypatch.context() as patch:
        patch.setattr(compiler, "compile_or_get_cached", forbidden)
        patch.setattr(dispatch, "apply_primitive", forbidden)
        patch.setattr(esmc, "_block_pool", forbidden)
        assert (
            esmc._hidden_state_storage_dtype(
                ids, sequences, parameters, "", _toy_settings()
            )
            == jnp.float32
        )
    del parameters
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("donate", [False, True])
@pytest.mark.parametrize("barrier", [False, True])
@pytest.mark.parametrize("native_autocast", [False, True])
def test_identical_81_layer_inputs_isolate_storage_from_encoder_variation(
    monkeypatch, donate, barrier, native_autocast
):
    """Replay finite captured values without asking the encoder to reproduce them.

    GPU execution of this same contract is needed to diagnose the reported
    discrepancy. CPU agreement cannot establish GPU kernel repeatability or
    reproduce an upstream layer difference.
    """
    if not donate:
        monkeypatch.setattr(
            esmc, "_store_token_state", jax.jit(esmc._store_token_state.__wrapped__)
        )
    features = _features()
    _, _, expand_map = esmc.pack_lm_inputs(*features, packed_length=24)
    rng = np.random.default_rng(732)
    captured = []
    for index in range(81):
        dtype = jnp.bfloat16 if native_autocast and index < 80 else jnp.float32
        captured.append(jnp.asarray(rng.normal(size=(2, 24, 65)) + index / 17, dtype))
    expected = esmc.hidden_states_for_tokens(jnp.stack(captured), expand_map, 9)
    storage = esmc._TokenStateStorage(expand_map, 80, expected.dtype)
    for index, state in enumerate(captured):
        storage(index, state)
        if barrier:
            storage.destination.block_until_ready()
    _same_bits(storage.destination, expected)


@pytest.mark.parametrize("storage", [None, "stream", False])
def test_storage_validation_precedes_any_encoding(storage):
    with pytest.raises(ValueError, match="state_storage"):
        esmc.lm_hidden_states(
            *_features(), {}, settings=_toy_settings(), state_storage=storage
        )
