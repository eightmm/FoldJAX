"""Host selection must preserve native draws while hiding row depth from JIT."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.backends.openfold3 import _padding_plan
from foldjax.models.openfold3 import streaming
from foldjax.models.openfold3.data import (
    compact_msa_features,
    prepare_msa_cycle_features,
)
from foldjax.models.openfold3.data.featurize import _MSA_CYCLE_INDICES
from foldjax.models.openfold3.models.input_embedders import msa_embedder
from foldjax.schema import PaddingConfig
from tests.models.openfold3.test_compact_msa_storage import (
    _params,
    _varied_msa_features,
)
from tests.models.openfold3.test_stable_compile import _config, _table


@pytest.mark.parametrize("rows", [1, 17, 1024, 1025, 4097])
def test_native_selection_preserved_and_cycle_capacity_is_fixed(rows):
    source = _varied_msa_features(rows=rows, tokens=2)
    prepared = prepare_msa_cycle_features(
        source,
        1024,
        num_recycles=4,
        rng=np.random.default_rng(13),
    )
    plan = _padding_plan(prepared, PaddingConfig())
    assert plan.target["msa"] == 1024
    host = streaming.HostMSACycles(prepared, depth=1024, cycles=4)
    assert not (host.common.keys() & streaming._MSA_KEYS)
    for i in range(4):
        selected = host.select(i)
        indices = prepared[_MSA_CYCLE_INDICES][i]
        assert selected["msa_mask"].shape == (1, 1024, 2)
        for name in ("msa", "msa_mask", "has_deletion", "deletion_value"):
            np.testing.assert_array_equal(
                selected[name][:, : len(indices)],
                prepared[name][:, indices],
            )
            assert not np.any(selected[name][:, len(indices) :])
    if rows > 1024:
        assert prepared["msa_mask"].shape[1] > 1024


def test_compact_padding_matches_dense_embedding_and_mask():
    prepared = prepare_msa_cycle_features(
        _varied_msa_features(rows=3),
        8,
        num_recycles=4,
        rng=np.random.default_rng(1),
    )
    dense = streaming.HostMSACycles(prepared, depth=8, cycles=4).select(0)
    compact = streaming.HostMSACycles(
        compact_msa_features(prepared),
        depth=8,
        cycles=4,
    ).select(0)
    params = _params(jnp.float32)
    single = jnp.ones((1, 5, 7))
    left = jax.jit(msa_embedder)(dense, single, params)
    right = jax.jit(msa_embedder)(compact, single, params)
    for a, b in zip(left, right, strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.any(np.asarray(right[1])[:, 3:])


class TinyParams(NamedTuple):
    trunk: jax.Array


@pytest.fixture
def stage_probe(monkeypatch):
    """Small noncommutative recurrence probes real staging and JIT reuse."""
    traces = []
    transfers = []
    events = []

    def initialize(batch, params, **kwargs):
        assert not batch.keys() & streaming._MSA_KEYS
        traces.append("initialize")
        x = batch["token_mask"] * params
        return x, x, x[..., None]

    def cycle(batch, msa, params, initial, carry, **kwargs):
        traces.append(("cycle", msa["msa_mask"].shape))
        assert not batch.keys() & streaming._MSA_KEYS
        signal = jnp.sum(msa["deletion_value"] * msa["msa_mask"], axis=1)
        return carry[0] + signal, carry[1] * 2 + signal[..., None] + initial[2] * params

    def finish(key, batch, params, config, table, *, trunk_output, **kwargs):
        assert not batch.keys() & streaming._MSA_KEYS
        traces.append("finish")
        return trunk_output

    original_replicate = streaming.replicate_tree

    def replicate(value):
        if isinstance(value, dict) and "msa_mask" in value:
            assert all(isinstance(a, np.ndarray) for a in value.values())
            transfers.append(value["msa_mask"].shape)
            events.append("transfer")
        return original_replicate(value)

    original_block = jax.block_until_ready

    def block(value):
        events.append("fence")
        return original_block(value)

    monkeypatch.setattr(streaming, "initialize_trunk", initialize)
    monkeypatch.setattr(streaming, "trunk_cycle", cycle)
    monkeypatch.setattr(streaming.inf, "_predict_from_trunk", finish)
    monkeypatch.setattr(streaming, "replicate_tree", replicate)
    monkeypatch.setattr(jax, "block_until_ready", block)
    streaming._compiled_streams.clear_cache()
    yield traces, transfers, events
    streaming._compiled_streams.clear_cache()


def test_host_depth_changes_reuse_all_stages_and_keep_native_carry(stage_probe):
    traces, transfers, events = stage_probe
    config = _config(n_token=2, msa_depth=8, num_recycles=4)
    for rows in (1, 3, 17):
        batch = prepare_msa_cycle_features(
            _varied_msa_features(rows=rows, tokens=2),
            8,
            num_recycles=4,
            rng=np.random.default_rng(8),
        )
        # Repeated factories must share their JIT owner too.
        run = streaming.compile_streamed_predict(config, _table())
        actual = run(jax.random.key(0), batch, TinyParams(jnp.asarray(0.5)))
        s = np.zeros((1, 2), dtype=np.float32)
        z = s.copy()
        for ids in batch[_MSA_CYCLE_INDICES]:
            signal = np.sum(
                batch["deletion_value"][:, ids] * batch["msa_mask"][:, ids],
                axis=1,
            )
            s = s + signal
            z = 2 * z + signal + 0.25
        np.testing.assert_allclose(actual[1], s, atol=1e-5)
        np.testing.assert_allclose(actual[2][..., 0], z, atol=1e-5)
    assert traces == ["initialize", ("cycle", (1, 8, 2)), "finish"]
    assert transfers == [(1, 8, 2)] * 12
    assert (
        events
        == [
            "transfer",
            "transfer",
            "fence",
            "transfer",
            "fence",
            "transfer",
            "fence",
            "fence",
        ]
        * 3
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((3, 2), dtype=np.int32),
        np.full((4, 2), -1, dtype=np.int32),
        np.zeros((4, 9), dtype=np.int32),
    ],
)
def test_invalid_indices_fail_before_device_transfer(bad):
    batch = _varied_msa_features(rows=3)
    batch[_MSA_CYCLE_INDICES] = bad
    with pytest.raises(ValueError, match="cycle indices"):
        streaming.HostMSACycles(batch, depth=8, cycles=4)


def test_eager_stream_preserves_the_compiled_cycle_order(stage_probe):
    config = _config(n_token=2, msa_depth=8, num_recycles=4)
    batch = prepare_msa_cycle_features(
        _varied_msa_features(rows=17, tokens=2),
        8,
        num_recycles=4,
        rng=np.random.default_rng(8),
    )
    args = (jax.random.key(0), batch, TinyParams(jnp.asarray(0.5)))
    compiled = streaming.compile_streamed_predict(config, _table())(*args)
    eager = streaming.compile_streamed_predict(config, _table(), compiled=False)(*args)
    for left, right in zip(compiled, eager, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)


def test_streamed_owner_partitions_chain_and_cache_scope(stage_probe, tmp_path):
    traces, _, _ = stage_probe
    batch = prepare_msa_cycle_features(
        _varied_msa_features(rows=3, tokens=2),
        8,
        num_recycles=4,
        rng=np.random.default_rng(8),
    )
    for chain, scope in ((1, "a"), (2, "a"), (2, "b"), (1, "a")):
        streaming.compile_streamed_predict(
            _config(n_token=2, msa_depth=8, num_recycles=4),
            _table(),
            n_chain=chain,
            cache_scope=str(tmp_path / scope),
        )(jax.random.key(0), batch, TinyParams(jnp.asarray(0.5)))
    assert len(traces) == 9
