"""CPU contracts for explicitly indexed execution-tape evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from bench.opendde_consumed_tape import (
    ConsumedTape,
    expected_events,
    observe_consumption,
)


def native_reference():
    tape = {
        "noise_schedule": np.linspace(2, 0.1, 201, dtype=np.float32),
        "init_noise": np.zeros((5, 2, 3), np.float32),
        "step_noises": np.zeros((200, 5, 2, 3), np.float32),
        "rotations": np.broadcast_to(
            np.eye(3, dtype=np.float32), (200, 5, 3, 3)
        ).copy(),
        "translations": np.zeros((200, 5, 3), np.float32),
    }
    cycles = [{"msa": np.full((2, 3), i, np.int32)} for i in range(10)]
    return tape, cycles


def test_complete_explicit_indices_allow_async_delivery():
    expected = expected_events(*native_reference())
    recorder = ConsumedTape(expected)
    for (kind, index), values in reversed(list(expected.items())):
        recorder.observe(kind, index, values)
    result = recorder.finish()
    assert result["passed"] and result["events"] == 211
    assert result["requires_uninstrumented_output_bridge"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "unexpected",
        "reordered",
        "value",
        "nan",
        "inf",
        "dtype",
        "shape",
        "field",
        "float_index",
        "signed_zero",
    ],
)
def test_strict_failure_contract(fault):
    expected = {("msa", i): {"msa": np.array([i], np.float32)} for i in range(2)}
    recorder = ConsumedTape(expected)
    for i in range(2):
        if fault == "missing" and i == 1:
            continue
        value = expected["msa", i]["msa"].copy()
        if i == 0:
            if fault == "value":
                value[0] = 7
            if fault == "nan":
                value[0] = np.nan
            if fault == "inf":
                value[0] = np.inf
            if fault == "dtype":
                value = value.astype(np.float64)
            if fault == "shape":
                value = value.reshape(1, 1)
            if fault == "signed_zero":
                value[0] = -0.0
        values = {"msa": value}
        if fault == "field" and i == 0:
            values = {"other": value}
        index = 1 - i if fault == "reordered" else i
        if fault == "float_index":
            index = float(index)
        recorder.observe("msa", index, values)
    if fault == "duplicate":
        recorder.observe("msa", 0, expected["msa", 0])
    if fault == "unexpected":
        recorder.observe("msa", 3, expected["msa", 0])
    assert not recorder.finish()["passed"]


def test_nonfinite_reference_cannot_pass():
    values = {"x": np.array([np.nan], np.float32)}
    recorder = ConsumedTape({("step", 0): values})
    recorder.observe("step", 0, values)
    assert not recorder.finish()["passed"]
    assert not ConsumedTape({}).finish()["passed"]


@pytest.mark.parametrize(
    "fault", ["length", "dtype", "nonfinite", "cycles", "cycle_nonfinite"]
)
def test_native_reference_validation(fault):
    tape, cycles = native_reference()
    if fault == "length":
        tape["noise_schedule"] = tape["noise_schedule"][:-1]
    if fault == "dtype":
        tape["init_noise"] = tape["init_noise"].astype(np.float64)
    if fault == "nonfinite":
        tape["translations"][0, 0, 0] = np.inf
    if fault == "cycles":
        cycles.pop()
    if fault == "cycle_nonfinite":
        cycles[0]["msa"] = np.array([np.nan])
    with pytest.raises(ValueError):
        expected_events(tape, cycles)


def test_real_sampler_scan_cpu_observation_and_output_bridge():
    import jax
    import jax.numpy as jnp

    from foldjax.models.opendde.models import sampling

    tape, cycles = native_reference()
    expected = {
        key: value
        for key, value in expected_events(tape, cycles).items()
        if key[0] != "msa"
    }
    recorder = ConsumedTape(expected)
    kwargs = dict(tape)
    schedule = kwargs.pop("noise_schedule")
    kwargs.update(num_samples=5, n_atom=2, key=jax.random.key(0), use_scan=True)
    original = sampling.sample_diffusion
    with jax.default_device(jax.devices("cpu")[0]):
        ordinary = original(lambda x, t: x * 0.5, schedule, **kwargs)
        with observe_consumption(recorder) as hashes:
            observed = sampling.sample_diffusion(
                lambda x, t: x * 0.5, schedule, **kwargs
            )
        np.testing.assert_array_equal(np.asarray(ordinary), np.asarray(observed))
        assert recorder.finish()["passed"], recorder.finish()
        assert recorder.finish()["events"] == 201
        assert len(hashes) == 2
        assert sampling.sample_diffusion is original
        with pytest.raises(ValueError, match="unpadded"):
            with observe_consumption(ConsumedTape(expected)):
                sampling.sample_diffusion(
                    lambda x, t: x, schedule, **kwargs, atom_mask=jnp.ones(2)
                )
        assert sampling.sample_diffusion is original


def test_real_cycle_scan_observes_consumer_cast_with_stubbed_neural_blocks(monkeypatch):
    import jax
    import jax.numpy as jnp

    from foldjax.models.protenix.models.trunk_blocks import trunk

    # Keep the actual recycling scan and consumer dtype conversion; replace only
    # neural computation so no weights or accelerator are required.
    monkeypatch.setattr(trunk, "_parameter_dtype", lambda p: jnp.float32)
    monkeypatch.setattr(
        trunk, "resolve_relative_position_features", lambda f: jnp.zeros((2, 2, 1))
    )
    monkeypatch.setattr(trunk, "shard_pair_rows", lambda x: x)
    monkeypatch.setattr(
        trunk, "trunk_initial_embeddings", lambda s, *a, **k: (s, jnp.zeros((2, 2, 1)))
    )
    monkeypatch.setattr(trunk, "recycle_embeddings", lambda s0, z0, s, z, p: (s, z))
    monkeypatch.setattr(
        trunk, "template_embedder", lambda f, z, *a, **k: jnp.zeros_like(z)
    )
    monkeypatch.setattr(trunk, "msa_module", lambda f, z, *a, **k: z)
    params = SimpleNamespace(
        trunk=SimpleNamespace(initial=None, recycling=None),
        template=None,
        msa=None,
        pairformer_stack=SimpleNamespace(blocks=()),
    )
    cycles = tuple(
        {
            "msa": jnp.full((2, 3), i, jnp.int8),
            "deletion_value": jnp.full((2, 3), i, jnp.bfloat16),
        }
        for i in range(10)
    )
    expected = {
        ("msa", i): {
            "msa": np.asarray(c["msa"]),
            "deletion_value": np.asarray(c["deletion_value"], np.float32),
        }
        for i, c in enumerate(cycles)
    }
    recorder = ConsumedTape(expected)
    with jax.default_device(jax.devices("cpu")[0]), observe_consumption(recorder):
        trunk.pairformer_output_from_s_inputs(
            {"token_bonds": jnp.zeros((2, 2))},
            jnp.zeros((2, 1)),
            params,
            num_recycles=10,
            cycle_msa_features=cycles,
        )
    assert recorder.finish()["passed"], recorder.finish()
