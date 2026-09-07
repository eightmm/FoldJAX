"""CPU-only contracts for the same-process OpenDDE repeat diagnostic."""

import argparse
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from bench.opendde_repeat_forward import (
    compare_arrays,
    normalized_inputs,
    observe_jit_owner,
    owner_evidence,
    repeat_compiled_forward,
    repeat_forward,
    tape_digest,
    validate_control,
)


def tape():
    return {
        "init_noise": np.zeros((5, 2, 3), np.float32),
        "step_noises": np.zeros((7, 5, 2, 3), np.float32),
        "rotations": np.broadcast_to(np.eye(3, dtype=np.float32), (7, 5, 3, 3)),
        "translations": np.zeros((7, 5, 3), np.float32),
        "cycle_msa_features": ({"msa": np.arange(4, dtype=np.int32)},),
    }


def test_identical_nested_arrays_and_numerical_difference():
    reference = {"coords": np.array([1, 2], np.float32), "score.0": np.array(7)}
    result = compare_arrays(reference, reference)
    assert result["all_finite_bitwise_equal"]
    assert result["reference_leaves"] == 2
    candidate = {**reference, "coords": np.array([1, 3], np.float32)}
    result = compare_arrays(reference, candidate)
    assert not result["all_bitwise_equal"]
    assert result["leaves"]["coords"]["max_abs"] == 1
    assert result["leaves"]["coords"]["rmse"] == pytest.approx(np.sqrt(0.5))


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "dtype", "signed_zero"])
def test_comparison_cannot_ignore_schema_or_bytes(fault):
    reference = {"coords": np.zeros(2, np.float32)}
    candidate = {name: value.copy() for name, value in reference.items()}
    if fault == "missing":
        candidate.clear()
    elif fault == "extra":
        candidate["extra"] = np.zeros(2)
    elif fault == "shape":
        candidate["coords"] = candidate["coords"].reshape(1, 2)
    elif fault == "dtype":
        candidate["coords"] = candidate["coords"].astype(np.float64)
    else:
        candidate["coords"][0] = -0.0
    result = compare_arrays(reference, candidate)
    assert not result["all_finite_bitwise_equal"]
    assert not result["all_bitwise_equal"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_equal_nonfinite_is_not_valid_evidence(value):
    arrays = {"coords": np.array([value], np.float32)}
    result = compare_arrays(arrays, arrays)
    assert result["all_bitwise_equal"]
    assert not result["all_finite_bitwise_equal"]
    assert result["leaves"]["coords"]["max_abs"] is None
    json.dumps(result, allow_nan=False)
    assert not compare_arrays({}, {})["all_finite_bitwise_equal"]


def test_repeat_reuses_every_argument_value_and_returns_first_object(tmp_path):
    features, params, kwargs = object(), object(), tape()
    outputs = [{"coords": np.array([i], np.float32)} for i in range(3)]
    calls, synced = [], []

    def infer(*args, **kw):
        assert args[0] is features and args[1] is params
        assert kw.keys() == kwargs.keys()
        assert all(kw[name] is kwargs[name] for name in kw)
        calls.append((args, kw))
        return outputs[len(calls) - 1]

    result = repeat_forward(
        infer,
        (features, params),
        kwargs,
        out=tmp_path,
        synchronize=synced.append,
        to_host=lambda x: x,
        observe=lambda: nullcontext([]),
    )
    assert result is outputs[0]
    assert len(calls) == len(synced) == 3
    assert all(value is outputs[i] for i, value in enumerate(synced))
    evidence = json.loads((tmp_path / "repeat-forward.json").read_text())
    assert evidence["explicit_tapes_unchanged"]
    assert not evidence["all_finite_bitwise_equal"]
    assert set(evidence["pairwise"]) == {"1_vs_2", "1_vs_3", "2_vs_3"}
    assert evidence["pairwise"]["1_vs_3"]["leaves"]["coords"]["max_abs"] == 2
    assert not evidence["compiled_owner_evidence"]["same_jit_owner_observed"]
    with np.load(tmp_path / "repeat-forward-1.npz") as saved:
        np.testing.assert_array_equal(saved["coords"], outputs[0]["coords"])


@pytest.mark.parametrize("mutation", ["value", "object"])
def test_reused_tape_mutation_fails_before_second_call(tmp_path, mutation):
    kwargs, calls = tape(), []

    def infer(**kw):
        calls.append(True)
        if mutation == "value":
            kw["translations"].flat[0] = 1
        else:
            cycle = kw["cycle_msa_features"][0]
            cycle["msa"] = cycle["msa"].copy()
        return {"coords": np.zeros(1)}

    with pytest.raises(RuntimeError, match="mutated"):
        repeat_forward(
            infer,
            (),
            kwargs,
            out=tmp_path,
            synchronize=lambda x: None,
            to_host=lambda x: x,
            observe=lambda: nullcontext([]),
        )
    assert calls == [True]
    assert not (tmp_path / "repeat-forward.json").exists()


def test_tape_hash_covers_nested_cycles_and_dtype_but_not_weights():
    kwargs = tape()
    before = tape_digest(kwargs, lambda x: x)
    kwargs["params"] = object()
    assert tape_digest(kwargs, lambda x: x) == before
    kwargs["init_noise"] = kwargs["init_noise"].astype(np.float64)
    assert tape_digest(kwargs, lambda x: x) != before
    kwargs["init_noise"] = kwargs["init_noise"].astype(np.float32)
    assert tape_digest(kwargs, lambda x: x) == before
    kwargs["cycle_msa_features"][0]["msa"][0] = 1
    assert tape_digest(kwargs, lambda x: x) != before
    kwargs["init_noise"] = None
    with pytest.raises(ValueError, match="complete explicit"):
        tape_digest(kwargs, lambda x: x)


def test_observer_returns_actual_owner_and_records_warm_lease():
    owner = SimpleNamespace(_cache_size=lambda: 1)
    pool = SimpleNamespace(_acquire=lambda identity: owner)
    acquire = pool._acquire
    runs = []
    for _ in range(3):
        with observe_jit_owner(pool) as records:
            assert pool._acquire(("fixed-shape", "high")) is owner
        runs.append({"jit_dispatches": records})
    assert pool._acquire is acquire
    result = owner_evidence(runs)
    assert result["same_jit_owner_observed"]
    assert result["warm_single_entry_owner_observed_on_repeats"]
    assert not result["runtime_executable_identity_verified"]
    runs[2]["jit_dispatches"][0]["owner_id"] += 1
    assert not owner_evidence(runs)["same_jit_owner_observed"]


@pytest.mark.parametrize(
    "arm,consumed,trunk",
    [("native", False, False), ("foldjax", True, False), ("foldjax", False, True)],
)
def test_reject_unsupported_controls(arm, consumed, trunk):
    args = SimpleNamespace(
        arm=arm, capture_consumed_tape=consumed, capture_trunk_boundary=trunk
    )
    with pytest.raises(SystemExit):
        validate_control(argparse.ArgumentParser(), args)


def test_snapshot_is_immutable_and_tree_container_changes_are_not_ignored(tmp_path):
    value = np.zeros(1, np.float32)
    outputs = iter([{"scores": [value]}, {"scores": (value,)}, {"scores": [value]}])

    def infer(**kw):
        value[0] += 1
        return next(outputs)

    repeat_forward(
        infer,
        (),
        tape(),
        out=tmp_path,
        synchronize=lambda x: None,
        to_host=lambda x: x,
        observe=lambda: nullcontext([]),
    )
    evidence = json.loads((tmp_path / "repeat-forward.json").read_text())
    assert not evidence["pairwise"]["1_vs_2"]["structure_equal"]
    assert evidence["pairwise"]["1_vs_3"]["leaves"]["scores.0"]["max_abs"] == 2
    with np.load(tmp_path / "repeat-forward-1.npz") as saved:
        np.testing.assert_array_equal(saved["scores.0"], [1])


@pytest.mark.parametrize("flag", ["--capture-consumed-tape", "--capture-trunk-b"])
def test_real_capture_parser_rejects_observers_before_creating_output(
    tmp_path, monkeypatch, flag
):
    from bench.opendde_repeat_forward import main

    out = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repeat-forward",
            "foldjax",
            "--input",
            "not-used.json",
            "--repo",
            ".",
            "--native-source",
            ".",
            "--legacy-driver",
            "not-used.py",
            "--out",
            str(out),
            flag,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not out.exists()


@pytest.mark.parametrize("boundary_args", [[], ["--repeat-boundary", "public"]])
def test_outer_wrapper_is_installed_before_capture_binds_infer(
    tmp_path, monkeypatch, boundary_args
):
    import jax

    from bench import opendde_closure_capture as capture
    from bench import opendde_repeat_forward as repeat
    from foldjax.models.opendde.cli import predict as cli
    from foldjax.models.opendde.models import model

    kwargs, calls = tape(), []
    expected = [{"coords": np.array([i], np.float32)} for i in range(3)]

    def underlying(*a, **kw):
        calls.append(kw)
        return expected[len(calls) - 1]

    def fake_capture():
        parser = argparse.ArgumentParser()
        parser.set_defaults(
            arm="foldjax",
            out=tmp_path,
            capture_consumed_tape=False,
            capture_trunk_boundary=False,
        )
        args = parser.parse_args(boundary_args)
        assert args.repeat_boundary == "public"
        # This reproduces closure_capture's late binding and subsequent replay.
        infer = cli._predict
        assert infer is not underlying
        result = infer(object(), **kwargs)
        assert result is expected[0]

    original_parse = argparse.ArgumentParser.parse_args
    monkeypatch.setattr(cli, "_predict", underlying)
    monkeypatch.setattr(capture, "main", fake_capture)
    monkeypatch.setattr(repeat, "observe_jit_owner", lambda pool: nullcontext([]))
    monkeypatch.setattr(jax, "block_until_ready", lambda x: x)
    monkeypatch.setattr(jax, "effects_barrier", lambda: None)
    monkeypatch.setattr(jax, "device_get", lambda x: x)
    assert model._compiled_opendde_infer is not None
    repeat.main()
    assert len(calls) == 3
    assert all(kw[name] is kwargs[name] for kw in calls for name in kwargs)
    assert cli._predict is underlying
    assert argparse.ArgumentParser.parse_args is original_parse
    assert (tmp_path / "repeat-forward-source.json").exists()
    evidence = json.loads((tmp_path / "repeat-forward.json").read_text())
    assert evidence["repeat_boundary"] == "public"
    assert not evidence["same_normalized_argument_objects"]


class FakePool:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []
        self.owner = SimpleNamespace(_cache_size=lambda: int(bool(self.calls)))

    def _acquire(self, identity):
        return self.owner

    def __call__(self, *args, **kwargs):
        self._acquire("fixed normalized signature")
        self.calls.append((args, kwargs))
        return self.outputs[len(self.calls) - 1]


def test_compiled_main_normalizes_once_and_bypasses_other_pools(tmp_path, monkeypatch):
    import jax

    from bench import opendde_closure_capture as capture
    from bench import opendde_repeat_forward as repeat
    from foldjax.models.opendde.cli import predict as cli
    from foldjax.models.opendde.models import model

    outputs = [{"coords": np.array([i], np.float32)} for i in range(3)]
    pool = FakePool(outputs)
    other_result = object()
    other_pool = FakePool([other_result])
    public_calls, normalized = [], {}
    tapes = tape()
    weight_sentinel = object()

    def underlying(features, params, **kw):
        public_calls.append((features, params, kw))
        normalized.update(
            features={"restype": np.ones((3, 32), np.float32)},
            schedule=np.linspace(2, 0, 8, dtype=np.float32),
            key=np.array([0, 101], np.uint32),
        )
        assert other_pool("unrelated") is other_result
        return model._compiled_opendde_infer(
            normalized["features"],
            (weight_sentinel,),
            normalized["schedule"],
            key=normalized["key"],
            **kw,
        )

    def fake_capture():
        parser = argparse.ArgumentParser()
        parser.set_defaults(
            arm="foldjax",
            out=tmp_path,
            capture_consumed_tape=False,
            capture_trunk_boundary=False,
        )
        args = parser.parse_args(["--repeat-boundary", "compiled"])
        assert args.repeat_boundary == "compiled"
        infer = cli._predict
        assert infer is not underlying
        assert infer(object(), weight_sentinel, **tapes) is outputs[0]

    monkeypatch.setattr(cli, "_predict", underlying)
    monkeypatch.setattr(model, "_compiled_opendde_infer", pool)
    monkeypatch.setattr(capture, "main", fake_capture)
    monkeypatch.setattr(jax, "block_until_ready", lambda x: x)
    monkeypatch.setattr(jax, "effects_barrier", lambda: None)
    monkeypatch.setattr(jax, "device_get", lambda x: x)
    original_call = FakePool.__call__
    repeat.main()
    assert FakePool.__call__ is original_call
    assert len(public_calls) == len(other_pool.calls) == 1
    assert len(pool.calls) == 3
    for args, kwargs in pool.calls:
        assert args[0] is normalized["features"]
        assert args[1] is pool.calls[0][0][1]
        assert args[2] is normalized["schedule"]
        assert kwargs["key"] is normalized["key"]
        assert all(kwargs[name] is tapes[name] for name in tapes)
    evidence = json.loads((tmp_path / "repeat-forward.json").read_text())
    assert evidence["repeat_boundary"] == "compiled"
    assert evidence["same_normalized_argument_objects"]
    assert evidence["normalized_inputs_unchanged"]
    assert not evidence["weight_bytes_hashed"]
    assert set(evidence["normalized_input_fields"]) == {
        "features",
        "noise_schedule",
        "key",
        *tapes,
    }
    digests = {
        run[field]
        for run in evidence["runs"]
        for field in ("normalized_input_sha256_before", "normalized_input_sha256_after")
    }
    assert len(digests) == 1 and None not in digests
    owners = evidence["compiled_owner_evidence"]
    assert owners["same_jit_owner_observed"]
    assert owners["warm_single_entry_owner_observed_on_repeats"]
    assert not owners["runtime_executable_identity_verified"]


@pytest.mark.parametrize("field", ["features", "noise_schedule", "key"])
def test_compiled_non_weight_input_mutation_fails(tmp_path, field):
    args = ({"feature": np.ones(2, np.float32)}, (object(),), np.ones(8, np.float32))
    kwargs = {**tape(), "key": np.array([0, 101], np.uint32)}
    normalized = normalized_inputs(args, kwargs)

    def infer(*args, **kwargs):
        value = normalized[field]
        if field == "features":
            value = value["feature"]
        value.flat[0] += 1
        return {"coords": np.zeros(1, np.float32)}

    with pytest.raises(RuntimeError, match="normalized input values mutated"):
        repeat_forward(
            infer,
            args,
            kwargs,
            boundary="compiled",
            out=tmp_path,
            synchronize=lambda x: None,
            to_host=lambda x: x,
            observe=lambda: nullcontext([]),
        )
    assert not (tmp_path / "repeat-forward.json").exists()


@pytest.mark.parametrize("count", [0, 2])
def test_compiled_control_requires_one_underlying_target_dispatch(tmp_path, count):
    pool = FakePool([{"coords": np.zeros(1)} for _ in range(3)])
    original = FakePool.__call__
    args = ({"feature": np.ones(2)}, (object(),), np.ones(8))
    kwargs = {**tape(), "key": np.array([0, 101], np.uint32)}

    def infer():
        for _ in range(count):
            pool(*args, **kwargs)

    with pytest.raises(RuntimeError, match="target pool"):
        repeat_compiled_forward(
            infer,
            (),
            {},
            pool=pool,
            out=tmp_path,
            synchronize=lambda x: None,
            to_host=lambda x: x,
            observe=lambda: observe_jit_owner(pool),
        )
    assert FakePool.__call__ is original
