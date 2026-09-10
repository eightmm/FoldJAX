from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from bench.protenix_foldjax_capture import (
    infer_boundary_report,
    msa_cycle_index_tape,
    msa_cycles,
    save_jax_boundary,
)


def _msa():
    fields = {
        "msa": np.arange(12, dtype=np.int64).reshape(3, 4),
        "has_deletion": np.zeros((3, 4), np.float32),
        "deletion_value": np.arange(12, dtype=np.float32).reshape(3, 4) / 17,
    }
    tape = {}
    for cycle in range(10):
        # Ragged on purpose: cycle 4 selects one row where the rest select two,
        # which is the shape the padded index tape has to represent.
        rows = np.asarray(
            [cycle % 3] if cycle == 4 else [cycle % 3, (cycle + 1) % 3], np.int64
        )
        tape[f"{cycle}.rows"] = rows
        tape.update(
            {
                f"{cycle}.selected.{name}": values[rows]
                for name, values in fields.items()
            }
        )
    return fields, tape


def test_msa_replay_uses_indices_into_independent_rows():
    features, tape = _msa()
    cycles = msa_cycles(features, tape)
    assert len(cycles) == 10
    for index, cycle in enumerate(cycles):
        for field, values in cycle.items():
            np.testing.assert_array_equal(values, tape[f"{index}.selected.{field}"])


def test_msa_index_tape_pads_short_cycles_and_masks_the_padding():
    features, tape = _msa()
    index_tape = msa_cycle_index_tape(features, tape)
    assert index_tape.row_indices.shape == (10, 2)
    assert index_tape.row_mask.shape == (10, 2)
    assert index_tape.row_indices.dtype == np.int32
    assert index_tape.row_mask.dtype == np.bool_
    for cycle in range(10):
        rows = tape[f"{cycle}.rows"]
        real = int(rows.size)
        assert index_tape.row_mask[cycle].sum() == real
        np.testing.assert_array_equal(index_tape.row_indices[cycle, :real], rows)
        # The ordinary sampler fills a short cycle with index zero and a false
        # mask; matching it keeps one meaning of "padded row" in the trunk.
        np.testing.assert_array_equal(index_tape.row_indices[cycle, real:], 0)
        np.testing.assert_array_equal(index_tape.row_mask[cycle, real:], False)
    assert index_tape.row_mask[4].tolist() == [True, False]


def test_msa_index_tape_gathers_native_rows_back_out_of_foldjax_inputs():
    """A tape nothing can reconstruct native's own values from is not evidence."""
    features, tape = _msa()
    index_tape = msa_cycle_index_tape(features, tape)
    for cycle in range(10):
        real = int(index_tape.row_mask[cycle].sum())
        rows = index_tape.row_indices[cycle, :real]
        for name, values in features.items():
            np.testing.assert_array_equal(
                np.asarray(values)[rows], tape[f"{cycle}.selected.{name}"]
            )


@pytest.mark.parametrize("defect", ["extra", "missing", "rows", "values"])
def test_msa_index_tape_rejects_the_same_unverified_tapes(defect):
    features, tape = _msa()
    if defect == "extra":
        tape["unknown"] = np.ones(1)
    elif defect == "missing":
        tape.pop("9.rows")
    elif defect == "rows":
        tape["0.rows"] = np.asarray([-1, 0])
    else:
        features["deletion_value"][0, 0] = 9
    with pytest.raises(ValueError):
        msa_cycle_index_tape(features, tape)


@pytest.mark.parametrize("defect", ["extra", "missing", "rows", "values"])
def test_msa_replay_rejects_unverified_or_incomplete_tapes(defect):
    features, tape = _msa()
    if defect == "extra":
        tape["unknown"] = np.ones(1)
    elif defect == "missing":
        tape.pop("9.rows")
    elif defect == "rows":
        tape["0.rows"] = np.asarray([-1, 0])
    else:
        features["deletion_value"][0, 0] = 9
    with pytest.raises(ValueError):
        msa_cycles(features, tape)


def test_bf16_output_storage_retains_dtype_and_all_numeric_bits(tmp_path):
    path = tmp_path / "boundary.npz"
    value = jnp.asarray([1.0078125, -0.0, 1.25], jnp.bfloat16)
    save_jax_boundary(path, {"bf": value, "integer": jnp.asarray([1, 2])})
    with np.load(path, allow_pickle=False) as archive:
        assert archive["bf"].dtype == np.float32
        assert archive["bf"].tobytes() == np.asarray(value, np.float32).tobytes()
        assert archive["integer"].dtype.kind == "i"
    metadata = json.loads(path.with_suffix(".tree.json").read_text())
    assert metadata["bf"]["dtype"] == "bfloat16"


@pytest.mark.parametrize("materialised", [False, True])
@pytest.mark.parametrize("graph_jit", [True, False])
@pytest.mark.parametrize("aggregation_control", [False, True, "mismatch"])
def test_replay_wires_complete_tape_through_public_prediction(
    monkeypatch, tmp_path, graph_jit, aggregation_control, materialised
):
    from bench import protenix_closure_report as report
    from bench import protenix_foldjax_capture as capture
    from foldjax.models.protenix.cli import predict as cli
    from foldjax.models.protenix.data import featurize_json
    from foldjax.models.protenix.models import predict

    reference, out = tmp_path / "reference", tmp_path / "out"
    reference.mkdir()
    out.mkdir()
    historical = out / "effective-options.json"
    historical.write_text("historical artifact must not be rewritten")
    input_path = tmp_path / "input.json"
    input_path.write_text("[{}]")
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"test-only mocked loader")
    (reference / "capture-complete.json").write_text(
        json.dumps(
            {
                "passed": True,
                "mc_dropout_applied": False,
            }
        )
    )
    (reference / "provenance.json").write_text(
        json.dumps(
            {
                "model_name": "protenix_base_default_v1.0.0",
                "input_sha256": capture.sha(input_path),
            }
        )
    )
    (reference / "effective-config.json").write_text(
        json.dumps({"enable_efficient_fusion": True})
    )
    (reference / "operator-policy.json").write_text(
        json.dumps({"fp32_atom_aggregation": aggregation_control is not False})
    )
    features, msa = _msa()
    np.savez(reference / "msa-tape.npz", **msa)
    tape = {
        "init_noise": np.ones((5, 3, 3), np.float32),
        "step_noises": np.ones((200, 5, 3, 3), np.float32),
        "rotations": np.broadcast_to(np.eye(3, dtype=np.float32), (200, 5, 3, 3)),
        "translations": np.ones((200, 5, 3), np.float32),
        "noise_schedule": np.linspace(4, 0, 201, dtype=np.float32),
    }
    np.savez(reference / "sampler-tape.npz", **tape)
    monkeypatch.setattr(capture, "record_features", lambda *a: None)
    monkeypatch.setattr(
        featurize_json, "featurize_protein_json", lambda *a, **k: features
    )
    monkeypatch.setattr(report, "check_inputs", lambda *a: {"passed": True})
    observed = {}

    def inference(features, params, noise_schedule, **kwargs):
        observed.update(kwargs)
        observed["noise_schedule"] = noise_schedule
        return {"coordinate": kwargs["init_noise"]}

    def fake_cli(argv):
        result = featurize_json.featurize_protein_json({})
        predict.protenix_predict_static(
            None,
            result,
            None,
            num_samples=5,
            num_sampling_steps=200,
            recycling_steps=10,
            trunk_dtype=jnp.bfloat16,
            graph_jit=graph_jit,
            use_pairformer_scan=False,
            use_confidence_scan=False,
            use_diffusion_scan=False,
        )

    monkeypatch.setattr(predict, "protenix_infer_compiled", inference)
    monkeypatch.setattr(predict, "protenix_infer_static", inference)
    monkeypatch.setattr(cli, "main", fake_cli)
    options = SimpleNamespace(
        reference=reference,
        out=out,
        input=input_path,
        weights=weights,
        fp32_atom_aggregation=aggregation_control is True,
        materialised_msa_cycles=materialised,
    )
    if aggregation_control == "mismatch":
        with pytest.raises(ValueError, match="aggregation policies differ"):
            capture.replay(options)
        return
    capture.replay(options)
    for name, expected in tape.items():
        np.testing.assert_array_equal(observed[name], expected)
    # Exactly one carrier reaches the trunk: the two are mutually exclusive
    # there, and the CLI's own freshly sampled tape must not survive either way.
    if materialised:
        assert len(observed["cycle_msa_features"]) == 10
        assert observed["cycle_msa_index_tape"] is None
    else:
        assert observed["cycle_msa_features"] is None
        index_tape = observed["cycle_msa_index_tape"]
        assert index_tape.row_indices.shape == (10, 2)
        np.testing.assert_array_equal(
            np.asarray(index_tape.row_indices[0]), msa["0.rows"]
        )
        np.testing.assert_array_equal(np.asarray(index_tape.row_mask[4]), [True, False])
    assert observed["use_diffusion_efficient_fusion"] is True
    assert json.loads((out / "capture-complete.json").read_text())["passed"]
    requested = json.loads((out / "requested-wrapper-options.json").read_text())
    boundary = json.loads((out / "infer-boundary.json").read_text())
    assert not requested["use_pairformer_scan"]
    assert requested["replay_msa_cycle_path"] == (
        "materialised" if materialised else "index_tape"
    )
    for name in ("use_pairformer_scan", "use_confidence_scan", "use_diffusion_scan"):
        assert boundary["options"][name] is graph_jit
        assert observed[name] is graph_jit
    route = "compiled" if graph_jit else "static"
    assert (
        boundary["boundary"] == f"predict wrapper -> protenix_infer_{route} host entry"
    )
    assert boundary["options"]["trunk_dtype"] == "bfloat16"
    assert ("padded_generated_schema" in boundary["options"]) is graph_jit
    assert boundary["jax_default_matmul_precision"] == "high"
    assert boundary["sampler_consumer_bytes_observed"] is False
    assert historical.read_text() == "historical artifact must not be rewritten"
    for name, expected in tape.items():
        signature = boundary["tape_inputs"][name]["leaves"][""]
        assert signature == {
            "shape": list(expected.shape),
            "dtype": str(expected.dtype),
            "content_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        }
    if materialised:
        assert boundary["tape_inputs"]["cycle_msa_index_tape"] == {"present": False}
        cycles = boundary["tape_inputs"]["cycle_msa_features"]
        assert len(cycles["leaves"]) == 30
        for index, cycle in enumerate(observed["cycle_msa_features"]):
            for field, value in cycle.items():
                signature = cycles["leaves"][f"{index}.{field}"]
                assert (
                    signature["content_sha256"]
                    == hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                )
    else:
        assert boundary["tape_inputs"]["cycle_msa_features"] == {"present": False}
        # A named tuple flattens positionally, so the leaf keys are ordinals;
        # the pytree string is what records which field each one is.
        carried = boundary["tape_inputs"]["cycle_msa_index_tape"]
        assert "MSACycleIndexTape" in carried["pytree"]
        assert set(carried["leaves"]) == {"0", "1"}
        for leaf, value in zip(
            ("0", "1"), observed["cycle_msa_index_tape"], strict=True
        ):
            assert (
                carried["leaves"][leaf]["content_sha256"]
                == hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
            )
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["infer_boundary_sha256"] == capture.sha(
        out / "infer-boundary.json"
    )


def test_infer_boundary_hashes_bf16_bits_without_widening_and_handles_sequences():
    value = jnp.asarray([1.0078125, -0.0], jnp.bfloat16)
    kwargs = {"step_noises": (value, value + 1), "init_noise": value}
    report = infer_boundary_report("protenix_infer_static", value, kwargs)
    signature = report["tape_inputs"]["init_noise"]["leaves"][""]
    assert signature["dtype"] == "bfloat16"
    assert (
        signature["content_sha256"]
        == hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
    )
    assert set(report["tape_inputs"]["step_noises"]["leaves"]) == {"0", "1"}
    assert report["tape_inputs"]["cycle_msa_index_tape"] == {"present": False}
    assert kwargs["init_noise"] is value


@pytest.mark.parametrize("location", ["schedule", "rotations"])
def test_infer_boundary_rejects_tracers_instead_of_recording_shapes_as_bytes(location):
    import jax

    def traced(value):
        schedule = value if location == "schedule" else np.ones(2)
        kwargs = {"rotations": value} if location == "rotations" else {}
        infer_boundary_report("protenix_infer_compiled", schedule, kwargs)
        return value

    with pytest.raises(TypeError, match="concrete arrays, not tracers"):
        jax.make_jaxpr(traced)(jnp.ones(2))


def test_infer_boundary_rejects_unknown_static_values():
    with pytest.raises(TypeError, match="JSON serializable"):
        infer_boundary_report(
            "protenix_infer_static", np.ones(2), {"unrecognized_option": object()}
        )
