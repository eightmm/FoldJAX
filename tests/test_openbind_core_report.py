import json

import numpy as np
import pytest

from bench.boltz_historical_replay import digest
from bench.openbind_core_report import compare, compare_raw_confidence


@pytest.fixture
def arms(tmp_path):
    native, candidate = tmp_path / "native", tmp_path / "candidate"
    native.mkdir()
    candidate.mkdir()
    predictions = native / "predictions"
    predictions.mkdir()
    fields = {
        "chain_id": ["A", "A", "L"],
        "res_id": [1, 1, 2],
        "res_name": ["GLY", "GLY", "LIG"],
        "atom_name": ["N", "CA", "C1"],
        "element": ["N", "C", "C"],
    }
    np.savez(
        native / "input.npz",
        atom_mask=np.ones((1, 3)),
        **{"atom_array.0.annotation." + k: np.array(v) for k, v in fields.items()},
    )
    np.savez(native / "tape.npz", noise=np.zeros(1))
    coordinates = np.tile(np.eye(3)[None], (5, 1, 1))
    np.savez(native / "coordinate.npz", coordinate=coordinates)
    np.savez(
        candidate / "prediction.npz",
        coordinates=coordinates,
        plddt=np.ones((5, 3)),
        ptm=np.ones(5),
        iptm=np.ones(5),
    )
    for i in range(1, 6):
        (predictions / f"sample_{i}_confidences.json").write_text(
            json.dumps({"plddt": [100, 100, 100]})
        )
        (predictions / f"sample_{i}_confidences_aggregated.json").write_text(
            json.dumps({"ptm": 1, "iptm": 1})
        )
    (candidate / "preflight.json").write_text(
        json.dumps(
            {
                "input_sha256": digest(native / "input.npz"),
                "tape_sha256": digest(native / "tape.npz"),
                "native_backend": "triton",
                "candidate_backend": "cueq",
            }
        )
    )
    (candidate / "finished.json").write_text(
        json.dumps(
            {
                "prediction_sha256": digest(candidate / "prediction.npz"),
            }
        )
    )
    return native, candidate


def test_identical_outputs_keep_non_admission_scope(arms):
    report = compare(*arms)
    assert max(report["coordinates"]["entity_max_rmsd"].values()) < 1e-12
    assert all(
        v["max_absolute_error"] == 0 for v in report["public_confidence"].values()
    )
    assert report["parity_admitted"] is False
    assert "plddt_logits" in report["raw_confidence"]["missing_from_native"]


def test_changed_tape_is_rejected(arms):
    np.savez(arms[0] / "tape.npz", noise=np.ones(1))
    with pytest.raises(ValueError, match="identity mismatch"):
        compare(*arms)


def test_confidence_shape_does_not_broadcast_silently(arms):
    path = arms[0] / "predictions/sample_1_confidences_aggregated.json"
    path.write_text(json.dumps({"ptm": [1], "iptm": 1}))
    with pytest.raises(ValueError):
        compare(*arms)


def test_raw_confidence_checks_digest_and_exact_batch_axis(tmp_path):
    path = tmp_path / "raw-output.npz"
    np.savez(path, pae_logits=np.ones((1, 5, 2, 2, 4)))
    (tmp_path / "trace.json").write_text(json.dumps({
        "raw_output_sha256": {"sha256": digest(path), "bytes": path.stat().st_size},
    }))
    report = compare_raw_confidence(tmp_path, {"pae_logits": np.ones((5, 2, 2, 4))})
    assert report["leaves"]["pae_logits"]["max_absolute_error"] == 0
    assert report["trace_binding"] == "not replay-bound"
    assert "plddt_logits" in report["missing_fields"]
    assert "plddt_logits" in report["missing_from_candidate"]
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_raw_confidence(tmp_path, {"pae_logits": np.ones((1, 5, 2, 2, 4))})
    np.savez(path, pae_logits=np.zeros((1, 5, 2, 2, 4)))
    with pytest.raises(ValueError, match="identity mismatch"):
        compare_raw_confidence(tmp_path, {})


def test_declared_raw_output_cannot_disappear(arms):
    (arms[0] / "trace.json").write_text(json.dumps({"raw_output_sha256": {}}))
    with pytest.raises(ValueError, match="declared native raw output is missing"):
        compare(*arms)


def test_replay_bound_public_artifacts_reject_changed_coordinates(arms):
    native, candidate = arms
    paths = [
        native / "coordinate.npz", *sorted((native / "predictions").glob("*.json"))
    ]
    trace = native / "trace.json"
    trace.write_text(json.dumps({"public_artifacts": {
        str(p.relative_to(native)): {"sha256": digest(p), "bytes": p.stat().st_size}
        for p in paths
    }}))
    preflight_path = candidate / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    preflight["capture_provenance"] = {"trace": {}, "sha256": digest(trace)}
    preflight_path.write_text(json.dumps(preflight))
    assert compare(*arms)["native_public_artifact_binding"].startswith("capture-time")
    np.savez(native / "coordinate.npz", coordinate=np.zeros((5, 3, 3)))
    with pytest.raises(ValueError, match="native public artifact identity mismatch"):
        compare(*arms)


def test_public_confidence_shape_guard_rejects_uniform_wrong_shape(arms):
    for path in (arms[0] / "predictions").glob("*confidences.json"):
        path.write_text(json.dumps({"plddt": [100, 100, 100, 100]}))
    with pytest.raises(ValueError, match="confidence shape mismatch: plddt"):
        compare(*arms)


def test_full_report_reads_raw_npz_and_attributes_missing_heads(arms):
    native, candidate = arms
    path = native / "raw-output.npz"
    np.savez(path, pae_logits=np.ones((1, 5, 2, 2, 4)),
             plddt_logits=np.ones((1, 5, 3, 50)))
    trace = native / "trace.json"
    trace.write_text(json.dumps({"raw_output_sha256": {
        "sha256": digest(path), "bytes": path.stat().st_size,
    }}))
    output_path = candidate / "prediction.npz"
    with np.load(output_path) as archive:
        outputs = dict(archive)
    outputs["pae_logits"] = np.zeros((5, 2, 2, 4))
    np.savez(output_path, **outputs)
    (candidate / "finished.json").write_text(json.dumps({
        "prediction_sha256": digest(output_path),
    }))
    preflight_path = candidate / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    preflight["capture_provenance"] = {"trace": {}, "sha256": digest(trace)}
    preflight_path.write_text(json.dumps(preflight))
    raw = compare(*arms)["raw_confidence"]
    assert raw["trace_binding"] == "replay-bound trace verified"
    assert raw["leaves"]["pae_logits"]["rmse"] == 1
    assert "plddt_logits" not in raw["missing_from_native"]
    assert "plddt_logits" in raw["missing_from_candidate"]
    trace.write_text("{}")
    with pytest.raises(ValueError, match="native trace identity mismatch"):
        compare(*arms)


@pytest.mark.parametrize("trace_exists", [False, True])
def test_undeclared_native_raw_output_is_rejected(tmp_path, trace_exists):
    np.savez(tmp_path / "raw-output.npz", pae_logits=np.zeros(1))
    if trace_exists:
        (tmp_path / "trace.json").write_text("{}")
    with pytest.raises(ValueError, match="undeclared native raw output"):
        compare_raw_confidence(tmp_path, {})


def test_bound_backend_request_must_match_effective_backend(arms):
    native, candidate = arms
    trace = native / "trace.json"
    trace.write_text(json.dumps({"requested_triangle_backend": "xla"}))
    path = candidate / "preflight.json"
    preflight = json.loads(path.read_text())
    preflight["capture_provenance"] = {"trace": {}, "sha256": digest(trace)}
    path.write_text(json.dumps(preflight))
    with pytest.raises(ValueError, match="request/effective mismatch"):
        compare(*arms)
