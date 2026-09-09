import hashlib

import numpy as np
import pytest

from bench.af3_closure_capture import flatten, sha


def test_cannot_disable_preprocessing_observers_for_audit(monkeypatch, capsys):
    from bench.af3_closure_capture import main

    monkeypatch.setattr("sys.argv", [
        "capture", "native", "--input", "unused", "--native-source", "unused",
        "--weights", "unused", "--out", "unused",
        "--no-preprocessing-observers",
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "requires performance mode" in capsys.readouterr().err


def test_flatten_preserves_confidence_nan_types_and_sample_order():
    result = flatten(
        [{"ptm": np.array([np.nan, 0.8]), "has_clash": False}, {"name": "B"}]
    )
    assert list(result) == ["0.ptm", "0.has_clash", "1.name"]
    assert np.isnan(result["0.ptm"][0])
    assert result["0.has_clash"].dtype == np.bool_
    assert result["1.name"].item() == "B"


def test_flatten_serializes_string_objects_without_pickle():
    result = flatten({"chains": np.array(["A", "B"], dtype=object)})
    assert result["chains"].dtype.kind == "U"
    with pytest.raises(TypeError, match="unhandled"):
        flatten({"unknown": object()})


def test_capture_sha_binds_artifact_bytes(tmp_path):
    artifact = tmp_path / "small"
    artifact.write_bytes(b"known bytes")
    assert sha(artifact) == hashlib.sha256(b"known bytes").hexdigest()


class _Config:
    def __init__(self, mapping):
        self._mapping = mapping

    def as_dict(self):
        return dict(self._mapping)


def test_both_arms_record_the_same_config_schema():
    from bench.af3_closure_capture import config_record

    native = _Config({"num_recycles": 10})
    port = _Config(
        {
            "num_recycles": 10,
            "foldjax_stop_after": "full",
            "foldjax_return_representations": [],
        }
    )
    assert config_record(native) == config_record(port)


def test_a_run_that_captured_something_extra_still_differs():
    from bench.af3_closure_capture import config_record

    native = _Config({"num_recycles": 10})
    for key, value in (
        ("foldjax_stop_after", "trunk"),
        ("foldjax_stop_after", "inputs"),
        ("foldjax_return_representations", ["single"]),
    ):
        port = _Config({"num_recycles": 10, key: value})
        assert config_record(native) != config_record(port), key


def test_config_record_does_not_mutate_the_config():
    from bench.af3_closure_capture import config_record

    mapping = {"num_recycles": 10}
    config = _Config(mapping)
    config_record(config)
    assert mapping == {"num_recycles": 10}
