import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench.opendde_closure_capture import (
    audit_request,
    confidence_feature_boundary,
    native_consumer_cycles,
    native_deterministic_policy,
    observer_policy,
    require_native_fp32_trunk_dtype,
)
from bench.opendde_confidence_boundary import validate_representatives


def test_observer_policy_binds_every_optional_switch():
    switches = dict.fromkeys((
        "capture_confidence_boundary", "capture_linear_policy",
        "capture_trunk_boundary", "capture_ffi_policy", "capture_consumed_tape",
    ), False)
    assert observer_policy(SimpleNamespace(**switches)) == switches
    for name in switches:
        changed = {**switches, name: True}
        assert observer_policy(SimpleNamespace(**changed)) == changed
    with pytest.raises(ValueError, match="explicit booleans"):
        observer_policy(SimpleNamespace(**{**switches, "capture_consumed_tape": 1}))


def test_audit_request_pins_seed_and_routes_native_fp32(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('[{"name":"tiny","modelSeeds":[101],"sequences":[]}]')
    weights = tmp_path / ".foldjax/weights/opendde/opendde.jax"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fixture")
    request = audit_request(
        SimpleNamespace(input=path, repo=tmp_path, out=tmp_path / "out")
    )
    assert request.seed == 101
    assert request.options == {
        "dtype": "float32",
        "include_raw": True,
        "matmul_precision": "high",
    }
    assert request.input_format == "native"
    from foldjax.backends.opendde import OpenDDEBackend

    invocation = OpenDDEBackend()._native_invocation(request)
    assert invocation.config_fields["trunk_dtype"] == "fp32"
    trunk_dtype_flag = invocation.argv.index("--trunk-dtype")
    assert invocation.argv[trunk_dtype_flag + 1] == "fp32"


@pytest.mark.parametrize("value", (None, "bf16", "bfloat16", "float32"))
def test_native_capture_rejects_non_fp32_public_trunk_route(value):
    with pytest.raises(ValueError, match="native FP32"):
        require_native_fp32_trunk_dtype(value)


def test_native_capture_accepts_fp32_public_trunk_route():
    require_native_fp32_trunk_dtype("fp32")


def test_confidence_boundary_captures_consumed_features_not_lazy_trunk_state():
    features = {
        key: np.asarray([0, 1])
        for key in (
            "distogram_rep_atom_mask",
            "structural_distogram_rep_atom_mask",
            "atom_to_token_idx",
            "atom_to_tokatom_idx",
        )
    }
    expected = dict(features)
    features["relp"] = object()
    assert confidence_feature_boundary(features).keys() == expected.keys()
    for key, value in confidence_feature_boundary(features).items():
        np.testing.assert_array_equal(value, expected[key])
    del features["atom_to_tokatom_idx"]
    with pytest.raises(KeyError):
        confidence_feature_boundary(features)


def test_native_policy_uses_recorded_config_for_early_capture(tmp_path):
    path = tmp_path / "effective-config.json"
    path.write_text(json.dumps({"deterministic": False}))
    assert native_deterministic_policy(tmp_path, {}) is False
    with pytest.raises(ValueError, match="conflicts"):
        native_deterministic_policy(tmp_path, {"native_deterministic": True})
    path.write_text(json.dumps({"deterministic": "false"}))
    with pytest.raises(ValueError, match="explicit boolean"):
        native_deterministic_policy(tmp_path, {})


def _native_msa():
    return {
        "input_msa_mask": np.ones((3, 4), dtype=bool),
        "rows": np.tile([2, 0], (10, 1)).astype(np.int64),
        "selected_msa": np.tile([[31] * 4, [0] * 4], (10, 1, 1)).astype(np.int64),
        "selected_has_deletion": np.zeros((10, 2, 4), dtype=np.float32),
        "selected_deletion_value": np.zeros((10, 2, 4), dtype=np.float32),
    }


def test_native_consumer_msa_mapping_is_explicit_and_value_preserving():
    native = _native_msa()
    result = native_consumer_cycles(native)
    assert len(result) == 10
    for cycle in result:
        assert cycle["msa"].dtype == np.int32
        assert cycle["msa_mask"].dtype == np.float32
        np.testing.assert_array_equal(cycle["msa"], native["selected_msa"][0])
        np.testing.assert_array_equal(cycle["msa_mask"], 1)
    assert native["selected_msa"].dtype == np.int64


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_msa_mask", np.zeros((3, 4), dtype=bool)),
        ("rows", np.full((10, 2), 3, dtype=np.int64)),
        ("rows", np.full((10, 2), -1, dtype=np.int64)),
        ("selected_msa", np.full((10, 2, 4), 32, dtype=np.int64)),
        ("selected_deletion_value", np.zeros((10, 2, 4), dtype=np.float64)),
    ],
)
def test_native_consumer_msa_rejects_unmapped_values(field, value):
    native = _native_msa()
    native[field] = value
    with pytest.raises(ValueError):
        native_consumer_cycles(native)


@pytest.mark.parametrize("mask", [[1, 0, 1], [1, 0, 0], [1, 1], [2, 0, 0]])
def test_confidence_representative_mask_cannot_be_padded_or_truncated(mask):
    values = {
        "selected_distogram_rep_atom_mask": np.asarray(mask),
        "x_pred_coords": np.zeros((5, 3, 3)),
        "s_inputs": np.zeros((2, 4)),
    }
    if mask == [1, 0, 1]:
        validate_representatives(values)
    else:
        with pytest.raises(ValueError, match="representative mask"):
            validate_representatives(values)
