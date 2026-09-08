import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench import openbind_triton_linear_probe as probe


def test_frozen_panel_covers_native_widths_row_tails_and_one_full_shape():
    cases = probe.case_specs()
    assert len(cases) == len({case.name for case in cases}) == 52
    assert {case.shape[0] for case in cases if len(case.shape) == 2} == {
        1,
        7,
        9,
        127,
        128,
        129,
    }
    assert {(case.shape[-1], case.outputs) for case in cases} == {
        (64, 64),
        (64, 128),
        (128, 64),
        (128, 128),
    }
    assert {case.arm for case in cases} == set(probe.ARMS)
    assert len([case for case in cases if len(case.shape) == 4]) == 1
    assert (
        probe.Case("k128-n128-pair437-plain", (1, 437, 437, 128), 128, "plain") in cases
    )


@pytest.mark.parametrize("arm", probe.ARMS)
def test_common_arms_use_native_bias_free_projection_contract(arm):
    case = probe.Case("test", (9, 128), 64, arm)
    first, second = probe.operands(case), probe.operands(case)
    probe.validate_operands(first, case)
    assert "bias" not in first
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert ("mask" in first) == (arm == "projection")
    assert ("add_tensor" in first) == (arm == "residual")
    if arm == "projection":
        assert first["mask"].shape == (9, 1)
        assert set(first["mask"].ravel()) == {0, 1}


def test_clamp_control_is_exact_bias_not_tf32_sensitive_input():
    case = probe.Case("clamp", (9, 64), 64, "sigmoid", "clamp")
    arrays = probe.operands(case)
    probe.validate_operands(arrays, case)
    assert np.count_nonzero(arrays["weight"]) == 0
    assert np.min(arrays["bias"]) < -20
    assert np.max(arrays["bias"]) > 20
    assert np.float32(-20) in arrays["bias"] and np.float32(20) in arrays["bias"]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("arm", probe.ARMS)
def test_dispatch_calls_actual_plain_or_fused_wrapper_twice(native, arm):
    calls = []
    case = probe.Case("test", (9, 64), 128, arm)
    name = probe.wrapper_name(case, native)
    module = SimpleNamespace(**{name: lambda **kwargs: calls.append(kwargs)})
    arrays = probe.operands(case)
    probe.call_wrapper(module, case, arrays, native=native)
    probe.call_wrapper(module, case, arrays, native=native)
    assert len(calls) == 2
    assert name == ("triton" if native else "native") + (
        "_linear" if arm == "plain" else "_linear_fused"
    )
    assert set(calls[0]) == set(arrays) | (
        set() if arm == "plain" else {"apply_sigmoid"}
    )
    if arm != "plain":
        assert calls[0]["apply_sigmoid"] == (arm in {"sigmoid", "residual"})
    for key, value in arrays.items():
        assert calls[0][key] is value and calls[1][key] is value


def test_operand_validation_rejects_missing_extra_dtype_shape_and_nonfinite():
    case = probe.Case("test", (9, 64), 128, "residual")
    arrays = probe.operands(case)
    for bad in (
        {k: v for k, v in arrays.items() if k != "other"},
        {**arrays, "mask": np.ones((9, 1), np.float32)},
    ):
        with pytest.raises(ValueError, match="operand keys"):
            probe.validate_operands(bad, case)
    for bad in (
        {**arrays, "x": arrays["x"].astype(np.float64)},
        {**arrays, "other": arrays["other"][:1]},
    ):
        with pytest.raises(ValueError, match="shape/dtype"):
            probe.validate_operands(bad, case)
    arrays["x"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        probe.validate_operands(arrays, case)


def test_default_candidate_operands_are_the_unchanged_archive_objects():
    arrays = probe.operands(probe.Case("test", (9, 64), 128, "residual"))
    assert probe.candidate_operands(arrays, "none") is arrays
    assert probe.operand_control_metadata("none") == {
        "mode": "none",
        "operands": [],
        "stage": "none",
        "rule": None,
        "runtime_operator_modified": False,
    }


def test_rtz_control_truncates_both_signs_and_preserves_non_dot_operands():
    # Include halfway cases, exact grid values, signed zero, subnormals and
    # the largest finite FP32 value; nearest rounding is observably different.
    bits = np.array(
        [
            0x3F801FFF,
            0xBF801FFF,
            0x3F801000,
            0xBF801000,
            0x3F802000,
            0,
            0x80000000,
            1,
            0x80000001,
            0x7F7FFFFF,
        ],
        np.uint32,
    )
    values = bits.view(np.float32)
    arrays = {
        "x": values,
        "weight": values.reshape(2, 5).T,
        "bias": values.copy(),
        "other": values.copy(),
        "mask": values.copy(),
        "add_tensor": values.copy(),
    }
    before = {key: value.tobytes() for key, value in arrays.items()}
    controlled = probe.candidate_operands(arrays, "rtz")
    for key in ("x", "weight"):
        np.testing.assert_array_equal(
            controlled[key].view(np.uint32),
            arrays[key].view(np.uint32) & np.uint32(0xFFFFE000),
        )
        assert controlled[key].shape == arrays[key].shape
        assert np.isfinite(controlled[key]).all()
        assert np.all(np.abs(controlled[key]) <= np.abs(arrays[key]))
    for key in ("bias", "other", "mask", "add_tensor"):
        assert controlled[key] is arrays[key]
    assert before == {key: value.tobytes() for key, value in arrays.items()}
    again = probe.candidate_operands(controlled, "rtz")
    assert all(
        again[key].tobytes() == value.tobytes() for key, value in controlled.items()
    )
    assert probe.operand_control_metadata("rtz") == {
        "mode": "rtz",
        "operands": ["x", "weight"],
        "stage": "host_before_device_transfer",
        "rule": "float32_uint32_bits & 0xffffe000",
        "runtime_operator_modified": False,
    }


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1], np.float64),
        np.array([np.nan], np.float32),
        np.array([np.inf], np.float32),
    ],
)
def test_rtz_control_rejects_unsupported_operand_dtype_or_values(bad):
    with pytest.raises(ValueError, match="finite FP32"):
        probe.candidate_operands({"x": bad, "weight": bad}, "rtz")


def test_operand_rounding_control_cannot_be_selected_for_native(tmp_path):
    with pytest.raises(ValueError, match="candidate-only"):
        probe.main(
            [
                "--mode",
                "native",
                "--source-root",
                str(tmp_path),
                "--out-dir",
                str(tmp_path / "out"),
                "--operand-rounding",
                "rtz",
            ]
        )
    assert not (tmp_path / "out").exists()


def test_capture_records_actual_kernel_metadata_and_launch():
    seen = []
    metadata = SimpleNamespace(
        name="linear_fused_kernel",
        num_warps=4,
        num_stages=3,
        default_dot_input_precision="tf32",
        enable_fp_fusion=True,
    )

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                seen.append((grid, args, kwargs))
                return SimpleNamespace(
                    metadata=metadata, asm={"ptx": "native compiled PTX"}
                )

            return launch

    capture = probe.LinearKernelCapture(Kernel())
    returned = capture[(2, 2)]("input", BLOCK_K=64)
    assert returned.metadata is metadata
    assert seen == [((2, 2), ("input",), {"BLOCK_K": 64})]
    assert capture.evidence["metadata"]["default_dot_input_precision"] == "tf32"
    assert capture.evidence["assembly"]["ptx"] == "native compiled PTX"


@pytest.fixture
def reference(tmp_path, monkeypatch):
    case = probe.Case("one", (9, 64), 128, "projection")
    monkeypatch.setattr(probe, "case_specs", lambda: [case])
    arrays = probe.operands(case)
    output = np.zeros((9, 128), np.float32)
    archive_sha = probe.save_arrays(
        tmp_path / "one.npz", **arrays, first=output, second=output
    )
    compiler = probe.write_compiler_evidence(
        tmp_path, "one", {"ptx": "actual native PTX"}
    )
    manifest = {
        "mode": "native",
        "completed": True,
        "panel": probe.PANEL,
        "dtype": "float32",
        "float32_matmul_precision": "high",
        "source": {
            "commit": probe.UPSTREAM_COMMIT,
            "operator_path": probe.NATIVE_OPERATOR,
        },
        "cases": {
            "one": {
                "spec": case.identity(),
                "wrapper": "triton_linear_fused",
                "archive_sha256": archive_sha,
                "native_repeat": probe.metrics(output, output),
                "compiler_evidence": compiler,
                "launch": {
                    "metadata": {
                        "name": "linear_fused_kernel",
                        "default_dot_input_precision": "tf32",
                        "enable_fp_fusion": True,
                        "num_warps": 4,
                        "num_stages": 3,
                    }
                },
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, case, manifest


def test_reference_validates_and_loads_same_payload_bytes(reference):
    root, case, manifest = reference
    checked = probe.verify_reference(
        root, expected_manifest_sha=probe.digest(root / "manifest.json")
    )
    assert checked == manifest
    arrays, output = probe.read_reference_case(root, case, manifest["cases"][case.name])
    assert set(arrays) == {"x", "weight", "other", "mask"}
    assert output.shape == (9, 128)


@pytest.mark.parametrize(
    "change,error",
    [
        ("dtype", "source/policy"),
        ("source", "source/policy"),
        ("incomplete", "incomplete"),
        ("wrapper", "wrapper"),
        ("spec", "specification"),
        ("repeat", "repeat"),
        ("compiler", "PTX"),
        ("precision", "compiler policy"),
        ("fusion", "compiler policy"),
        ("rounding", "source/policy"),
    ],
)
def test_reference_rejects_changed_contract_metadata(reference, change, error):
    root, case, manifest = reference
    record = manifest["cases"][case.name]
    if change == "dtype":
        manifest["dtype"] = "bfloat16"
    elif change == "source":
        manifest["source"]["operator_path"] = "src/norm.py"
    elif change == "incomplete":
        manifest["cases"] = {}
    elif change == "wrapper":
        record["wrapper"] = "triton_linear"
    elif change == "spec":
        record["spec"]["outputs"] = 64
    elif change == "repeat":
        record["native_repeat"]["bitwise_equal"] = False
    elif change == "compiler":
        record["compiler_evidence"] = {}
    elif change == "precision":
        record["launch"]["metadata"]["default_dot_input_precision"] = "ieee"
    elif change == "fusion":
        record["launch"]["metadata"]["enable_fp_fusion"] = False
    elif change == "rounding":
        manifest["operand_control"] = probe.operand_control_metadata("rtz")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=error):
        probe.verify_reference(root)


def test_reference_rejects_original_manifest_hash_after_valid_metadata_change(
    reference,
):
    root, case, manifest = reference
    original = probe.digest(root / "manifest.json")
    manifest["unrelated"] = "new metadata must not bypass the initial identity"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest changed"):
        probe.verify_reference(root, expected_manifest_sha=original)


@pytest.mark.parametrize("artifact", ["one.npz", "one.ptx.txt"])
def test_reference_rejects_changed_archive_or_compiler_payload(reference, artifact):
    root, case, manifest = reference
    (root / artifact).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        probe.verify_reference(root)


def test_reference_does_not_trust_rehashed_wrong_output_dtype(reference):
    root, case, manifest = reference
    arrays = probe.operands(case)
    output = np.zeros((9, 128), np.float64)
    with (root / "one.npz").open("wb") as stream:
        np.savez(stream, **arrays, first=output, second=output)
    manifest["cases"]["one"]["archive_sha256"] = probe.digest(root / "one.npz")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="output shape/dtype"):
        probe.verify_reference(root)


def test_source_identity_selects_linear_not_norm_and_tracks_all_source(tmp_path):
    path = tmp_path / probe.CANDIDATE_OPERATOR
    path.parent.mkdir(parents=True)
    (path.parent / "native_triton_norm.py").write_text("norm = 1\n")
    with pytest.raises(FileNotFoundError):
        probe.source_identity(tmp_path, False)
    path.write_text("linear = 1\n")
    first = probe.source_identity(tmp_path, False)
    assert first["operator_path"] == probe.CANDIDATE_OPERATOR
    assert first["operator_sha256"] == probe.digest(path)
    (path.parent / "native_triton_norm.py").write_text("norm = 2\n")
    second = probe.source_identity(tmp_path, False)
    assert first["operator_sha256"] == second["operator_sha256"]
    assert first["python_source_sha256"] != second["python_source_sha256"]


def test_source_escape_is_rejected(tmp_path):
    root = tmp_path / "source"
    path = root / probe.CANDIDATE_OPERATOR
    path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("outside = 1\n")
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        probe.source_identity(root, False)


def test_final_guards_check_source_helpers_and_initial_reference(
    reference, monkeypatch
):
    root, case, manifest = reference
    identity, helpers = {"operator": "original"}, {"harness": "original"}
    monkeypatch.setattr(probe, "source_identity", lambda *args: identity.copy())
    monkeypatch.setattr(probe, "helper_identities", lambda: helpers.copy())
    initial = probe.digest(root / "manifest.json")
    probe.verify_unchanged(root, False, identity, helpers, root, initial)
    with pytest.raises(RuntimeError, match="source changed"):
        probe.verify_unchanged(root, False, {}, helpers, root, initial)
    with pytest.raises(RuntimeError, match="helper source changed"):
        probe.verify_unchanged(root, False, identity, {}, root, initial)
    with pytest.raises(ValueError, match="manifest changed"):
        probe.verify_unchanged(root, False, identity, helpers, root, "0" * 64)


def test_metrics_report_strict_and_exact_without_conflating_them():
    reference = np.array([0, 1], np.float32)
    nearby = np.array([1e-5, 1], np.float32)
    result = probe.metrics(nearby, reference)
    assert result["strict_1e4_allclose"] and not result["bitwise_equal"]
    with pytest.raises(ValueError, match="nonfinite"):
        probe.metrics(np.array([np.inf, 1], np.float32), reference)


def test_output_archives_are_never_overwritten(tmp_path):
    path = tmp_path / "arrays.npz"
    probe.save_arrays(path, x=np.ones(1, np.float32))
    with pytest.raises(FileExistsError):
        probe.save_arrays(path, x=np.zeros(1, np.float32))
