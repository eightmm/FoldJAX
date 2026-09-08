import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench import openbind_triangle_ops_probe as probe


def test_frozen_real_weight_panel_52_cases_four_families_and_four_operators():
    cases = probe.case_specs()
    assert len(cases) == len({c.name for c in cases}) == 52
    assert {(c.family, c.operator, c.length) for c in cases if c.length != 437} == {
        (family, op, n)
        for family in probe.FAMILIES
        for op in probe.OPERATORS
        for n in (16, 17, 65)
    }
    full = [c for c in cases if c.length == 437]
    assert {c.operator for c in full} == set(probe.OPERATORS)
    assert all(c.shape == (1, 437, 437, 128) and c.family == "pairformer" for c in full)
    assert len({c.weight_id for c in cases}) == 16
    assert all(c.finite for c in cases)
    assert "synthetic activations" in probe.__doc__


@pytest.mark.parametrize("family", probe.FAMILIES)
@pytest.mark.parametrize("operator", probe.OPERATORS)
def test_real_weight_schema_and_output_contract(family, operator):
    case = probe.Case(family, operator, 17)
    params = {
        name: np.ones(shape, np.float32)
        for name, shape in probe.parameter_shapes(case).items()
    }
    meta = probe.parameter_metadata(params, case)
    assert len(meta) == (10 if case.multiplication else 8)
    assert all(
        v["checkpoint_key"] == case.prefix + "." + key for key, v in meta.items()
    )
    assert case.identity()["output_contract"] == (
        "fused_residual" if case.multiplication else "update_only"
    )
    for key, value in params.items():
        assert meta[key]["sha256"] == hashlib.sha256(value.tobytes()).hexdigest()


@pytest.mark.parametrize("damage", ["extra", "missing", "shape", "dtype", "nan"])
def test_malformed_checkpoint_weights_rejected(damage):
    case = probe.Case("template", "tri_mul_out", 17)
    params = {
        key: np.ones(shape, np.float32)
        for key, shape in probe.parameter_shapes(case).items()
    }
    key = "linear_a_g.weight"
    if damage == "extra":
        params["extra"] = np.zeros(1, np.float32)
    if damage == "missing":
        del params[key]
    if damage == "shape":
        params[key] = params[key][:1]
    if damage == "dtype":
        params[key] = params[key].astype(np.float64)
    if damage == "nan":
        params[key][0, 0] = np.nan
    with pytest.raises(ValueError):
        probe.parameter_metadata(params, case)


def test_operands_deterministic_asymmetric_and_untransformed():
    case = probe.Case("template", "tri_att_end", 17)
    arrays = probe.operands(case)
    probe.validate_operands(arrays, case)
    for key, value in probe.operands(case).items():
        np.testing.assert_array_equal(value, arrays[key])
        assert not np.array_equal(value, value.swapaxes(1, 2))
    arrays["mask"] = arrays["mask"].swapaxes(1, 2)
    with pytest.raises(ValueError, match="mask changed"):
        probe.validate_operands(arrays, case)


class Tensor:
    def __init__(self, array):
        self.array = array

    def clone(self):
        return Tensor(self.array.copy())

    def transpose(self, i, j):
        return Tensor(self.array.swapaxes(i, j))


@pytest.mark.parametrize("operator", probe.OPERATORS)
def test_native_invokes_real_module_flags_with_fresh_input_and_end_orientation(
    operator,
):
    case = probe.Case("template", operator, 17)
    arrays = {k: Tensor(v) for k, v in probe.operands(case).items()}
    original = arrays["z"].array.copy()
    seen = []

    def module(z, mask, **kwargs):
        seen.append((z.array.copy(), mask.array.copy(), kwargs))
        z.array += 3
        return z

    first = probe.call_native(module, case, arrays).array
    second = probe.call_native(module, case, arrays).array
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(arrays["z"].array, original)
    np.testing.assert_array_equal(first, original + 3)
    np.testing.assert_array_equal(
        seen[0][0], original.swapaxes(1, 2) if operator == "tri_att_end" else original
    )
    assert seen[0][2] == probe.native_kwargs(case)
    assert seen[0][2]["use_triton_triangle_kernels"] is True
    if case.multiplication:
        assert seen[0][2]["_add_with_inplace"] is True
    else:
        assert seen[0][2]["chunk_size"] == 1024
        assert seen[0][2]["transpose_bias"] == (operator == "tri_att_end")


def calls(case):
    if case.multiplication:
        n = case.length
        a = [min(256, n - i) for i in range(0, n, 256)]
        half = (n + 1) // 2
        b = [
            min(256, end - i)
            for start, end in ((0, half), (half, n))
            for i in range(start, end, 256)
        ]
        events = [
            {"weight": "linear_a_g.weight", "input_shape": [1, r, n, case.width]}
            for r in a
        ]
        events += [
            {
                "weight": "linear_b_g.weight",
                "input_shape": [1, r, n, case.width]
                if case.operator == "tri_mul_out"
                else [1, n, r, case.width],
            }
            for r in b
        ]
        count = 3 * len(a) + 7 * len(b)
    else:
        events = [
            {
                "name": "mha.linear_q",
                "input_shape": [case.length, case.length, case.width],
            }
        ]
        count = int(case.length > 16)
    kernels = [
        {
            "compiler_evidence": {"ptx": "a" * 64},
            "metadata": {
                "name": "linear_fused_kernel" if case.multiplication else "_attn_fwd",
                "num_warps": 4,
                "num_stages": 3 if case.multiplication else 1,
                "default_dot_input_precision": "tf32",
                "enable_fp_fusion": True,
            },
            "grid": [(case.length + 63) // 64, case.length * 4, 1],
            "launch_kwargs": {
                "BLOCK_SIZE_Q": 64,
                "BLOCK_SIZE_KV": 16,
                "BLOCK_DIM": 32,
                "USE_EXP2": False,
                "HAS_PAIR_BIAS": True,
                "HEAD": 4,
                "N_SEQ": case.length,
                "SEQ_LEN": case.length,
                "DIM": case.width // 4,
                "softmax_scale": (case.width // 4) ** -0.5,
                "num_warps": 4,
                "num_stages": 1,
            },
        }
        for _ in range(count)
    ]
    call = {
        "module_kwargs": probe.native_kwargs(case),
        "events": events,
        "kernels": kernels,
    }
    return [call, copy.deepcopy(call)]


@pytest.mark.parametrize("operator", probe.OPERATORS)
@pytest.mark.parametrize("length", [16, 17, 437])
def test_actual_call_chunk_metadata_and_stock_dispatch(operator, length):
    case = probe.Case("pairformer", operator, length)
    record = {"calls": calls(case)}
    probe.validate_calls(record, case)
    record["calls"][1]["module_kwargs"]["inplace_safe"] = False
    with pytest.raises(ValueError, match="repeats differ"):
        probe.validate_calls(record, case)


def test_halfway_chunk_must_not_be_replaced_by_plain_256_schedule():
    case = probe.Case("pairformer", "tri_mul_out", 437)
    record = {"calls": calls(case)}
    for call in record["calls"]:
        b = [e for e in call["events"] if e["weight"] == "linear_b_g.weight"]
        b[0]["input_shape"][1] = 256
        b[1]["input_shape"][1] = 181
    with pytest.raises(ValueError, match="b projection chunk schedule"):
        probe.validate_calls(record, case)


@pytest.mark.parametrize("operator", ["tri_mul_out", "tri_att_start"])
def test_native_kernel_policy_must_match_even_if_repeats_agree(operator):
    case = probe.Case("template", operator, 17)
    record = {"calls": calls(case)}
    for call in record["calls"]:
        call["kernels"][0]["metadata"]["default_dot_input_precision"] = "ieee"
    with pytest.raises(ValueError, match="compiler policy differs"):
        probe.validate_calls(record, case)


def reference(tmp_path, monkeypatch):
    case = probe.Case("template", "tri_att_start", 16)
    monkeypatch.setattr(probe, "case_specs", lambda: [case])
    params = {
        k: np.ones(shape, np.float32)
        for k, shape in probe.parameter_shapes(case).items()
    }
    weight = {
        "prefix": case.prefix,
        "parameters": probe.parameter_metadata(params, case),
        "archive_sha256": probe.save_arrays(
            tmp_path / f"weights-{case.weight_id}.npz", **params
        ),
    }
    arrays = probe.operands(case)
    output = np.zeros(case.shape, np.float32)
    record = {
        "spec": case.identity(),
        "status": "ok",
        "calls": calls(case),
        "native_repeat": probe.compare_outputs(output, output, case),
        "archive_sha256": probe.save_arrays(
            tmp_path / f"{case.name}.npz", **arrays, first=output, second=output
        ),
    }
    manifest = {
        "panel": probe.PANEL,
        "mode": "native",
        "completed": True,
        "policy": probe.POLICY,
        "source": {
            "commit": probe.UPSTREAM_COMMIT,
            "operator_path": probe.NATIVE_OPERATOR,
        },
        "cases": {case.name: record},
        "weights": {case.weight_id: weight},
        "checkpoint": {
            "relative_path": probe.CHECKPOINT,
            "sha256": "b" * 64,
            "size": 100,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return case, manifest


def test_valid_reference_archive_is_hashed_and_decoded_from_same_bytes(
    tmp_path, monkeypatch
):
    case, manifest = reference(tmp_path, monkeypatch)
    assert probe.verify_reference(tmp_path) == manifest
    path = tmp_path / f"{case.name}.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        probe.verify_reference(tmp_path)


@pytest.mark.parametrize(
    "damage",
    [
        "policy",
        "prefix",
        "tensor_hash",
        "checkpoint",
        "status",
        "panel",
        "manifest_hash",
    ],
)
def test_reference_identity_damage_rejected(tmp_path, monkeypatch, damage):
    case, manifest = reference(tmp_path, monkeypatch)
    original_sha = probe.digest(tmp_path / "manifest.json")
    if damage == "policy":
        manifest["policy"] = {**manifest["policy"], "attention_chunk_size": 256}
    if damage == "prefix":
        manifest["weights"][case.weight_id]["prefix"] = "other"
    if damage == "tensor_hash":
        next(iter(manifest["weights"][case.weight_id]["parameters"].values()))[
            "sha256"
        ] = "0" * 64
    if damage == "checkpoint":
        manifest["checkpoint"]["relative_path"] = "wrong.pt"
    if damage == "status":
        manifest["cases"][case.name]["status"] = "error"
    if damage == "panel":
        manifest["cases"] = {}
    if damage == "manifest_hash":
        manifest["extra"] = "changed"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        probe.verify_reference(
            tmp_path,
            expected_manifest_sha=original_sha if damage == "manifest_hash" else None,
        )


def test_nonfinite_outputs_are_preserved_as_failure_not_finite_pass():
    case = probe.Case("template", "tri_att_start", 16)
    output = np.full(case.shape, np.nan, np.float32)
    result = probe.compare_outputs(output, output, case)
    assert result["bitwise_equal"] and result["nonfinite_pattern_equal"]
    assert not result["both_finite"] and not result["strict_1e4_allclose"]


def test_probe_helpers_and_runtime_identity_cover_new_operator_and_dependencies():
    root = probe.Path(__file__).resolve().parents[1]
    identity = probe.source_identity(root, False)
    assert identity["operator_path"] == probe.CANDIDATE_OPERATOR
    assert identity["operator_sha256"] == probe.digest(root / probe.CANDIDATE_OPERATOR)
    assert {probe.Path(p).name for p in identity["dependency_sha256"]} >= {
        "native_triton_attention.py",
        "native_triton_norm.py",
        "native_triton_linear.py",
        "torch_mapping.py",
        "primitives.py",
        "native_amp_norm.py",
        "_cp.py",
    }
    assert "openbind_triangle_ops_probe.py" in probe.helper_identities()


def test_kernel_recorder_preserves_each_actual_launch_and_deduplicates_ptx(tmp_path):
    compiled = SimpleNamespace(
        asm={"ptx": "actual ptx"}, metadata={"name": "fake", "num_warps": 4}
    )

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: compiled

    recorder = object.__new__(probe.NativeRecorder)
    recorder.kernels = []
    recorder.out_dir = tmp_path
    capture = recorder._capture(Kernel())
    assert capture[(1,)](BLOCK_SIZE=64) is compiled
    assert capture[(2,)](BLOCK_SIZE=128) is compiled
    assert [x["grid"] for x in recorder.kernels] == [[1], [2]]
    assert [x["launch_kwargs"]["BLOCK_SIZE"] for x in recorder.kernels] == [64, 128]
    assert len(list(tmp_path.iterdir())) == 1
