import io
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from bench import openbind_triton_attention_probe as probe


def _launch(case):
    return {
        "grid": [(case.length + 63) // 64, case.rows * 4, 1],
        "launch_kwargs": {
            "BLOCK_SIZE_Q": 64,
            "BLOCK_SIZE_KV": 16,
            "BLOCK_DIM": 32,
            "USE_EXP2": False,
            "HAS_PAIR_BIAS": True,
            "HEAD": 4,
            "N_SEQ": case.rows,
            "SEQ_LEN": case.length,
            "DIM": case.dim,
            "softmax_scale": case.dim**-0.5,
            "num_warps": 4,
            "num_stages": 1,
        },
        "metadata": {
            "name": "_attn_fwd",
            "num_warps": 4,
            "num_stages": 1,
            "default_dot_input_precision": "tf32",
            "enable_fp_fusion": True,
        },
    }


def test_panel_has_frozen_seventeen_finite_and_four_separate_semantic_cases():
    cases = probe.case_specs()
    assert len(cases) == len({c.name for c in cases}) == 21
    assert sum(c.finite for c in cases) == 17
    assert {
        (c.dim, c.length) for c in cases if c.rows == 2 and c.profile == "normal"
    } == {(d, n) for d in (16, 32) for n in (17, 31, 32, 33, 63, 64, 65)}
    full = [c for c in cases if c.rows > 2]
    assert len(full) == 1 and full[0].shape == (1, 437, 437, 4, 32)
    assert "repeats two row activations" in probe.__doc__


@pytest.mark.parametrize(
    "profile", ["normal", "finite_masked", "full_inf", "initial_inf"]
)
def test_operands_are_deterministic_and_only_masks_can_be_nonfinite(profile):
    case = probe.Case("case", 33, 16, profile=profile)
    arrays = probe.operands(case)
    probe.validate_operands(arrays, case)
    for name, value in probe.operands(case).items():
        np.testing.assert_array_equal(value, arrays[name])
    assert not np.array_equal(arrays["query"][:, 0], arrays["query"][:, 1])
    assert not np.array_equal(arrays["pair_bias"], arrays["pair_bias"].swapaxes(-1, -2))
    if profile == "normal":
        assert not np.array_equal(
            arrays["additive_mask"][:, 0], arrays["additive_mask"][:, 1]
        )
    elif profile == "initial_inf":
        assert np.isneginf(arrays["additive_mask"][..., :16]).all()
        assert np.isfinite(arrays["additive_mask"][..., 16:]).all()


def test_repeated_rows_are_explicit_synthetic_compression_not_new_samples():
    case = probe.Case("rows", 17, 16, rows=5)
    arrays = probe.operands(case)
    for key in (*probe.KEYS[:3], "additive_mask"):
        np.testing.assert_array_equal(arrays[key][:, 0], arrays[key][:, 2])
        np.testing.assert_array_equal(arrays[key][:, 1], arrays[key][:, 3])


def test_nan_semantics_never_count_as_finite_gate_pass():
    finite = probe.Case("finite", 17, 16)
    semantic = probe.Case("inf", 17, 16, profile="full_inf")
    nan = np.full(finite.shape, np.nan, np.float32)
    zero = np.zeros(finite.shape, np.float32)
    finite_failure = probe.compare_outputs(nan, nan.copy(), finite)
    semantic_match = probe.compare_outputs(nan, nan.copy(), semantic)
    assert not finite_failure["strict_1e4_allclose"]
    assert semantic_match["strict_1e4_allclose"] is None
    assert semantic_match["semantic_equal"] and semantic_match["bitwise_equal"]
    mismatch = probe.compare_outputs(nan, zero, semantic)
    assert not mismatch["semantic_equal"] and not mismatch["nonfinite_pattern_equal"]
    records = {
        "finite": {
            "status": "ok",
            "spec": finite.identity(),
            "native_comparison": finite_failure,
        },
        "semantic": {
            "status": "ok",
            "spec": semantic.identity(),
            "native_comparison": semantic_match,
        },
    }
    summary = probe.summarize(records, False)
    assert summary["finite_cases"] == 1 and summary["finite_strict_pass"] == 0
    assert summary["semantic_cases"] == summary["semantic_equal"] == 1
    assert probe.compare_outputs(zero + np.float32(1e-5), zero, finite)[
        "strict_1e4_allclose"
    ]


def test_semantic_comparison_checks_inf_sign_nan_locations_and_finite_values():
    case = probe.Case("inf", 17, 16, profile="full_inf")
    first = np.zeros(case.shape, np.float32)
    first.flat[:3] = [np.nan, np.inf, -np.inf]
    for index, value in ((0, 0), (1, -np.inf), (2, np.inf), (3, 1)):
        second = first.copy()
        second.flat[index] = value
        assert not probe.compare_outputs(first, second, case)["semantic_equal"]


@pytest.mark.parametrize("native", [False, True])
def test_calls_exact_native_apply_or_candidate_with_unchanged_operands_twice(native):
    calls = []
    arrays = probe.operands(probe.Case("test", 17, 16))
    module = SimpleNamespace(
        EvoformerAttention=SimpleNamespace(apply=lambda *a: calls.append(a)),
        native_triangle_attention=lambda **kw: calls.append(
            tuple(kw[k] for k in probe.KEYS)
        ),
    )
    for _ in range(2):
        probe.call_wrapper(module, arrays, native)
    assert all(a is arrays[k] for call in calls for a, k in zip(call, probe.KEYS))


def test_native_callable_grid_capture_omits_tensors_but_preserves_launch():
    seen = []
    case = probe.Case("test", 17, 16)
    contract = _launch(case)
    tensor = object()

    class Kernel:
        def __getitem__(self, grid):
            def launch(**kwargs):
                seen.append((list(grid), kwargs))
                return SimpleNamespace(
                    metadata=contract["metadata"], asm={"ptx": "native PTX"}
                )

            return launch

    capture = probe.AttentionKernelCapture(Kernel())

    def grid(kw):
        return ((kw["SEQ_LEN"] + 63) // 64, kw["N_SEQ"] * 4, 1)

    capture[grid](Q=tensor, **contract["launch_kwargs"])
    assert seen[0][0] == contract["grid"] and seen[0][1]["Q"] is tensor
    assert "Q" not in capture.evidence["launch_kwargs"]
    probe.validate_launch(capture.evidence, case)
    assert capture.evidence["assembly"]["ptx"] == "native PTX"


@pytest.fixture
def reference(tmp_path, monkeypatch):
    case = probe.Case("one", 17, 16)
    monkeypatch.setattr(probe, "case_specs", lambda: [case])
    arrays = probe.operands(case)
    output = np.zeros(case.shape, np.float32)
    record = {
        "status": "ok",
        "spec": case.identity(),
        "wrapper": "EvoformerAttention.apply",
        "archive_sha256": probe.save_arrays(
            tmp_path / "one.npz", **arrays, first=output, second=output
        ),
        "native_repeat": probe.compare_outputs(output, output, case),
        "launch": _launch(case),
        "compiler_evidence": probe.write_compiler_evidence(
            tmp_path, "one", {"ptx": "native PTX"}
        ),
    }
    manifest = {
        "mode": "native",
        "completed": True,
        "panel": probe.PANEL,
        "policy": probe.POLICY.copy(),
        "source": {
            "commit": probe.UPSTREAM_COMMIT,
            "operator_path": probe.NATIVE_OPERATOR,
        },
        "cases": {case.name: record},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, case, manifest


def test_reference_validates_original_payload_and_manifest_hash(reference):
    root, case, manifest = reference
    assert (
        probe.verify_reference(
            root, expected_manifest_sha=probe.digest(root / "manifest.json")
        )
        == manifest
    )
    arrays, first = probe.read_reference_case(root, case, manifest["cases"][case.name])
    assert set(arrays) == set(probe.KEYS) and first.shape == case.shape
    with pytest.raises(ValueError, match="manifest changed"):
        probe.verify_reference(root, expected_manifest_sha="0" * 64)


@pytest.mark.parametrize(
    "change,error",
    [
        ("policy", "source/policy"),
        ("source", "source/policy"),
        ("completed", "source/policy"),
        ("panel", "incomplete"),
        ("wrapper", "specification"),
        ("spec", "specification"),
        ("launch", "launch policy"),
        ("compiler", "compiler policy"),
        ("ptx", "actual native PTX"),
        ("repeat", "native repeat"),
    ],
)
def test_reference_contract_mutations_fail_closed(reference, change, error):
    root, case, manifest = reference
    record = manifest["cases"][case.name]
    if change == "policy":
        manifest["policy"]["use_exp2"] = True
    elif change == "source":
        manifest["source"]["operator_path"] = probe.CANDIDATE_OPERATOR
    elif change == "completed":
        manifest["completed"] = False
    elif change == "panel":
        manifest["cases"] = {}
    elif change == "wrapper":
        record["wrapper"] = "stock_attention"
    elif change == "spec":
        record["spec"]["dim"] = 32
    elif change == "launch":
        record["launch"]["launch_kwargs"]["BLOCK_SIZE_KV"] = 32
    elif change == "compiler":
        record["launch"]["metadata"]["num_stages"] = 3
    elif change == "ptx":
        record["compiler_evidence"] = {}
    elif change == "repeat":
        record["native_repeat"]["bitwise_equal"] = False
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=error):
        probe.verify_reference(root)


@pytest.mark.parametrize("artifact", ["one.npz", "one.ptx.txt"])
def test_changed_archive_and_compiler_bytes_are_rejected(reference, artifact):
    root, _, _ = reference
    (root / artifact).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        probe.verify_reference(root)


def test_rehashed_wrong_input_mask_and_dtype_do_not_bypass_validation(reference):
    root, case, manifest = reference
    arrays = probe.operands(case)
    for key, value in (
        ("query", arrays["query"].astype(np.float64)),
        ("additive_mask", np.zeros_like(arrays["additive_mask"])),
        ("pair_bias", np.full_like(arrays["pair_bias"], np.nan)),
    ):
        with pytest.raises(ValueError):
            probe.validate_operands({**arrays, key: value}, case)


def test_candidate_source_identity_binds_attention_and_imported_linear_helper(tmp_path):
    for relative in (probe.CANDIDATE_OPERATOR, probe.CANDIDATE_HELPER):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n")
    identity = probe.source_identity(tmp_path, False)
    assert identity["operator_path"] == probe.CANDIDATE_OPERATOR
    assert identity["dependency_sha256"][probe.CANDIDATE_HELPER] == probe.digest(
        tmp_path / probe.CANDIDATE_HELPER
    )
    (tmp_path / probe.CANDIDATE_HELPER).write_text("value = 2\n")
    changed = probe.source_identity(tmp_path, False)
    assert identity["python_source_sha256"] != changed["python_source_sha256"]
    assert identity["operator_sha256"] == changed["operator_sha256"]
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("outside = 1\n")
    (tmp_path / "src" / "escape.py").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        probe.source_identity(tmp_path, False)


def test_initial_source_helper_and_reference_identities_are_rechecked(
    reference, monkeypatch
):
    root, _, _ = reference
    identity, helpers = {"source": "initial"}, {"helper": "initial"}
    monkeypatch.setattr(probe, "source_identity", lambda *a: identity.copy())
    monkeypatch.setattr(probe, "helper_identities", lambda: helpers.copy())
    sha = probe.digest(root / "manifest.json")
    probe.verify_unchanged(root, False, identity, helpers, root, sha)
    for stale_source, stale_helpers, error in (
        ({}, helpers, "source changed"),
        (identity, {}, "helper source"),
    ):
        with pytest.raises(RuntimeError, match=error):
            probe.verify_unchanged(root, False, stale_source, stale_helpers, root, sha)


@pytest.mark.parametrize("live", [True, False])
def test_hlo_bytecode_decoder_checks_actual_carried_operand_not_marker_text(live):
    from jax._src.lib.mlir import ir
    from jax._src.lib.mlir.dialects import func

    from foldjax.models.openfold3.models import native_triton_attention as attention

    with attention.triton_lowering._new_ir_context(), ir.Location.unknown():
        module = ir.Module.create()
        tensor = ir.RankedTensorType.get((16, 16), ir.F32Type.get())
        with ir.InsertionPoint(module.body):
            function = func.FuncOp("proof", ir.FunctionType.get([tensor] * 3, [tensor]))
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                attention._carried_dot_lowering(None, *block.arguments)
                scaled = attention.tt_dialect.elementwise_inline_asm(
                    [tensor],
                    "mul.rn.f32 $0, $1, $2;",
                    constraints="=f,f,f",
                    pure=True,
                    packed_element=1,
                    args=[block.arguments[2], block.arguments[0]],
                )
                result = attention._carried_dot_lowering(
                    None,
                    block.arguments[0],
                    block.arguments[1],
                    scaled if live else block.arguments[2],
                )
                func.ReturnOp([result])
        stream = io.BytesIO()
        module.operation.write_bytecode(stream)
    hlo = (
        'backend_config={ir = "'
        + "".join(f"\\{byte:02X}" for byte in stream.getvalue())
        + '"}'
    )
    if live:
        text, report = probe.decode_triton_ir(hlo)
        assert "tt.dot" in text and report == {
            "dot_count": 2,
            "live_rescaled_accumulator_dots": 1,
        }
    else:
        with pytest.raises(ValueError, match="live rescaled"):
            probe.decode_triton_ir(hlo)


@pytest.mark.parametrize(
    "hlo", ["no embedded module", 'ir = "\\XX"', 'ir = "a" ir = "b"']
)
def test_missing_or_malformed_embedded_ir_is_explicit_failure(hlo):
    with pytest.raises(ValueError):
        probe.decode_triton_ir(hlo)


def test_native_case_failures_are_all_retained_and_no_success_manifest(
    monkeypatch, tmp_path
):
    for key in ("OF3_TRITON_EXP2", "OF3_TRITON_DYNAMIC_SHAPES"):
        monkeypatch.delenv(key, raising=False)
    cases = [probe.Case("first", 17, 16), probe.Case("second", 31, 16)]
    monkeypatch.setattr(probe, "case_specs", lambda: cases)
    monkeypatch.setattr(
        probe, "source_identity", lambda *a: {"operator_path": probe.NATIVE_OPERATOR}
    )
    monkeypatch.setattr(
        probe,
        "helper_identities",
        lambda: {"openbind_triton_attention_probe.py": "same"},
    )
    module = ModuleType("fake_native")
    module.__file__ = str(tmp_path / probe.NATIVE_OPERATOR)
    module._attn_fwd = None
    monkeypatch.setattr(probe.importlib, "import_module", lambda name: module)
    torch = SimpleNamespace(
        __version__="fake",
        version=SimpleNamespace(cuda="fake", hip=None),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda _: "fake",
        ),
        set_float32_matmul_precision=lambda _: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="fake"))

    def fail(case):
        raise ValueError(f"preserved failure {case.name}")

    monkeypatch.setattr(probe, "operands", fail)
    out = tmp_path / "out"
    assert (
        probe.main(
            ["--mode", "native", "--source-root", str(tmp_path), "--out-dir", str(out)]
        )
        == 1
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["all_cases_attempted"] and not manifest["completed"]
    assert manifest["summary"]["execution_errors"] == 2
    for case in cases:
        record = json.loads((out / f"{case.name}.json").read_text())
        assert record["status"] == "error" and case.name in record["error"]


def test_native_nondefault_environment_is_rejected_before_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("OF3_TRITON_EXP2", "1")
    with pytest.raises(ValueError, match="default"):
        probe.main(
            [
                "--mode",
                "native",
                "--source-root",
                str(tmp_path),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert not (tmp_path / "out").exists()
