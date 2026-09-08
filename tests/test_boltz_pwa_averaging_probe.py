import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.boltz_pwa_averaging_probe import (
    bitwise_comparison,
    einsum_average,
    fp32_accumulator_average,
    jax_profiles,
    load_reference,
    loaded_cublas_libraries,
    lowering_evidence,
    main,
    native_profiles,
    profiled_call,
    profiler_evidence,
    raw_fp32_output_control,
    save_array,
    sha,
    source_binding,
    validate_operands,
    wide_average,
)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "function", [einsum_average, wide_average, fp32_accumulator_average]
)
def test_averaging_layout_preserves_msa_token_and_channel_axes(function, compiled):
    weights = (np.arange(25).reshape(1, 1, 5, 5) % 2).astype(np.float32)
    values = (np.arange(60).reshape(1, 3, 5, 4) % 3).astype(np.float32)
    expected = np.einsum("bhij,bhsjd->bhsid", weights, values[:, None])[:, 0]
    actual = (jax.jit(function) if compiled else function)(
        jnp.asarray(weights, jnp.bfloat16), jnp.asarray(values, jnp.bfloat16)
    )
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(actual, np.float32), expected)


def test_fp32_control_keeps_bf16_operands_and_rounds_only_dot_output():
    weights = jnp.ones((1, 1, 5, 5), jnp.bfloat16)
    values = jnp.ones((1, 3, 5, 4), jnp.bfloat16)
    graph = jax.make_jaxpr(fp32_accumulator_average)(weights, values)
    dot = next(eq for eq in graph.jaxpr.eqns if eq.primitive.name == "dot_general")
    assert all(value.aval.dtype == jnp.bfloat16 for value in dot.invars)
    assert dot.params["preferred_element_type"] == jnp.float32
    assert dot.outvars[0].aval.dtype == jnp.float32
    assert graph.jaxpr.outvars[0].aval.dtype == jnp.bfloat16


def test_accumulation_backend_factorial_is_opt_in_and_preserves_baselines():
    baseline = jax_profiles()
    profiles = jax_profiles(True)
    assert profiles[:2] == baseline
    assert len(profiles) == 5
    assert [
        (fn is fp32_accumulator_average, flags) for _, fn, flags in profiles[2:]
    ] == [
        (True, {}),
        (False, {"xla_gpu_enable_triton_gemm": False}),
        (True, {"xla_gpu_enable_triton_gemm": False}),
    ]
    assert native_profiles(True, True) == [
        ("einsum", True),
        ("native_layout_bmm", True),
        ("einsum_reduction_disabled", False),
    ]
    assert native_profiles(False) == [("einsum", False), ("native_layout_bmm", False)]


def test_lowering_evidence_uses_actual_dispatch_not_requested_flags():
    triton = lowering_evidence(
        'ROOT d = bf16[2,3] dot(a,b)\nkind="__triton_nested_gemm_fusion"'
    )
    assert triton["contains_triton_gemm"]
    assert triton["dot_instructions"] == ["ROOT d = bf16[2,3] dot(a,b)"]
    cublas = lowering_evidence('custom-call(x), custom_call_target="__cublas$gemm"')
    assert not cublas["contains_triton_gemm"]
    assert cublas["custom_call_targets"] == ["__cublas$gemm"]


@pytest.mark.parametrize("kind", ["__triton_gemm", "__triton_nested_gemm_fusion"])
def test_lowering_evidence_recognizes_json_gemm_backend_kind(kind):
    assert lowering_evidence(
        'backend_config={"fusion_backend_config":{"kind":"' + kind + '"}}'
    )["contains_triton_gemm"]


def test_lowering_evidence_does_not_call_cublas_conversion_fusion_a_triton_gemm():
    evidence = lowering_evidence(
        'custom-call(x), custom_call_target="__cublas$lt$matmul"\n'
        "fusion(x), kind=kCustom, calls=%fused_transpose, "
        'backend_config={"fusion_backend_config":{"kind":"__triton"}}'
    )
    assert not evidence["contains_triton_gemm"]
    assert evidence["custom_call_targets"] == ["__cublas$lt$matmul"]
    assert evidence["dot_instructions"] == []


@pytest.mark.parametrize(
    "arm,flag",
    [
        ("native", "--accumulation-controls"),
        ("foldjax", "--reduction-controls"),
        ("foldjax", "--raw-fp32-control"),
    ],
)
def test_precision_controls_reject_the_wrong_framework(
    monkeypatch, tmp_path, arm, flag
):
    argv = [
        "probe",
        arm,
        "--source-root",
        str(tmp_path),
        "--reference",
        str(tmp_path),
        "--out",
        str(tmp_path / "new"),
        flag,
    ]
    if arm == "native":
        argv += ["--msa-reference", str(tmp_path), "--upstream", str(tmp_path)]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("problem", ["dtype", "shape", "nan", "rounding", "heads"])
def test_operand_validation_rejects_unreviewed_inputs(problem):
    weights = np.ones((1, 1, 5, 5), np.float32)
    values = np.ones((1, 3, 5, 4), np.float32)
    expected = np.ones_like(values)
    if problem == "dtype":
        values = values.astype(np.float64)
    elif problem == "shape":
        expected = expected[:, :2]
    elif problem == "nan":
        weights[0, 0, 0, 0] = np.nan
    elif problem == "rounding":
        values[0, 0, 0, 0] = 1.0001
    else:
        weights = np.repeat(weights, 2, axis=1)
    with pytest.raises(ValueError):
        validate_operands(weights, values, expected)


@pytest.fixture
def native_reference(tmp_path):
    root, msa_root = tmp_path / "pwa", tmp_path / "msa"
    root.mkdir()
    tree = msa_root / "layers/00/input_m.tree.json"
    tree.parent.mkdir(parents=True)
    tree.write_text(
        json.dumps(
            {
                "": {
                    "shape": [1, 2, 385, 64],
                    "native_dtype": "torch.bfloat16",
                    "storage_dtype": "float32",
                }
            }
        )
    )
    msa = {
        "arm": "native",
        "passed": True,
        "native_source": {},
        "runtime": {"torch": "test"},
        "stages": {"layers/00/input_m": {"tree_sha256": sha(tree)}},
    }
    (msa_root / "report.json").write_text(json.dumps(msa))
    artifact = save_array(
        root / "stages.npz",
        **{
            "head2/weights": np.ones((1, 1, 385, 385), np.float32),
            "head2/v": np.ones((1, 2, 385, 32), np.float32),
            "head2/averaged": np.ones((1, 2, 385, 32), np.float32),
            # Reading this member would raise with allow_pickle=False.
            "unrelated": np.array([object()], dtype=object),
        },
    )
    report = {
        "arm": "native",
        "passed": True,
        "native_decomposition": {"values_equal": True},
        "row_slice_vs_full": {"values_equal": True},
        "rows": 2,
        "native_source": {},
        "torch": "test",
        "native_msa_report_sha256": sha(msa_root / "report.json"),
        "artifacts": {"stages.npz": artifact},
    }
    (root / "report.json").write_text(json.dumps(report))
    return root, msa_root


def test_reference_reads_only_selected_archive_members(native_reference):
    _, _, weights, values, expected = load_reference(*native_reference, 2)
    assert weights.shape == (1, 1, 385, 385)
    assert values.shape == expected.shape == (1, 2, 385, 32)


@pytest.mark.parametrize("problem", ["incomplete", "rows", "hash", "head"])
def test_reference_rejects_unbound_or_partial_capture(native_reference, problem):
    root, msa_root = native_reference
    path = root / "report.json"
    report = json.loads(path.read_text())
    if problem == "incomplete":
        report["native_decomposition"]["values_equal"] = False
    elif problem == "rows":
        report["rows"] = 1
    elif problem == "hash":
        report["artifacts"]["stages.npz"] = "invalid"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_reference(root, msa_root, 8 if problem == "head" else 2)


def test_probe_refuses_existing_output_before_importing_framework(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "foldjax",
            "--source-root",
            str(tmp_path),
            "--reference",
            str(tmp_path),
            "--out",
            str(tmp_path),
        ],
    )
    with pytest.raises(FileExistsError):
        main()


def test_source_binding_rejects_another_snapshot(tmp_path):
    with pytest.raises(ValueError, match="snapshot"):
        source_binding(tmp_path)
    root = Path(__file__).resolve().parents[1]
    assert "bench/boltz_pwa_averaging_probe.py" in source_binding(root)


def test_private_raw_control_rejects_output_inside_source(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "native",
            "--source-root",
            str(tmp_path),
            "--reference",
            str(tmp_path),
            "--msa-reference",
            str(tmp_path),
            "--upstream",
            str(tmp_path),
            "--out",
            str(tmp_path / "capture"),
            "--raw-fp32-control",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "capture").exists()


def test_cublas_identity_hashes_only_distinct_mapped_libraries(tmp_path):
    library = tmp_path / "libcublas.so.13"
    library.write_bytes(b"test library")
    info = library.stat()
    device = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}"
    line = f"0-1 r-xp 0 {device} {info.st_ino} {library}\n"
    maps = tmp_path / "maps"
    maps.write_text(line * 2 + "0-1 r-xp 0 00:00 1 /irrelevant/libc.so.6\n")
    identity = loaded_cublas_libraries(maps)
    assert len(identity) == 1
    assert identity[0]["path"] == str(library)
    assert identity[0]["sha256"] == sha(library)
    maps.write_text(line.replace(str(info.st_ino), str(info.st_ino + 1), 1))
    with pytest.raises(ValueError, match="replaced"):
        loaded_cublas_libraries(maps)
    maps.write_text(line.rstrip() + " (deleted)\n")
    with pytest.raises(ValueError, match="unavailable"):
        loaded_cublas_libraries(maps)
    maps.write_text("")
    with pytest.raises(ValueError, match="no loaded"):
        loaded_cublas_libraries(maps)


def test_profiler_records_cuda_kernels_and_cpu_operator_shapes():
    cpu = SimpleNamespace(
        device_type="DeviceType.CPU",
        name="aten::bmm",
        input_shapes=[[1, 3, 3]],
        kernels=[SimpleNamespace(name="actual_gemm")],
    )
    cuda = SimpleNamespace(device_type="DeviceType.CUDA", name="actual_gemm")
    evidence = profiler_evidence([cpu, cuda])
    assert evidence["kernel_names"] == ["actual_gemm"]
    assert evidence["cpu_events"] == [
        {"name": "aten::bmm", "input_shapes": [[1, 3, 3]]}
    ]
    assert evidence["kernel_capture_complete"]
    assert not profiler_evidence([])["kernel_capture_complete"]


def test_profiled_call_requests_shapes_and_synchronizes():
    seen, syncs = [], []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: syncs.append(True)),
        profiler=SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CPU="CPU", CUDA="CUDA"),
            profile=lambda **kwargs: (
                seen.append(kwargs) or nullcontext(SimpleNamespace(events=lambda: []))
            ),
        ),
    )
    result, evidence = profiled_call(torch, lambda: "result")
    assert result == "result"
    assert seen == [{"activities": ["CPU", "CUDA"], "record_shapes": True}]
    assert len(syncs) == 2
    assert not evidence["kernel_capture_complete"]


def test_bridge_gate_is_bitwise_and_rejects_nonfinite_output():
    expected = np.array([0], np.float32)
    assert bitwise_comparison(expected, expected)["bitwise_equal"]
    assert not bitwise_comparison(-expected, expected)["bitwise_equal"]
    assert bitwise_comparison(-expected, expected)["numerical"]["values_equal"]
    assert not bitwise_comparison(np.array([np.nan], np.float32), expected)["finite"]


class _CpuTensor:
    """Small API double; real CUDA out_dtype execution belongs to the GPU gate."""

    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape, self.dtype, self.device = self.data.shape, self.data.dtype, "cpu"

    def __getitem__(self, index):
        return _CpuTensor(self.data[index])

    def bfloat16(self):
        return _CpuTensor(self.data.astype(jnp.bfloat16))

    def float(self):
        return _CpuTensor(self.data.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def reshape(self, *shape):
        return _CpuTensor(self.data.reshape(shape))

    def permute(self, *axes):
        return _CpuTensor(self.data.transpose(axes))

    def stride(self):
        return tuple(value // self.data.itemsize for value in self.data.strides)

    def storage_offset(self):
        return 0

    def is_contiguous(self):
        return self.data.flags.c_contiguous


@pytest.mark.parametrize("failure", [None, "bridge", "baseline"])
def test_raw_control_preserves_operands_and_retains_failed_bridge(
    monkeypatch, tmp_path, failure
):
    import bench.boltz_pwa_averaging_probe as probe

    calls, autocast = [], []
    matmul = SimpleNamespace(allow_bf16_reduced_precision_reduction=True)

    def bmm(lhs, rhs, **kwargs):
        calls.append((lhs, rhs, kwargs))
        value = np.matmul(lhs.data.astype(np.float32), rhs.data.astype(np.float32))
        if kwargs:
            value += 0.125 if failure == "bridge" else 0.0001
            return _CpuTensor(value)
        if failure == "baseline":
            value += 0.125
        return _CpuTensor(value).bfloat16()

    torch = SimpleNamespace(
        bfloat16=np.dtype(jnp.bfloat16),
        float32=np.dtype(np.float32),
        bmm=bmm,
        version=SimpleNamespace(git_version="source"),
        cuda=SimpleNamespace(tunable=SimpleNamespace(is_enabled=lambda: False)),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=matmul, preferred_blas_library=lambda: "Cublas")
        ),
        get_float32_matmul_precision=lambda: "highest",
        set_float32_matmul_precision=lambda value: None,
        autocast=lambda *args, **kwargs: autocast.append(kwargs) or nullcontext(),
    )
    monkeypatch.setattr(
        probe,
        "profiled_call",
        lambda _, fn: (
            fn(),
            {"kernel_names": ["test"], "kernel_capture_complete": True},
        ),
    )
    monkeypatch.setattr(
        probe, "loaded_cublas_libraries", lambda: [{"path": "/private/libcublas.so"}]
    )
    weights = _CpuTensor(np.ones((1, 1, 3, 3), np.float32))
    values = _CpuTensor(np.ones((1, 2, 3, 2), np.float32)).bfloat16()
    expected = np.full(values.shape, 3, np.float32)
    if failure == "baseline":
        with pytest.raises(ValueError, match="baseline"):
            raw_fp32_output_control(
                torch,
                weights,
                values,
                expected,
                {"bf16_reduced_precision_reduction": True},
                tmp_path,
                provenance={"inputs_sha256": "bound"},
            )
        assert len(calls) == 1
        assert (tmp_path / "raw_fp32_failed_reproduction.json").exists()
        assert not (tmp_path / "raw_fp32_output.npz").exists()
        return
    report = raw_fp32_output_control(
        torch,
        weights,
        values,
        expected,
        {"bf16_reduced_precision_reduction": True},
        tmp_path,
        provenance={"inputs_sha256": "bound"},
    )
    assert autocast == [{"enabled": False}]
    assert len(calls) == 2
    assert calls[0][0] is calls[1][0] and calls[0][1] is calls[1][1]
    assert calls[0][2] == {} and calls[1][2] == {"out_dtype": torch.float32}
    assert report["bridge_gate_passed"] is (failure is None)
    assert report["internal_accumulator_identity_proven"] is False
    assert report["profiling_complete"] is True
    assert "loaded_cublas_libraries" not in report
    with np.load(tmp_path / "raw_fp32_output.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"raw_output", "rounded_output"}
        assert archive["raw_output"].dtype == np.float32
        assert np.all(archive["raw_output"] > expected)
    private_path = tmp_path / "raw_fp32_control.private.json"
    assert sha(private_path) == report["private_metadata_sha256"]
    assert json.loads(private_path.read_text())["loaded_cublas_libraries"]
