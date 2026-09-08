import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench import boltz_pwa_full_mma_probe as probe
from bench import boltz_pwa_mma_probe as tiled


def test_schedule_is_the_proved_fixed_tile_residue_control():
    a, b, _, _ = tiled.basis_operands()
    _, _, expected = tiled.k8_control_operands(a, b, "residue_first_32")
    np.testing.assert_array_equal(probe.residue_k(np.arange(448), np), expected)
    assert (probe.ROWS, probe.TOKENS, probe.CHANNELS, probe.CHUNK_ROWS) == (
        4436,
        437,
        32,
        64,
    )
    assert probe.schedule_proof() == {
        "original_terms_once_in_order": 437,
        "inserted_zero_positions": list(range(21, 32)),
        "mma_steps": 56,
    }


def test_schedule_proof_rejects_a_missing_or_duplicate_term(monkeypatch):
    monkeypatch.setattr(probe, "residue_k", lambda position, xp: position % 437)
    with pytest.raises(ValueError, match="every original term"):
        probe.schedule_proof()


@pytest.mark.parametrize("tile_m,tile_n", [(0, 0), (1, 32), (3, 54)])
def test_on_device_gather_indices_match_proved_tile_registers(tile_m, tile_n):
    rng = np.random.default_rng(728)
    weights = tiled.bf16_round(rng.normal(size=(437, 437)).astype(np.float32))
    values = tiled.bf16_round(rng.normal(size=(2, 437, 32)).astype(np.float32))
    values[0, 21, 0] = -0.0
    wbits, vbits = probe.bf16_words(weights), probe.bf16_words(values)
    mm = np.arange(tile_m * 16, tile_m * 16 + 16)
    nn = np.arange(tile_n * 8, tile_n * 8 + 8)
    a = values[mm // 32, :, mm % 32][None]
    b = np.zeros((1, 437, 8), np.float32)
    b[0, :, nn < 437] = weights[nn[nn < 437]]
    aa, bb, _ = tiled.k8_control_operands(a, b, "residue_first_32")
    expected = tiled.pack_operands(aa, bb, 8)[0]
    actual = np.empty_like(expected)
    for step in range(56):
        mr, nr, kr = probe.fragment_indices(tile_m, tile_n, step, np)
        for register, m in enumerate(mr):
            halves = [
                np.where(k >= 0, vbits[m // 32, np.maximum(k, 0), m % 32], 0).astype(
                    np.uint32
                )
                for k in kr
            ]
            actual[step, register] = halves[0] | halves[1] << 16
        halves = [
            np.where(
                (k >= 0) & (nr < 437), wbits[np.minimum(nr, 436), np.maximum(k, 0)], 0
            ).astype(np.uint32)
            for k in kr
        ]
        actual[step, 2] = halves[0] | halves[1] << 16
    np.testing.assert_array_equal(actual, expected)


def test_output_mapping_writes_each_valid_element_once_and_masks_only_token_tail():
    counts = np.zeros((2, 437, 32), np.int32)
    lane = np.arange(32)
    invalid = 0
    for tile_m in range(4):
        for tile_n in range(55):
            for register in range(4):
                m = tile_m * 16 + lane // 4 + (register // 2) * 8
                n = tile_n * 8 + 2 * (lane % 4) + register % 2
                valid = n < 437
                np.add.at(counts, (m[valid] // 32, n[valid], m[valid] % 32), 1)
                invalid += int(np.count_nonzero(~valid))
    np.testing.assert_array_equal(counts, np.ones_like(counts))
    assert invalid == 2 * 32 * 3


def test_full_kernel_jaxpr_has_on_device_words_and_genuine_carried_k8_c():
    graph = jax.make_jaxpr(probe.full_mma_call)(
        jax.ShapeDtypeStruct((437, 437), jnp.uint16),
        jax.ShapeDtypeStruct((2, 437, 32), jnp.uint16),
        jax.ShapeDtypeStruct((1,), jnp.float32),
    )
    call = graph.jaxpr.eqns[0]
    assert call.primitive.name == "pallas_call"
    assert call.params["compiler_params"].num_warps == 1
    assert call.params["grid_mapping"].grid == (4, 55)
    assert not call.params["interpret"]
    kernel = call.params["jaxpr"]
    assert [v.aval.dtype for v in kernel.invars] == [
        jnp.uint16,
        jnp.uint16,
        jnp.float32,
        jnp.float32,
        jnp.int32,
    ]
    loop = next(eq for eq in kernel.eqns if eq.primitive.name == "scan")
    assert loop.params["length"] == 56
    body = loop.params["jaxpr"].jaxpr
    mma = next(
        eq for eq in body.eqns if eq.primitive.name == "elementwise_inline_asm_p"
    )
    assert (mma.params["asm"], mma.params["constraints"]) == tiled.mma_assembly(8)
    assert [v.aval.dtype for v in mma.invars[:3]] == [jnp.uint32] * 3
    assert mma.invars[-4:] == body.invars[-4:]
    assert body.outvars[-4:] == mma.outvars
    assert sum(eq.primitive.name == "masked_load" for eq in body.eqns) == 6
    for eq in body.eqns:
        assert "dot" not in eq.primitive.name
        if eq.primitive.name == "add":
            assert all(var.aval.dtype == jnp.int32 for var in eq.invars)
    lane = next(
        eq for eq in kernel.eqns if eq.params.get("asm") == "mov.u32 $0, %laneid;"
    )
    assert lane.outvars[0].aval.dtype == jnp.int32


@pytest.mark.parametrize("problem", ["weights", "rows", "tokens", "dtype", "initial"])
def test_full_kernel_rejects_unreviewed_shape_or_storage_before_lowering(problem):
    weights = jax.ShapeDtypeStruct((437, 437), jnp.uint16)
    values = jax.ShapeDtypeStruct((2, 437, 32), jnp.uint16)
    initial = jax.ShapeDtypeStruct((1,), jnp.float32)
    if problem == "weights":
        weights = jax.ShapeDtypeStruct((1, 437, 437), jnp.uint16)
    elif problem == "rows":
        values = jax.ShapeDtypeStruct((0, 437, 32), jnp.uint16)
    elif problem == "tokens":
        values = jax.ShapeDtypeStruct((2, 436, 32), jnp.uint16)
    elif problem == "dtype":
        values = jax.ShapeDtypeStruct((2, 437, 32), jnp.bfloat16)
    else:
        initial = jax.ShapeDtypeStruct((2,), jnp.float32)
    with pytest.raises(ValueError, match="requires BF16 words"):
        probe.full_mma_call(weights, values, initial)


def test_host_word_conversion_is_lossless_including_signed_zero():
    words = np.array([0, 0x8000, 0x3F80, 0xBF80, 1, 0x7F7F], np.uint32)
    value = (words << 16).view(np.float32).reshape(3, 2)
    np.testing.assert_array_equal(
        probe.bf16_words(value, chunk_rows=1), words.reshape(3, 2)
    )


@pytest.mark.parametrize("problem", ["fp64", "non_bf16", "nan", "inf", "zero_chunk"])
def test_host_word_conversion_never_silently_rounds_or_accepts_nonfinite(problem):
    value = np.zeros((2, 3), np.float32)
    chunk = 1
    if problem == "fp64":
        value = value.astype(np.float64)
    elif problem == "zero_chunk":
        chunk = 0
    else:
        value[1, 2] = {"non_bf16": 1.00001, "nan": np.nan, "inf": np.inf}[problem]
    with pytest.raises(ValueError):
        probe.bf16_words(value, chunk_rows=chunk)


def test_same_kernel_basis_has_dense_all_k_one_hot_tail_and_nonzero_c():
    weights, values, initial, expected = probe.basis_inputs()
    assert values.shape == (2, 437, 32)
    assert initial[0] == 3
    assert np.all(np.any(weights != 0, axis=0))
    np.testing.assert_array_equal(weights[:16], np.eye(437, dtype=np.float32)[:16])
    np.testing.assert_array_equal(weights[-1], np.eye(437, dtype=np.float32)[-1])
    np.testing.assert_array_equal(
        values[1, :, 16:], np.eye(437, dtype=np.float32)[:, -16:]
    )
    expected_by_steps = np.full_like(expected, initial[0])
    positions = probe.residue_k(np.arange(448), np)
    for start in range(0, 448, 8):
        k = positions[start : start + 8]
        k = k[k >= 0]
        expected_by_steps += np.einsum("ij,sjd->sid", weights[:, k], values[:, k])
    np.testing.assert_array_equal(expected_by_steps, expected)


def test_chunked_statistics_retain_bitwise_signed_zero_and_numerical_metrics():
    target = np.zeros(5, np.float32)
    actual = np.array([-0.0, 3, -4, 0, 0], np.float32)
    stats = probe.empty_stats()
    probe.add_stats(stats, actual[:2], target[:2])
    probe.add_stats(stats, actual[2:], target[2:])
    assert probe.finish_stats(stats) == {
        "count": 5,
        "bitwise_unequal": 3,
        "nonfinite": 0,
        "bitwise_equal": False,
        "rmse": np.sqrt(5),
        "max_abs": 4,
    }
    assert not probe.finish_stats(probe.empty_stats())["bitwise_equal"]


def test_nonfinite_statistics_never_pass_even_identical_nan_bits():
    value = np.array([1, np.nan, np.inf], np.float32)
    stats = probe.empty_stats()
    probe.add_stats(stats, value, value.copy())
    result = probe.finish_stats(stats)
    assert result["count"] == 3 and result["bitwise_unequal"] == 0
    assert result["nonfinite"] == 2 and not result["bitwise_equal"]
    assert result["rmse"] is result["max_abs"] is None


def test_failed_nonfinite_rounding_keeps_original_payloads_without_cast_claim():
    bits = np.array([0x7FC00003, 0x7F800000, 0xFF800000], np.uint32)
    values = np.concatenate((bits.view(np.float32), np.array([1.0001], np.float32)))
    result = probe.round_finite_output(values)
    np.testing.assert_array_equal(result[:3].view(np.uint32), bits)
    assert result[3] == probe.bf16_round(values[3])


@pytest.mark.parametrize("nonfinite", [False, True])
def test_stream_compares_and_archives_every_output_in_bounded_rows(tmp_path, nonfinite):
    target = np.zeros((130, 3, 2), np.float32)
    actual = target.copy()
    actual[64, 1, 0] = np.nan if nonfinite else 1.0001
    reads = []

    def read_chunk(start, stop):
        reads.append((start, stop))
        return actual[start:stop]

    report = {}
    probe.compare_stream(target, read_chunk, tmp_path, report)
    assert reads == [(0, 64), (64, 128), (128, 130)]
    for chunk in report["chunks"]:
        path = tmp_path / chunk["file"]
        assert chunk["sha256"] == probe.sha(path)
        with np.load(path, allow_pickle=False) as archive:
            assert set(archive.files) == {"raw_output", "rounded_output"}
            expected = actual[chunk["start"] : chunk["stop"]]
            np.testing.assert_array_equal(
                archive["raw_output"].view(np.uint32), expected.view(np.uint32)
            )
            np.testing.assert_array_equal(
                archive["rounded_output"].view(np.uint32),
                probe.round_finite_output(expected).view(np.uint32),
            )
    for key in ("raw_fp32", "bf16"):
        assert report[key]["count"] == target.size
        assert report[key]["bitwise_unequal"] == 1
        assert report[key]["nonfinite"] == int(nonfinite)
        assert not report[key]["bitwise_equal"]


@pytest.fixture
def reference(tmp_path, monkeypatch):
    # Only the test fixture uses fewer MSA rows; the CLI contract remains S4436.
    monkeypatch.setattr(probe, "ROWS", 2)
    root = tmp_path / "native"
    root.mkdir()
    weights = np.ones((1, 1, 437, 437), np.float32)
    values = np.ones((1, 2, 437, 32), np.float32)
    raw = np.full_like(values, 437)
    inputs_hash = probe.save_array(root / "inputs.npz", weights=weights, values=values)
    raw_hash = probe.save_array(
        root / "raw_fp32_output.npz",
        raw_output=raw,
        # Selective loading must not deserialize this unrelated member.
        rounded_output=np.array([object()], dtype=object),
    )
    source = {"bench/native_probe.py": "frozen"}
    metadata = {
        "provenance": {
            "inputs_sha256": inputs_hash,
            "selected_head": 2,
            "source": source,
        },
        "baseline": {"comparison": {"bitwise_equal": True}},
        "bridge_gate_passed": True,
        "capture_complete": True,
        "output_sha256": raw_hash,
    }
    probe.save_new(root / "raw_fp32_control.private.json", metadata)
    probe.save_new(
        root / "report.json",
        {
            "arm": "native",
            "passed": True,
            "head": 2,
            "source": source,
            "inputs_sha256": inputs_hash,
            "input_shapes": {
                "weights": list(weights.shape),
                "values": list(values.shape),
            },
            "raw_fp32_control": {
                "bridge_gate_passed": True,
                "capture_complete": True,
                "private_metadata_sha256": probe.sha(
                    root / "raw_fp32_control.private.json"
                ),
                "output_sha256": raw_hash,
            },
        },
    )
    return root


def test_reference_loader_binds_native_bridge_and_selects_only_raw_member(reference):
    weights, values, target, binding = probe.load_reference(reference)
    assert weights.dtype == values.dtype == np.uint16
    assert weights.shape == (437, 437) and values.shape == (2, 437, 32)
    np.testing.assert_array_equal(values, np.full_like(values, 0x3F80))
    np.testing.assert_array_equal(target, np.full_like(target, 437))
    assert binding == probe.reference_binding(reference)


@pytest.mark.parametrize(
    "problem", ["rows", "bridge", "source", "hash", "head", "passed"]
)
def test_reference_loader_rejects_unbound_or_unapproved_reference(reference, problem):
    path = reference / "report.json"
    report = json.loads(path.read_text())
    if problem == "rows":
        report["input_shapes"]["values"][1] = 4435
    elif problem == "bridge":
        report["raw_fp32_control"]["bridge_gate_passed"] = False
    elif problem == "source":
        report["source"] = {"different": "source"}
    elif problem == "hash":
        report["raw_fp32_control"]["output_sha256"] = "wrong"
    elif problem == "head":
        report["head"] = 3
    else:
        report["passed"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        probe.load_reference(reference)


def rebind_raw(reference, **arrays):
    path = reference / "raw_fp32_output.npz"
    np.savez(path, **arrays)
    report = json.loads((reference / "report.json").read_text())
    private_path = reference / "raw_fp32_control.private.json"
    private = json.loads(private_path.read_text())
    private["output_sha256"] = report["raw_fp32_control"]["output_sha256"] = probe.sha(
        path
    )
    private_path.write_text(json.dumps(private))
    report["raw_fp32_control"]["private_metadata_sha256"] = probe.sha(private_path)
    (reference / "report.json").write_text(json.dumps(report))


@pytest.mark.parametrize("problem", ["schema", "shape", "dtype", "nonfinite"])
def test_reference_loader_rejects_hash_bound_bad_raw_arrays(reference, problem):
    raw = np.zeros((1, 2, 437, 32), np.float32)
    arrays = {"raw_output": raw, "rounded_output": raw}
    if problem == "schema":
        arrays["unknown"] = np.zeros(1)
    elif problem == "shape":
        arrays["raw_output"] = raw[:, :1]
    elif problem == "dtype":
        arrays["raw_output"] = raw.astype(np.float64)
    else:
        raw[0, 1, 1, 1] = np.nan
    rebind_raw(reference, **arrays)
    with pytest.raises(ValueError):
        probe.load_reference(reference)


def setup_cli(monkeypatch, reference, out):
    root = Path(probe.__file__).resolve().parents[1]
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--source-root",
            str(root),
            "--reference",
            str(reference),
            "--out",
            str(out),
        ],
    )
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "devices", lambda: [object()])


@pytest.mark.parametrize(
    "failure", ["unsupported", "basis", "full_nonfinite", "binding", None]
)
def test_cli_gates_basis_before_full_and_records_failures_without_fallback(
    monkeypatch, tmp_path, reference, failure
):
    out = tmp_path / "full-probe"
    setup_cli(monkeypatch, reference, out)
    calls = []

    def fake_compile(weights, values, initial, directory, label):
        calls.append(label)
        assert weights.dtype == values.dtype == np.uint16
        if failure == "unsupported":
            (directory / "basis.0.triton.mlir").write_text("failed compiler evidence")
            raise NotImplementedError("unsupported opcode")
        if label == "basis":
            result = probe.basis_inputs()[-1]
            if failure == "basis":
                result[0, 0, 0] += 1
        else:
            assert initial[0] == 0
            result = np.full(values.shape, 437, np.float32)
            if failure == "full_nonfinite":
                result[0, 0, 0] = np.nan
        return result, {"physical_lane_check_first_tile": True}

    monkeypatch.setattr(probe, "compile_and_run", fake_compile)
    if failure == "binding":
        reads = []

        def source_binding(root):
            reads.append(root)
            return {"snapshot": str(len(reads))}

        monkeypatch.setattr(probe, "source_binding", source_binding)
    if failure is None:
        probe.main()
    else:
        with pytest.raises(SystemExit, match="1"):
            probe.main()
    report = json.loads((out / "report.json").read_text())
    assert report["passed"] == (failure is None)
    assert report["bindings_unchanged"] == (failure != "binding")
    assert report["capture_complete"] == (failure not in ("unsupported", "basis"))
    assert not report["native_private_cta_identity_proven"]
    assert report["not_model_parity_admission"]
    assert report["schedule_proof"]["original_terms_once_in_order"] == 437
    assert calls == (
        ["basis"] if failure in ("unsupported", "basis") else ["basis", "full"]
    )
    if failure == "unsupported":
        assert not report["error"]["fallback_used"]
        assert report["error"]["type"] == "NotImplementedError"
        assert report["failed_compiler_artifacts"] == {
            "basis.0.triton.mlir": probe.sha(out / "basis.0.triton.mlir")
        }
    if failure == "full_nonfinite":
        assert report["full"]["raw_fp32"]["nonfinite"] == 1
    if failure is None:
        assert report["full"]["raw_fp32"]["count"] == 2 * 437 * 32
        assert report["full"]["raw_fp32"]["bitwise_equal"]
        assert report["full"]["bf16"]["bitwise_equal"]


def test_cli_never_overwrites_an_existing_output(monkeypatch, tmp_path):
    setup_cli(monkeypatch, tmp_path / "missing-native", tmp_path)
    with pytest.raises(FileExistsError):
        probe.main()


@pytest.mark.parametrize("cast_error", [False, True])
def test_runtime_cli_compares_actual_device_bf16_and_raw_outputs(
    monkeypatch, tmp_path, reference, cast_error
):
    import sys

    out = tmp_path / "runtime-probe"
    setup_cli(monkeypatch, reference, out)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--implementation", "runtime"])
    calls = []

    def runtime(weights, values, initial, directory, label):
        calls.append(label)
        raw = (
            probe.basis_inputs()[-1]
            if label == "basis"
            else np.full(values.shape, 437, np.float32)
        )
        rounded = probe.bf16_round(raw)
        if cast_error and label == "full":
            rounded[0, 0, 0] += 2
        return {"raw": raw, "rounded": rounded}, {
            "probe_only_multi_result_ir_unwrap": False
        }

    monkeypatch.setattr(probe, "runtime_compile_and_run", runtime)
    monkeypatch.setattr(
        probe, "compile_and_run", lambda *a: pytest.fail("bench compiler arm used")
    )
    if cast_error:
        with pytest.raises(SystemExit, match="1"):
            probe.main()
    else:
        probe.main()
    report = json.loads((out / "report.json").read_text())
    assert report["implementation"] == "runtime"
    assert report["capture_complete"] and report["bindings_unchanged"]
    assert report["passed"] == (not cast_error)
    assert report["full"]["raw_fp32"]["bitwise_equal"]
    assert report["full"]["bf16"]["bitwise_equal"] == (not cast_error)
    assert report["full"]["device_bf16_rounding_bridge"]["bitwise_equal"] == (
        not cast_error
    )
    assert calls == ["basis", "full"]
    assert report["basis"]["device_bf16"]["bitwise_equal"]
    assert (
        "src/foldjax/models/boltz2/models/primitives/native_pwa_mma.py"
        in report["source"]
    )
    assert "src/foldjax/models/boltz2/models/trunk_blocks/msa.py" in report["source"]
    assert (
        report["storage"]["mandatory_gpu_array_bytes"]
        == 437 * 437 * 2 + 2 * 437 * 32 * 8
    )


def test_runtime_compile_arm_never_replaces_a_compiler_rule(monkeypatch, tmp_path):
    from foldjax.models.boltz2.models.primitives import native_pwa_mma as native

    registry = native.triton_lowering.triton_lowering_rules.copy()
    w, v, initial, expected = probe.basis_inputs()
    operands_seen = []

    class Executable:
        def as_text(self):
            return "compiled HLO"

        def __call__(self, *operands):
            return (
                jnp.asarray(expected),
                jnp.asarray(expected, jnp.bfloat16),
                jnp.arange(32, dtype=jnp.int32),
            )

    class Lowered:
        def lower(self, *operands):
            operands_seen.append(operands)
            assert operands[0].dtype == operands[1].dtype == jnp.bfloat16
            return self

        def compile(self):
            print("The Triton module for pallas_call")
            print(f"tt.elementwise_inline_asm {native._ASM}")
            return Executable()

    def compile_runtime(function, **kwargs):
        assert function.func is native._pwa_outputs
        assert function.keywords == {"debug": True}
        assert not kwargs["compiler_options"]["xla_allow_excess_precision"]
        return Lowered()

    monkeypatch.setattr(jax, "jit", compile_runtime)
    output, evidence = probe.runtime_compile_and_run(
        probe.bf16_words(w), probe.bf16_words(v), initial, tmp_path, "basis"
    )
    assert len(operands_seen) == 1
    assert native.triton_lowering.triton_lowering_rules == registry
    np.testing.assert_array_equal(output["raw"], expected)
    assert output["rounded"].dtype == jnp.bfloat16
    assert evidence["own_runtime_mma_primitive"]
    assert not evidence["probe_only_multi_result_ir_unwrap"]
    for name, digest in evidence["artifacts"].items():
        assert probe.sha(tmp_path / name) == digest


def test_runtime_source_binding_rejects_another_imported_snapshot(
    monkeypatch, tmp_path
):
    from foldjax.models.boltz2.models.primitives import native_pwa_mma as native

    root = Path(probe.__file__).resolve().parents[1]
    monkeypatch.setattr(native, "__file__", str(tmp_path / "native_pwa_mma.py"))
    with pytest.raises(ValueError, match="another source snapshot"):
        probe.runtime_source_binding(root)


def test_default_row_control_has_only_the_four_actual_5sak_intervals():
    assert probe.DEFAULT_ROW_SLICES == (
        (0, 1199),
        (1199, 2398),
        (2398, 3597),
        (3597, 4436),
    )
    assert [stop - start for start, stop in probe.DEFAULT_ROW_SLICES] == [
        1199,
        1199,
        1199,
        839,
    ]


@pytest.fixture
def row_operands(monkeypatch):
    # Compact storage-only fixture; K437/D32 and runtime dispatch remain fixed.
    monkeypatch.setattr(probe, "ROWS", 13)
    monkeypatch.setattr(
        probe, "DEFAULT_ROW_SLICES", ((0, 4), (4, 8), (8, 12), (12, 13))
    )
    weights = np.arange(437 * 437, dtype=np.uint16).reshape(437, 437)
    values = np.arange(13 * 437 * 32, dtype=np.uint16).reshape(13, 437, 32)
    target = np.arange(values.size, dtype=np.float32).reshape(values.shape)
    return weights, values, target


def test_native_row_views_preserve_all_operand_bits_strides_and_full_output_offsets(
    row_operands,
):
    w, v, target = row_operands
    pieces = []
    for start, stop in probe.DEFAULT_ROW_SLICES:
        ww, vv, yy, binding = probe.native_row_slice(w, v, target, start, stop)
        assert ww is w and np.shares_memory(vv, v) and np.shares_memory(yy, target)
        assert vv.strides == v.strides and yy.strides == target.strides
        assert binding["values_byte_offset"] == start * v.strides[0]
        assert binding["target_byte_offset"] == start * target.strides[0]
        assert binding["values"] == probe.array_identity(v[start:stop])
        assert binding["target"] == probe.array_identity(target[start:stop])
        np.testing.assert_array_equal(vv, v[start:stop])
        np.testing.assert_array_equal(
            yy.view(np.uint32), target[start:stop].view(np.uint32)
        )
        pieces.append(vv)
    np.testing.assert_array_equal(np.concatenate(pieces), v)


@pytest.mark.parametrize("bounds", [(-1, 4), (0, 5), (4, 4), (13, 14)])
def test_native_row_slice_rejects_unapproved_bounds(row_operands, bounds):
    with pytest.raises(ValueError, match="frozen row slice"):
        probe.native_row_slice(*row_operands, *bounds)


@pytest.mark.parametrize("problem", ["shape", "dtype", "strides"])
def test_native_row_slice_rejects_changed_operand_storage(row_operands, problem):
    w, v, target = row_operands
    if problem == "shape":
        v = v[:, :-1]
    elif problem == "dtype":
        v = v.astype(np.float32)
    else:
        v = v[:, :, ::-1]
    with pytest.raises(ValueError):
        probe.native_row_slice(w, v, target, 0, 4)


@pytest.mark.parametrize(
    "slices",
    [((0, 4), (5, 13)), ((0, 4), (3, 13)), ((0, 4), (4, 4), (4, 13)), ((0, 12),)],
)
def test_row_control_rejects_gaps_overlap_empty_or_incomplete_cover(
    monkeypatch, tmp_path, row_operands, slices
):
    monkeypatch.setattr(probe, "DEFAULT_ROW_SLICES", slices)
    monkeypatch.setattr(
        probe,
        "runtime_compile_and_run",
        lambda *a, **k: pytest.fail("compiled invalid slices"),
    )
    with pytest.raises(ValueError):
        probe.run_default_row_chunks(*row_operands, tmp_path, {})


@pytest.mark.parametrize("failure", [None, "nonfinite", "mutation"])
def test_row_control_compares_each_original_full_output_once_and_retains_failures(
    monkeypatch, tmp_path, row_operands, failure
):
    w, v, target = row_operands
    calls = []

    def execute(ww, vv, initial, out, label, *, cache):
        start, stop = probe.DEFAULT_ROW_SLICES[len(calls)]
        calls.append((start, stop, id(cache)))
        assert ww is w and np.shares_memory(vv, v)
        np.testing.assert_array_equal(vv, v[start:stop])
        result = target[start:stop].copy()
        if failure == "nonfinite" and start == 4:
            result[0, 0, 0] = np.nan
        if failure == "mutation":
            vv[0, 0, 0] += 1
        return {"raw": result, "rounded": probe.round_finite_output(result)}, {
            "label": label
        }

    monkeypatch.setattr(probe, "runtime_compile_and_run", execute)
    progress = {}
    if failure == "mutation":
        with pytest.raises(ValueError, match="operands changed"):
            probe.run_default_row_chunks(*row_operands, tmp_path, progress)
        assert len(progress["row_slices"]) == 1
        assert progress["row_slices"][0]["chunks"]
        return
    probe.run_default_row_chunks(*row_operands, tmp_path, progress)
    assert [(start, stop) for start, stop, _ in calls] == list(probe.DEFAULT_ROW_SLICES)
    assert len({cache for _, _, cache in calls}) == 1
    assert not progress["native_subshape_replayed"]
    assert not progress["production_dispatcher_widened"]
    for key in ("raw_fp32", "bf16", "device_bf16_rounding_bridge"):
        assert progress[key]["count"] == target.size
        assert progress[key]["bitwise_equal"] == (failure is None)
        assert progress[key]["nonfinite"] == int(failure == "nonfinite")
    for entry in progress["row_slices"]:
        assert entry["slice_binding_unchanged"]
        for chunk in entry["chunks"]:
            path = tmp_path / entry["artifact_directory"] / chunk["file"]
            assert chunk["sha256"] == probe.sha(path)


def test_same_shape_executable_cache_uses_each_fresh_row_operand(monkeypatch, tmp_path):
    from foldjax.models.boltz2.models.primitives import native_pwa_mma as native

    compilations, executions = [], []

    class Executable:
        def as_text(self):
            return "compiled HLO"

        def __call__(self, w, v, initial):
            executions.append(np.asarray(v, np.float32).copy())
            raw = v.astype(jnp.float32) + initial[0]
            return raw, raw.astype(jnp.bfloat16), jnp.arange(32, dtype=jnp.int32)

    class Lowered:
        def lower(self, *operands):
            compilations.append(tuple(x.shape for x in operands))
            return self

        def compile(self):
            print("The Triton module for pallas_call")
            print(f"tt.elementwise_inline_asm {native._ASM}")
            return Executable()

    monkeypatch.setattr(jax, "jit", lambda *a, **k: Lowered())
    weights = probe.bf16_words(np.ones((437, 437), np.float32))
    cache = {}
    outputs = []
    for index in (1, 2):
        value = np.full((2, 437, 32), index, np.float32)
        output, evidence = probe.runtime_compile_and_run(
            weights,
            probe.bf16_words(value),
            np.zeros(1, np.float32),
            tmp_path,
            f"full.{index}",
            cache=cache,
        )
        np.testing.assert_array_equal(output["raw"], value)
        outputs.append(output)
    assert len(compilations) == 1 and len(executions) == 2
    assert not np.array_equal(executions[0], executions[1])
    assert not np.array_equal(outputs[0]["raw"], outputs[1]["raw"])
    assert evidence["compilation_reused_from"] == "full.1"
    assert evidence["artifacts"]


def test_row_cli_requires_runtime_arm_before_loading_inputs(monkeypatch, tmp_path):
    import sys

    setup_cli(monkeypatch, tmp_path / "missing", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--default-row-chunks"])
    with pytest.raises(SystemExit, match="2"):
        probe.main()


def test_row_cli_runs_basis_then_slices_and_summarizes_full_target(
    monkeypatch, tmp_path, reference
):
    import sys

    out = tmp_path / "row-probe"
    setup_cli(monkeypatch, reference, out)
    monkeypatch.setattr(
        sys, "argv", [*sys.argv, "--implementation", "runtime", "--default-row-chunks"]
    )
    monkeypatch.setattr(probe, "DEFAULT_ROW_SLICES", ((0, 1), (1, 2)))
    calls = []

    def runtime(w, v, initial, directory, label, *, cache=None):
        calls.append(label)
        raw = (
            probe.basis_inputs()[-1]
            if label == "basis"
            else np.full(v.shape, 437, np.float32)
        )
        return {"raw": raw, "rounded": probe.bf16_round(raw)}, {"label": label}

    monkeypatch.setattr(probe, "runtime_compile_and_run", runtime)
    probe.main()
    report = json.loads((out / "report.json").read_text())
    assert (
        report["row_chunk_control"]
        and report["passed"]
        and report["bindings_unchanged"]
    )
    assert calls == ["basis", "full.rows-0000-0001", "full.rows-0001-0002"]
    assert report["grid"] == [[2, 55], [2, 55]]
    assert report["full"]["raw_fp32"]["count"] == 2 * 437 * 32
    assert len(report["full"]["row_slices"]) == 2
    assert not report["full"]["production_dispatcher_widened"]
