import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench import boltz_pwa_mma_probe as probe


@pytest.mark.parametrize("count", [1, 4])
def test_probe_inline_asm_unwrap_preserves_ir_values(count):
    values = [object() for _ in range(count)]
    ctx = SimpleNamespace(avals_out=[None] * count)
    nested = values if count == 1 else [tuple(values)]
    rule = probe._flatten_inline_asm_results(lambda *a, **k: nested)
    assert rule(ctx) == values


def test_probe_inline_asm_unwrap_rejects_wrong_arity():
    rule = probe._flatten_inline_asm_results(lambda *a, **k: [(1, 2)])
    with pytest.raises(ValueError, match="arity"):
        rule(SimpleNamespace(avals_out=[None] * 4))


@pytest.mark.parametrize("k_width", [8, 16])
def test_lane_fragments_cover_each_matrix_element_exactly_once(k_width):
    for coordinates, shape in zip(
        probe.fragment_coordinates(k_width),
        ((16, k_width), (k_width, 8), (16, 8)),
        strict=True,
    ):
        counts = np.zeros(shape, np.int32)
        for rows, columns in coordinates:
            np.add.at(counts, (rows, columns), 1)
        np.testing.assert_array_equal(counts, np.ones(shape, np.int32))


def decode_register_stream(packed, k_width):
    """Independent literal PTX lane formula, not the production coordinate helper."""
    tiles, steps = packed.shape[:2]
    a = np.empty((tiles, 16, steps * k_width), np.float32)
    b = np.empty((tiles, steps * k_width, 8), np.float32)
    for lane in range(32):
        group, thread = divmod(lane, 4)
        for step in range(steps):
            for register in range(k_width // 4):
                word = packed[:, step, register, lane]
                for half in range(2):
                    element = 2 * register + half
                    row = group + (8 if element % 4 >= 2 else 0)
                    k = step * k_width + 2 * thread + half
                    if element >= 4:
                        k += 8
                    bits = ((word >> (16 * half)) & 0xFFFF) << 16
                    a[:, row, k] = bits.view(np.float32)
            for register in range(k_width // 8):
                word = packed[:, step, k_width // 4 + register, lane]
                for half in range(2):
                    k = step * k_width + 2 * thread + half + 8 * register
                    bits = ((word >> (16 * half)) & 0xFFFF) << 16
                    b[:, k, group] = bits.view(np.float32)
    return a, b


@pytest.mark.parametrize("k_width", [8, 16])
def test_packing_preserves_all_bf16_bits_and_full_437_tail(k_width):
    rng = np.random.default_rng(718)
    a = probe.bf16_round(rng.normal(size=(2, 16, 437)).astype(np.float32))
    b = probe.bf16_round(rng.normal(size=(2, 437, 8)).astype(np.float32))
    a[0, 0, 0] = -0.0
    packed = probe.pack_operands(a, b, k_width)
    actual_a, actual_b = decode_register_stream(packed, k_width)
    np.testing.assert_array_equal(
        actual_a[:, :, :437].view(np.uint32), a.view(np.uint32)
    )
    np.testing.assert_array_equal(actual_b[:, :437].view(np.uint32), b.view(np.uint32))
    assert not actual_a[:, :, 437:].view(np.uint32).any()
    assert not actual_b[:, 437:].view(np.uint32).any()


@pytest.mark.parametrize("k_width", [8, 16])
def test_basis_oracle_checks_full_k_lane_layout_tail_and_nonzero_c(k_width):
    a, b, initial, expected = probe.basis_operands()
    decoded_a, decoded_b = decode_register_stream(
        probe.pack_operands(a, b, k_width), k_width
    )
    result = initial.copy()
    # Integer-valued controls are exact: this checks mapping, not hardware rounding.
    for k in range(0, decoded_a.shape[2], k_width):
        result += decoded_a[:, :, k : k + k_width] @ decoded_b[:, k : k + k_width]
    np.testing.assert_array_equal(result, expected)
    assert np.any(initial)
    assert np.count_nonzero(a[2]) == 16
    assert np.count_nonzero(b[2]) == 8


def test_residue_controls_are_additive_and_exactly_the_two_approved_arms():
    assert probe.probe_arms() == [("k8", 8, None), ("k16", 16, None)]
    assert probe.probe_arms(True) == [
        ("k8", 8, None),
        ("k16", 16, None),
        ("k8_zero_tail_448", 8, "zero_tail_448"),
        ("k8_residue_first_32", 8, "residue_first_32"),
    ]


@pytest.mark.parametrize("layout", ["zero_tail_448", "residue_first_32"])
def test_k8_controls_preserve_each_original_term_once_and_all_operand_bits(layout):
    rng = np.random.default_rng(914)
    a = probe.bf16_round(rng.normal(size=(2, 16, 437)).astype(np.float32))
    b = probe.bf16_round(rng.normal(size=(2, 437, 8)).astype(np.float32))
    a[0, 0, 21] = -0.0
    original_a, original_b = a.copy(), b.copy()
    aa, bb, positions = probe.k8_control_operands(a, b, layout)
    real = positions >= 0
    np.testing.assert_array_equal(positions[real], np.arange(437))
    np.testing.assert_array_equal(aa[:, :, real].view(np.uint32), a.view(np.uint32))
    np.testing.assert_array_equal(bb[:, real].view(np.uint32), b.view(np.uint32))
    np.testing.assert_array_equal(a.view(np.uint32), original_a.view(np.uint32))
    np.testing.assert_array_equal(b.view(np.uint32), original_b.view(np.uint32))
    zeros = np.arange(437, 448) if layout == "zero_tail_448" else np.arange(21, 32)
    np.testing.assert_array_equal(np.flatnonzero(~real), zeros)
    assert not aa[:, :, ~real].view(np.uint32).any()
    assert not bb[:, ~real].view(np.uint32).any()
    packed = probe.pack_operands(aa, bb, 8)
    assert packed.shape == (2, 56, 3, 32)
    decoded_a, decoded_b = decode_register_stream(packed, 8)
    np.testing.assert_array_equal(decoded_a.view(np.uint32), aa.view(np.uint32))
    np.testing.assert_array_equal(decoded_b.view(np.uint32), bb.view(np.uint32))
    if layout == "zero_tail_448":
        np.testing.assert_array_equal(packed[:, :55], probe.pack_operands(a, b, 8))
        assert not packed[:, 55].any()


@pytest.mark.parametrize("layout", ["zero_tail_448", "residue_first_32"])
def test_each_control_retains_basis_tail_and_carried_c_oracle(layout):
    a, b, initial, expected = probe.basis_operands()
    aa, bb, _ = probe.k8_control_operands(a, b, layout)
    aa, bb = decode_register_stream(probe.pack_operands(aa, bb, 8), 8)
    result = initial.copy()
    for start in range(0, 448, 8):
        result += aa[:, :, start : start + 8] @ bb[:, start : start + 8]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("problem", ["other_layout", "k438", "fp64"])
def test_controls_reject_any_unapproved_layout_or_input(problem):
    a, b, _, _ = probe.basis_operands(k=438 if problem == "k438" else 437)
    layout = "residue_first_64" if problem == "other_layout" else "zero_tail_448"
    if problem == "fp64":
        a = a.astype(np.float64)
    with pytest.raises(ValueError):
        probe.k8_control_operands(a, b, layout)


def test_accumulator_round_trip_and_literal_ptx_coordinates():
    value = np.arange(2 * 16 * 8, dtype=np.float32).reshape(2, 16, 8)
    packed = probe.pack_accumulator(value)
    for lane in range(32):
        group, thread = divmod(lane, 4)
        for register in range(4):
            np.testing.assert_array_equal(
                packed[:, register, lane],
                value[:, group + 8 * (register // 2), 2 * thread + register % 2],
            )
    np.testing.assert_array_equal(probe.unpack_accumulator(packed), value)


@pytest.mark.parametrize("k_width", [8, 16])
def test_jaxpr_passes_each_mma_output_directly_as_next_c_without_float_add(k_width):
    a, b, initial, _ = probe.basis_operands()
    packed = probe.pack_operands(a, b, k_width)
    graph = jax.make_jaxpr(lambda p, c: probe.mma_call(p, c, k_width=k_width))(
        jax.ShapeDtypeStruct(packed.shape, jnp.uint32),
        jax.ShapeDtypeStruct(probe.pack_accumulator(initial).shape, jnp.float32),
    )
    call = graph.jaxpr.eqns[0]
    assert call.primitive.name == "pallas_call"
    assert call.params["compiler_params"].num_warps == 1
    assert not call.params["interpret"]
    kernel = call.params["jaxpr"]
    loop = next(eq for eq in kernel.eqns if eq.primitive.name == "scan")
    assert loop.params["length"] == (437 + k_width - 1) // k_width
    body = loop.params["jaxpr"].jaxpr
    mma = next(
        eq for eq in body.eqns if eq.primitive.name == "elementwise_inline_asm_p"
    )
    asm, constraints = probe.mma_assembly(k_width)
    assert mma.params["asm"] == asm
    assert mma.params["constraints"] == constraints
    assert f"m16n8k{k_width}.row.col.f32.bf16.bf16.f32" in asm
    assert constraints.split(",") == ["=f"] * 4 + ["r"] * (3 * k_width // 8) + ["f"] * 4
    assert mma.invars[-4:] == body.invars[-4:]
    assert body.outvars[-4:] == mma.outvars
    for eq in body.eqns:
        assert "dot" not in eq.primitive.name
        if eq.primitive.name == "add":
            assert all(var.aval.dtype == jnp.int32 for var in eq.invars)
    assert any(eq.params.get("asm") == "mov.u32 $0, %laneid;" for eq in kernel.eqns)


@pytest.mark.parametrize("problem", ["width", "shape", "fp64", "non_bf16", "nan"])
def test_operand_packing_rejects_unreviewed_inputs(problem):
    a, b, _, _ = probe.basis_operands()
    width = 8
    if problem == "width":
        width = 32
    elif problem == "shape":
        b = b[:, :-1]
    elif problem == "fp64":
        a = a.astype(np.float64)
    elif problem == "non_bf16":
        a[0, 0, 0] = 1.00001
    else:
        b[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        probe.pack_operands(a, b, width)


def test_deterministic_mismatch_selection_and_native_tile_orientation():
    n = 437
    weights = (np.arange(n * n).reshape(1, 1, n, n) % 7).astype(np.float32)
    values = (np.arange(2 * n * 32).reshape(1, 2, n, 32) % 5).astype(np.float32)
    raw = np.einsum("ij,sjd->sid", weights[0, 0], values[0])[None]
    previous = probe.bf16_round(raw)
    for row, token, channel in ((0, 0, 0), (0, 8, 0), (0, 16, 0), (1, 436, 16)):
        previous[0, row, token, channel] += 16
    selected = probe.select_tiles(raw, previous)
    assert selected == [(0, 0), (0, 8), (0, 16), (48, 432)]
    a, b, target, prior, valid = probe.extract_tiles(
        weights, values, raw, previous, selected
    )
    assert a.shape == (4, 16, 437)
    assert b.shape == (4, 437, 8)
    np.testing.assert_array_equal(a @ b, target)
    for tile, (m0, n0) in enumerate(selected):
        for m in range(16):
            row, channel = divmod(m0 + m, 32)
            for column in range(8):
                token = n0 + column
                assert valid[tile, m, column] == (token < n)
                if token < n:
                    assert target[tile, m, column] == raw[0, row, token, channel]
                    assert prior[tile, m, column] == previous[0, row, token, channel]
                else:
                    assert target[tile, m, column] == prior[tile, m, column] == 0
    assert not valid[-1, :, 5:].any()


@pytest.mark.parametrize("count", [0, 1, 4])
def test_selection_rejects_missing_mismatch_tiles(count):
    value = np.zeros((1, 1, 437, 32), np.float32)
    with pytest.raises(ValueError):
        probe.select_tiles(value, value, count=count)


@pytest.fixture
def references(tmp_path):
    native, candidate = tmp_path / "native", tmp_path / "jax"
    native.mkdir()
    candidate.mkdir()
    weights = np.ones((1, 1, 437, 437), np.float32)
    values = np.ones((1, 1, 437, 32), np.float32)
    raw = np.zeros_like(values)
    output = raw.copy()
    output[0, 0, [0, 8, 16, 436], [0, 0, 0, 16]] = 1
    inputs_hash = probe.save_array(
        native / "inputs.npz", weights=weights, values=values
    )
    raw_hash = probe.save_array(
        native / "raw_fp32_output.npz",
        raw_output=raw,
        # Reading this unrelated member with allow_pickle=False must fail.
        rounded_output=np.array([object()], dtype=object),
    )
    output_hash = probe.save_array(candidate / "einsum.npz", output=output)
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
    probe.save_new(native / "raw_fp32_control.private.json", metadata)
    report = {
        "arm": "native",
        "passed": True,
        "head": 2,
        "source": source,
        "inputs_sha256": inputs_hash,
        "input_shapes": {"weights": list(weights.shape), "values": list(values.shape)},
        "raw_fp32_control": {
            "bridge_gate_passed": True,
            "capture_complete": True,
            "private_metadata_sha256": probe.sha(
                native / "raw_fp32_control.private.json"
            ),
            "output_sha256": raw_hash,
        },
    }
    probe.save_new(native / "report.json", report)
    probe.save_new(
        candidate / "report.json",
        {
            "arm": "foldjax",
            "head": 2,
            "capture_complete": True,
            "inputs_sha256": inputs_hash,
            "arms": {"einsum": {"output_sha256": output_hash}},
        },
    )
    return native, candidate


def test_reference_loader_binds_raw_bridge_and_reads_only_selected_npz_members(
    references,
):
    native, candidate = references
    arrays, binding = probe.load_tiles(native, candidate)
    assert arrays[0].shape == (4, 16, 437)
    assert arrays[1].shape == (4, 437, 8)
    assert binding["full_k"] == 437
    assert binding["not_new_native_submatrix_reference"]
    assert binding["reference_sha256"] == probe.sha(native / "report.json")
    assert binding["tiles_native_transposed_gemm_mn"] == [
        (0, 0),
        (0, 8),
        (0, 16),
        (16, 432),
    ]


@pytest.mark.parametrize("problem", ["bridge", "head", "source", "hash", "jax_input"])
def test_reference_loader_rejects_unbound_or_unproven_capture(references, problem):
    native, candidate = references
    path = (candidate if problem == "jax_input" else native) / "report.json"
    report = json.loads(path.read_text())
    if problem == "bridge":
        report["raw_fp32_control"]["bridge_gate_passed"] = False
    elif problem == "head":
        report["head"] = 3
    elif problem == "source":
        report["source"] = {"different": "source"}
    elif problem == "hash":
        report["raw_fp32_control"]["private_metadata_sha256"] = "wrong"
    else:
        report["inputs_sha256"] = "different"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        probe.load_tiles(native, candidate)


def test_reference_loader_rejects_even_hash_bound_unknown_npz_schema(references):
    native, candidate = references
    path = candidate / "einsum.npz"
    with np.load(path) as archive:
        output = archive["output"]
    np.savez(path, output=output, unknown=output)
    path = candidate / "report.json"
    report = json.loads(path.read_text())
    report["arms"]["einsum"]["output_sha256"] = probe.sha(candidate / "einsum.npz")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="unreviewed capture schema"):
        probe.load_tiles(native, candidate)


def test_cli_records_finite_unsupported_and_failed_arms_without_fallback(
    monkeypatch, tmp_path, references
):
    native, candidate = references
    out = tmp_path / "probe-output"
    root = Path(probe.__file__).resolve().parents[1]
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--source-root",
            str(root),
            "--reference",
            str(native),
            "--jax-reference",
            str(candidate),
            "--out",
            str(out),
        ],
    )
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "devices", lambda: [object()])
    calls = []

    def unsupported(packed, initial, k_width, directory, label):
        calls.append(k_width)
        (directory / f"{label}.triton.mlir").write_text("actual failed lowering")
        if k_width == 8:
            raise NotImplementedError("unsupported opcode")
        raise ValueError("failed lane oracle")

    monkeypatch.setattr(probe, "run_compiled", unsupported)
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 1
    assert calls == [8, 16]
    report = json.loads((out / "report.json").read_text())
    assert report["capture_complete"] is False
    assert report["native_instruction_identity_proven"] is False
    assert report["arms"]["k8"]["status"] == "unsupported"
    assert report["arms"]["k16"]["status"] == "failed"
    for k_width in (8, 16):
        arm = report["arms"][f"k{k_width}"]
        assert arm["fallback_used"] is False
        assert arm["artifacts"] == {
            f"k{k_width}-basis.triton.mlir": probe.sha(
                out / f"k{k_width}-basis.triton.mlir"
            )
        }


def test_cli_additive_controls_share_original_targets_and_gate_every_basis(
    monkeypatch, tmp_path, references
):
    native, candidate = references
    out = tmp_path / "four-arms"
    root = Path(probe.__file__).resolve().parents[1]
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--source-root",
            str(root),
            "--reference",
            str(native),
            "--jax-reference",
            str(candidate),
            "--out",
            str(out),
            "--k8-residue-controls",
        ],
    )
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "devices", lambda: [object()])
    calls = []

    def mapping_oracle(packed, initial, k_width, directory, label):
        calls.append((label, packed.shape[1]))
        a, b = decode_register_stream(packed, k_width)
        return initial + a @ b, {"cpu_mapping_oracle_only": True}

    monkeypatch.setattr(probe, "run_compiled", mapping_oracle)
    probe.main()
    report = json.loads((out / "report.json").read_text())
    assert report["capture_complete"]
    assert report["k8_residue_controls"]
    assert report["native_private_cta_identity_proven"] is False
    expected_calls = []
    targets = []
    for name, width, layout in probe.probe_arms(True):
        steps = 56 if layout else (437 + width - 1) // width
        expected_calls += [(f"{name}-basis", steps), (f"{name}-native-tiles", steps)]
        arm = report["arms"][name]
        assert arm["basis"]["comparison"]["bitwise_equal"]
        with np.load(out / f"{name}-output.npz") as archive:
            targets.append(archive["target"])
        if layout:
            proof = arm["k_operand_layout"]
            assert proof["original_terms_once_in_order"]
            assert proof["operands_preserved_bitwise"]
            assert proof["same_basis_and_actual_layout"]
            assert proof["native_private_cta_identity_proven"] is False
            assert proof["positions_sha256"] == probe.sha(
                out / f"{name}-k-positions.npz"
            )
        else:
            assert "k_operand_layout" not in arm
    assert calls == expected_calls
    for target in targets[1:]:
        np.testing.assert_array_equal(target, targets[0])


def test_source_binding_rejects_another_snapshot(tmp_path):
    with pytest.raises(ValueError, match="another source snapshot"):
        probe.source_binding(tmp_path)
    root = Path(probe.__file__).resolve().parents[1]
    assert probe.source_binding(root)["bench/boltz_pwa_mma_probe.py"] == probe.sha(
        Path(probe.__file__)
    )


def test_cli_rejects_existing_output_before_device_or_reference_access(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--source-root",
            "missing",
            "--reference",
            "missing",
            "--jax-reference",
            "missing",
            "--out",
            str(tmp_path),
        ],
    )
    with pytest.raises(FileExistsError):
        probe.main()
