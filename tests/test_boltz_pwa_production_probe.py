import json
from pathlib import Path

import numpy as np
import pytest

from bench import boltz_pwa_production_probe as probe


@pytest.mark.parametrize("count", [7, 8, 9])
def test_native_logits_intervention_is_exactly_eight_calls_and_restores(
    monkeypatch, count
):
    from foldjax.models.boltz2.models.trunk_blocks import msa

    original = msa._linear
    logits = np.broadcast_to(np.arange(8, dtype=np.float32), (1, 437, 437, 8))
    value = np.broadcast_to(np.float32(0), (1, 437, 437, 128))
    kernel = np.zeros((128, 1), dtype=np.float32)

    def forward(params, values):
        return [msa._linear(value, kernel)[0, 0, 0, 0] for _ in range(count)]

    monkeypatch.setattr(probe, "production_forward", forward)
    if count == 8:
        result = probe.native_logits_forward({}, {"native_logits": logits})
        assert result == list(range(8))
    else:
        with pytest.raises(ValueError, match="projection"):
            probe.native_logits_forward({}, {"native_logits": logits})
    assert msa._linear is original


def save_json(path, value):
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("count", [7, 8, 9])
def test_native_softmax_intervention_restores_and_counts(monkeypatch, count):
    import jax

    original = jax.nn.softmax
    weights = np.broadcast_to(
        np.arange(8, dtype=np.float32)[None, :, None, None], (1, 8, 437, 437)
    )
    value = np.broadcast_to(np.float32(0), (1, 1, 437, 437))

    def forward(params, values):
        return [jax.nn.softmax(value)[0, 0, 0, 0] for _ in range(count)]

    monkeypatch.setattr(probe, "production_forward", forward)
    if count == 8:
        result = probe.native_softmax_forward({}, {"native_softmax": weights})
        assert result == list(range(8))
    else:
        with pytest.raises(ValueError):
            probe.native_softmax_forward({}, {"native_softmax": weights})
    assert jax.nn.softmax is original


@pytest.mark.parametrize("count", [7, 8, 9])
@pytest.mark.parametrize("native_logits", [False, True])
def test_warp_softmax_restores_and_counts(monkeypatch, count, native_logits):
    import jax

    from bench import boltz_pwa_softmax_probe

    original = jax.nn.softmax
    value = np.broadcast_to(np.float32(0), (1, 1, 437, 437))
    calls = []

    def computed(x):
        assert x is value
        calls.append(x)
        return 17

    monkeypatch.setattr(boltz_pwa_softmax_probe, "warp_softmax", computed)
    route = "native_logits_forward" if native_logits else "production_forward"
    monkeypatch.setattr(
        probe, route,
        lambda p, v: [jax.nn.softmax(value) for _ in range(count)],
    )
    values = {"native_logits": object()} if native_logits else {}
    if count == 8:
        assert probe.warp_softmax_forward({}, values) == [17] * 8
        assert len(calls) == 8
    else:
        with pytest.raises(ValueError):
            probe.warp_softmax_forward({}, values)
    assert jax.nn.softmax is original


def save_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)


@pytest.mark.parametrize("count", [7, 8, 9])
def test_computed_logits_counts_and_restores(monkeypatch, count):
    from bench import boltz_pwa_logits_probe
    from foldjax.models.boltz2.models.trunk_blocks import msa

    original = msa._linear
    x = np.broadcast_to(np.float32(0), (1, 437, 437, 128))
    w = np.zeros((128, 1), np.float32)
    monkeypatch.setattr(boltz_pwa_logits_probe, "strided_four_forward", lambda a, b: 9)
    monkeypatch.setattr(
        probe, "warp_softmax_forward",
        lambda p, v: [msa._linear(x, w) for _ in range(count)],
    )
    if count == 8:
        assert probe.computed_logits_warp_forward({}, {}) == [9] * 8
    else:
        with pytest.raises(ValueError):
            probe.computed_logits_warp_forward({}, {})
    assert msa._linear is original


@pytest.fixture
def reference(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROWS", 3)
    monkeypatch.setattr(probe, "TOKENS", 2)
    root, msa = tmp_path / "pwa", tmp_path / "msa"
    root.mkdir()
    msa.mkdir()
    m = np.ones((1, 3, 2, 64), np.float32)
    z = np.ones((1, 2, 2, 128), np.float32)
    mask = np.array([[1, 0]], np.float32)
    weights = {
        name: np.ones(shape, np.float32)
        for name, shape in probe.weight_shapes().items()
    }
    save_npz(root / "inputs.npz", m=m, z=z, mask=mask)
    save_npz(root / "weights.npz", **weights)
    save_npz(root / "stages.npz", actual=m, unused_decomposition=np.ones(7))
    save_npz(msa / "operands.npz", input_z=z, token_pad_mask=mask, unused=np.ones(5))
    prefix = "msa_module.layers.0.pair_weighted_averaging."
    save_npz(msa / "native-weights.npz", **{prefix + k: v for k, v in weights.items()})
    report = {
        "arm": "native",
        "passed": True,
        "native_source": {"native.py": "abc"},
        "runtime": {"torch": "pinned"},
        "stages": {},
        "artifacts": {
            name: probe.sha(msa / name)
            for name in ("operands.npz", "native-weights.npz")
        },
    }
    for stage in ("input_m", "pwa"):
        name = f"layers/00/{stage}"
        save_npz(msa / f"{name}.npz", **{"": m})
        save_json(
            msa / f"{name}.tree.json",
            {
                "": {
                    "shape": list(m.shape),
                    "native_dtype": "torch.bfloat16",
                    "storage_dtype": "float32",
                }
            },
        )
        report["stages"][name] = {
            "arrays_sha256": probe.sha(msa / f"{name}.npz"),
            "tree_sha256": probe.sha(msa / f"{name}.tree.json"),
        }
    save_json(msa / "report.json", report)
    report = {
        "arm": "native",
        "passed": True,
        "rows": 3,
        "native_source": report["native_source"],
        "torch": "pinned",
        "native_msa_report_sha256": probe.sha(msa / "report.json"),
        "native_decomposition": {"values_equal": True},
        "row_slice_vs_full": {"values_equal": True},
        "artifacts": {
            name: probe.sha(root / name)
            for name in ("inputs.npz", "weights.npz", "stages.npz")
        },
    }
    save_json(root / "report.json", report)
    return root, msa


def test_reference_binds_exact_native_arrays_and_selects_only_actual(
    reference, monkeypatch
):
    selected = []
    original = probe.select_arrays

    def select(path, names, **kwargs):
        selected.append((path.name, tuple(names)))
        return original(path, names, **kwargs)

    monkeypatch.setattr(probe, "select_arrays", select)
    inputs, weights, target, files, binding = probe.load_reference(*reference)
    assert inputs["m"].dtype == np.float32
    assert target.shape == (1, 3, 2, 64)
    assert set(weights) == set(probe.weight_shapes())
    assert [names for file, names in selected if file == "stages.npz"] == [("actual",)]
    assert binding == {name: probe.sha(path) for name, path in files.items()}


@pytest.mark.parametrize(
    "key,value",
    [
        ("arm", "foldjax"),
        ("passed", False),
        ("rows", 2),
        ("native_source", {}),
        ("torch", "other"),
        ("native_decomposition", {"values_equal": False}),
        ("row_slice_vs_full", {"values_equal": False}),
    ],
)
def test_unproven_native_reference_rejected(reference, key, value):
    path = reference[0] / "report.json"
    report = json.loads(path.read_text())
    report[key] = value
    save_json(path, report)
    with pytest.raises(ValueError, match="reproduction"):
        probe.load_reference(*reference)


@pytest.mark.parametrize("filename", ["inputs.npz", "weights.npz", "stages.npz"])
def test_changed_native_artifact_rejected(reference, filename):
    (reference[0] / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        probe.load_reference(*reference)


def test_true_fp32_native_carry_cannot_be_relabeled_as_bf16(reference):
    root, msa = reference
    path = msa / "layers/00/input_m.tree.json"
    metadata = json.loads(path.read_text())
    metadata[""]["native_dtype"] = "torch.float32"
    save_json(path, metadata)
    report = json.loads((msa / "report.json").read_text())
    report["stages"]["layers/00/input_m"]["tree_sha256"] = probe.sha(path)
    save_json(msa / "report.json", report)
    report = json.loads((root / "report.json").read_text())
    report["native_msa_report_sha256"] = probe.sha(msa / "report.json")
    save_json(root / "report.json", report)
    with pytest.raises(ValueError, match="not true FP32"):
        probe.load_reference(root, msa)


@pytest.mark.parametrize("value", [np.nan, np.inf, np.float32(1.0000001)])
def test_bf16_capture_rejects_nonfinite_or_unrepresentable_values(value):
    with pytest.raises(ValueError):
        probe.validate_fp32(np.array([value], np.float32), bf16=True)


def test_same_byte_gate_rejects_signed_zero_and_dtype_changes():
    with pytest.raises(ValueError, match="differ"):
        probe.require_same_bytes(
            np.array([0.0], np.float32), np.array([-0.0], np.float32)
        )
    with pytest.raises(ValueError, match="differ"):
        probe.require_same_bytes(np.ones(2, np.float32), np.ones(2, np.float64))


def test_forward_passes_exact_fp32_values_to_actual_default_pwa(monkeypatch):
    from foldjax.models.boltz2.models.trunk_blocks import msa

    values = {
        "m": np.ones((1, 3, 2, 64), np.float32),
        "z": np.ones((1, 2, 2, 128), np.float32),
        "mask": np.array([[1, 0]], np.float32),
    }
    seen = []
    monkeypatch.setattr(
        msa,
        "pair_weighted_averaging_forward",
        lambda *a, **k: seen.append((a, k)) or "result",
    )
    assert probe.production_forward({}, values) == "result"
    args, kwargs = seen.pop()
    assert args[1] is values["m"] and args[2] is values["z"]
    assert kwargs == {"row_chunk_size": None}
    np.testing.assert_array_equal(args[3], [[[1, 0], [0, 0]]])


@pytest.mark.parametrize(
    "failure", [None, "drift", "nan", "runtime", "input", "source", "reference"]
)
def test_main_retains_numeric_failures_and_rejects_binding_changes(
    reference, tmp_path, monkeypatch, failure
):
    root, msa = reference
    out = tmp_path / "result"
    source = Path(probe.__file__).resolve().parents[1]
    source_version = ["original"]
    monkeypatch.setattr(
        probe, "source_binding", lambda _: {"source.py": source_version[0]}
    )

    def run(
        source, inputs, weights, destination, *,
        warp_reduction=False, computed_logits=False,
    ):
        assert warp_reduction is False
        assert computed_logits is False
        if failure == "runtime":
            raise RuntimeError("compiler refused")
        result = inputs["m"].copy()
        if failure in ("drift", "nan"):
            result.flat[0] = 2 if failure == "drift" else np.nan
        elif failure == "input":
            inputs["m"].flat[0] = 2
        elif failure == "source":
            source_version[0] = "changed"
        elif failure == "reference":
            (root / "weights.npz").write_bytes(b"changed")
        hlo = destination / "compiled.hlo.txt"
        hlo.write_text("fixture, not GPU evidence")
        return result, {"compiled_hlo_sha256": probe.sha(hlo)}

    monkeypatch.setattr(probe, "run_production", run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--source-root",
            str(source),
            "--reference",
            str(root),
            "--msa-reference",
            str(msa),
            "--out",
            str(out),
        ],
    )
    if failure in ("runtime", "input", "source", "reference"):
        with pytest.raises((RuntimeError, ValueError)):
            probe.main()
    else:
        probe.main()
    report = json.loads((out / "report.json").read_text())
    assert report["passed"] is (failure is None)
    assert report["capture_complete"] is (failure not in ("runtime", "input"))
    assert report["bindings_unchanged"] is (failure not in ("source", "reference"))
    assert report["input_m_policy"]["native_true_fp32_carry"] is False
    assert report["input_m_policy"]["candidate_embedding_recomputed"] is False
    if failure == "nan":
        assert report["comparison"]["nonfinite"] == 1
        assert report["comparison"]["rmse"] is None
        with np.load(out / report["output_chunks"][0]["file"]) as archive:
            assert np.isnan(archive["actual"].flat[0])


def test_chunk_comparison_preserves_every_row_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "CHUNK_ROWS", 2)
    actual = np.ones((1, 5, 2, 3), np.float32)
    comparison, chunks = probe.save_comparison(actual, actual.copy(), tmp_path)
    assert [(x["start"], x["stop"]) for x in chunks] == [(0, 2), (2, 4), (4, 5)]
    assert comparison["count"] == actual.size and comparison["bitwise_equal"]
    assert all(probe.sha(tmp_path / x["file"]) == x["sha256"] for x in chunks)
    with pytest.raises(FileExistsError):
        probe.save_comparison(actual, actual, tmp_path)
