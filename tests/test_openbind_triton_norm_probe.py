import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench import openbind_triton_norm_probe as probe


def test_panel_contains_precision_boundaries_and_native_row_tails():
    specs = probe.case_specs()
    assert len(specs) == 38
    assert len({name for name, _, _ in specs}) == len(specs)
    assert {shape[-1] for _, shape, _ in specs} == {64, 128}
    assert {shape[0] for _, shape, _ in specs if len(shape) == 2} == {7, 8, 9}
    assert (1, 437, 437, 128) in {shape for _, shape, _ in specs}


@pytest.mark.parametrize("profile", probe.PROFILES)
def test_synthetic_operands_are_deterministic_fp32(profile):
    shape = (9, 64)
    first = probe.operands(shape, profile)
    probe.validate_operands(first, shape)
    second = probe.operands(shape, profile)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    if profile in {"below_eps", "at_eps", "above_eps"}:
        ratio = {"below_eps": 0.25, "at_eps": 1.0, "above_eps": 4.0}[profile]
        np.testing.assert_allclose(
            np.var(first["x"], axis=-1), probe.EPS * ratio, rtol=1e-6
        )


def test_wrong_dtype_and_shape_are_not_silently_cast():
    arrays = probe.operands((7, 64), "normal")
    with pytest.raises(ValueError, match="shape/dtype"):
        probe.validate_operands(
            {**arrays, "x": arrays["x"].astype(np.float64)}, (7, 64)
        )
    with pytest.raises(ValueError, match="shape/dtype"):
        probe.validate_operands(arrays, (8, 64))


def test_metrics_keep_exact_strict_and_finite_separate():
    left = np.array([0.0, 1.0], np.float32)
    signed = np.array([-0.0, 1.0], np.float32)
    report = probe.metrics(left, signed)
    assert report["array_equal"]
    assert not report["bitwise_equal"]
    assert report["strict_1e4_allclose"]
    with pytest.raises(ValueError, match="nonfinite"):
        probe.metrics(left, np.array([np.nan, 1], np.float32))


def test_kernel_capture_invokes_actual_indexed_launcher():
    seen = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                seen.append((grid, args, kwargs))
                return SimpleNamespace(asm={"ptx": "actual compiled PTX"})

            return launch

    proxy = probe.KernelCapture(Kernel())
    proxy[(2,)]("input", ROWS_PER_PROGRAM=8)
    assert seen == [((2,), ("input",), {"ROWS_PER_PROGRAM": 8})]
    assert proxy.evidence["assembly"]["ptx"] == "actual compiled PTX"


def test_reference_requires_native_repeat_and_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "case_specs", lambda: [("one", (7, 64), "normal")])
    archive = tmp_path / "one.npz"
    operands = probe.operands((7, 64), "normal")
    output = np.zeros((7, 64), np.float32)
    sha = probe.save_arrays(archive, **operands, first=output, second=output)
    manifest = {
        "mode": "native",
        "completed": True,
        "source": {"commit": probe.UPSTREAM_COMMIT},
        "epsilon": probe.EPS,
        "cases": {
            "one": {
                "archive_sha256": sha,
                "native_repeat": probe.metrics(output, output),
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert probe.verify_reference(tmp_path)["completed"]
    manifest["cases"]["one"]["native_repeat"]["bitwise_equal"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="repeat differs"):
        probe.verify_reference(tmp_path)
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        probe.verify_reference(tmp_path)


def test_output_archives_are_not_overwritten(tmp_path):
    path = tmp_path / "arrays.npz"
    probe.save_arrays(path, x=np.zeros((1,), np.float32))
    with pytest.raises(FileExistsError):
        probe.save_arrays(path, x=np.ones((1,), np.float32))
