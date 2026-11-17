"""CPU unit tests for ``bench/cueq_upgrade_check.py``."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "bench" / "cueq_upgrade_check.py"


def _load_check_module(name: str = "cueq_upgrade_check_under_test"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_imports_on_cpu_without_cuequivariance_installed(monkeypatch) -> None:
    for mod_name in (
        "cuequivariance",
        "cuequivariance_jax",
        "cuequivariance_ops_jax",
    ):
        monkeypatch.setitem(sys.modules, mod_name, None)

    mod = _load_check_module("cueq_upgrade_check_no_cueq")
    specs = mod.default_case_specs()
    families = {spec.family for spec in specs}
    assert families == {
        "raw_triangle_attention",
        "cueq_attention_core",
        "fused_triangle_multiplication",
        "boltz2_amp_triangle_multiplication",
        "layer_norm_transpose",
    }
    raw_att = [s for s in specs if s.family == "raw_triangle_attention"]
    assert {s.length for s in raw_att} == {256, 1003, 1024, 2100, 3012}
    assert {s.dtype for s in raw_att} == {"bf16", "fp32"}
    assert {s.variant for s in raw_att} == {"all_valid", "last5pct_invalid"}

    core_att = [s for s in specs if s.family == "cueq_attention_core"]
    assert len(core_att) == 10
    assert {s.dtype for s in core_att} == {"bf16"}

    fused_mul = [s for s in specs if s.family == "fused_triangle_multiplication"]
    assert {s.length for s in fused_mul} == {256, 1024, 2100}
    assert {s.variant for s in fused_mul} == {"outgoing", "incoming"}

    ln_cases = [s for s in specs if s.family == "layer_norm_transpose"]
    assert {s.variant for s in ln_cases} == {"bijd->bijd", "dbij->bijd"}
    assert {s.length for s in ln_cases} == {256, 1024, 2100}


def test_compare_cli_on_two_synthetic_json_files(tmp_path: Path, capsys) -> None:
    mod = _load_check_module()
    sample_a1 = np.array([1.0, -2.0, 0.0, 4.0], dtype=np.float32)
    sample_b1 = np.array([1.0, -2.0, 0.0, 4.0], dtype=np.float32)
    sample_a2 = np.array([2.0, -4.0, 0.0, 8.0], dtype=np.float32)
    sample_b2 = np.array([2.5, -4.0, 0.0, 8.0], dtype=np.float32)

    sha_s_a1 = mod.write_sample_npy(tmp_path / "a.case_same.npy", sample_a1)
    sha_s_b1 = mod.write_sample_npy(tmp_path / "b.case_same.npy", sample_b1)
    sha_s_a2 = mod.write_sample_npy(tmp_path / "a.case_diff.npy", sample_a2)
    sha_s_b2 = mod.write_sample_npy(tmp_path / "b.case_diff.npy", sample_b2)
    mod.write_sample_npy(tmp_path / "a.case_zero_time.npy", sample_a1)
    mod.write_sample_npy(tmp_path / "b.case_zero_time.npy", sample_b1)

    report_a = {
        "versions": {
            "jax": "0.11.1",
            "jaxlib": "0.11.1",
            "cuequivariance": "0.11.1",
            "cuequivariance_jax": "0.11.1",
            "cuequivariance_ops_jax": "0.11.1",
        },
        "gpu_name": "Synthetic GPU",
        "cases": {
            "case_same": {
                "status": "ok",
                "median_ms": 10.0,
                "shape": [1, 4],
                "dtype": "bfloat16",
                "sha256": "aa" * 32,
                "max_abs": 4.0,
                "sample_file": "a.case_same.npy",
                "sample_sha256": sha_s_a1,
            },
            "case_diff": {
                "status": "ok",
                "median_ms": 20.0,
                "shape": [1, 4],
                "dtype": "float32",
                "sha256": "bb" * 32,
                "max_abs": 8.0,
                "sample_file": "a.case_diff.npy",
                "sample_sha256": sha_s_a2,
            },
            "case_zero_time": {
                "status": "ok",
                "median_ms": 0.0,
                "shape": [1, 4],
                "dtype": "float32",
                "sha256": "dd" * 32,
                "max_abs": 4.0,
                "sample_file": "a.case_zero_time.npy",
            },
            "case_err": {
                "status": "error",
                "error_type": "RuntimeError",
                "error": "unsupported",
            },
            "case_only_a": {
                "status": "ok",
                "median_ms": 5.0,
                "shape": [1, 4],
                "dtype": "float32",
                "sha256": "ee" * 32,
                "sample_file": "a.case_same.npy",
            },
        },
    }
    report_b = {
        "versions": {
            "jax": "0.11.1",
            "jaxlib": "0.11.1",
            "cuequivariance": "0.13.0",
            "cuequivariance_jax": "0.13.0",
            "cuequivariance_ops_jax": "0.13.0",
        },
        "gpu_name": "Synthetic GPU",
        "cases": {
            "case_same": {
                "status": "ok",
                "median_ms": 8.0,
                "shape": [1, 4],
                "dtype": "bfloat16",
                "sha256": "aa" * 32,
                "max_abs": 4.0,
                "sample_file": "b.case_same.npy",
                "sample_sha256": sha_s_b1,
            },
            "case_diff": {
                "status": "ok",
                "median_ms": 15.0,
                "shape": [1, 4],
                "dtype": "float32",
                "sha256": "cc" * 32,
                "max_abs": 8.0,
                "sample_file": "b.case_diff.npy",
                "sample_sha256": sha_s_b2,
            },
            "case_zero_time": {
                "status": "ok",
                "median_ms": 1.0,
                "shape": [1, 4],
                "dtype": "float32",
                "sha256": "dd" * 32,
                "max_abs": 4.0,
                "sample_file": "b.case_zero_time.npy",
            },
            "case_err": {
                "status": "ok",
                "median_ms": 3.0,
                "shape": [1, 4],
                "dtype": "float32",
                "sha256": "ff" * 32,
                "sample_file": "b.case_same.npy",
            },
            "case_only_b": {
                "status": "error",
                "error_type": "ValueError",
                "error": "bad",
            },
        },
    }
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(json.dumps(report_a), encoding="utf-8")
    path_b.write_text(json.dumps(report_b), encoding="utf-8")

    rc = mod.main(["--compare", str(path_a), str(path_b)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cueq_jax=0.11.1" in out
    assert "cueq_jax=0.13.0" in out
    assert "case_same" in out and "0.800x" in out and "yes" in out
    assert "case_diff" in out and "0.750x" in out and "no" in out
    assert "5.000e-01" in out and "2.000e-01" in out
    assert "case_zero_time" in out and "inf" in out
    assert "case_err" in out and "ERROR" in out
    assert "case_only_a" in out and "MISSING" in out
    assert "case_only_b" in out


def test_extract_sample_and_summarize_array_edges() -> None:
    mod = _load_check_module()

    for bad_limit in (0, -1, False, True):
        with pytest.raises(ValueError, match="max_elements"):
            mod.extract_sample(np.ones((4,), dtype=np.float32), max_elements=bad_limit)

    empty, stride_empty = mod.extract_sample(
        np.empty((0, 4), dtype=np.float32), max_elements=10
    )
    assert empty.shape == (0,) and empty.dtype == np.float32 and stride_empty == 1

    exact, stride_exact = mod.extract_sample(
        np.arange(10, dtype=np.float32), max_elements=10
    )
    assert exact.shape == (10,) and stride_exact == 1
    np.testing.assert_array_equal(exact, np.arange(10, dtype=np.float32))

    over, stride_over = mod.extract_sample(
        np.arange(11, dtype=np.float32), max_elements=10
    )
    assert stride_over == 2 and over.shape == (6,)
    np.testing.assert_array_equal(over, np.array([0, 2, 4, 6, 8, 10], dtype=np.float32))

    at_default, stride_def = mod.extract_sample(
        np.zeros((1_000_000,), dtype=np.float32)
    )
    assert at_default.shape == (1_000_000,) and stride_def == 1

    above_default, stride_above = mod.extract_sample(
        np.zeros((1_000_001,), dtype=np.float32)
    )
    assert stride_above == 2 and above_default.shape == (500_001,)

    summary_empty, _ = mod.summarize_array(
        np.empty((0,), dtype=np.float32), dtype_name="float32"
    )
    assert summary_empty["max_abs"] == 0.0 and summary_empty["all_finite"] is True

    summary_nan, _ = mod.summarize_array(
        np.array([1.0, float("nan")], dtype=np.float32), dtype_name="float32"
    )
    assert summary_nan["max_abs"] is None and summary_nan["all_finite"] is False
    json.dumps(summary_nan, allow_nan=False)


def test_compute_sample_diffs_edges() -> None:
    mod = _load_check_module()

    with pytest.raises(ValueError, match="shape mismatch"):
        mod.compute_sample_diffs(
            np.zeros((2,), dtype=np.float32), np.zeros((3,), dtype=np.float32)
        )

    assert mod.compute_sample_diffs(
        np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    ) == (0.0, 0.0)

    assert mod.compute_sample_diffs(
        np.zeros((4,), dtype=np.float32), np.zeros((4,), dtype=np.float32)
    ) == (0.0, 0.0)

    assert mod.compute_sample_diffs(
        np.array([0.0], dtype=np.float32), np.array([2.0], dtype=np.float32)
    ) == (2.0, 1.0)

    abs_nan, rel_nan = mod.compute_sample_diffs(
        np.array([float("nan")], dtype=np.float32), np.array([1.0], dtype=np.float32)
    )
    assert math.isnan(abs_nan) and math.isnan(rel_nan)

    assert mod.compute_sample_diffs(
        np.array([float("inf")], dtype=np.float32),
        np.array([float("inf")], dtype=np.float32),
    ) == (0.0, 0.0)

    abs_inf, rel_inf = mod.compute_sample_diffs(
        np.array([float("inf")], dtype=np.float32), np.array([1.0], dtype=np.float32)
    )
    assert math.isinf(abs_inf) and math.isinf(rel_inf)


def test_run_config_and_filter_validation_and_cpu_refusal(
    tmp_path: Path, capsys
) -> None:
    mod = _load_check_module()

    mod.validate_run_config(warmup=0, repeats=1, max_sample_elements=1)
    for kwargs in (
        {"warmup": -1, "repeats": 1, "max_sample_elements": 1},
        {"warmup": True, "repeats": 1, "max_sample_elements": 1},
        {"warmup": 0, "repeats": 0, "max_sample_elements": 1},
        {"warmup": 0, "repeats": False, "max_sample_elements": 1},
        {"warmup": 0, "repeats": 1, "max_sample_elements": 0},
        {"warmup": 0, "repeats": 1, "max_sample_elements": True},
    ):
        with pytest.raises(ValueError):
            mod.validate_run_config(**kwargs)

    specs = mod.default_case_specs()
    assert len(mod.select_cases(specs, "")) == len(specs)
    assert len(mod.select_cases(specs, "a_raw_tri_att_n256")) == 4
    with pytest.raises(ValueError, match="no benchmark cases matched"):
        mod.select_cases(specs, "does_not_exist_xyz")

    with pytest.raises(SystemExit) as exc_info:
        mod.main([])
    assert exc_info.value.code == 2

    out_json = tmp_path / "should_not_exist.json"
    assert mod.main(["--out", str(out_json), "--warmup", "-1"]) == 1
    assert not out_json.exists()

    rc_cpu = mod.main(["--out", str(out_json)])
    assert rc_cpu == 1
    err = capsys.readouterr().err
    assert "GPU backend is required" in err
    assert not out_json.exists()


@pytest.mark.parametrize(
    "mutator,expected_msg",
    [
        ("missing_file", "does not exist"),
        ("bad_json", "Expecting"),
        ("list_root", "JSON object"),
        ("empty_cases", "non-empty 'cases'"),
        ("bad_status", "invalid status"),
        ("neg_time", "invalid median_ms"),
        ("bool_time", "invalid median_ms"),
        ("bad_shape", "invalid shape"),
        ("empty_dtype", "invalid dtype"),
        ("empty_sha", "invalid sha256"),
        ("parent_escape", "stay relative"),
        ("abs_sample", "stay relative"),
        ("missing_npy", "missing sample file"),
        ("tampered_sha", "SHA-256 mismatch"),
        ("corrupt_npy", "corrupt sample file"),
        ("non_1d_npy", "1D floating array"),
        ("shape_mismatch", "sample shape mismatch"),
    ],
)
def test_compare_rejects_malformed_inputs(
    tmp_path: Path, capsys, mutator: str, expected_msg: str
) -> None:
    mod = _load_check_module()
    sample_ok = np.array([1.0, 2.0], dtype=np.float32)
    sha_ok = mod.write_sample_npy(tmp_path / "ok.npy", sample_ok)

    def make_valid() -> dict:
        return {
            "cases": {
                "c1": {
                    "status": "ok",
                    "median_ms": 1.0,
                    "shape": [2],
                    "dtype": "float32",
                    "sha256": "11" * 32,
                    "sample_file": "ok.npy",
                    "sample_sha256": sha_ok,
                }
            }
        }

    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    doc_a = make_valid()
    doc_b = make_valid()

    if mutator == "missing_file":
        path_b.write_text(json.dumps(doc_b), encoding="utf-8")
        assert mod.main(["--compare", str(tmp_path / "absent.json"), str(path_b)]) == 1
        assert expected_msg in capsys.readouterr().err
        return
    if mutator == "bad_json":
        path_a.write_text("{broken", encoding="utf-8")
        path_b.write_text(json.dumps(doc_b), encoding="utf-8")
        assert mod.main(["--compare", str(path_a), str(path_b)]) == 1
        assert expected_msg in capsys.readouterr().err
        return
    if mutator == "list_root":
        doc_a = []  # type: ignore[assignment]
    elif mutator == "empty_cases":
        doc_a["cases"] = {}
    elif mutator == "bad_status":
        doc_a["cases"]["c1"]["status"] = "unknown"
    elif mutator == "neg_time":
        doc_a["cases"]["c1"]["median_ms"] = -0.1
    elif mutator == "bool_time":
        doc_a["cases"]["c1"]["median_ms"] = True
    elif mutator == "bad_shape":
        doc_a["cases"]["c1"]["shape"] = [-1]
    elif mutator == "empty_dtype":
        doc_a["cases"]["c1"]["dtype"] = ""
    elif mutator == "empty_sha":
        doc_a["cases"]["c1"]["sha256"] = ""
    elif mutator == "parent_escape":
        doc_a["cases"]["c1"]["sample_file"] = "../escape.npy"
    elif mutator == "abs_sample":
        doc_a["cases"]["c1"]["sample_file"] = os.sep + "tmp" + os.sep + "bad.npy"
    elif mutator == "missing_npy":
        doc_a["cases"]["c1"]["sample_file"] = "absent.npy"
    elif mutator == "tampered_sha":
        doc_a["cases"]["c1"]["sample_sha256"] = "00" * 32
    elif mutator == "corrupt_npy":
        (tmp_path / "corrupt.npy").write_bytes(b"not-an-npy")
        doc_a["cases"]["c1"]["sample_file"] = "corrupt.npy"
        doc_a["cases"]["c1"].pop("sample_sha256")
    elif mutator == "non_1d_npy":
        bad_2d = tmp_path / "mat.npy"
        np.save(bad_2d, np.ones((2, 2), dtype=np.float32), allow_pickle=False)
        doc_a["cases"]["c1"]["sample_file"] = "mat.npy"
        doc_a["cases"]["c1"].pop("sample_sha256")
    elif mutator == "shape_mismatch":
        mod.write_sample_npy(
            tmp_path / "three.npy", np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        doc_b["cases"]["c1"]["sample_file"] = "three.npy"
        doc_b["cases"]["c1"].pop("sample_sha256")

    path_a.write_text(json.dumps(doc_a), encoding="utf-8")
    path_b.write_text(json.dumps(doc_b), encoding="utf-8")
    assert mod.main(["--compare", str(path_a), str(path_b)]) == 1
    assert expected_msg in capsys.readouterr().err


def test_run_benchmark_end_to_end_with_fake_cueq_on_cpu(
    tmp_path: Path, monkeypatch
) -> None:
    import jax.numpy as jnp

    from foldjax.models import _cueq as cueq_mod

    mod = _load_check_module()

    def make_fake_cueq(ver: str, shift: float):
        cuex = types.ModuleType("cuequivariance_jax")
        cuex.__version__ = ver
        cuex.TriMulPrecision = types.SimpleNamespace(
            DEFAULT="DEFAULT", TF32="TF32", TF32x3="TF32x3", IEEE="IEEE"
        )

        def triangle_attention(*, q, k, v, bias, mask, scale, precision=None):
            del k, v, bias, mask, scale, precision
            out = q + jnp.asarray(shift, dtype=q.dtype)
            stats = jnp.zeros((*q.shape[:-1], 1), dtype=jnp.float32)
            return out, stats, stats

        def triangle_multiplicative_update(**kwargs):
            x = kwargs["x"]
            return x + jnp.asarray(shift, dtype=x.dtype)

        cuex.triangle_attention = triangle_attention
        cuex.triangle_multiplicative_update = triangle_multiplicative_update

        ln_mod = types.ModuleType("cuequivariance_jax.triangle._layer_norm_transpose")

        def layer_norm_transpose(
            x, w, b, eps=1e-5, layout="bijd->bijd", fallback=False
        ):
            del w, b, eps, fallback
            moved = jnp.transpose(x, (1, 2, 3, 0)) if layout == "dbij->bijd" else x
            return moved + jnp.asarray(shift, dtype=moved.dtype)

        ln_mod.layer_norm_transpose = layer_norm_transpose

        gemm_mod = types.ModuleType(
            "cuequivariance_jax.triangle._sigmoid_gated_dual_gemm"
        )

        def sigmoid_gated_dual_gemm(
            x, w1, w2, *, mask=None, transpose_out=False, precision=None, fallback=False
        ):
            del w2, mask, precision, fallback
            proj = x @ w1.T
            return jnp.transpose(proj, (3, 0, 1, 2)) if transpose_out else proj

        def sigmoid_gated_dual_gemm_dual_x(
            x1, x2, w1, w2, *, precision=None, fallback=False
        ):
            del x1, w1, precision, fallback
            return (x2 @ w2.T) + jnp.asarray(shift, dtype=x2.dtype)

        gemm_mod.sigmoid_gated_dual_gemm = sigmoid_gated_dual_gemm
        gemm_mod.sigmoid_gated_dual_gemm_dual_x = sigmoid_gated_dual_gemm_dual_x

        tri_pkg = types.ModuleType("cuequivariance_jax.triangle")
        tri_pkg._layer_norm_transpose = ln_mod
        tri_pkg._sigmoid_gated_dual_gemm = gemm_mod
        cuex.triangle = tri_pkg
        return cuex, tri_pkg, ln_mod, gemm_mod

    tiny_specs = [
        mod.CaseSpec(
            "raw_att", "raw_triangle_attention", 8, "fp32", "last5pct_invalid"
        ),
        mod.CaseSpec("core_att", "cueq_attention_core", 8, "bf16", "all_valid"),
        mod.CaseSpec(
            "fused_mul", "fused_triangle_multiplication", 8, "bf16_fp32norm", "outgoing"
        ),
        mod.CaseSpec(
            "boltz_mul",
            "boltz2_amp_triangle_multiplication",
            8,
            "fp32_bf16amp",
            "incoming",
        ),
        mod.CaseSpec("ln_bijd", "layer_norm_transpose", 8, "fp32", "bijd->bijd"),
        mod.CaseSpec("ln_dbij", "layer_norm_transpose", 8, "bf16", "dbij->bijd"),
        mod.CaseSpec("bad_fam", "unsupported_family", 8, "fp32", "none"),
    ]

    for label, ver, shift in (("a", "0.11.1", 0.0), ("b", "0.13.0", 0.25)):
        cuex, tri_pkg, ln_mod, gemm_mod = make_fake_cueq(ver, shift)
        monkeypatch.setitem(sys.modules, "cuequivariance_jax", cuex)
        monkeypatch.setitem(sys.modules, "cuequivariance_jax.triangle", tri_pkg)
        monkeypatch.setitem(
            sys.modules,
            "cuequivariance_jax.triangle._layer_norm_transpose",
            ln_mod,
        )
        monkeypatch.setitem(
            sys.modules,
            "cuequivariance_jax.triangle._sigmoid_gated_dual_gemm",
            gemm_mod,
        )
        def fake_loader(mod_ref=cuex):
            return mod_ref

        monkeypatch.setattr(cueq_mod, "load_cueq", fake_loader)
        report = mod.run_benchmark(
            tmp_path / f"run_{label}.json",
            specs=tiny_specs,
            warmup=0,
            repeats=1,
            max_sample_elements=64,
            _allow_non_gpu=True,
        )
        assert report["versions"]["cuequivariance_jax"] == ver
        assert report["cases"]["bad_fam"]["status"] == "error"
        for ok_name in (
            "raw_att",
            "core_att",
            "fused_mul",
            "boltz_mul",
            "ln_bijd",
            "ln_dbij",
        ):
            assert report["cases"][ok_name]["status"] == "ok"
            assert report["cases"][ok_name]["sample_elements"] <= 64

    table = mod.compare_reports(tmp_path / "run_a.json", tmp_path / "run_b.json")
    assert "raw_att" in table and "2.500e-01" in table
    assert "ln_bijd" in table and "bad_fam" in table and "ERROR" in table


def test_run_mode_returns_2_on_case_failure_and_still_writes_report(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import jax
    import jax.numpy as jnp

    from foldjax.models import _cueq as cueq_mod

    mod = _load_check_module()
    env_before = (
        os.environ.get("XDG_CACHE_HOME"),
        os.environ.get("TRITON_CACHE_DIR"),
    )

    cuex = types.ModuleType("cuequivariance_jax")
    cuex.__version__ = "0.13.0"
    cuex.TriMulPrecision = types.SimpleNamespace(
        DEFAULT="DEFAULT", TF32="TF32", TF32x3="TF32x3", IEEE="IEEE"
    )
    calls = {"count": 0}

    def triangle_attention(**kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("synthetic kernel failure")
        q = kwargs["q"]
        stats = jnp.zeros((*q.shape[:-1], 1), dtype=jnp.float32)
        return q, stats, stats

    cuex.triangle_attention = triangle_attention
    monkeypatch.setitem(sys.modules, "cuequivariance_jax", cuex)
    monkeypatch.setattr(cueq_mod, "load_cueq", lambda: cuex)
    orig_devices = jax.devices
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "devices", lambda *_a, **_kw: orig_devices())

    out_fail = tmp_path / "partial_fail.json"
    rc = mod.main(
        [
            "--out",
            str(out_fail),
            "--case-filter",
            "a_raw_tri_att_n256_fp32",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--max-sample-elements",
            "16",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert (
        "error: 1 of 2 cases failed: a_raw_tri_att_n256_fp32_last5pct_invalid"
        in captured.err
    )
    assert out_fail.is_file()
    loaded = mod.load_report(out_fail)
    assert loaded["cases"]["a_raw_tri_att_n256_fp32_all_valid"]["status"] == "ok"
    assert (
        loaded["cases"]["a_raw_tri_att_n256_fp32_last5pct_invalid"]["status"]
        == "error"
    )
    assert (
        tmp_path / "partial_fail.a_raw_tri_att_n256_fp32_all_valid.npy"
    ).is_file()

    out_ok = tmp_path / "all_ok.json"
    rc_ok = mod.main(
        [
            "--out",
            str(out_ok),
            "--case-filter",
            "a_raw_tri_att_n256_fp32_all_valid",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--max-sample-elements",
            "16",
        ]
    )
    assert rc_ok == 0
    assert capsys.readouterr().err == ""
    assert out_ok.is_file()
    assert (
        os.environ.get("XDG_CACHE_HOME"),
        os.environ.get("TRITON_CACHE_DIR"),
    ) == env_before
    assert mod.__doc__ is not None
    assert "XDG_CACHE_HOME" in mod.__doc__
    assert "TRITON_CACHE_DIR" in mod.__doc__
    assert mod.main.__doc__ is not None
    assert "2" in mod.main.__doc__
    help_text = mod.build_parser().format_help()
    assert "fresh PATH" in help_text and "exits 2" in help_text


def test_main_failed_case_summary_limits_and_malformed_status(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    mod = _load_check_module()

    def make_stub_report(case_statuses: list[tuple[str, object]]):
        return {"cases": {name: {"status": st} for name, st in case_statuses}}

    five_failures = [(f"c{idx}", "error") for idx in range(1, 6)]
    monkeypatch.setattr(
        mod, "run_benchmark", lambda *_a, **_kw: make_stub_report(five_failures)
    )
    rc_five = mod.main(["--out", str(tmp_path / "five.json")])
    err_five = capsys.readouterr().err.strip()
    assert rc_five == 2
    assert err_five == "error: 5 of 5 cases failed: c1, c2, c3, c4, c5"

    six_failures = [
        ("ok0", "ok"),
        *[(f"c{idx}", "error") for idx in range(1, 6)],
        ("c6", None),
    ]
    monkeypatch.setattr(
        mod, "run_benchmark", lambda *_a, **_kw: make_stub_report(six_failures)
    )
    rc_six = mod.main(["--out", str(tmp_path / "six.json")])
    err_six = capsys.readouterr().err.strip()
    assert rc_six == 2
    assert err_six == "error: 6 of 7 cases failed: c1, c2, c3, c4, c5 (+1 more)"


def test_run_mode_refuses_existing_report_or_sidecars(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import jax

    from foldjax.models import _cueq as cueq_mod

    mod = _load_check_module()
    orig_devices = jax.devices
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "devices", lambda *_a, **_kw: orig_devices())
    monkeypatch.setattr(
        cueq_mod,
        "load_cueq",
        lambda: pytest.fail("run_benchmark should refuse before loading cueq"),
    )

    existing_report = tmp_path / "existing.json"
    original_json = '{"sentinel": true}\n'
    existing_report.write_text(original_json, encoding="utf-8")
    rc_report = mod.main(["--out", str(existing_report)])
    err_report = capsys.readouterr().err
    assert rc_report == 1
    assert "refusing to overwrite existing output" in err_report
    assert "fresh --out" in err_report
    assert existing_report.read_text(encoding="utf-8") == original_json
    assert list(tmp_path.glob("existing.*.npy")) == []

    dangling_report = tmp_path / "dangling.json"
    dangling_report.symlink_to(tmp_path / "missing_target.json")
    assert mod.main(["--out", str(dangling_report)]) == 1
    assert "refusing to overwrite existing output" in capsys.readouterr().err
    assert dangling_report.is_symlink()

    sidecar_target = tmp_path / "with_sidecar.json"
    sidecar_file = tmp_path / "with_sidecar.raw_att.npy"
    original_bytes = b"untouched-sidecar-bytes"
    sidecar_file.write_bytes(original_bytes)
    rc_sidecar = mod.main(["--out", str(sidecar_target)])
    err_sidecar = capsys.readouterr().err
    assert rc_sidecar == 1
    assert "refusing to overwrite existing output" in err_sidecar
    assert "with_sidecar.raw_att.npy" in err_sidecar
    assert sidecar_file.read_bytes() == original_bytes
    assert not sidecar_target.exists()

    dir_sidecar_target = tmp_path / "with_dir.json"
    sidecar_dir = tmp_path / "with_dir.case_a.npy"
    sidecar_dir.mkdir()
    rc_dir = mod.main(["--out", str(dir_sidecar_target)])
    err_dir = capsys.readouterr().err
    assert rc_dir == 1
    assert "refusing to overwrite existing output" in err_dir
    assert sidecar_dir.is_dir()
    assert not dir_sidecar_target.exists()

    rc_empty = mod.main(["--out", ""])
    assert rc_empty == 1
    assert "refusing to overwrite existing output" in capsys.readouterr().err

    clean_dir = tmp_path / "neighbors"
    clean_dir.mkdir()
    (clean_dir / "other.case1.npy").write_bytes(b"x")
    (clean_dir / "run_extra.case1.npy").write_bytes(b"x")
    (clean_dir / "run.npy").write_bytes(b"x")
    (clean_dir / "run1.case1.npy").write_bytes(b"x")
    assert mod._existing_outputs(clean_dir / "run.json") == []
    assert mod._existing_outputs(clean_dir / "run[1].json") == []

    limit_dir = tmp_path / "limits"
    limit_dir.mkdir()
    for idx in range(1, 6):
        (limit_dir / f"lim.c{idx}.npy").write_bytes(b"x")
    with pytest.raises(ValueError) as exc_five:
        mod.run_benchmark(limit_dir / "lim.json", warmup=0, repeats=1)
    msg_five = str(exc_five.value)
    assert "lim.c5.npy" in msg_five and "more" not in msg_five

    (limit_dir / "lim.c6.npy").write_bytes(b"x")
    with pytest.raises(ValueError) as exc_six:
        mod.run_benchmark(limit_dir / "lim.json", warmup=0, repeats=1)
    msg_six = str(exc_six.value)
    assert "+1 more" in msg_six
