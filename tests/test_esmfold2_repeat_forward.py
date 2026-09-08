import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.esmfold2_tape import (
    _save_npz,
    compare_saved_output_bytes,
    compile_replay,
    configure_autotune_control,
    finish_autotune_control,
)


@pytest.mark.parametrize(
    "change", [None, "value", "dtype", "shape", "missing", "signzero"]
)
def test_repeat_bytes_detect_semantic_and_storage_changes(tmp_path, change):
    left, right = tmp_path / "a", tmp_path / "b"
    left.mkdir()
    right.mkdir()
    a = np.array([0, 2], np.float32)
    b = a.copy()
    if change == "value":
        b[1] = 3
    elif change == "dtype":
        b = b.astype(np.float64)
    elif change == "shape":
        b = b[None]
    elif change == "signzero":
        b[0] = -0.0
    for filename in ("jax_coords.npz", "jax_confidence.npz"):
        _save_npz(left / filename, {"value": a})
        _save_npz(right / filename, {} if change == "missing" else {"value": b})
    results = compare_saved_output_bytes(left, right)
    assert all(
        r["all_storage_bytes_equal"] == (change is None) for r in results.values()
    )


def test_compiled_replay_reuses_executable_with_static_arguments_removed():
    def fn(x, *, settings, n_chains, tape):
        return x * settings + tape + n_chains

    predict = jax.jit(fn, static_argnames=("settings", "n_chains"))
    args = (jnp.array([1.0, 2.0]),)
    dynamic = {"tape": jnp.array([3.0, 4.0])}
    compiled = compile_replay(predict, args, {"settings": 2, "n_chains": 1}, dynamic)
    first = compiled(*args, **dynamic)
    second = compiled(*args, **dynamic)
    np.testing.assert_array_equal(first, [6, 9])
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_matching_nonfinite_bytes_are_not_finite_evidence(tmp_path, value):
    for filename in ("jax_coords.npz", "jax_confidence.npz"):
        _save_npz(tmp_path / filename, {"value": np.array([value], np.float32)})
    results = compare_saved_output_bytes(tmp_path, tmp_path)
    assert all(r["all_storage_bytes_equal"] for r in results.values())
    assert all(not r["all_finite"] for r in results.values())


def test_autotune_control_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "sentinel")
    assert configure_autotune_control(tmp_path) is None
    assert os.environ["XLA_FLAGS"] == "sentinel"


def test_autotune_strict_load_binds_input_and_dump(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "jax")
    monkeypatch.setenv(
        "XLA_FLAGS", "--xla_gpu_enable_scatter_determinism_expander=true"
    )
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "old-cache")
    monkeypatch.setenv("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "all")
    source = tmp_path / "source.textproto"
    source.write_text("results {}")
    output = tmp_path / "new"
    control = configure_autotune_control(output, load=source)
    assert control["strict_complete_load"]
    assert "require_complete_aot_autotune_results=true" in os.environ["XLA_FLAGS"]
    assert "scatter_determinism_expander=true" in os.environ["XLA_FLAGS"]
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == str(output / "jax-cache")
    assert os.environ["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] == "none"
    with pytest.raises(RuntimeError, match="missing or empty"):
        finish_autotune_control(control)
    output.mkdir()
    (output / "xla-autotune.textproto").write_text("results {}")
    completed = finish_autotune_control(control)
    assert completed["dump_sha256"] == control["load_sha256"]
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        finish_autotune_control(control)


@pytest.mark.parametrize("flag", ["autotune_results", "autotune_cache"])
def test_autotune_rejects_inherited_controls(tmp_path, monkeypatch, flag):
    monkeypatch.delitem(sys.modules, "jax")
    monkeypatch.setenv("XLA_FLAGS", f"--xla_gpu_{flag}=value")
    with pytest.raises(ValueError, match="conflict"):
        configure_autotune_control(tmp_path / "new", enabled=True)


def test_autotune_requires_fresh_output_and_existing_input(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "jax")
    monkeypatch.setenv("XLA_FLAGS", "")
    with pytest.raises(ValueError, match="fresh"):
        configure_autotune_control(tmp_path, enabled=True)
    with pytest.raises(FileNotFoundError):
        configure_autotune_control(tmp_path / "new", load=tmp_path / "absent")


def test_autotune_rejects_already_imported_jax(tmp_path, monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "unchanged")
    with pytest.raises(RuntimeError, match="before importing JAX"):
        configure_autotune_control(tmp_path / "new", enabled=True)
    assert os.environ["XLA_FLAGS"] == "unchanged"
