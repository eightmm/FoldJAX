"""CPU contracts for the subprocess peak observer's engine selection."""

from __future__ import annotations

import atexit
import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from bench.provenance import execution_identity

HOOK = Path(__file__).parents[1] / "bench/peakhook/sitecustomize.py"


def _hook(monkeypatch):
    callbacks = []
    monkeypatch.delenv("BENCH_PEAK_FILE", raising=False)
    monkeypatch.delenv("BENCH_PEAK_ENGINE", raising=False)
    monkeypatch.setattr(atexit, "register", callbacks.append)
    spec = importlib.util.spec_from_file_location("test_peakhook", HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, callbacks


def test_default_engine_remains_torch(monkeypatch):
    hook, _callbacks = _hook(monkeypatch)

    assert hook._peak_engine() == "torch"
    monkeypatch.setenv("BENCH_PEAK_ENGINE", "jax")
    assert hook._peak_engine() == "jax"


def test_unloaded_jax_returns_no_peak_and_does_not_import_it(monkeypatch):
    hook, _callbacks = _hook(monkeypatch)
    monkeypatch.delitem(sys.modules, "jax._src.xla_bridge", raising=False)

    assert hook._jax_peak() is None
    assert "jax._src.xla_bridge" not in sys.modules


def test_initialized_cuda_jax_peak_uses_live_allocator_stat(monkeypatch):
    hook, _callbacks = _hook(monkeypatch)
    device = SimpleNamespace(memory_stats=lambda: {"peak_bytes_in_use": 1234})
    backend = SimpleNamespace(platform="cuda", local_devices=lambda: [device])
    monkeypatch.setitem(
        sys.modules,
        "jax._src.xla_bridge",
        SimpleNamespace(_backends={"cuda": backend}),
    )
    monkeypatch.setattr(
        hook, "_torch_peak", lambda: (_ for _ in ()).throw(AssertionError)
    )

    assert hook._peak_for_engine("jax") == 1234


def test_initialized_gpu_platform_jax_peak_uses_live_allocator_stat(monkeypatch):
    hook, _callbacks = _hook(monkeypatch)
    device = SimpleNamespace(memory_stats=lambda: {"peak_bytes_in_use": 1234})
    backend = SimpleNamespace(platform="gpu", local_devices=lambda: [device])
    monkeypatch.setitem(
        sys.modules,
        "jax._src.xla_bridge",
        SimpleNamespace(_backends={"gpu": backend}),
    )

    assert hook._jax_peak() == 1234


def test_missing_jax_peak_field_writes_no_fabricated_zero(monkeypatch, tmp_path):
    hook, _callbacks = _hook(monkeypatch)
    destination = tmp_path / "peak.txt"
    device = SimpleNamespace(memory_stats=lambda: {"bytes_in_use": 99})
    backend = SimpleNamespace(platform="cuda", local_devices=lambda: [device])
    monkeypatch.setitem(
        sys.modules,
        "jax._src.xla_bridge",
        SimpleNamespace(_backends={"cuda": backend}),
    )

    hook._report(str(destination), "jax")

    assert not destination.exists()


def test_jax_peak_engine_is_bound_in_execution_provenance():
    identity = execution_identity(
        {"BENCH_PEAK_ENGINE": "jax"},
        timing_state="cold-or-unspecified",
        traced=False,
    )

    assert identity["environment"]["BENCH_PEAK_ENGINE"] == "jax"


def test_jax_import_registers_cleanup_before_observer_without_device_init(
    monkeypatch, tmp_path
):
    callbacks = []
    imports = []
    destination = tmp_path / "peak.txt"
    bridge = SimpleNamespace(
        _backends={
            "cuda": SimpleNamespace(
                platform="gpu",
                local_devices=lambda: [
                    SimpleNamespace(memory_stats=lambda: {"peak_bytes_in_use": 1234})
                ],
            )
        }
    )

    def cleanup():
        bridge._backends.clear()

    real_import = builtins.__import__

    def importing(name, *args, **kwargs):
        imports.append(name)
        if name == "jax":
            atexit.register(cleanup)
            monkeypatch.setitem(sys.modules, "jax._src.xla_bridge", bridge)
            return SimpleNamespace(
                devices=lambda: (_ for _ in ()).throw(AssertionError)
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("BENCH_PEAK_FILE", str(destination))
    monkeypatch.setenv("BENCH_PEAK_ENGINE", "jax")
    monkeypatch.setattr(atexit, "register", callbacks.append)
    monkeypatch.setattr(builtins, "__import__", importing)
    spec = importlib.util.spec_from_file_location("test_peakhook_order", HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert imports.count("jax") == 1
    assert callbacks[-2:] == [cleanup, callbacks[-1]]
    assert "torch" not in imports
    for callback in reversed(callbacks):
        callback()
    assert destination.read_text(encoding="utf-8") == "1234"
    assert bridge._backends == {}


def test_jax_import_failure_keeps_observer_fail_closed(monkeypatch, tmp_path):
    callbacks = []
    imports = []
    real_import = builtins.__import__

    def importing(name, *args, **kwargs):
        imports.append(name)
        if name == "jax":
            raise ImportError("missing jax")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("BENCH_PEAK_FILE", str(tmp_path / "peak.txt"))
    monkeypatch.setenv("BENCH_PEAK_ENGINE", "jax")
    monkeypatch.setattr(atexit, "register", callbacks.append)
    monkeypatch.setattr(builtins, "__import__", importing)
    spec = importlib.util.spec_from_file_location("test_peakhook_missing_jax", HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert imports.count("jax") == 1
    assert len(callbacks) == 1


def test_no_destination_does_not_import_jax(monkeypatch):
    imports = []
    real_import = builtins.__import__

    def importing(name, *args, **kwargs):
        imports.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.delenv("BENCH_PEAK_FILE", raising=False)
    monkeypatch.setenv("BENCH_PEAK_ENGINE", "jax")
    monkeypatch.setattr(builtins, "__import__", importing)
    spec = importlib.util.spec_from_file_location("test_peakhook_no_destination", HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "jax" not in imports
