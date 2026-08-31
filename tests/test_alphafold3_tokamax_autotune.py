from __future__ import annotations

import dataclasses
import errno
import functools
import json
import multiprocessing
import stat
import time
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest

from foldjax.backends import _tokamax_autotune as persistent


class _FakeData(dict[str, float | Exception]):
    @property
    def fastest_config(self) -> str:
        valid = [
            (name, value)
            for name, value in self.items()
            if not isinstance(value, Exception)
        ]
        if not valid:
            raise ValueError("no successful config")
        return min(valid, key=lambda item: item[1])[0]

    def prune(self) -> _FakeData:
        name = self.fastest_config
        return _FakeData({name: self[name]})


class _FakeBoundArguments:
    def __init__(self, key: str, data: _FakeData | None = None) -> None:
        self.key = key
        self.cached_autotuning_data = data
        self.op = _FAKE_OP

    @property
    def autotuning_cache_key(self) -> str:
        return self.key


class _FakeOp:
    pass


_FAKE_OP = _FakeOp()


class _FakeBatchedShapeDtype(jax.ShapeDtypeStruct):
    __slots__ = ("vmap_axes",)

    def __init__(self, shape, dtype, vmap_axes) -> None:
        super().__init__(shape, dtype)
        self.vmap_axes = tuple(vmap_axes)


@dataclasses.dataclass(frozen=True)
class _DataclassBoundArguments:
    arguments: dict[str, Any]


class _FakeResult:
    active: list[_FakeResult] = []

    def __init__(
        self,
        device_kind: str,
        data: tuple[tuple[_FakeBoundArguments, _FakeData], ...],
        tokamax_version: str = "0.0.13",
    ) -> None:
        self.device_kind = device_kind
        self.data = data
        self.tokamax_version = tokamax_version
        self._previous: list[tuple[_FakeBoundArguments, _FakeData | None]] = []

    def __enter__(self) -> _FakeResult:
        self._previous = []
        for bound_arg, data in self.data:
            self._previous.append((bound_arg, bound_arg.cached_autotuning_data))
            bound_arg.cached_autotuning_data = data
        self.active.append(self)
        return self

    def __exit__(self, *_args: Any) -> None:
        assert self.active.pop() is self
        for bound_arg, previous in self._previous:
            bound_arg.cached_autotuning_data = previous
        self._previous = []

    def dumps(self, *, prune_errors: bool = False) -> str:
        assert prune_errors
        return json.dumps(
            {
                "device_kind": self.device_kind,
                "tokamax_version": self.tokamax_version,
                "entries": [
                    {
                        "key": bound_arg.key,
                        "data": [
                            [name, value]
                            for name, value in data.items()
                            if not isinstance(value, Exception)
                        ],
                    }
                    for bound_arg, data in self.data
                ],
            }
        )

    @classmethod
    def loads(cls, payload: str) -> _FakeResult:
        value = json.loads(payload)
        entries = tuple(
            (
                _FakeBoundArguments(entry["key"]),
                _FakeData(dict(entry["data"])),
            )
            for entry in value["entries"]
        )
        return cls(
            value["device_kind"],
            entries,
            tokamax_version=value["tokamax_version"],
        )


class _FakeTokamax:
    __version__ = "0.0.13"
    AutotuningResult = _FakeResult

    def __init__(self) -> None:
        self.tune_calls = 0
        self.fallbacks: list[str] = []
        self.failed_keys: set[str] = set()
        self.autotuning = SimpleNamespace(get_bound_args=lambda lowered: lowered)
        self.config = SimpleNamespace(autotuning_cache_miss_fallback=self._fallback)

    @contextmanager
    def _fallback(self, strategy: str):
        self.fallbacks.append(strategy)
        yield

    def autotune(
        self,
        bound_args: tuple[_FakeBoundArguments, ...],
        *,
        progress_bar: bool,
    ) -> _FakeResult:
        assert not progress_bar
        self.tune_calls += 1
        entries = tuple(
            (
                bound_arg,
                _FakeData({f"{bound_arg.key}-slow": 2.0, f"{bound_arg.key}-fast": 1.0}),
            )
            for bound_arg in bound_args
            if bound_arg.cached_autotuning_data is None
            and bound_arg.key not in self.failed_keys
        )
        return _FakeResult("mock-gpu", entries)


class _FakeStore:
    compatibility_key = ("fake",)

    def __init__(self) -> None:
        self.lowered = None

    def call(self, *, lower, invoke, **_kwargs):
        self.lowered = lower()
        return invoke()


def _store(
    root: Path,
    *,
    strategy: str = "autotune",
    can_tune: bool = True,
    identity: dict[str, Any] | None = None,
) -> persistent._PersistentTokamaxStore:
    root.mkdir(mode=0o700, exist_ok=True)
    return persistent._PersistentTokamaxStore(
        root,
        identity=identity or {"runtime": "test", "tokamax": "0.0.13"},
        strategy=strategy,
        device_kind="mock-gpu",
        can_tune=can_tune,
    )


def _inputs(shape: tuple[int, ...] = (2, 3)) -> tuple[np.ndarray, dict[str, Any]]:
    return (
        np.zeros((2,), dtype=np.uint32),
        {"x": np.zeros(shape, dtype=np.float32), "flag": np.asarray(1)},
    )


def _signature(
    store: persistent._PersistentTokamaxStore,
    rng_key: Any,
    batch: Any,
) -> str:
    descriptor = persistent._call_descriptor(rng_key, batch)
    return persistent._sha256(
        persistent._canonical_bytes({"identity": store._identity, "call": descriptor})
    )


def _lock_worker(
    cache_dir: str,
    signature: str,
    barrier,
    builders,
    results,
) -> None:
    with persistent._open_cache_directory(Path(cache_dir)) as directory_fd:
        barrier.wait()
        with persistent._exclusive_lock(directory_fd, signature):
            payload = persistent._read_bytes(directory_fd, "shared.json")
            if payload is None:
                with builders.get_lock():
                    builders.value += 1
                time.sleep(0.05)
                persistent._atomic_write(directory_fd, "shared.json", b"complete")
            results.put(persistent._read_bytes(directory_fd, "shared.json"))


def test_call_signature_uses_structure_shape_and_dtype_but_not_values() -> None:
    rng_key, first = _inputs()
    second = {
        "flag": np.asarray(999),
        "x": np.ones((2, 3), dtype=np.float32),
    }

    assert persistent._call_descriptor(rng_key, first) == persistent._call_descriptor(
        rng_key + 7, second
    )
    assert persistent._call_descriptor(rng_key, first) != persistent._call_descriptor(
        rng_key, {"x": np.ones((3, 2), dtype=np.float32), "flag": np.asarray(1)}
    )
    assert persistent._call_descriptor(rng_key, first) != persistent._call_descriptor(
        rng_key, {"x": np.ones((2, 3), dtype=np.float16), "flag": np.asarray(1)}
    )


def test_all_nested_pre_vmap_lookups_are_reconstructed() -> None:
    bound_arg = _DataclassBoundArguments(
        {
            "x": _FakeBatchedShapeDtype((132, 384), np.float32, ((0, 1), (0, 4))),
            "weights": _FakeBatchedShapeDtype((384, 2, 768), np.float32, (None, None)),
            "activation": "silu",
        }
    )

    intermediate, unbatched = persistent._pre_vmap_variants(bound_arg)

    assert intermediate.arguments["activation"] == "silu"
    assert intermediate.arguments["x"].vmap_axes == ((0, 4),)
    assert intermediate.arguments["weights"].vmap_axes == (None,)
    for name in ("x", "weights"):
        value = unbatched.arguments[name]
        assert type(value) is jax.ShapeDtypeStruct
        assert not hasattr(value, "vmap_axes")
        assert value.shape == bound_arg.arguments[name].shape
        assert value.dtype == bound_arg.arguments[name].dtype


@pytest.mark.parametrize(
    ("managed", "cache", "platform", "snapshot"),
    (
        (False, True, "gpu", (("source",),)),
        (True, False, "gpu", (("source",),)),
        (True, True, "cpu", (("source",),)),
        (True, True, "gpu", None),
    ),
)
def test_store_activation_requires_managed_gpu_cache_and_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    managed: bool,
    cache: bool,
    platform: str,
    snapshot: tuple[Any, ...] | None,
) -> None:
    monkeypatch.setattr(persistent, "_filesystem_supported", lambda: True)
    monkeypatch.setattr(
        persistent,
        "_runtime_identity",
        lambda **_kwargs: pytest.fail("ineligible routes must not build identity"),
    )
    device = SimpleNamespace(platform=platform, device_kind="mock-gpu")

    assert (
        persistent.create_store(
            managed=managed,
            cache_dir=tmp_path if cache else None,
            device=device,
            source_snapshot=snapshot,
            config_identity=(),
            buckets=None,
            padding=False,
            strategy="autotune",
        )
        is None
    )


def test_nondefault_gpu_can_read_but_cannot_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(persistent, "_filesystem_supported", lambda: True)
    monkeypatch.setattr(
        persistent, "_runtime_identity", lambda **_kwargs: {"runtime": "test"}
    )
    monkeypatch.setattr(persistent, "_can_tune_on_device", lambda _device: False)
    device = SimpleNamespace(platform="gpu", device_kind="mock-gpu")

    store = persistent.create_store(
        managed=True,
        cache_dir=tmp_path,
        device=device,
        source_snapshot=(("source",),),
        config_identity=(),
        buckets=None,
        padding=False,
        strategy="autotune",
    )

    assert store is not None
    assert not store._can_tune


def test_nondefault_gpu_autotune_miss_never_runs_inline_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    store = _store(tmp_path / "cache", strategy="autotune", can_tune=False)
    rng_key, batch = _inputs()

    with pytest.raises(RuntimeError, match="non-default GPU"):
        store.call(
            rng_key=rng_key,
            batch=batch,
            lower=lambda: pytest.fail("unsafe worker tuning must not lower"),
            invoke=lambda: pytest.fail("unsafe inline autotune must not run"),
        )

    assert tokamax.tune_calls == 0


def test_nondefault_gpu_requires_an_installed_persistent_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(persistent, "_can_tune_on_device", lambda _device: False)
    device = SimpleNamespace(platform="gpu")

    with pytest.raises(RuntimeError, match="non-default GPU"):
        persistent.ensure_safe_autotuning_route(
            device=device,
            strategy="autotune",
            persistent_installed=False,
        )

    persistent.ensure_safe_autotuning_route(
        device=device,
        strategy="autotune",
        persistent_installed=True,
    )
    persistent.ensure_safe_autotuning_route(
        device=device,
        strategy="heuristics",
        persistent_installed=False,
    )


def test_runner_wrapper_lowers_the_bound_parameter_callable() -> None:
    calls = []

    class _Compiled:
        def __call__(self, params, rng_key, batch):
            calls.append(("invoke", params, rng_key, batch))
            return "prediction"

        def lower(self, params, rng_key, batch):
            calls.append(("lower", params, rng_key, batch))
            return "lowered"

    original = functools.partial(_Compiled(), "params")
    runner = SimpleNamespace(_model=original)
    store = _FakeStore()

    assert persistent.install_store(runner, store)
    assert runner._model("rng", {"batch": 1}) == "prediction"
    assert store.lowered == "lowered"
    assert calls == [
        ("lower", "params", "rng", {"batch": 1}),
        ("invoke", "params", "rng", {"batch": 1}),
    ]
    assert persistent.install_store(runner, store)


def test_complete_result_is_pruned_persisted_and_reused_across_store_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    root = tmp_path / "cache"
    store = _store(root)
    rng_key, batch = _inputs()
    packaged = _FakeBoundArguments(
        "packaged", _FakeData({"packaged-slow": 3.0, "packaged-fast": 1.0})
    )
    missing = _FakeBoundArguments("missing")
    bound_args = (packaged, missing)

    def active_id() -> int | None:
        return id(_FakeResult.active[-1]) if _FakeResult.active else None

    first_id = store.call(
        rng_key=rng_key,
        batch=batch,
        lower=lambda: bound_args,
        invoke=active_id,
    )
    second_id = store.call(
        rng_key=rng_key,
        batch=batch,
        lower=lambda: pytest.fail("an in-memory hit must not lower"),
        invoke=active_id,
    )

    assert tokamax.tune_calls == 1
    assert first_id == second_id
    result = next(iter(store._results.values()))
    assert len(result.data) == 2
    assert all(len(data) == 1 for _bound_arg, data in result.data)
    files = list((root / persistent._CACHE_DIRECTORY).glob("*.json"))
    assert len(files) == 1
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600

    fresh = _store(root, strategy="error", can_tune=False)
    loaded_id = fresh.call(
        rng_key=rng_key,
        batch=batch,
        lower=lambda: pytest.fail("a disk hit must not lower"),
        invoke=active_id,
    )
    assert loaded_id is not None
    assert tokamax.tune_calls == 1


def test_discovery_reaches_strict_coverage_after_overlay_changes_the_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    root = tmp_path / "cache"
    store = _store(root)
    rng_key, batch = _inputs()
    initial = _FakeBoundArguments("initial")
    revealed_by_overlay = _FakeBoundArguments("revealed-by-overlay")
    lower_calls = 0

    def lower():
        nonlocal lower_calls
        lower_calls += 1
        active_keys = {
            bound_arg.key
            for result in _FakeResult.active
            for bound_arg, _data in result.data
        }
        if "initial" in active_keys:
            return (initial, revealed_by_overlay)
        return (initial,)

    assert store.call(
        rng_key=rng_key,
        batch=batch,
        lower=lower,
        invoke=lambda: bool(_FakeResult.active),
    )

    assert lower_calls == 4
    assert tokamax.tune_calls == 2
    manifest = json.loads(
        next((root / persistent._CACHE_DIRECTORY).glob("*.json")).read_text()
    )
    assert manifest["coverage"] == {
        "bound_arguments": 2,
        "configs_per_entry": 1,
        "entries": 2,
    }


def test_discovery_keeps_superseded_result_ids_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    completed: list[weakref.ReferenceType[_FakeResult]] = []
    real_complete = persistent._complete_result

    def record_complete(*args, **kwargs):
        result = real_complete(*args, **kwargs)
        completed.append(weakref.ref(result))
        return result

    monkeypatch.setattr(persistent, "_complete_result", record_complete)
    root = tmp_path / "cache"
    store = _store(root)
    rng_key, batch = _inputs()
    initial = _FakeBoundArguments("initial")
    revealed = _FakeBoundArguments("revealed")
    lower_calls = 0

    def lower():
        nonlocal lower_calls
        lower_calls += 1
        if lower_calls == 4:
            assert completed[0]() is not None
        active_keys = {
            bound_arg.key
            for result in _FakeResult.active
            for bound_arg, _data in result.data
        }
        return (initial, revealed) if "initial" in active_keys else (initial,)

    assert store.call(
        rng_key=rng_key,
        batch=batch,
        lower=lower,
        invoke=lambda: bool(_FakeResult.active),
    )
    assert lower_calls == 4


def test_duplicate_operation_identity_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    bound_arg = _FakeBoundArguments("duplicate")
    result = _FakeResult(
        "mock-gpu",
        (
            (bound_arg, _FakeData({"first": 1.0})),
            (bound_arg, _FakeData({"second": 1.0})),
        ),
    )

    with pytest.raises(
        persistent._InvalidManifestError,
        match="duplicate operation entry",
    ):
        persistent._manifest_bytes(
            result,
            signature="a" * 64,
            call_sha256="b" * 64,
            identity={"runtime": "test", "tokamax": "0.0.13"},
            expected_count=2,
            device_kind="mock-gpu",
        )


def test_published_result_excludes_failed_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    bound_arg = _FakeBoundArguments("op")
    tuned = _FakeResult(
        "mock-gpu",
        (
            (
                bound_arg,
                _FakeData(
                    {
                        "fast": 1.0,
                        "serialization-failure-marker": RuntimeError("failed"),
                    }
                ),
            ),
        ),
    )

    complete = persistent._complete_result((bound_arg,), tuned, "mock-gpu")
    payload = persistent._manifest_bytes(
        complete,
        signature="a" * 64,
        call_sha256="b" * 64,
        identity={"runtime": "test", "tokamax": "0.0.13"},
        expected_count=1,
        device_kind="mock-gpu",
    )

    assert complete.data[0][1] == {"fast": 1.0}
    assert b"serialization-failure-marker" not in payload


def test_empty_capture_skips_public_autotune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    root = tmp_path / "cache"
    store = _store(root)
    rng_key, batch = _inputs()

    assert (
        store.call(
            rng_key=rng_key,
            batch=batch,
            lower=lambda: (),
            invoke=lambda: "fallback",
        )
        == "fallback"
    )
    assert tokamax.tune_calls == 0
    assert not list((root / persistent._CACHE_DIRECTORY).glob("*.json"))


def test_incomplete_coverage_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokamax = _FakeTokamax()
    tokamax.failed_keys.add("missing")
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    root = tmp_path / "cache"
    store = _store(root)
    rng_key, batch = _inputs()

    assert (
        store.call(
            rng_key=rng_key,
            batch=batch,
            lower=lambda: (
                _FakeBoundArguments("covered"),
                _FakeBoundArguments("missing"),
            ),
            invoke=lambda: "fallback",
        )
        == "fallback"
    )
    assert tokamax.tune_calls == 1
    assert not list((root / persistent._CACHE_DIRECTORY).glob("*.json"))


def test_serialization_failure_keeps_the_complete_in_memory_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)

    def fail_manifest(*_args, **_kwargs):
        raise persistent._InvalidManifestError("simulated serialization failure")

    monkeypatch.setattr(persistent, "_manifest_bytes", fail_manifest)
    root = tmp_path / "cache"
    store = _store(root)
    rng_key, batch = _inputs()
    bound_args = (_FakeBoundArguments("one"),)

    assert store.call(
        rng_key=rng_key,
        batch=batch,
        lower=lambda: bound_args,
        invoke=lambda: bool(_FakeResult.active),
    )
    assert tokamax.tune_calls == 1
    assert len(store._results) == 1
    assert not list((root / persistent._CACHE_DIRECTORY).glob("*.json"))


@pytest.mark.parametrize(
    "corruption", ("schema", "signature", "identity", "result_version", "device")
)
def test_manifest_identity_and_versions_are_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    root = tmp_path / "cache"
    store = _store(root)
    rng_key, batch = _inputs()
    bound_args = (_FakeBoundArguments("one"),)
    store.call(
        rng_key=rng_key,
        batch=batch,
        lower=lambda: bound_args,
        invoke=lambda: "first",
    )
    path = next((root / persistent._CACHE_DIRECTORY).glob("*.json"))
    manifest = json.loads(path.read_text())
    if corruption == "schema":
        manifest["schema"] += 1
    elif corruption == "signature":
        manifest["signature"] = "0" * 64
    elif corruption == "identity":
        manifest["identity"] = {"runtime": "other"}
    elif corruption == "result_version":
        manifest["result"]["tokamax_version"] = "9.9.9"
    else:
        manifest["result"]["device_kind"] = "other-gpu"
    if corruption in {"result_version", "device"}:
        manifest["result_sha256"] = persistent._sha256(
            persistent._canonical_bytes(manifest["result"])
        )
    path.write_text(persistent._canonical_json(manifest))
    path.chmod(0o600)

    fresh = _store(root, strategy="error", can_tune=False)
    assert (
        fresh.call(
            rng_key=rng_key,
            batch=batch,
            lower=lambda: pytest.fail("error policy must not lower"),
            invoke=lambda: "miss",
        )
        == "miss"
    )
    assert tokamax.tune_calls == 1


@pytest.mark.parametrize("entry", ("result", "lock"))
def test_symlink_entries_are_never_followed_or_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    root = tmp_path / "cache"
    store = _store(root)
    rng_key, batch = _inputs()
    signature = _signature(store, rng_key, batch)
    directory = root / persistent._CACHE_DIRECTORY
    directory.mkdir(mode=0o700)
    target = tmp_path / "outside"
    target.write_text("untouched")
    name = f"{signature}.json" if entry == "result" else f"{signature}.lockfile"
    (directory / name).symlink_to(target)

    assert (
        store.call(
            rng_key=rng_key,
            batch=batch,
            lower=lambda: (_FakeBoundArguments("one"),),
            invoke=lambda: "fallback",
        )
        == "fallback"
    )
    assert target.read_text() == "untouched"
    assert (directory / name).is_symlink()
    assert tokamax.tune_calls == 0


def test_oversize_and_nonprivate_entries_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    monkeypatch.setattr(persistent, "_MAX_RESULT_BYTES", 16)
    root = tmp_path / "cache"
    store = _store(root, strategy="error", can_tune=False)
    rng_key, batch = _inputs()
    signature = _signature(store, rng_key, batch)
    directory = root / persistent._CACHE_DIRECTORY
    directory.mkdir(mode=0o700)
    path = directory / f"{signature}.json"
    path.write_bytes(b"x" * 17)
    path.chmod(0o600)

    assert (
        store.call(
            rng_key=rng_key,
            batch=batch,
            lower=lambda: pytest.fail("error policy must not lower"),
            invoke=lambda: "oversize",
        )
        == "oversize"
    )

    path.write_text("{}")
    path.chmod(0o644)
    fresh = _store(root, strategy="error", can_tune=False)
    assert (
        fresh.call(
            rng_key=rng_key,
            batch=batch,
            lower=lambda: pytest.fail("error policy must not lower"),
            invoke=lambda: "permissions",
        )
        == "permissions"
    )


def test_atomic_publish_preserves_last_good_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    root.mkdir(mode=0o700)
    with persistent._open_cache_directory(root) as directory_fd:
        persistent._atomic_write(directory_fd, "value.json", b"old")
        path = root / persistent._CACHE_DIRECTORY / "value.json"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

        def fail_replace(*_args, **_kwargs):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(persistent.os, "replace", fail_replace)
        with pytest.raises(persistent._CacheError, match="atomically publish"):
            persistent._atomic_write(directory_fd, "value.json", b"new")
        assert persistent._read_bytes(directory_fd, "value.json") == b"old"


def test_open_cache_directory_repairs_a_collaborative_umask(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir(mode=0o770)
    root.chmod(0o770)

    with persistent._open_cache_directory(root):
        pass

    assert stat.S_IMODE(root.stat().st_mode) == 0o750
    private = root / persistent._CACHE_DIRECTORY
    assert stat.S_IMODE(private.stat().st_mode) == 0o700


def test_filesystem_fstat_failure_falls_back_without_tuning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    store = _store(tmp_path / "cache")
    rng_key, batch = _inputs()

    def fail_fstat(_fd: int):
        raise OSError(errno.EIO, "simulated stale filesystem")

    monkeypatch.setattr(persistent.os, "fstat", fail_fstat)
    assert (
        store.call(
            rng_key=rng_key,
            batch=batch,
            lower=lambda: pytest.fail("unavailable cache must not lower"),
            invoke=lambda: "fallback",
        )
        == "fallback"
    )
    assert tokamax.tune_calls == 0


def test_nondefault_gpu_filesystem_failure_never_invokes_inline_autotune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    store = _store(tmp_path / "cache", can_tune=False)
    rng_key, batch = _inputs()

    def fail_fstat(_fd: int):
        raise OSError(errno.EIO, "simulated stale filesystem")

    monkeypatch.setattr(persistent.os, "fstat", fail_fstat)
    with pytest.raises(RuntimeError, match="non-default GPU"):
        store.call(
            rng_key=rng_key,
            batch=batch,
            lower=lambda: pytest.fail("unavailable cache must not lower"),
            invoke=lambda: pytest.fail("unsafe inline autotune must not run"),
        )
    assert tokamax.tune_calls == 0


def test_filesystem_flock_failure_falls_back_without_tuning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    store = _store(tmp_path / "cache")
    rng_key, batch = _inputs()
    assert persistent.fcntl is not None
    real_flock = persistent.fcntl.flock

    def fail_exclusive_lock(fd: int, operation: int):
        if operation == persistent.fcntl.LOCK_EX:
            raise OSError(errno.ENOLCK, "simulated unsupported lock")
        return real_flock(fd, operation)

    monkeypatch.setattr(persistent.fcntl, "flock", fail_exclusive_lock)
    assert (
        store.call(
            rng_key=rng_key,
            batch=batch,
            lower=lambda: (_FakeBoundArguments("one"),),
            invoke=lambda: "fallback",
        )
        == "fallback"
    )
    assert tokamax.tune_calls == 0


def test_filesystem_read_failure_ignores_the_entry_and_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    root = tmp_path / "cache"
    rng_key, batch = _inputs()
    bound_args = (_FakeBoundArguments("one"),)
    _store(root).call(
        rng_key=rng_key,
        batch=batch,
        lower=lambda: bound_args,
        invoke=lambda: "writer",
    )
    fresh = _store(root, strategy="error", can_tune=False)

    def fail_read(_fd: int, _size: int):
        raise OSError(errno.EIO, "simulated stale cache entry")

    monkeypatch.setattr(persistent.os, "read", fail_read)
    assert (
        fresh.call(
            rng_key=rng_key,
            batch=batch,
            lower=lambda: pytest.fail("error-policy reader must not lower"),
            invoke=lambda: "fallback",
        )
        == "fallback"
    )
    assert tokamax.tune_calls == 1


def test_flock_recheck_allows_only_one_cross_process_builder(tmp_path: Path) -> None:
    if not persistent._filesystem_supported():
        pytest.skip("requires POSIX no-follow file descriptors and flock")
    root = tmp_path / "cache"
    root.mkdir(mode=0o700)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    builders = context.Value("i", 0)
    results = context.Queue()
    signature = "a" * 64
    processes = [
        context.Process(
            target=_lock_worker,
            args=(str(root), signature, barrier, builders, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    assert builders.value == 1
    assert {results.get(timeout=2) for _ in processes} == {b"complete"}
