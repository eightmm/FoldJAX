"""Shared Tokamax hits stay shared; complete misses persist in a private tier."""

from __future__ import annotations

import errno
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from foldjax import _cache_tiers
from foldjax._cache_tiers import CacheLocations, cache_locations
from foldjax.backends import _tokamax_autotune as persistent
from tests.test_alphafold3_tokamax_autotune import (
    _FakeBoundArguments,
    _FakeResult,
    _FakeTokamax,
    _inputs,
    _store,
)


@pytest.fixture
def runtime(monkeypatch):
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    monkeypatch.setenv("FOLDJAX_TRUST_SHARED_COMPILE_CACHE", "1")
    yield tokamax
    persistent.clear_record()


def _layered(shared, private, *, available=True, strategy="autotune", can_tune=True):
    return persistent._PersistentTokamaxStore(
        shared,
        identity={"runtime": "test", "tokamax": "0.0.13"},
        strategy=strategy,
        device_kind="mock-gpu",
        can_tune=can_tune,
        cache_locations=CacheLocations(shared, shared if available else None, private),
    )


def _call(store, *, lower=None, batch=None, invoke=None):
    rng_key, default_batch = _inputs()
    arguments = (_FakeBoundArguments("one"),)
    persistent.start_record(strategy=store._strategy, persistent_installed=True)
    value = store.call(
        rng_key=rng_key,
        batch=default_batch if batch is None else batch,
        lower=lower or (lambda: arguments),
        invoke=invoke or (lambda: bool(_FakeResult.active)),
    )
    return value, persistent.recorded()


def _files(root):
    return list((root / persistent._CACHE_DIRECTORY).glob("*.json"))


def _mode_writable(fd):
    # Model an unprivileged peer even when the suite itself runs as root.
    return bool(os.fstat(fd).st_mode & 0o200)


def _readonly(root):
    directory = root / persistent._CACHE_DIRECTORY
    if directory.exists():
        directory.chmod(0o555)
        for entry in directory.iterdir():
            entry.chmod(0o444)
    root.chmod(0o555)


def test_shared_hit_precedes_private_and_creates_no_private_copy(tmp_path, runtime):
    shared, private = tmp_path / "shared", tmp_path / "private"
    _call(_store(shared))
    payload = _files(shared)[0].read_bytes()
    for _ in range(2):
        value, record = _call(
            _layered(shared, private, strategy="error"),
            lower=lambda: pytest.fail("a shared hit must not lower"),
        )
        assert value
        assert record["sources"] == {"store": 1}
        assert record["calls"][0]["cache_read_tiers"] == {"shared": 1}
        assert "cache_write_tiers" not in record["calls"][0]
    assert not private.exists()
    assert _files(shared)[0].read_bytes() == payload
    assert runtime.tune_calls == 1

    _call(_layered(shared, private, available=False))
    assert _files(private)
    _, record = _call(_layered(shared, private, strategy="error"))
    assert record["calls"][0]["cache_read_tiers"] == {"shared": 1}


def test_writable_shared_tier_keeps_umask_permissions(tmp_path, runtime):
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir(mode=0o755)
    previous = os.umask(0o022)
    try:
        _, record = _call(_layered(shared, private))
    finally:
        os.umask(previous)
    assert record["sources"] == {"measured": 1}
    assert record["calls"][0]["cache_write_tiers"] == {"shared": 1}
    directory = shared / persistent._CACHE_DIRECTORY
    assert stat.S_IMODE(directory.stat().st_mode) == 0o755
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o644 for path in directory.iterdir()
    )
    assert len(_files(shared)) == 1 and not _files(private)
    assert runtime.tune_calls == 1


@pytest.mark.parametrize("tier", ["shared", "private"])
def test_in_memory_store_reuse_does_not_count_as_another_disk_read(
    tmp_path, runtime, tier
):
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir()
    store = _layered(shared, private, available=tier == "shared")
    _, first = _call(store)
    assert first["sources"] == {"measured": 1}
    assert first["calls"][0]["cache_write_tiers"] == {tier: 1}
    _, second = _call(
        store, lower=lambda: pytest.fail("an in-memory result must not be lowered")
    )
    assert second["sources"] == {"store": 1}
    call = second["calls"][0]
    assert call["cache_reuse_tiers"] == {tier: 1}
    assert "cache_read_tiers" not in call
    assert "cache_write_tiers" not in call
    assert runtime.tune_calls == 1


@pytest.mark.parametrize("foreign_owner", [0, 123456])
def test_readonly_foreign_shared_miss_persists_for_two_fresh_readers(
    tmp_path, monkeypatch, runtime, foreign_owner
):
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir()
    monkeypatch.setattr(persistent, "_directory_writable", _mode_writable)
    original_fstat = persistent._fstat

    def other_owner(fd, *, subject):
        info = original_fstat(fd, subject=subject)
        if Path(os.readlink(f"/proc/self/fd/{fd}")).is_relative_to(shared):
            fields = list(info)
            fields[4] = foreign_owner
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(persistent, "_fstat", other_owner)
    _readonly(shared)
    try:
        _, record = _call(_layered(shared, private))
        assert record["sources"] == {"measured": 1}
        assert record["calls"][0]["cache_write_tiers"] == {"private": 1}
        assert not _files(shared)
        assert stat.S_IMODE(private.stat().st_mode) == 0o700
        directory = private / persistent._CACHE_DIRECTORY
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o600 for path in directory.iterdir()
        )
        for _ in range(2):
            value, record = _call(
                _layered(shared, private, can_tune=False),
                lower=lambda: pytest.fail("a private hit must not lower"),
            )
            assert value
            assert record["sources"] == {"store": 1}
            assert record["calls"][0]["cache_read_tiers"] == {"private": 1}
        assert runtime.tune_calls == 1
    finally:
        shared.chmod(0o755)


def test_missing_shared_namespace_uses_the_active_private_namespace(
    tmp_path, monkeypatch, runtime
):
    shared, private = tmp_path / "missing" / "shared", tmp_path / "private"
    monkeypatch.setattr(
        persistent, "_runtime_identity",
        lambda **_: {"runtime": "test", "tokamax": "0.0.13"},
    )
    monkeypatch.setattr(persistent, "_can_tune_on_device", lambda _: False)
    locations = CacheLocations(shared, None, private)
    with cache_locations(locations):
        store = persistent.create_store(
            managed=True, cache_dir=shared,
            device=SimpleNamespace(platform="gpu", device_kind="mock-gpu"),
            source_snapshot=(("source",),), config_identity=(),
            buckets=None, padding=False, strategy="autotune",
        )
    assert store is not None
    assert store._cache_locations == locations
    value, record = _call(store)
    assert value
    assert runtime.tune_calls == 0  # The selected-device synchronous route ran.
    assert record["sources"] == {"measured": 1}
    assert record["calls"][0]["cache_write_tiers"] == {"private": 1}
    assert _files(private) and not shared.exists()


@pytest.mark.parametrize("failure", ["lock", "publish"])
def test_actual_shared_write_failure_falls_back_without_retuning(
    tmp_path, monkeypatch, runtime, failure
):
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir()
    real_replace = persistent.os.replace
    real_flock = persistent.fcntl.flock

    def replace(*args, **kwargs):
        if failure == "publish" and not persistent._PRIVATE_CACHE_IO.get():
            raise OSError(errno.EROFS, "shared filesystem became read-only")
        return real_replace(*args, **kwargs)

    def flock(fd, operation):
        if (
            failure == "lock" and operation == persistent.fcntl.LOCK_EX
            and not persistent._PRIVATE_CACHE_IO.get()
        ):
            raise OSError(errno.ENOLCK, "shared lock is unavailable")
        return real_flock(fd, operation)

    monkeypatch.setattr(persistent.os, "replace", replace)
    monkeypatch.setattr(persistent.fcntl, "flock", flock)
    _, record = _call(_layered(shared, private))
    assert runtime.tune_calls == 1
    assert record["sources"] == {"measured": 1}
    assert record["calls"][0]["cache_write_tiers"] == {"private": 1}
    assert not _files(shared) and len(_files(private)) == 1
    assert not list(shared.rglob("*.tmp"))
    value, record = _call(
        _layered(shared, private, strategy="error"),
        lower=lambda: pytest.fail("the complete fallback must survive a fresh store"),
    )
    assert value and record["sources"] == {"store": 1}


def test_private_lock_recheck_uses_a_peer_publication(tmp_path, monkeypatch, runtime):
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir()
    _call(_layered(shared, private, available=False))
    path = _files(private)[0]
    payload, name = path.read_bytes(), path.name
    path.unlink()
    _readonly(shared)
    monkeypatch.setattr(persistent, "_directory_writable", _mode_writable)
    real_lock = persistent._exclusive_lock

    @contextmanager
    def peer_publication(fd, signature):
        assert persistent._PRIVATE_CACHE_IO.get()
        with real_lock(fd, signature):
            persistent._atomic_write(fd, name, payload)
            yield

    monkeypatch.setattr(persistent, "_exclusive_lock", peer_publication)
    try:
        value, record = _call(
            _layered(shared, private),
            lower=lambda: pytest.fail("the write-target lock must recheck both tiers"),
        )
        assert value and record["sources"] == {"store": 1}
        assert record["calls"][0]["cache_read_tiers"] == {"private": 1}
        assert runtime.tune_calls == 1
    finally:
        shared.chmod(0o755)


def test_invalid_shared_entry_continues_to_valid_private_hit(tmp_path, runtime):
    shared, private = tmp_path / "shared", tmp_path / "private"
    _call(_store(shared))
    _call(_layered(shared, private, available=False))
    _files(shared)[0].write_text("{\"incomplete\":true}")
    _, record = _call(
        _layered(shared, private, strategy="error"),
        lower=lambda: pytest.fail("a valid lower tier must be tried"),
    )
    assert record["calls"][0]["cache_read_tiers"] == {"private": 1}
    assert runtime.tune_calls == 2


def test_partial_overlay_is_never_published_in_either_tier(tmp_path, runtime):
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir()
    runtime.failed_keys.add("missing")
    store = _layered(shared, private)
    arguments = (_FakeBoundArguments("covered"), _FakeBoundArguments("missing"))
    _, record = _call(store, lower=lambda: arguments)
    assert record["sources"] == {"partial": 1}
    assert store._partial_results and not _files(shared) and not _files(private)
    _, record = _call(
        store, lower=lambda: pytest.fail("partial overlays remain process-local")
    )
    assert record["sources"] == {"partial": 1}
    assert runtime.tune_calls == 1


@pytest.mark.parametrize("unsafe", ["symlink", "public-mode", "store-symlink"])
def test_private_trust_does_not_inherit_shared_opt_in(
    tmp_path, monkeypatch, runtime, unsafe
):
    shared, private, outside = (
        tmp_path / "shared", tmp_path / "private", tmp_path / "outside"
    )
    shared.mkdir()
    outside.mkdir(mode=0o700)
    if unsafe == "symlink":
        private.symlink_to(outside, target_is_directory=True)
    elif unsafe == "public-mode":
        private.mkdir(mode=0o755)
        private.chmod(0o755)
    else:
        private.mkdir(mode=0o700)
        (private / persistent._CACHE_DIRECTORY).symlink_to(
            outside, target_is_directory=True
        )
    _readonly(shared)
    monkeypatch.setattr(persistent, "_directory_writable", _mode_writable)
    try:
        value, record = _call(_layered(shared, private))
        assert value and record["sources"] == {"measured": 1}
        assert "cache_write_tiers" not in record["calls"][0]
        assert not list(outside.iterdir())
        assert not _files(shared) and not _files(private)
    finally:
        shared.chmod(0o755)


def test_another_active_namespace_does_not_change_standalone_store(
    tmp_path, monkeypatch, runtime
):
    shared, private = tmp_path / "shared", tmp_path / "private"
    monkeypatch.setattr(
        persistent, "_runtime_identity", lambda **_: {"runtime": "test"}
    )
    monkeypatch.setattr(persistent, "_can_tune_on_device", lambda _: True)
    with cache_locations(CacheLocations(tmp_path / "other", None, private)):
        store = persistent.create_store(
            managed=True, cache_dir=shared,
            device=SimpleNamespace(platform="gpu", device_kind="mock-gpu"),
            source_snapshot=(("source",),), config_identity=(), buckets=None,
            padding=False, strategy="autotune",
        )
    assert store is not None and store._cache_locations is None
    assert not private.exists()


def test_shared_symlink_cannot_redirect_the_store(tmp_path, runtime):
    shared, private, outside = (
        tmp_path / "shared", tmp_path / "private", tmp_path / "outside"
    )
    outside.mkdir()
    shared.symlink_to(outside, target_is_directory=True)
    value, record = _call(_layered(shared, private))
    assert value and record["calls"][0]["cache_write_tiers"] == {"private": 1}
    assert not list(outside.iterdir())


def test_private_ancestor_swap_does_not_reopen_outside_the_held_fd(
    tmp_path, monkeypatch, runtime
):
    shared, parent, outside = (
        tmp_path / "shared", tmp_path / "private-root", tmp_path / "outside"
    )
    shared.mkdir()
    _readonly(shared)
    monkeypatch.setattr(persistent, "_directory_writable", _mode_writable)
    outside.mkdir(mode=0o700)
    (outside / "profile").mkdir(mode=0o700)
    private = parent / "profile"
    retired = tmp_path / "retired-private-root"
    original_open = _cache_tiers.open_directory
    swapped = False

    @contextmanager
    def swap_ancestor(path, **kwargs):
        nonlocal swapped
        with original_open(path, **kwargs) as fd:
            if path == private and not swapped:
                parent.rename(retired)
                parent.symlink_to(outside, target_is_directory=True)
                swapped = True
            yield fd

    monkeypatch.setattr(_cache_tiers, "open_directory", swap_ancestor)
    try:
        value, record = _call(_layered(shared, private))
        assert swapped and value and record["sources"] == {"measured": 1}
        assert list(outside.iterdir()) == [outside / "profile"]
        assert not list((outside / "profile").iterdir())
        assert (retired / "profile" / persistent._CACHE_DIRECTORY).is_dir()
    finally:
        shared.chmod(0o755)


def test_model_oserror_is_not_retried_as_a_cache_failure(tmp_path, runtime):
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir()
    calls = []

    def invoke():
        calls.append(True)
        raise OSError("model execution failed")

    with pytest.raises(OSError, match="model execution failed"):
        _call(_layered(shared, private), invoke=invoke)
    assert calls == [True]
    assert runtime.tune_calls == 1
