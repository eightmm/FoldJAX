"""Two-tier I/O boundaries and real JAX reuse in independent CPU processes."""

from __future__ import annotations

import errno
import json
import os
import stat
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import jax
import pytest
from jax._src import compilation_cache as jax_cache
from jax._src import compiler

from foldjax import cache
from foldjax._cache_tiers import (
    CachePathError,
    current_cache_locations,
    prepare_private_directory,
)


@pytest.fixture(autouse=True)
def _record():
    cache.clear_record()
    yield
    cache.clear_record()


def _adapter():
    result, _ = jax_cache.get_file_cache(jax.config.jax_compilation_cache_dir)
    return result


def test_private_root_does_not_follow_shared_home_or_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("FOLDJAX_PRIVATE_COMPILE_CACHE", raising=False)
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "shared"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache.private_compile_cache_root() == tmp_path / "xdg/foldjax/compile"
    monkeypatch.setenv("FOLDJAX_PRIVATE_COMPILE_CACHE", str(tmp_path / "override"))
    assert cache.private_compile_cache_root() == tmp_path / "override"


@pytest.mark.parametrize("mask", [0o022, 0o002])
def test_layered_shared_namespaces_are_readable_without_group_write(
    tmp_path, monkeypatch, trust_ancestors_above_tmp_path, mask
):
    monkeypatch.delenv(cache.TRUST_SHARED_COMPILE_CACHE_ENV, raising=False)
    shared = tmp_path / "compile" / "protenix" / "weights" / "digest"
    private = tmp_path / "private"
    previous = os.umask(mask)
    try:
        with cache.compilation_cache_scope(shared, private_directory=private):
            _adapter().put("key", b"compiled")
    finally:
        os.umask(previous)
    for directory in (shared, *list(shared.parents)[:3]):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o755
    assert stat.S_IMODE((shared / "key-cache").stat().st_mode) == 0o644
    assert stat.S_IMODE(private.stat().st_mode) == 0o700


@pytest.mark.parametrize("mask", [0o002, 0o007])
def test_layered_trusted_group_namespaces_keep_umask_and_setgid(
    tmp_path, monkeypatch, mask
):
    store = tmp_path / "compile"
    store.mkdir()
    store.chmod(0o2775)
    monkeypatch.setenv(cache.TRUST_SHARED_COMPILE_CACHE_ENV, "1")
    shared = store / "boltz2" / "weights" / "digest"
    previous = os.umask(mask)
    try:
        with cache.compilation_cache_scope(
            shared, private_directory=tmp_path / "private"
        ):
            _adapter().put("key", b"compiled")
    finally:
        os.umask(previous)
    for directory in (shared, shared.parent, shared.parent.parent):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o2777 & ~mask
    assert stat.S_IMODE((shared / "key-cache").stat().st_mode) == 0o666 & ~mask
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cache.cache_snapshot(shared, warn_unshareable=True)
    assert not caught


def test_shared_first_no_copy_and_private_write_on_readonly_shared(
    tmp_path, monkeypatch
):
    shared, private = tmp_path / "shared", tmp_path / "private"
    with cache.compilation_cache_scope(shared, private_directory=private):
        adapter = _adapter()
        adapter.put("shared-key", b"shared")
        assert (shared / "shared-key-cache").read_bytes() == b"shared"
        (private / "shared-key-cache").write_bytes(b"private")
        (private / "shared-key-cache").chmod(0o600)
        assert adapter.get("shared-key") == b"shared"
        assert cache.recorded()["loads"] == {}
        assert jax.config.jax_persistent_cache_enable_xla_caches == "none"
        original_access = os.access
        monkeypatch.setattr(
            os,
            "access",
            lambda path, mode: (
                False if Path(path) == shared else original_access(path, mode)
            ),
        )
        adapter.put("new-key", b"new")
        assert adapter.get("new-key") == b"new"
        assert (private / "shared-key-cache").read_bytes() == b"private"
        assert not (shared / "new-key-cache").exists()
    record = cache.recorded()
    assert record["writes"]["private"]["success"] == 1
    assert record["lookups"]["shared"]["hit"] == 1


@pytest.mark.parametrize("failure", (errno.EACCES, errno.EROFS, errno.ENOSPC))
def test_actual_shared_write_failure_falls_back(tmp_path, monkeypatch, failure):
    shared, private = tmp_path / "shared", tmp_path / "private"
    with cache.compilation_cache_scope(shared, private_directory=private):
        adapter = _adapter()
        original = adapter._put

        def put(tier, key, value):
            if tier == "shared":
                raise OSError(failure, "simulated write failure after access succeeded")
            return original(tier, key, value)

        monkeypatch.setattr(adapter, "_put", put)
        with pytest.warns(
            RuntimeWarning, match="shared compilation cache write failed"
        ):
            adapter.put("key", b"complete")
        assert (private / "key-cache").read_bytes() == b"complete"
        assert cache.recorded()["writes"]["shared"]["error"] == 1
        assert cache.recorded()["writes"]["private"]["success"] == 1


def test_missing_shared_namespace_and_two_unavailable_tiers(tmp_path, monkeypatch):
    shared, private = tmp_path / "missing/shared", tmp_path / "private"
    monkeypatch.setattr(cache, "_layered_shared_directory", lambda _: None)
    with cache.compilation_cache_scope(shared, private_directory=private):
        assert current_cache_locations().shared is None
        _adapter().put("key", b"private")
        assert (private / "key-cache").read_bytes() == b"private"
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 0
    cache.clear_record()
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o755)
    with pytest.warns(RuntimeWarning, match="private compile cache"):
        with cache.compilation_cache_scope(shared, private_directory=unsafe):
            assert jax.config.jax_compilation_cache_dir is None
            assert current_cache_locations().private is None
    assert cache.recorded()["writes"]["private"]["unavailable"] == 1


def test_private_owner_mode_symlink_and_hardlink_guards(tmp_path, monkeypatch):
    private = tmp_path / "private"
    prepare_private_directory(private)
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    alias = tmp_path / "alias"
    alias.symlink_to(private, target_is_directory=True)
    with pytest.raises(OSError):
        prepare_private_directory(alias)
    original_uid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: original_uid + 1)
    with pytest.raises(CachePathError):
        prepare_private_directory(private)
    monkeypatch.undo()
    with cache.compilation_cache_scope(None, private_directory=private):
        adapter = _adapter()
        adapter.put("valid", b"bytes")
        os.link(private / "valid-cache", private / "linked-cache")
        (private / "symlink-cache").symlink_to(private / "valid-cache")
        with pytest.warns(RuntimeWarning):
            assert adapter.get("linked") is None
        with pytest.warns(RuntimeWarning):
            assert adapter.get("symlink") is None


def test_alias_dedup_preserves_existing_shared_mode(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    alias = tmp_path / "alias"
    alias.symlink_to(shared, target_is_directory=True)
    with cache.compilation_cache_scope(shared, private_directory=alias):
        assert current_cache_locations().private is None
        _adapter().put("key", b"bytes")
    assert stat.S_IMODE(shared.stat().st_mode) == 0o755
    assert (shared / "key-cache").exists()


def test_nested_inheritance_different_single_scope_and_host_restoration(tmp_path):
    original_factory = jax_cache.get_file_cache
    original_load = jax_cache.get_executable_and_time
    original_compile = compiler.backend_compile_and_load
    original_directory = jax.config.jax_compilation_cache_dir
    shared, private, other = (tmp_path / name for name in ("a", "private", "b"))
    with pytest.raises(RuntimeError, match="unwind"):
        with cache.compilation_cache_scope(shared, private_directory=private):
            active_factory = jax_cache.get_file_cache
            with cache.compilation_cache_scope(shared, min_entry_size_bytes=-1):
                assert jax_cache.get_file_cache is active_factory
                assert current_cache_locations().private == private
                _adapter().put("outer", b"a")
            with cache.compilation_cache_scope(other):
                _adapter().put("inner", b"b")
                assert (other / "inner-cache").read_bytes() == b"b"
                assert not (shared / "inner-cache").exists()
            with cache.compilation_cache_scope(None):
                assert jax.config.jax_compilation_cache_dir is None
                assert current_cache_locations().private is None
            assert current_cache_locations().private == private
            _adapter().put("after", b"a")
            raise RuntimeError("unwind")
    assert jax_cache.get_file_cache is original_factory
    assert jax_cache.get_executable_and_time is original_load
    assert compiler.backend_compile_and_load is original_compile
    assert jax.config.jax_compilation_cache_dir == original_directory
    assert current_cache_locations() is None


def test_contents_verification_and_maximum_size_are_preserved(tmp_path):
    old_check = jax.config.jax_compilation_cache_check_contents
    old_size = jax.config.jax_compilation_cache_max_size
    try:
        jax.config.update("jax_compilation_cache_check_contents", True)
        jax.config.update("jax_compilation_cache_max_size", 8)
        with cache.compilation_cache_scope(
            tmp_path / "shared", private_directory=tmp_path / "private"
        ):
            adapter = _adapter()
            assert isinstance(adapter, jax_cache.VerificationCache)
            base = adapter._base_cache
            base.put("first", b"123456")
            base.put("second", b"abcdef")
            assert base.get("first") is None
            assert base.get("second") == b"abcdef"
            base.put("too-large", b"x" * 9)
            assert base.get("too-large") is None
            old = jax_cache.compress_executable(
                jax_cache.combine_executable_and_time(b"old", 1)
            )
            new = jax_cache.compress_executable(
                jax_cache.combine_executable_and_time(b"new", 1)
            )
            base.max_size = -1
            base.put("check", old)
            assert adapter.get("check") is None
            with pytest.raises(jax_cache.CacheVerificationError):
                adapter.put("check", new)
    finally:
        jax.config.update("jax_compilation_cache_check_contents", old_check)
        jax.config.update("jax_compilation_cache_max_size", old_size)


def test_bounded_writable_shared_cache_keeps_a_recently_read_entry(tmp_path):
    old_size = jax.config.jax_compilation_cache_max_size
    try:
        jax.config.update("jax_compilation_cache_max_size", 8)
        shared, private = tmp_path / "shared", tmp_path / "private"
        with cache.compilation_cache_scope(shared, private_directory=private):
            adapter = _adapter()
            adapter.put("a", b"aaaa")
            adapter.put("b", b"bbbb")
            assert adapter.get("a") == b"aaaa"
            adapter.put("c", b"cccc")
            assert adapter.get("a") == b"aaaa"
            assert adapter.get("b") is None
            assert adapter.get("c") == b"cccc"
    finally:
        jax.config.update("jax_compilation_cache_max_size", old_size)


_PROCESS = textwrap.dedent("""
    import json,os,sys
    from pathlib import Path
    import jax,jax.numpy as jnp,numpy as np
    from foldjax import cache
    shared,private,size=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3])
    if sys.argv[4]=='readonly' and os.geteuid()==0:
        original=cache.os.access
        def access(path,mode):
            return False if Path(path)==shared else original(path,mode)
        cache.os.access=access
    cache.clear_record()
    @jax.jit
    def layered_probe(x):
        return jnp.sin(x)+x*2
    with cache.compilation_cache_scope(
        shared,private_directory=private,min_entry_size_bytes=-1
    ):
        value=np.arange(size,dtype=np.float32)
        result=layered_probe.lower(value).compile()(value)
        np.testing.assert_allclose(np.asarray(result),np.sin(value)+value*2,rtol=1e-6)
    print(json.dumps(cache.recorded()))
""")


def _process(shared, private, size=16, readonly=False):
    environment = dict(os.environ, JAX_PLATFORMS="cpu")
    previous_mode = stat.S_IMODE(shared.stat().st_mode) if readonly else None
    try:
        if readonly:
            shared.chmod(0o555)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _PROCESS,
                str(shared),
                str(private),
                str(size),
                "readonly" if readonly else "write",
            ],
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
        )
    finally:
        if previous_mode is not None:
            shared.chmod(previous_mode)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_fresh_cpu_processes_load_shared_then_persist_and_reuse_private(tmp_path):
    shared, private = tmp_path / "shared", tmp_path / "private"
    first = _process(shared, private)
    assert first["writes"]["shared"]["success"] >= 1
    assert first["compiles"]["success"] >= 1
    shared_files = {p.name: p.read_bytes() for p in shared.glob("*-cache")}
    second = _process(shared, private, readonly=True)
    assert second["loads"]["shared"]["success"] >= 1
    assert not list(private.glob("*-cache"))
    missing = _process(shared, private, 32, readonly=True)
    assert missing["writes"]["private"]["success"] >= 1
    assert missing["compiles"]["success"] >= 1
    reused = _process(shared, private, 32, readonly=True)
    assert reused["loads"]["private"]["success"] >= 1
    assert not reused["compiles"]
    assert shared_files == {p.name: p.read_bytes() for p in shared.glob("*-cache")}


def test_corrupt_shared_executable_retries_a_valid_private_entry(tmp_path):
    shared, private = tmp_path / "shared", tmp_path / "private"
    _process(shared, private)
    files = list(shared.glob("*-cache"))
    assert files
    for source in files:
        target = private / source.name
        target.write_bytes(source.read_bytes())
        target.chmod(0o600)
        source.write_bytes(b"corrupt serialized executable")
    result = _process(shared, private, readonly=True)
    assert result["loads"]["shared"]["error"] >= 1
    assert result["loads"]["private"]["success"] >= 1
    assert not result["compiles"]
