"""Untrusted shared entries cannot shadow private entries or expose partial writes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import jax
import pytest
from jax._src import compilation_cache as jax_cache

from foldjax import cache


@pytest.mark.parametrize("problem", ["owner", "writable", "hardlink"])
@pytest.mark.parametrize("trusted", [False, True])
def test_shared_entry_fstat_trust_and_explicit_opt_in(
    monkeypatch, tmp_path, problem, trusted
):
    if trusted:
        monkeypatch.setenv(cache.TRUST_SHARED_COMPILE_CACHE_ENV, "1")
    else:
        monkeypatch.delenv(cache.TRUST_SHARED_COMPILE_CACHE_ENV, raising=False)
    shared, private = tmp_path / "shared", tmp_path / "private"
    cache.clear_record()
    try:
        with cache.compilation_cache_scope(shared, private_directory=private):
            adapter, _ = jax_cache.get_file_cache(jax.config.jax_compilation_cache_dir)
            adapter.put("entry", b"shared")
            fallback = private / "entry-cache"
            fallback.write_bytes(b"private")
            fallback.chmod(0o600)
            inode = (shared / "entry-cache").stat().st_ino
            original = os.fstat

            def fstat(fd):
                info = original(fd)
                if info.st_ino != inode:
                    return info
                return SimpleNamespace(
                    st_mode=info.st_mode | (0o022 if problem == "writable" else 0),
                    st_uid=info.st_uid + (1048576 if problem == "owner" else 0),
                    st_gid=info.st_gid,
                    st_nlink=2 if problem == "hardlink" else info.st_nlink,
                )

            monkeypatch.setattr(os, "fstat", fstat)
            if trusted and problem != "hardlink":
                assert adapter.get("entry") == b"shared"
            else:
                with pytest.warns(
                    RuntimeWarning, match="shared compilation cache read"
                ):
                    assert adapter.get("entry") == b"private"
    finally:
        cache.clear_record()


@pytest.mark.parametrize("suffix", ["", "new/model/profile"])
def test_unsafe_shared_alias_does_not_suppress_a_safe_private_fallback(
    monkeypatch, tmp_path, suffix
):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(private, target_is_directory=True)
    monkeypatch.setenv(cache.TRUST_SHARED_COMPILE_CACHE_ENV, "1")
    cache.clear_record()
    try:
        with pytest.warns(RuntimeWarning, match="safe directory access failed"):
            with cache.compilation_cache_scope(
                alias / suffix, private_directory=private
            ):
                adapter, _ = jax_cache.get_file_cache(
                    jax.config.jax_compilation_cache_dir
                )
                adapter.put("entry", b"private")
                assert adapter.get("entry") == b"private"
        assert cache.recorded()["locations"] == {
            "shared": None,
            "private": str(private),
        }
        assert (private / "entry-cache").stat().st_mode & 0o777 == 0o600
        assert not (private / "new").exists()
    finally:
        cache.clear_record()


_CONCURRENT = textwrap.dedent(
    """
    import json, sys, time
    from pathlib import Path
    import jax
    from jax._src import compilation_cache as jax_cache
    from foldjax import cache

    root, role = Path(sys.argv[1]), sys.argv[2]
    cache.clear_record()
    with cache.compilation_cache_scope(root/'shared', private_directory=root/role):
        adapter, _ = jax_cache.get_file_cache(jax.config.jax_compilation_cache_dir)
        (root/f'ready-{role}').touch()
        deadline = time.monotonic()+15
        while len(list(root.glob('ready-*'))) != 3:
            if time.monotonic() > deadline:
                raise RuntimeError('publication test barrier timed out')
            time.sleep(0.01)
        def payload(key):
            return (str(key).encode()+b'!')*32768
        if role != 'reader':
            for key in range(12):
                adapter.put(str(key), payload(key))
        seen = set()
        while len(seen) != 12:
            for key in range(12):
                value = adapter.get(str(key))
                if value is not None:
                    assert value == payload(key), 'partial or mixed cache entry'
                    seen.add(key)
            if time.monotonic() > deadline:
                raise RuntimeError('publication reader timed out')
            time.sleep(0.01)
    print(json.dumps({'seen':len(seen), 'record':cache.recorded()}))
    """
)


def test_concurrent_writers_and_reader_only_observe_complete_entries(tmp_path):
    processes = []
    try:
        for role in ("writer-a", "writer-b", "reader"):
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", _CONCURRENT, str(tmp_path), role],
                    env=dict(os.environ, JAX_PLATFORMS="cpu"),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )
        results = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stdout + stderr
            results.append(json.loads(stdout.strip().splitlines()[-1]))
        assert all(result["seen"] == 12 for result in results)
        assert (
            sum(
                result["record"]["writes"].get("shared", {}).get("success", 0)
                for result in results
            )
            == 12
        )
        assert all(
            not list((tmp_path / role).glob("*-cache"))
            for role in ("writer-a", "writer-b", "reader")
        )
        assert not list((tmp_path / "shared").glob("*.tmp"))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
