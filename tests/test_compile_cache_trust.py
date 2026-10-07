"""A compile cache is used only when no other account can write into it.

JAX loads whatever executable it finds under a key, so a cache directory that
another account can write is a way to run that account's code in this one.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from foldjax import cache
from foldjax.cache import (
    TRUST_SHARED_COMPILE_CACHE_ENV,
    compilation_cache_scope,
    trusted_compile_cache_dir,
)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRUST_SHARED_COMPILE_CACHE_ENV, raising=False)
    monkeypatch.setattr(cache, "_WARNED_CACHE_DIRS", set())


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_a_private_cache_is_created_without_group_write(
    tmp_path: Path, trust_ancestors_above_tmp_path
) -> None:
    target = tmp_path / "compile" / "boltz2" / "weights" / "digest"
    previous = os.umask(0o002)
    try:
        assert trusted_compile_cache_dir(target) == target
    finally:
        os.umask(previous)
    for directory in (target, *list(target.parents)[:3]):
        assert not _mode(directory) & 0o022, directory


def test_a_world_writable_cache_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "compile"
    target.mkdir()
    target.chmod(0o777)
    with pytest.warns(RuntimeWarning, match="world-writable"):
        assert trusted_compile_cache_dir(target / "boltz2") is None


def test_a_cache_below_a_shared_group_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "store"
    shared.mkdir()
    shared.chmod(0o2775)
    monkeypatch.setattr(cache, "_group_is_only_this_user", lambda gid: False)
    with pytest.warns(RuntimeWarning, match="group with other members"):
        assert trusted_compile_cache_dir(shared / "compile") is None


def test_a_private_group_may_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    store.chmod(0o775)
    monkeypatch.setattr(cache, "_group_is_only_this_user", lambda gid: True)
    assert trusted_compile_cache_dir(store / "compile") == store / "compile"


def test_a_cache_owned_by_another_user_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache.os, "geteuid", lambda: os.getuid() + 1)
    with pytest.warns(RuntimeWarning, match="belongs to another user"):
        assert trusted_compile_cache_dir(tmp_path / "compile") is None


def test_a_symlink_into_a_writable_directory_is_judged_by_its_target(
    tmp_path: Path,
) -> None:
    open_dir = tmp_path / "open"
    open_dir.mkdir()
    open_dir.chmod(0o777)
    link = tmp_path / "compile"
    link.symlink_to(open_dir, target_is_directory=True)
    with pytest.warns(RuntimeWarning, match="world-writable"):
        assert trusted_compile_cache_dir(link) is None


def test_the_environment_opts_in_to_a_shared_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "compile"
    target.mkdir()
    target.chmod(0o777)
    monkeypatch.setenv(TRUST_SHARED_COMPILE_CACHE_ENV, "1")
    assert trusted_compile_cache_dir(target / "boltz2") == target / "boltz2"


def test_a_refused_cache_creates_no_namespace(tmp_path: Path) -> None:
    """The decision comes first: a refused store gains no empty directories.

    They used to be made 0755 before the refusal, and a later opt-in run by
    another member of the group then could not write into them.
    """
    shared = tmp_path / "compile"
    shared.mkdir()
    shared.chmod(0o777)
    with pytest.warns(RuntimeWarning, match="world-writable"):
        assert trusted_compile_cache_dir(shared / "boltz2" / "w" / "digest") is None
    assert not (shared / "boltz2").exists()


def test_a_shared_store_namespace_takes_the_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "compile"
    store.mkdir()
    store.chmod(0o2775)
    monkeypatch.setenv(TRUST_SHARED_COMPILE_CACHE_ENV, "1")
    target = store / "boltz2" / "w" / "digest"
    previous = os.umask(0o002)
    try:
        assert trusted_compile_cache_dir(target) == target
    finally:
        os.umask(previous)
    for directory in (target, target.parent, target.parent.parent):
        assert _mode(directory) & 0o2070 == 0o2070, directory


def test_a_shared_store_that_cannot_be_created_compiles_without_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another account's 0755 namespace: slower, never a crash."""
    theirs = tmp_path / "compile"
    theirs.mkdir()
    theirs.chmod(0o555)
    monkeypatch.setenv(TRUST_SHARED_COMPILE_CACHE_ENV, "1")
    try:
        with pytest.warns(RuntimeWarning, match="cannot be created"):
            assert trusted_compile_cache_dir(theirs / "boltz2") is None
    finally:
        theirs.chmod(0o755)


def test_a_shared_namespace_this_account_cannot_write_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "compile" / "boltz2"
    namespace.mkdir(parents=True)
    namespace.chmod(0o555)
    monkeypatch.setenv(TRUST_SHARED_COMPILE_CACHE_ENV, "1")
    try:
        with pytest.warns(RuntimeWarning, match="chmod -R g\\+w"):
            assert trusted_compile_cache_dir(namespace) == namespace
    finally:
        namespace.chmod(0o755)


def test_the_trust_verdict_is_read_without_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "compile"
    shared.mkdir()
    shared.chmod(0o777)
    target = shared / "boltz2" / "w"
    assert "world-writable" in cache.compile_cache_untrusted_reason(target)
    assert not (shared / "boltz2").exists()
    monkeypatch.setenv(TRUST_SHARED_COMPILE_CACHE_ENV, "1")
    assert cache.compile_cache_untrusted_reason(target) is None


def test_the_scope_compiles_without_a_cache_it_refuses(
    tmp_path: Path, trust_ancestors_above_tmp_path
) -> None:
    import jax

    target = tmp_path / "compile"
    target.mkdir()
    target.chmod(0o777)
    with (
        pytest.warns(RuntimeWarning, match="FOLDJAX_TRUST_SHARED_COMPILE_CACHE"),
        compilation_cache_scope(target / "ns"),
    ):
        assert jax.config.jax_compilation_cache_dir is None
    trusted = tmp_path / "private"
    with compilation_cache_scope(trusted):
        assert jax.config.jax_compilation_cache_dir == str(trusted)


def test_one_directory_warns_once(tmp_path: Path) -> None:
    import warnings

    target = tmp_path / "compile"
    target.mkdir()
    target.chmod(0o777)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trusted_compile_cache_dir(target)
        trusted_compile_cache_dir(target)
    assert len(caught) == 1


def test_no_backend_latches_the_compile_cache_outside_the_shared_scope() -> None:
    """Every in-process run selects its cache through `compilation_cache_scope`.

    A raw `jax.config.update` of the directory skips the trust check's
    refusal path and leaves the setting in force after the call. Allowed:
    the scope itself, OpenFold3's documented process-wide opt-in for direct
    library callers, and upstream's byte-identical runner.
    """
    import foldjax

    root = Path(foldjax.__file__).parent
    allowed = {
        root / "cache.py",
        root / "models" / "openfold3" / "compilation.py",
        root / "backends" / "_alphafold3_upstream" / "run_alphafold.py",
    }
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path not in allowed
        and '"jax_compilation_cache_dir", str(' in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_boltz2_predict_scopes_its_compile_cache(
    tmp_path: Path, trust_ancestors_above_tmp_path
) -> None:
    import jax

    from foldjax.models.boltz2.api import _scoped_compile_cache

    seen: list[object] = []

    @_scoped_compile_cache
    def run(*, compile_cache=None):
        seen.append(jax.config.jax_compilation_cache_dir)

    before = jax.config.jax_compilation_cache_dir
    run(compile_cache=tmp_path / "private")
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    with pytest.warns(RuntimeWarning, match="world-writable"):
        run(compile_cache=shared / "ns")
    run()
    assert seen == [str((tmp_path / "private").resolve()), None, before]
    assert jax.config.jax_compilation_cache_dir == before
