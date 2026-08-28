from __future__ import annotations

import threading

import pytest

from foldjax.models import _managed_memory


def test_nested_owners_clear_collect_and_trim_only_after_the_last_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def release() -> bool:
        events.append("clear")
        return True

    monkeypatch.setattr(
        _managed_memory.gc, "collect", lambda: events.append("collect") or 0
    )
    monkeypatch.setattr(
        _managed_memory, "_malloc_trim", lambda: events.append("trim")
    )

    with _managed_memory.lease("shared", release):
        with _managed_memory.lease("shared", release):
            assert events == []
        assert events == []

    assert events == ["clear", "collect", "trim"]


def test_an_unloaded_cache_skips_collection_and_trim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        _managed_memory.gc, "collect", lambda: events.append("collect") or 0
    )
    monkeypatch.setattr(
        _managed_memory, "_malloc_trim", lambda: events.append("trim")
    )

    with _managed_memory.lease(
        "empty", lambda: events.append("clear") or False
    ):
        pass

    assert events == ["clear"]


def test_cleanup_failures_never_replace_the_prediction_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PredictionInterrupted(BaseException):
        pass

    def loaded_release() -> bool:
        return True

    def broken_collect() -> int:
        raise RuntimeError("collect failed")

    monkeypatch.setattr(_managed_memory.gc, "collect", broken_collect)
    monkeypatch.setattr(
        _managed_memory,
        "_malloc_trim",
        lambda: (_ for _ in ()).throw(RuntimeError("trim failed")),
    )

    with pytest.raises(PredictionInterrupted):
        with _managed_memory.lease("broken", loaded_release):
            raise PredictionInterrupted


def test_release_failure_is_swallowed_without_running_later_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_release() -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        _managed_memory.gc,
        "collect",
        lambda: pytest.fail("unknown loaded state must not be collected"),
    )
    monkeypatch.setattr(
        _managed_memory,
        "_malloc_trim",
        lambda: pytest.fail("unknown loaded state must not be trimmed"),
    )

    with _managed_memory.lease("release-error", broken_release):
        pass


def test_one_key_rejects_two_release_helpers() -> None:
    first_calls = []

    def first() -> bool:
        first_calls.append("clear")
        return False

    def second() -> bool:
        raise AssertionError("the conflicting helper must never run")

    with _managed_memory.lease("collision", first):
        with pytest.raises(RuntimeError, match="two release helpers"):
            with _managed_memory.lease("collision", second):
                pass

    assert first_calls == ["clear"]


def test_loaded_cleanup_swallows_an_allocator_trim_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_managed_memory.gc, "collect", lambda: 0)
    monkeypatch.setattr(
        _managed_memory,
        "_malloc_trim",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with _managed_memory.lease("trim-error", lambda: True):
        pass


def test_acquire_waits_until_cleanup_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    first_entered = threading.Event()
    leave_first = threading.Event()
    cleanup_started = threading.Event()
    finish_cleanup = threading.Event()
    second_entered = threading.Event()
    errors: list[BaseException] = []
    releases = 0

    def release() -> bool:
        nonlocal releases
        releases += 1
        if releases == 1:
            cleanup_started.set()
            if not finish_cleanup.wait(5):
                raise TimeoutError("cleanup test did not resume")
        return False

    monkeypatch.setattr(_managed_memory.gc, "collect", lambda: 0)

    def first_owner() -> None:
        try:
            with _managed_memory.lease("concurrent", release):
                first_entered.set()
                if not leave_first.wait(5):
                    raise TimeoutError("first owner did not exit")
        except BaseException as error:
            errors.append(error)

    def second_owner() -> None:
        try:
            with _managed_memory.lease("concurrent", release):
                second_entered.set()
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=first_owner)
    first.start()
    assert first_entered.wait(5)
    leave_first.set()
    assert cleanup_started.wait(5)

    second = threading.Thread(target=second_owner)
    second.start()
    assert not second_entered.wait(0.05)
    finish_cleanup.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert second_entered.is_set()
    assert releases == 2


def test_malloc_trim_is_a_noop_when_platform_or_symbol_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_managed_memory.sys, "platform", "darwin")
    monkeypatch.setattr(
        _managed_memory.ctypes,
        "CDLL",
        lambda *_args: pytest.fail("non-Linux must not open libc"),
    )
    _managed_memory._malloc_trim()

    monkeypatch.setattr(_managed_memory.sys, "platform", "linux")
    monkeypatch.setattr(_managed_memory.platform, "libc_ver", lambda: ("musl", "1"))
    _managed_memory._malloc_trim()

    monkeypatch.setattr(
        _managed_memory.platform, "libc_ver", lambda: ("glibc", "2.40")
    )
    monkeypatch.setattr(_managed_memory.ctypes, "CDLL", lambda *_args: object())
    _managed_memory._malloc_trim()

    monkeypatch.setattr(
        _managed_memory.ctypes,
        "CDLL",
        lambda *_args: (_ for _ in ()).throw(OSError("unavailable")),
    )
    _managed_memory._malloc_trim()

    monkeypatch.setattr(
        _managed_memory.platform,
        "libc_ver",
        lambda: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    _managed_memory._malloc_trim()
