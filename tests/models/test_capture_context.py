from __future__ import annotations

import threading

import pytest

from foldjax.models import _capture


def test_crossed_thread_captures_do_not_overwrite_or_leak_context() -> None:
    first_entered = threading.Event()
    second_entered = threading.Event()
    first_exited = threading.Event()
    results: dict[str, tuple[frozenset[str], dict[str, object]]] = {}
    errors: list[BaseException] = []

    def first() -> None:
        try:
            with _capture.capturing(("first",)) as captured:
                first_entered.set()
                assert second_entered.wait(timeout=2)
                _capture.capture("first", 1)
                _capture.capture("second", "wrong")
                results["first"] = (_capture.wanted(), dict(captured))
            first_exited.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def second() -> None:
        try:
            assert first_entered.wait(timeout=2)
            with _capture.capturing(("second",)) as captured:
                second_entered.set()
                assert first_exited.wait(timeout=2)
                _capture.capture("second", 2)
                _capture.capture("first", "wrong")
                results["second"] = (_capture.wanted(), dict(captured))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    assert results == {
        "first": (frozenset({"first"}), {"first": 1}),
        "second": (frozenset({"second"}), {"second": 2}),
    }
    assert _capture.wanted() == frozenset()
    assert _capture.collected() == {}


def test_nested_and_failed_capture_restores_the_outer_context() -> None:
    with _capture.capturing(("outer",)) as outer:
        _capture.capture("outer", 1)
        with pytest.raises(RuntimeError, match="synthetic"):
            with _capture.capturing(("inner",)) as inner:
                _capture.capture("inner", 2)
                assert inner == {"inner": 2}
                raise RuntimeError("synthetic")
        assert _capture.wanted() == frozenset({"outer"})
        _capture.capture("outer", 3)
        assert outer == {"outer": 3}

    assert _capture.wanted() == frozenset()
    assert _capture.collected() == {}
