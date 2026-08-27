from __future__ import annotations

import pytest


def test_kalign_cache_is_bounded_and_reuses_recent_alignment(monkeypatch):
    pytest.importorskip("kalign")

    from foldjax.models.openfold3._upstream.openfold3.core.data.tools import (
        kalign as kalign_tool,
    )

    calls: list[tuple[str, ...]] = []

    def fake_align(sequences: list[str]) -> list[str]:
        calls.append(tuple(sequences))
        return [sequence[::-1] for sequence in sequences]

    monkeypatch.setattr(kalign_tool.kalign, "align", fake_align)
    kalign_tool._run_kalign_cached.cache_clear()
    try:
        first = ["QUERY-0", "TEMPLATE-0"]
        expected = kalign_tool.run_kalign(first)
        assert kalign_tool.run_kalign(first) is expected
        assert calls == [tuple(first)]

        for index in range(1, kalign_tool._KALIGN_CACHE_LIMIT + 1):
            kalign_tool.run_kalign([f"QUERY-{index}", f"TEMPLATE-{index}"])

        info = kalign_tool._run_kalign_cached.cache_info()
        assert info.maxsize == kalign_tool._KALIGN_CACHE_LIMIT
        assert info.currsize == kalign_tool._KALIGN_CACHE_LIMIT

        # The oldest entry was evicted, while a recent retry remains reusable.
        kalign_tool.run_kalign(first)
        assert calls.count(tuple(first)) == 2
        recent = [
            f"QUERY-{kalign_tool._KALIGN_CACHE_LIMIT}",
            f"TEMPLATE-{kalign_tool._KALIGN_CACHE_LIMIT}",
        ]
        before = len(calls)
        kalign_tool.run_kalign(recent)
        assert len(calls) == before
    finally:
        kalign_tool._run_kalign_cached.cache_clear()
