"""Search caches survive concurrent runs, damaged entries and shared stores.

Publishing renamed a staged directory onto the entry and caught only
``FileExistsError``; a non-empty winner raises ``ENOTEMPTY``, so the second of
two runs on one sequence failed. Nothing stopped both from asking the server.
A damaged entry failed every later run of its sequence, and entries were
published 0700 whatever the umask.
"""

from __future__ import annotations

import os
import shutil
import stat
import threading
import time
from pathlib import Path

import pytest

from foldjax.search import MsaPayload, MsaSearchPipeline
from foldjax.search.templates import (
    StructureStore,
    TemplateHitsPayload,
    TemplateHitsPipeline,
)

SEQUENCE = "MQIFVKTLTGKTITLEVEPSD"
M8 = "101\t1abc_A\t1.0\t20\t0\t0\t1\t20\t1\t20\t1e-30\t200"


class _Backend:
    name = "fake"
    version = "1"

    def __init__(self, before_return=None, delay: float = 0.0) -> None:
        self.calls = 0
        self.before_return = before_return
        self.delay = delay

    def search(self, sequence: str) -> MsaPayload:
        self.calls += 1
        time.sleep(self.delay)
        if self.before_return is not None:
            self.before_return()
        a3m = f">query\n{sequence}\n>hit\n{sequence}\n"
        return MsaPayload(a3m, a3m)


def _entries(cache: Path) -> list[Path]:
    return [path for path in cache.iterdir() if not path.name.startswith(".")]


def test_losing_a_publish_race_reads_the_winners_entry(tmp_path: Path) -> None:
    winner_cache = tmp_path / "winner"
    (winner,) = MsaSearchPipeline(winner_cache, _Backend()).search([SEQUENCE])
    entry = Path(winner["provenancePath"]).parent
    cache = tmp_path / "cache"

    def another_run_publishes_first() -> None:
        shutil.copytree(entry, cache / entry.name)

    backend = _Backend(before_return=another_run_publishes_first)
    (result,) = MsaSearchPipeline(cache, backend).search([SEQUENCE])

    assert Path(result["provenancePath"]).parent == (cache / entry.name).resolve()
    assert not [p for p in cache.iterdir() if p.name.startswith(f".{entry.name}.")
                and not p.name.endswith(".lock")]


def test_concurrent_runs_on_one_sequence_ask_the_server_once(tmp_path: Path) -> None:
    backend = _Backend(delay=0.3)
    pipeline = MsaSearchPipeline(tmp_path, backend)
    results: list[object] = []
    threads = [
        threading.Thread(target=lambda: results.append(pipeline.search([SEQUENCE])))
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert backend.calls == 1
    assert len(results) == 3 and all(result == results[0] for result in results)


def test_a_damaged_entry_is_set_aside_and_searched_again(tmp_path: Path) -> None:
    backend = _Backend()
    pipeline = MsaSearchPipeline(tmp_path, backend)
    (first,) = pipeline.search([SEQUENCE])
    entry = Path(first["provenancePath"]).parent
    Path(first["unpairedMsaPath"]).write_text(f">query\n{SEQUENCE}\n")

    with pytest.warns(UserWarning, match=str(entry)):
        (second,) = pipeline.search([SEQUENCE])

    assert backend.calls == 2
    assert second == first
    assert (tmp_path / f".{entry.name}.damaged").is_dir()
    assert pipeline.search([SEQUENCE]) == [first] and backend.calls == 2


def test_an_entry_is_published_with_the_umask(tmp_path: Path) -> None:
    previous = os.umask(0o002)
    try:
        (result,) = MsaSearchPipeline(tmp_path, _Backend()).search([SEQUENCE])
        hits = TemplateHitsPipeline(
            tmp_path / "hits", _HitsBackend(), options={}
        ).search(SEQUENCE)
    finally:
        os.umask(previous)
    for path in (Path(result["provenancePath"]), Path(hits["hitsPath"])):
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o775
        assert stat.S_IMODE(path.stat().st_mode) == 0o664


class _HitsBackend:
    name = "fake-hits"
    version = "1"

    def __init__(self) -> None:
        self.calls = 0

    def search(self, sequence: str) -> TemplateHitsPayload:
        self.calls += 1
        return TemplateHitsPayload(M8)


def test_a_damaged_template_hits_entry_is_searched_again(tmp_path: Path) -> None:
    backend = _HitsBackend()
    pipeline = TemplateHitsPipeline(tmp_path, backend)
    first = pipeline.search(SEQUENCE)
    Path(first["hitsPath"]).write_text("")
    with pytest.warns(UserWarning, match="damaged"):
        assert pipeline.search(SEQUENCE) == first
    assert backend.calls == 2


def test_a_failed_structure_publish_leaves_no_debris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foldjax.search.msa import HttpResponse

    store = StructureStore(
        tmp_path,
        transport=lambda *a: HttpResponse(
            200, b"data_1ABC\nloop_\n_atom_site.id\n1\n"
        ),
    )

    def disk_full(*_args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("foldjax.search.templates.os.replace", disk_full)
    with pytest.raises(OSError, match="No space"):
        store.path("1abc")
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
