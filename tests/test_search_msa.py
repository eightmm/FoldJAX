"""The shared MSA pipeline, with a fake backend so nothing touches the network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from foldjax.search import MsaPayload, MsaSearchPipeline, SearchError

SEQUENCE = "MQIFVKTLTGKTITLEVEPSD"


class _Backend:
    """Counts calls, so a cache hit is observable rather than assumed."""

    name = "fake"
    version = "1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, sequence: str, **_options) -> MsaPayload:
        self.calls.append(sequence)
        return MsaPayload(
            paired=f">query\n{sequence}\n>p1\n{sequence}\n",
            unpaired=f">query\n{sequence}\n>u1\n{sequence}\n",
        )


def test_search_writes_alignments_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _Backend()
    pipeline = MsaSearchPipeline(cache_dir=tmp_path, backend=backend)

    first = pipeline.search([SEQUENCE])
    assert len(first) == 1
    paired = Path(first[0]["pairedMsaPath"])
    unpaired = Path(first[0]["unpairedMsaPath"])
    assert paired.name == "pairing.a3m"
    assert unpaired.name == "non_pairing.a3m"
    assert SEQUENCE in paired.read_text()
    assert backend.calls == [SEQUENCE]

    # A second search must be served from the cache.
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: pytest.fail(f"cache validation read the whole file: {path}"),
    )
    second = pipeline.search([SEQUENCE])
    assert second == first
    assert backend.calls == [SEQUENCE], "the backend was called again"


def test_cache_validation_checks_utf8_after_the_first_entry(tmp_path: Path) -> None:
    backend = _Backend()
    pipeline = MsaSearchPipeline(cache_dir=tmp_path, backend=backend)
    result = pipeline.search([SEQUENCE])[0]
    paired = Path(result["pairedMsaPath"])
    raw = paired.read_bytes() + b"\xff"
    paired.write_bytes(raw)
    provenance_path = Path(result["provenancePath"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["files"][paired.name]["sha256"] = hashlib.sha256(raw).hexdigest()
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(UnicodeDecodeError):
        pipeline.search([SEQUENCE])


def test_the_cache_layout_is_fixed(tmp_path: Path) -> None:
    """Filenames are the pipeline's, not the caller's.

    Making them caller-chosen looks free, since the alignments are identical bytes,
    but the cache key excludes them: a rename makes every later cache hit fail
    looking for files the cache never wrote. Consumers that need other names -- and
    OpenFold3 does, since it selects alignment files by stem -- link to them.
    """
    import inspect

    signature = inspect.signature(MsaSearchPipeline.__init__)
    assert "paired_name" not in signature.parameters
    assert "unpaired_name" not in signature.parameters

    result = MsaSearchPipeline(cache_dir=tmp_path, backend=_Backend()).search(
        [SEQUENCE]
    )[0]
    assert Path(result["pairedMsaPath"]).name == "pairing.a3m"
    assert Path(result["unpairedMsaPath"]).name == "non_pairing.a3m"


def test_a_different_backend_misses_the_cache(tmp_path: Path) -> None:
    """The cache key includes backend identity, so results cannot be crossed."""
    first = _Backend()
    MsaSearchPipeline(cache_dir=tmp_path, backend=first).search([SEQUENCE])

    class _Other(_Backend):
        name = "other"

    second = _Other()
    MsaSearchPipeline(cache_dir=tmp_path, backend=second).search([SEQUENCE])
    assert second.calls == [SEQUENCE], "a different backend reused a cached result"


def test_options_are_part_of_the_identity(tmp_path: Path) -> None:
    backend = _Backend()
    MsaSearchPipeline(cache_dir=tmp_path, backend=backend).search([SEQUENCE])
    MsaSearchPipeline(
        cache_dir=tmp_path, backend=backend, options={"pairing": "greedy"}
    ).search([SEQUENCE])
    assert len(backend.calls) == 2, "changed options reused a cached result"


def test_unserializable_options_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON-serializable"):
        MsaSearchPipeline(
            cache_dir=tmp_path, backend=_Backend(), options={"fn": object()}
        )


def test_a_response_whose_query_differs_is_refused(tmp_path: Path) -> None:
    """A backend returning someone else's alignment is worse than an error."""

    class _Wrong(_Backend):
        def search(self, sequence: str, **_options) -> MsaPayload:
            self.calls.append(sequence)
            return MsaPayload(paired=">q\nAAAA\n", unpaired=">q\nAAAA\n")

    pipeline = MsaSearchPipeline(cache_dir=tmp_path, backend=_Wrong())
    with pytest.raises(SearchError, match="does not match requested"):
        pipeline.search([SEQUENCE])


def test_an_empty_response_is_refused(tmp_path: Path) -> None:
    class _Empty(_Backend):
        def search(self, sequence: str, **_options) -> MsaPayload:
            return MsaPayload(paired="", unpaired="")

    with pytest.raises(SearchError, match="is missing"):
        MsaSearchPipeline(cache_dir=tmp_path, backend=_Empty()).search([SEQUENCE])


def test_a_rate_limited_server_is_waited_out_not_failed(monkeypatch) -> None:
    """HTTP 429 is this API saying "at capacity", which is a wait, not an error.

    A public MMseqs2 server is at capacity often. Failing on it ended the search
    for a condition whose whole meaning is that it is temporary, and took the
    queue position of every sequence behind it with it.
    """
    from foldjax.search.msa import HttpResponse, RemoteMMseqs2Client

    slept: list[float] = []
    monkeypatch.setattr("foldjax.search.msa.time.sleep", slept.append)
    responses = iter(
        [
            HttpResponse(429, b""),
            HttpResponse(429, b""),
            HttpResponse(200, b'{"status":"COMPLETE","id":"u-1"}'),
        ]
    )
    client = RemoteMMseqs2Client(
        "https://msa.invalid",
        version="api-v1",
        poll_interval=2.0,
        transport=lambda *_: next(responses),
    )

    assert client._json("POST", "ticket/msa")["status"] == "COMPLETE"
    assert slept == [2.0, 4.0], "the wait has to back off, not hammer"


def test_a_server_that_never_recovers_still_ends_the_search(monkeypatch) -> None:
    from foldjax.search.msa import HttpResponse, RemoteMMseqs2Client

    monkeypatch.setattr("foldjax.search.msa.time.sleep", lambda _: None)
    client = RemoteMMseqs2Client(
        "https://msa.invalid",
        version="api-v1",
        poll_interval=1.0,
        max_wait_seconds=3.0,
        transport=lambda *_: HttpResponse(429, b""),
    )
    with pytest.raises(SearchError, match="rate-limited for the whole"):
        client._json("POST", "ticket/msa")
