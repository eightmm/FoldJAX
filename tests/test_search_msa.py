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

    # Detected, and the damaged entry is set aside and searched again rather
    # than failing every later run of the sequence.
    with pytest.warns(UserWarning, match="UnicodeDecodeError|can't decode"):
        again = pipeline.search([SEQUENCE])[0]
    assert backend.calls == [SEQUENCE, SEQUENCE]
    assert Path(again["pairedMsaPath"]).read_bytes() == raw[:-1]


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


# Complex pairing: OpenFold3 v0.5.0 pairs a heteromer in one ColabFold job.

OTHER = "GSHMLEDPVDAFQLGKVLNQ"


class _ComplexBackend(_Backend):
    """A backend that can also pair a complex, row-aligned across its chains."""

    def __init__(self) -> None:
        super().__init__()
        self.complex_calls: list[list[str]] = []

    def search_complex(self, sequences):
        from foldjax.search import ComplexPairPayload

        self.complex_calls.append(list(sequences))
        return ComplexPairPayload(
            tuple(
                f">{101 + i}\n{s}\n>hit\n{'A' * len(s)}\n"
                for i, s in enumerate(sequences)
            )
        )


def test_colabfold_pair_a3m_splits_on_the_nul_separator_only() -> None:
    """Upstream's gather loop: only the header after a NUL (or the first) is a query."""
    from foldjax.search.msa import _split_colabfold_a3m

    text = (
        ">101\nAAAA\n>UniRef100_X\t123\nAAAC\n"
        "\x00>102\nCCCC\n>UniRef100_Y\t456\nCCCA\n"
    )
    assert _split_colabfold_a3m(text, "paired MSA") == {
        101: ">101\nAAAA\n>UniRef100_X\t123\nAAAC\n",
        102: ">102\nCCCC\n>UniRef100_Y\t456\nCCCA\n",
    }


def test_remote_complex_pairing_is_one_pairgreedy_job() -> None:
    """One ``ticket/pair`` job, mode ``pairgreedy-env``, queries 101 and 102."""
    import io
    import tarfile
    import urllib.parse

    from foldjax.search.msa import HttpResponse, RemoteMMseqs2Client

    pair = f">101\n{SEQUENCE}\n>h\t1\n{SEQUENCE}\n\x00>102\n{OTHER}\n>h\t1\n{OTHER}\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        raw = pair.encode()
        info = tarfile.TarInfo("pair.a3m")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    requests: list[tuple[str, str, bytes | None]] = []

    def transport(method, url, data, _headers, _timeout):
        requests.append((method, url, data))
        if url.endswith("ticket/pair"):
            return HttpResponse(200, b'{"status":"COMPLETE","id":"p-1"}')
        return HttpResponse(200, buffer.getvalue())

    client = RemoteMMseqs2Client(
        "https://msa.invalid", version="api-v1", poll_interval=0, transport=transport
    )
    payload = client.search_complex([SEQUENCE, OTHER])

    (method, url, data), download = requests
    assert (method, url) == ("POST", "https://msa.invalid/ticket/pair")
    assert urllib.parse.parse_qs(data.decode()) == {
        "q": [f">101\n{SEQUENCE}\n>102\n{OTHER}\n"],
        "mode": ["pairgreedy-env"],
    }
    assert download[1] == "https://msa.invalid/result/download/p-1"
    assert payload.paired == (
        f">101\n{SEQUENCE}\n>h\t1\n{SEQUENCE}\n",
        f">102\n{OTHER}\n>h\t1\n{OTHER}\n",
    )


def test_complex_pairing_is_cached_and_maps_repeats(tmp_path: Path) -> None:
    backend = _ComplexBackend()
    pipeline = MsaSearchPipeline(cache_dir=tmp_path, backend=backend)

    first = pipeline.search_complex([SEQUENCE, OTHER, SEQUENCE])
    assert backend.complex_calls == [[SEQUENCE, OTHER]]
    assert first[0] == first[2] and first[0] != first[1]
    assert Path(first[1]["pairedMsaPath"]).read_text().startswith(f">102\n{OTHER}\n")

    assert pipeline.search_complex([SEQUENCE, OTHER, SEQUENCE]) == first
    assert backend.complex_calls == [[SEQUENCE, OTHER]], "the cache was not used"
    # The per-sequence cache is a different identity and is untouched.
    assert backend.calls == []


def test_complex_pairing_needs_two_sequences_and_a_capable_backend(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="two distinct"):
        MsaSearchPipeline(tmp_path / "a", _ComplexBackend()).search_complex(
            [SEQUENCE, SEQUENCE]
        )
    plain = MsaSearchPipeline(tmp_path / "b", _Backend())
    assert not plain.pairs_complexes
    with pytest.raises(SearchError, match="cannot pair a complex"):
        plain.search_complex([SEQUENCE, OTHER])


def test_a_complex_block_for_the_wrong_query_is_refused(tmp_path: Path) -> None:
    class _Swapped(_ComplexBackend):
        def search_complex(self, sequences):
            payload = super().search_complex(sequences)
            return type(payload)(tuple(reversed(payload.paired)))

    with pytest.raises(SearchError, match="does not match"):
        MsaSearchPipeline(tmp_path, _Swapped()).search_complex([SEQUENCE, OTHER])
