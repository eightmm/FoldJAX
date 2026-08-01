from __future__ import annotations

import io
import json
import tarfile
import urllib.parse
from pathlib import Path

import pytest

from foldjax.models.chai.data.search import (
    HttpResponse,
    LocalMsaBackend,
    MsaPayload,
    MsaSearchPipeline,
    RemoteMMseqs2Backend,
    SearchError,
)


class RecordingBackend:
    name = "recording"
    version = "uniref-2026-07"
    server_url = None

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, sequence: str) -> MsaPayload:
        self.calls.append(sequence)
        return MsaPayload(
            paired=f">query\n{sequence}\n>paired\n{sequence}\n",
            unpaired=f">query\n{sequence}\n>hit\n{sequence[:-1]}-\n",
            source={"database": self.version},
        )


def test_cache_normalizes_deduplicates_and_hits_without_backend(
    tmp_path: Path,
) -> None:
    backend = RecordingBackend()
    pipeline = MsaSearchPipeline(
        tmp_path, backend, options={"mode": "complete", "max_seqs": 2048}
    )

    first = pipeline.search([" acd e\n", "ACDE"])
    second = pipeline.search(["acde"])

    assert backend.calls == ["ACDE"]
    assert first[0] == first[1] == second[0]
    provenance = json.loads(Path(first[0]["provenancePath"]).read_text())
    assert provenance["sequence"] == "ACDE"
    assert provenance["backend"] == {
        "name": "recording",
        "version": "uniref-2026-07",
    }
    assert provenance["options"] == {"max_seqs": 2048, "mode": "complete"}
    for filename in ("pairing.a3m", "non_pairing.a3m"):
        assert provenance["files"][filename]["sha256"]


def test_cache_identity_changes_with_options_backend_and_server(tmp_path: Path) -> None:
    first_backend = RecordingBackend()
    first = MsaSearchPipeline(
        tmp_path, first_backend, options={"mode": "env"}
    ).search(["ACDE"])[0]
    second = MsaSearchPipeline(
        tmp_path, RecordingBackend(), options={"mode": "complete"}
    ).search(["ACDE"])[0]

    server_one = RecordingBackend()
    server_one.server_url = "https://msa-one.invalid"
    server_two = RecordingBackend()
    server_two.server_url = "https://msa-two.invalid"
    third = MsaSearchPipeline(tmp_path, server_one).search(["ACDE"])[0]
    fourth = MsaSearchPipeline(tmp_path, server_two).search(["ACDE"])[0]

    directories = {
        Path(result["provenancePath"]).parent
        for result in (first, second, third, fourth)
    }
    assert len(directories) == 4


@pytest.mark.parametrize("damage", ["hash", "missing", "provenance"])
def test_corrupt_or_incomplete_cache_fails_without_search(
    tmp_path: Path, damage: str
) -> None:
    backend = RecordingBackend()
    pipeline = MsaSearchPipeline(tmp_path, backend)
    result = pipeline.search(["ACDE"])[0]
    directory = Path(result["provenancePath"]).parent
    if damage == "hash":
        (directory / "pairing.a3m").write_text(">query\nAAAA\n")
    elif damage == "missing":
        (directory / "non_pairing.a3m").unlink()
    else:
        (directory / "provenance.json").write_text("{broken")
    backend.calls.clear()

    with pytest.raises(SearchError, match="cache|provenance|hash|incomplete"):
        pipeline.search(["ACDE"])
    assert backend.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        MsaPayload(">query\nAAAA\n", ">query\nACDE\n"),
        MsaPayload(">query\nACDE\n", ">query\nACDF\n"),
        MsaPayload("not-a3m", ">query\nACDE\n"),
    ],
)
def test_payload_query_must_match_requested_sequence(
    tmp_path: Path, payload: MsaPayload
) -> None:
    backend = RecordingBackend()
    backend.search = lambda _sequence: payload
    with pytest.raises(SearchError, match="query|header"):
        MsaSearchPipeline(tmp_path, backend).search(["ACDE"])


def test_a3m_query_validation_ignores_insertions_and_gaps(tmp_path: Path) -> None:
    backend = RecordingBackend()
    backend.search = lambda _sequence: MsaPayload(
        ">query\nACd-DE\n", ">query\nAC-DE\n"
    )
    result = MsaSearchPipeline(tmp_path, backend).search(["ACDE"])[0]
    assert Path(result["pairedMsaPath"]).is_file()


def test_local_backend_command_and_failure_boundaries() -> None:
    calls = []

    def runner(command: list[str], **kwargs: object):
        calls.append((command, kwargs))
        output = Path(command[command.index("--output") + 1])
        (output / "pairing.a3m").write_text(">query\nACDE\n")
        (output / "non_pairing.a3m").write_text(">query\nACDE\n")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    backend = LocalMsaBackend(["fake-mmseqs-wrapper"], version="db-v1", runner=runner)
    payload = backend.search("ACDE")
    assert payload.paired == ">query\nACDE\n"
    assert calls[0][0][0] == "fake-mmseqs-wrapper"
    assert calls[0][1]["shell"] is False

    def incomplete(command: list[str], **_kwargs: object):
        del command
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with pytest.raises(SearchError, match="did not produce"):
        LocalMsaBackend(["fake"], version="v1", runner=incomplete).search("ACDE")


def test_local_command_is_part_of_cache_identity(tmp_path: Path) -> None:
    def unused(*_args, **_kwargs):
        raise AssertionError("cache identity test must not execute the backend")

    one = LocalMsaBackend(["wrapper-one"], version="v1", runner=unused)
    two = LocalMsaBackend(["wrapper-two"], version="v1", runner=unused)
    assert MsaSearchPipeline(tmp_path, one)._identity("ACDE")[0] != (
        MsaSearchPipeline(tmp_path, two)._identity("ACDE")[0]
    )


def _archive(files: dict[str, str], *, unsafe: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as output:
        for name, content in files.items():
            raw = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            output.addfile(info, io.BytesIO(raw))
        if unsafe is not None:
            raw = b"bad"
            info = tarfile.TarInfo(unsafe)
            info.size = len(raw)
            output.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()


def test_remote_ticket_poll_download_and_archive_validation() -> None:
    responses = iter(
        [
            HttpResponse(200, b'{"status":"PENDING","id":"u-1"}'),
            HttpResponse(200, b'{"status":"RUNNING"}'),
            HttpResponse(200, b'{"status":"COMPLETE"}'),
            HttpResponse(
                200,
                _archive(
                    {
                        "uniref.a3m": ">101\nACDE\n",
                        "bfd.mgnify30.metaeuk30.smag30.a3m": ">101\nACDE\n",
                    }
                ),
            ),
            HttpResponse(200, b'{"status":"COMPLETE","id":"p-1"}'),
            HttpResponse(200, _archive({"pair.a3m": ">101\nACDE\n"})),
        ]
    )
    requests = []

    def transport(method, url, data, headers, timeout):
        requests.append((method, url, data, headers, timeout))
        return next(responses)

    payload = RemoteMMseqs2Backend(
        "https://msa.invalid/api",
        version="api-v1",
        transport=transport,
        poll_interval=0,
    ).search("ACDE")

    assert payload.source == {"paired_job_id": "p-1", "unpaired_job_id": "u-1"}
    assert payload.paired.startswith(">101\nACDE")
    assert [(method, url) for method, url, *_ in requests] == [
        ("POST", "https://msa.invalid/api/ticket/msa"),
        ("GET", "https://msa.invalid/api/ticket/u-1"),
        ("GET", "https://msa.invalid/api/ticket/u-1"),
        ("GET", "https://msa.invalid/api/result/download/u-1"),
        ("POST", "https://msa.invalid/api/ticket/pair"),
        ("GET", "https://msa.invalid/api/result/download/p-1"),
    ]
    post_bodies = [
        urllib.parse.parse_qs(data.decode())
        for method, _url, data, *_ in requests
        if method == "POST"
    ]
    assert post_bodies[0]["mode"] == ["env"]
    assert post_bodies[1]["mode"] == ["pairgreedy-env"]


def test_remote_template_hits_are_archive_checked_and_content_cached(
    tmp_path: Path,
) -> None:
    m8 = "101\t1abc_A\t100\t4\t0\t0\t1\t4\t1\t4\t1e-20\t50\t0\n"
    responses = iter(
        [
            HttpResponse(200, b'{"status":"COMPLETE","id":"u-templates"}'),
            HttpResponse(
                200,
                _archive(
                    {
                        "uniref.a3m": ">101\nACDE\n",
                        "bfd.mgnify30.metaeuk30.smag30.a3m": ">101\nACDE\n",
                        "pdb70.m8": m8,
                    }
                ),
            ),
            HttpResponse(200, b'{"status":"COMPLETE","id":"p"}'),
            HttpResponse(200, _archive({"pair.a3m": ">101\nACDE\n"})),
        ]
    )
    backend = RemoteMMseqs2Backend(
        "https://msa.invalid",
        version="api-v1",
        transport=lambda *_: next(responses),
        poll_interval=0,
        use_templates=True,
    )

    result = MsaSearchPipeline(tmp_path, backend).search(["ACDE"])[0]

    assert Path(result["templateHitsPath"]).read_text() == m8
    provenance = json.loads(Path(result["provenancePath"]).read_text())
    assert provenance["backend_options"]["use_templates"] is True
    assert provenance["files"]["pdb70.m8"]["sha256"]


def test_remote_template_archive_requires_exactly_one_regular_m8() -> None:
    responses = iter(
        [
            HttpResponse(200, b'{"status":"COMPLETE","id":"u"}'),
            HttpResponse(
                200,
                _archive(
                    {
                        "uniref.a3m": ">101\nACDE\n",
                        "bfd.mgnify30.metaeuk30.smag30.a3m": ">101\nACDE\n",
                    }
                ),
            ),
        ]
    )
    with pytest.raises(SearchError, match="pdb70.m8"):
        RemoteMMseqs2Backend(
            "https://msa.invalid",
            version="api-v1",
            transport=lambda *_: next(responses),
            poll_interval=0,
            use_templates=True,
        ).search("ACDE")


def test_remote_rejects_http_status_terminal_state_and_bad_archive() -> None:
    def http_failure(*_args):
        return HttpResponse(503, b"unavailable")

    with pytest.raises(SearchError, match="HTTP 503"):
        RemoteMMseqs2Backend(
            "https://msa.invalid", version="v1", transport=http_failure
        ).search("ACDE")

    def terminal(*_args):
        return HttpResponse(200, b'{"status":"ERROR","id":"bad"}')

    with pytest.raises(SearchError, match="ERROR"):
        RemoteMMseqs2Backend(
            "https://msa.invalid", version="v1", transport=terminal
        ).search("ACDE")

    responses = iter(
        [
            HttpResponse(200, b'{"status":"COMPLETE","id":"u"}'),
            HttpResponse(
                200,
                _archive(
                    {
                        "uniref.a3m": ">101\nACDE\n",
                        "bfd.mgnify30.metaeuk30.smag30.a3m": ">101\nACDE\n",
                    },
                    unsafe="../escape",
                ),
            ),
        ]
    )
    with pytest.raises(SearchError, match="unsafe|archive"):
        RemoteMMseqs2Backend(
            "https://msa.invalid", version="v1", transport=lambda *_: next(responses)
        ).search("ACDE")
