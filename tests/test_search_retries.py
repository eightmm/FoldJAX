"""The remote MSA client rides out what a busy public server does.

A submission answered RATELIMIT or UNKNOWN was polled as if it were a job; a
503, a reset connection or a truncated body ended the search on the first
attempt, and a truncated body escaped as ``http.client.IncompleteRead``, which
no caller catches. The one-hour ceiling was fixed, and a timeout did not say
which server it was waiting on.
"""

from __future__ import annotations

import http.client
import io
import json
import tarfile
from collections.abc import Callable, Iterator

import pytest

from foldjax import progress
from foldjax.search.msa import (
    MAX_WAIT_ENV,
    HttpResponse,
    RemoteMMseqs2Client,
    SearchError,
)

SEQUENCE = "MKTAYIAKQR"


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr("foldjax.search.msa.time.sleep", slept.append)
    monkeypatch.delenv(MAX_WAIT_ENV, raising=False)
    return slept


@pytest.fixture
def lines() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    progress.enable(stream)
    try:
        yield stream
    finally:
        progress.disable()


def _json(payload: dict) -> HttpResponse:
    return HttpResponse(200, json.dumps(payload).encode())


def _archive(text: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        raw = text.encode()
        info = tarfile.TarInfo("pair.a3m")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()


def _client(
    answers: Callable[[str, str], HttpResponse], **kwargs
) -> tuple[RemoteMMseqs2Client, list[tuple[str, str]]]:
    calls: list[tuple[str, str]] = []

    def transport(method, url, data, headers, timeout):
        path = url.removeprefix("https://msa.invalid/")
        calls.append((method, path))
        return answers(method, path)

    kwargs.setdefault("poll_interval", 1.0)
    return RemoteMMseqs2Client(
        "https://msa.invalid", version="v", transport=transport, **kwargs
    ), calls


def test_a_rate_limited_submission_is_resubmitted(lines: io.StringIO) -> None:
    submissions = iter(
        [
            _json({"status": "RATELIMIT", "id": "not-a-job"}),
            _json({"status": "UNKNOWN"}),
            _json({"status": "PENDING", "id": "job-7"}),
        ]
    )

    def answers(method: str, path: str) -> HttpResponse:
        if method == "POST":
            return next(submissions)
        if path.startswith("ticket/"):
            return _json({"status": "COMPLETE"})
        return HttpResponse(200, _archive(f">101\n{SEQUENCE}\n"))

    client, calls = _client(answers)
    text, job_id = client._run(SEQUENCE, paired=True)

    assert job_id == "job-7"
    posts = [call for call in calls if call[0] == "POST"]
    assert posts == [("POST", "ticket/pair")] * 3
    assert ("GET", "ticket/not-a-job") not in calls
    assert "resubmitting" in lines.getvalue()


def test_a_server_that_never_takes_the_ticket_ends_the_search() -> None:
    client, _ = _client(
        lambda method, path: _json({"status": "RATELIMIT"}), max_wait_seconds=5.0
    )
    with pytest.raises(SearchError, match="msa.invalid answered RATELIMIT"):
        client._run(SEQUENCE, paired=True)


@pytest.mark.parametrize(
    "failure",
    [
        HttpResponse(503, b"busy"),
        ConnectionResetError("reset by peer"),
        http.client.IncompleteRead(b"partial", 100),
    ],
    ids=["503", "reset", "incomplete-read"],
)
def test_a_transient_failure_is_retried(
    failure: object, _no_waiting: list[float], lines: io.StringIO
) -> None:
    attempts = iter([failure, failure, _json({"status": "COMPLETE", "id": "j"})])

    def answers(method: str, path: str) -> HttpResponse:
        answer = next(attempts)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    client, calls = _client(answers, poll_interval=2.0)
    assert client._json("POST", "ticket/msa")["id"] == "j"
    assert len(calls) == 3
    assert _no_waiting == [2.0, 4.0]
    assert "msa.invalid" in lines.getvalue() and "retrying" in lines.getvalue()


def test_a_failure_that_persists_is_a_search_error_naming_the_server() -> None:
    def answers(method: str, path: str) -> HttpResponse:
        raise http.client.IncompleteRead(b"", 10)

    client, calls = _client(answers)
    with pytest.raises(SearchError, match=r"msa\.invalid failed after 6 attempts"):
        client._json("POST", "ticket/msa")
    assert len(calls) == 6


def test_a_permanent_answer_is_not_retried() -> None:
    client, calls = _client(lambda method, path: HttpResponse(400, b"bad"))
    with pytest.raises(SearchError, match="HTTP 400"):
        client._json("POST", "ticket/msa")
    assert len(calls) == 1


def test_a_poll_timeout_names_the_server_and_the_knob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter(range(0, 10_000, 10))
    monkeypatch.setattr("foldjax.search.msa.time.monotonic", lambda: next(clock))

    def answers(method: str, path: str) -> HttpResponse:
        return _json({"status": "RUNNING", "id": "job-1"})

    client, _ = _client(answers, max_wait_seconds=60.0)
    with pytest.raises(TimeoutError, match=f"msa.invalid.*{MAX_WAIT_ENV}"):
        client._run(SEQUENCE, paired=True)


def test_the_ceiling_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MAX_WAIT_ENV, "7200")
    assert RemoteMMseqs2Client("https://msa.invalid", version="v").max_wait_seconds == (
        7200.0
    )
    explicit = RemoteMMseqs2Client(
        "https://msa.invalid", version="v", max_wait_seconds=10.0
    )
    assert explicit.max_wait_seconds == 10.0
    monkeypatch.setenv(MAX_WAIT_ENV, "soon")
    with pytest.raises(ValueError, match=MAX_WAIT_ENV):
        RemoteMMseqs2Client("https://msa.invalid", version="v")
