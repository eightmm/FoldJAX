"""What the remote search and structure clients refuse to send or accept.

A redirect used to carry the request's credentials to whatever host the
``Location`` named, ``http://`` included; plain-http servers were accepted
without being asked for; a response, an archive member or a server job id was
used however large or strange it was; and a cached mmCIF was returned whatever
it held.
"""

from __future__ import annotations

import gzip
import http.server
import io
import tarfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from foldjax.models.boltz2.data.msa import mmseqs2
from foldjax.search import msa as search_msa
from foldjax.search.msa import (
    ALLOW_INSECURE_HTTP_ENV,
    HttpResponse,
    RemoteMMseqs2Client,
    SearchError,
    _urllib_transport,
    require_https,
)
from foldjax.search.templates import StructureStore

SEQUENCE = "MKTAYIAKQR"


@pytest.fixture(autouse=True)
def _no_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOW_INSECURE_HTTP_ENV, raising=False)


# ---------------------------------------------------------------- https only


@pytest.mark.parametrize(
    "url",
    ["https://api.colabfold.com", "http://localhost:8080", "http://127.0.0.1/x"],
)
def test_https_and_loopback_http_are_accepted(url: str) -> None:
    assert require_https(url, what="url") == url


@pytest.mark.parametrize(
    "url", ["http://msa.example.org", "ftp://msa.example.org", "msa.example.org", ""]
)
def test_anything_else_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match=ALLOW_INSECURE_HTTP_ENV):
        require_https(url, what="url")


def test_plain_http_is_an_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOW_INSECURE_HTTP_ENV, "1")
    assert require_https("http://msa.lab", what="url") == "http://msa.lab"


def test_the_refusal_does_not_echo_credentials() -> None:
    with pytest.raises(ValueError) as caught:
        require_https("http://user:hunter2@msa.example.org", what="url")
    assert "hunter2" not in str(caught.value)


def test_every_client_checks_its_url(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="remote MSA host URL"):
        RemoteMMseqs2Client("http://msa.example.org", version="v")
    with pytest.raises(ValueError, match="template structure URL"):
        StructureStore(Path("/nonexistent"), base_url="http://files.example.org")

    def unreachable(*_args, **_kwargs):
        raise AssertionError("a request was sent to a plain-http server")

    monkeypatch.setattr(mmseqs2.requests, "post", unreachable)
    with pytest.raises(ValueError, match="MSA server URL"):
        mmseqs2.run_mmseqs2(["ACD"], "unused", host_url="http://msa.example.org")


# ---------------------------------------------------------- redirects, size


class _Server(http.server.BaseHTTPRequestHandler):
    seen: list[tuple[str, str | None]] = []
    body = b""

    def do_GET(self) -> None:  # noqa: N802
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        if self.path == "/moved":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.2:9/stolen")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(type(self).body)))
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *_args) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    _Server.seen = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_redirect_is_returned_not_followed(server: str) -> None:
    response = _urllib_transport(
        "GET", f"{server}/moved", None, {"Authorization": "Basic c2VjcmV0"}, 5.0
    )
    assert response.status == 302
    assert _Server.seen == [("/moved", "Basic c2VjcmV0")]


def test_the_remote_client_turns_a_redirect_into_an_error(server: str) -> None:
    client = RemoteMMseqs2Client(
        server, version="v", username="u", password="p", poll_interval=0
    )
    with pytest.raises(SearchError, match="HTTP 302"):
        client._request("GET", "moved")
    assert len(_Server.seen) == 1


def test_an_oversized_response_is_refused(
    server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search_msa, "MAX_REMOTE_BYTES", 16)
    _Server.body = b"x" * 17
    with pytest.raises(SearchError, match="exceeds 16 bytes"):
        _urllib_transport("GET", f"{server}/big", None, {}, 5.0)
    _Server.body = b"x" * 16
    assert _urllib_transport("GET", f"{server}/ok", None, {}, 5.0).body == b"x" * 16


def _archive(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, raw in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()


def _client(job_id: str, archive: bytes) -> tuple[RemoteMMseqs2Client, list[str]]:
    urls: list[str] = []

    def transport(method, url, data, headers, timeout):
        urls.append(url)
        if url.endswith("ticket/pair"):
            return HttpResponse(
                200, f'{{"status":"COMPLETE","id":"{job_id}"}}'.encode()
            )
        return HttpResponse(200, archive)

    client = RemoteMMseqs2Client(
        "https://msa.invalid", version="v", poll_interval=0, transport=transport
    )
    return client, urls


def test_an_oversized_archive_member_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(search_msa, "MAX_REMOTE_BYTES", 32)
    client, _ = _client("p-1", _archive({"pair.a3m": b">101\n" + b"A" * 64}))
    with pytest.raises(SearchError, match="exceeds 32 bytes"):
        client._run(SEQUENCE, paired=True)


@pytest.mark.parametrize("job_id", ["../ticket/x", "a/b", "id?x=1", "id#f", " p "])
def test_a_job_id_that_is_not_a_url_segment_is_refused(job_id: str) -> None:
    client, urls = _client(job_id, b"")
    with pytest.raises(SearchError, match="invalid job id"):
        client._run(SEQUENCE, paired=True)
    assert urls == ["https://msa.invalid/ticket/pair"]


class _Response:
    def __init__(self, *, json_data=None, status_code=200):
        self._json_data = json_data
        self.status_code = status_code
        self.text = ""
        self.content = b""

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise mmseqs2.requests.HTTPError(f"HTTP {self.status_code}")


def test_boltz2_client_refuses_redirects_and_odd_job_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict] = []

    def redirected(url, **kwargs):
        seen.append(kwargs)
        return _Response(status_code=307)

    monkeypatch.setattr(mmseqs2.requests, "post", redirected)
    with pytest.raises(RuntimeError, match="redirected"):
        mmseqs2.run_mmseqs2(
            ["ACD"],
            str(tmp_path / "a"),
            host_url="https://msa.example.test",
            auth_headers={"X-API-Key": "secret"},
            max_retries=0,
        )
    assert seen[0]["allow_redirects"] is False

    monkeypatch.setattr(
        mmseqs2.requests,
        "post",
        lambda url, **kwargs: _Response(
            json_data={"status": "COMPLETE", "id": "../../x"}
        ),
    )
    monkeypatch.setattr(
        mmseqs2.requests,
        "get",
        lambda *a, **k: pytest.fail("an invalid job id reached a URL"),
    )
    with pytest.raises(RuntimeError, match="invalid job id"):
        mmseqs2.run_mmseqs2(
            ["ACD"], str(tmp_path / "b"), host_url="https://msa.example.test"
        )


# ----------------------------------------------------------- mmCIF cache


def _cif(entry: str) -> str:
    return (
        f"data_{entry.upper()}\n_entry.id {entry.upper()}\n"
        "loop_\n_atom_site.id\n_atom_site.type_symbol\n1 C\n"
    )


def test_a_cached_mmcif_of_another_entry_is_fetched_again(tmp_path: Path) -> None:
    calls: list[str] = []

    def transport(method, url, data, headers, timeout):
        calls.append(url)
        return HttpResponse(200, _cif("1abc").encode())

    store = StructureStore(tmp_path / "cache", transport=transport)
    first = store.path("1abc")
    assert calls and store.path("1abc") == first and len(calls) == 1

    first.write_text(_cif("9xyz"))
    assert store.path("1abc").read_text() == _cif("1abc")
    assert len(calls) == 2

    first.write_text("data_1ABC\nloop_\n_atom_site.id\n'unterminated\n")
    store.path("1abc")
    assert len(calls) == 3


def test_a_download_naming_another_entry_is_refused(tmp_path: Path) -> None:
    store = StructureStore(
        tmp_path / "cache",
        transport=lambda *a: HttpResponse(200, _cif("9xyz").encode()),
    )
    with pytest.raises(SearchError, match="not that entry's mmCIF"):
        store.path("1abc")
    assert not list((tmp_path / "cache").rglob("*.cif"))


def test_a_stale_unpacked_mirror_entry_is_unpacked_again(tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "1abc.cif.gz").write_bytes(gzip.compress(_cif("1abc").encode()))
    store = StructureStore(
        tmp_path / "cache",
        local_dir=mirror,
        transport=lambda *a: pytest.fail("a mirrored entry was downloaded"),
    )
    unpacked = store.path("1abc")
    unpacked.write_text(_cif("9xyz"))
    assert store.path("1abc").read_text() == _cif("1abc")
