import io
import tarfile
from pathlib import Path

import pytest

from foldjax.models.boltz2.data.msa import mmseqs2


class _Response:
    def __init__(self, *, json_data=None, content=b"", status_code=200, text=""):
        self._json_data = json_data
        self.content = content
        self.status_code = status_code
        self.text = text

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise mmseqs2.requests.HTTPError(f"HTTP {self.status_code}")


def _result_tar(files: dict[str, str]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, value in files.items():
            payload = value.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def test_run_mmseqs2_full_remote_contract_and_api_key(tmp_path, monkeypatch) -> None:
    calls = []
    archive = _result_tar(
        {
            "uniref.a3m": ">101\nACD\n\x00>102\nEFG\n",
            "bfd.mgnify30.metaeuk30.smag30.a3m": ">101\nAXD\n\x00>102\nEYG\n",
        }
    )

    def post(url, **kwargs):
        calls.append(("POST", url, kwargs))
        return _Response(json_data={"status": "COMPLETE", "id": "job-1"})

    def get(url, **kwargs):
        calls.append(("GET", url, kwargs))
        assert url.endswith("/result/download/job-1")
        return _Response(content=archive)

    monkeypatch.setattr(mmseqs2.requests, "post", post)
    monkeypatch.setattr(mmseqs2.requests, "get", get)

    result = mmseqs2.run_mmseqs2(
        ["ACD", "EFG", "ACD"],
        str(tmp_path / "msa"),
        host_url="https://msa.example.test/",
        auth_headers={"X-API-Key": "secret"},
    )

    assert result == [
        ">101\nACD\n>101\nAXD\n",
        ">102\nEFG\n>102\nEYG\n",
        ">101\nACD\n>101\nAXD\n",
    ]
    assert calls[0][1] == "https://msa.example.test/ticket/msa"
    assert calls[0][2]["headers"]["X-API-Key"] == "secret"
    assert calls[0][2]["data"]["mode"] == "env"
    assert calls[1][2]["headers"]["User-Agent"].startswith("boltz-jax/")


def test_run_mmseqs2_retries_transient_http_error(tmp_path, monkeypatch) -> None:
    archive = _result_tar({"uniref.a3m": ">101\nACD\n"})
    responses = iter(
        [
            _Response(status_code=503),
            _Response(json_data={"status": "COMPLETE", "id": "job-2"}),
        ]
    )
    monkeypatch.setattr(mmseqs2.requests, "post", lambda *a, **k: next(responses))
    monkeypatch.setattr(
        mmseqs2.requests,
        "get",
        lambda *a, **k: _Response(content=archive),
    )
    monkeypatch.setattr(mmseqs2.time, "sleep", lambda _: None)

    assert mmseqs2.run_mmseqs2(
        "ACD", str(tmp_path / "retry"), use_env=False, max_retries=1
    ) == [">101\nACD\n"]


def test_run_mmseqs2_pairing_mode_contract(tmp_path, monkeypatch) -> None:
    calls = []
    archive = _result_tar(
        {"pair.a3m": ">101\nACD\n>paired\nAXD\n\x00>102\nEFG\n>paired\nEYG\n"}
    )

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response(json_data={"status": "COMPLETE", "id": "paired-1"})

    monkeypatch.setattr(mmseqs2.requests, "post", post)
    monkeypatch.setattr(
        mmseqs2.requests, "get", lambda *a, **k: _Response(content=archive)
    )

    result = mmseqs2.run_mmseqs2(
        ["ACD", "EFG"],
        str(tmp_path / "paired"),
        use_env=False,
        use_pairing=True,
        pairing_strategy="complete",
    )

    assert result == [
        ">101\nACD\n>paired\nAXD\n",
        ">102\nEFG\n>paired\nEYG\n",
    ]
    assert calls[0][0].endswith("/ticket/pair")
    assert calls[0][1]["data"]["mode"] == "paircomplete"


def test_run_mmseqs2_reuses_completed_cache_without_network(
    tmp_path, monkeypatch
) -> None:
    prefix = tmp_path / "cached"
    result_dir = Path(f"{prefix}_all")
    result_dir.mkdir()
    (result_dir / "out.tar.gz").write_bytes(b"already-downloaded")
    (result_dir / "uniref.a3m").write_text(">101\nACD\n")
    monkeypatch.setattr(
        mmseqs2.requests,
        "post",
        lambda *a, **k: pytest.fail("completed MSA cache must not use the network"),
    )

    assert mmseqs2.run_mmseqs2(
        "ACD", str(prefix), use_env=False
    ) == [">101\nACD\n"]


@pytest.mark.parametrize(
    ("username", "password"), [("user", None), (None, "password")]
)
def test_run_mmseqs2_rejects_partial_basic_auth(
    tmp_path, username, password
) -> None:
    with pytest.raises(ValueError, match="username and password"):
        mmseqs2.run_mmseqs2(
            "ACD",
            str(tmp_path / "auth"),
            msa_server_username=username,
            msa_server_password=password,
        )


def test_run_mmseqs2_removes_corrupt_cached_archive(tmp_path) -> None:
    prefix = tmp_path / "broken"
    result_dir = Path(f"{prefix}_env")
    result_dir.mkdir()
    archive = result_dir / "out.tar.gz"
    archive.write_bytes(b"not a tar archive")

    with pytest.raises(RuntimeError, match="invalid MSA result archive"):
        mmseqs2.run_mmseqs2("ACD", str(prefix))

    assert not archive.exists()


def test_run_mmseqs2_rejects_two_authentication_methods(tmp_path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        mmseqs2.run_mmseqs2(
            "ACD",
            str(tmp_path / "auth"),
            msa_server_username="user",
            msa_server_password="password",
            auth_headers={"X-API-Key": "secret"},
        )


def test_run_mmseqs2_rejects_tar_traversal(tmp_path) -> None:
    prefix = tmp_path / "unsafe"
    result_dir = Path(f"{prefix}_env")
    result_dir.mkdir()
    archive = result_dir / "out.tar.gz"
    archive.write_bytes(_result_tar({"../outside.a3m": "malicious"}))

    with pytest.raises(RuntimeError, match="unsafe tar entry"):
        mmseqs2.run_mmseqs2("ACD", str(prefix))

    assert not (tmp_path / "outside.a3m").exists()
