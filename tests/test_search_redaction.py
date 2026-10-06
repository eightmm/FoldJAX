"""Search provenance and `doctor` never print a credential from the search setup.

A local search command's argv (``FOLDJAX_MSA_COMMAND``,
``FOLDJAX_TEMPLATE_COMMAND``) and a server URL can carry a password or token.
The cache identity keeps the raw values -- redacting it would orphan every
existing cache entry -- but what is written for people to read does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foldjax.search import MsaPayload, MsaSearchPipeline
from foldjax.search.templates import TemplateHitsPayload, TemplateHitsPipeline

SEQUENCE = "MQIFVKTLTGKTITLEVEPSD"
SECRETS = ("hunter2", "s3cr3t", "tok-123")
COMMAND = ["/opt/search.sh", "--password", "hunter2", "--api-key=s3cr3t"]
HOST = "https://user:tok-123@msa.example.org"
M8 = "101\t1abc_A\t1.0\t20\t0\t0\t1\t20\t1\t20\t1e-30\t200"


def _no_secret(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text, secret


class _MsaBackend:
    name = "local"
    version = "1"

    def __init__(self) -> None:
        self.calls = 0

    def search(self, sequence: str) -> MsaPayload:
        self.calls += 1
        a3m = f">query\n{sequence}\n"
        return MsaPayload(a3m, a3m, {"command": COMMAND, "host": HOST})


class _HitsBackend:
    name = "local-template"
    version = "1"

    def __init__(self) -> None:
        self.calls = 0

    def search(self, sequence: str) -> TemplateHitsPayload:
        self.calls += 1
        return TemplateHitsPayload(M8, {"command": COMMAND})


def test_msa_provenance_is_redacted_and_the_cache_still_hits(tmp_path: Path) -> None:
    backend = _MsaBackend()
    pipeline = MsaSearchPipeline(
        tmp_path, backend, options={"command": COMMAND, "host": HOST}
    )
    (first,) = pipeline.search([SEQUENCE])
    text = Path(first["provenancePath"]).read_text()
    _no_secret(text)
    assert "/opt/search.sh" in text and "msa.example.org" in text

    assert pipeline.search([SEQUENCE]) == [first]
    assert backend.calls == 1, "the redacted provenance broke the cache hit"


def test_template_hits_provenance_is_redacted(tmp_path: Path) -> None:
    backend = _HitsBackend()
    pipeline = TemplateHitsPipeline(tmp_path, backend, options={"command": COMMAND})
    first = pipeline.search(SEQUENCE)
    _no_secret(Path(first["provenancePath"]).read_text())
    assert pipeline.search(SEQUENCE) == first
    assert backend.calls == 1


def test_doctor_reports_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    from foldjax.msa_search import msa_search_backend
    from foldjax.template_search import template_search_backend

    monkeypatch.setenv("FOLDJAX_MSA_COMMAND", " ".join(COMMAND))
    monkeypatch.setenv("FOLDJAX_TEMPLATE_COMMAND", " ".join(COMMAND))
    _no_secret(repr(msa_search_backend()))
    _no_secret(repr(template_search_backend()))
    assert msa_search_backend()["protein"]["command"][0] == "/opt/search.sh"

    monkeypatch.delenv("FOLDJAX_MSA_COMMAND")
    monkeypatch.delenv("FOLDJAX_TEMPLATE_COMMAND")
    monkeypatch.setenv("FOLDJAX_MSA_SERVER_URL", HOST)
    _no_secret(repr(msa_search_backend()))
    _no_secret(repr(template_search_backend()))


def test_doctor_reports_a_refused_structure_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from foldjax.template_search import template_search_backend

    monkeypatch.delenv("FOLDJAX_ALLOW_INSECURE_HTTP", raising=False)
    monkeypatch.setenv("FOLDJAX_TEMPLATE_STRUCTURE_URL", "http://files.example.org")
    structures = template_search_backend()["structures"]
    assert structures["url"] is None
    assert "https" in structures["error"]
