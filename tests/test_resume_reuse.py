"""`--resume` reuses what it can prove, and says why when it cannot."""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path

import pytest

import foldjax
from foldjax import progress
from foldjax.manifest import MANIFEST_NAME, _input_dependencies, request_mismatch
from tests.test_resume_manifest import _backends, _request

SEQUENCE = "MKTAYIAKQRQISFVK"


class _StubSearch:
    name = "stub"
    version = "1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, sequence: str):
        from foldjax.search.msa import MsaPayload

        self.calls.append(sequence)
        return MsaPayload(
            paired=f">query\n{sequence}\n",
            unpaired=f">query\n{sequence}\n>hit\n{sequence}\n",
            source={"stub": True},
        )


def _bare_job(tmp_path: Path) -> Path:
    job = tmp_path / "bare.json"
    job.write_text(
        json.dumps(
            {
                "name": "bare",
                "entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}],
            }
        ),
        encoding="utf-8",
    )
    return job


@pytest.fixture
def stub_search(tmp_path: Path, monkeypatch) -> _StubSearch:
    from foldjax.search.msa import MsaSearchPipeline

    backend = _StubSearch()
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline",
        lambda: MsaSearchPipeline(tmp_path / "msa-cache", backend),
    )
    monkeypatch.delenv("FOLDJAX_RNA_MSA_COMMAND", raising=False)
    return backend


@pytest.fixture
def messages(monkeypatch) -> io.StringIO:
    stream = io.StringIO()
    monkeypatch.setattr(progress, "_enabled", True)
    monkeypatch.setattr(progress, "_stream", stream)
    return stream


def test_a_single_sequence_run_is_verifiable_and_reused(tmp_path: Path) -> None:
    request = _request(tmp_path, input=_bare_job(tmp_path), msa="single")
    assert _input_dependencies(request)["verifiable"] is True

    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))
    assert len(calls) == 1
    assert resumed.skipped == (request.output_dir,)


def test_a_searched_run_records_its_cached_alignment_and_is_reused(
    tmp_path: Path, stub_search: _StubSearch
) -> None:
    request = _request(tmp_path, input=_bare_job(tmp_path), msa="auto")
    # Nothing cached yet: the run would search, so nothing can be proved.
    assert _input_dependencies(request)["verifiable"] is False

    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        manifest = json.loads((request.output_dir / MANIFEST_NAME).read_text())
        recorded = manifest["input_dependencies"]
        assert recorded["verifiable"] is True
        names = {Path(item["path"]).name for item in recorded["artifacts"]}
        assert {"non_pairing.a3m", "pairing.a3m", "provenance.json"} <= names

        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))
    assert len(calls) == 1
    assert stub_search.calls == [SEQUENCE]
    assert resumed.skipped == (request.output_dir,)


def test_a_changed_cached_alignment_forces_a_rerun_and_says_why(
    tmp_path: Path, stub_search: _StubSearch, messages: io.StringIO
) -> None:
    request = _request(tmp_path, input=_bare_job(tmp_path), msa="auto")
    calls: list[tuple[str, str, int]] = []
    with _backends(calls):
        foldjax.predict(request)
        cached = next((tmp_path / "msa-cache").rglob("non_pairing.a3m"))
        cached.write_text(f">query\n{SEQUENCE}\n>other\n{SEQUENCE}\n")
        foldjax.predict_batch(dataclasses.replace(request, resume=True))
    assert len(calls) == 2
    assert "not resumable: the finished run" in messages.getvalue()
    assert "changed" in messages.getvalue()


def test_a_fresh_directory_is_not_reported(
    tmp_path: Path, messages: io.StringIO
) -> None:
    request = _request(tmp_path, resume=True)
    with _backends([]):
        foldjax.predict_batch(request)
    assert "not resumable" not in messages.getvalue()


def test_request_mismatch_names_the_difference(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with _backends([]):
        foldjax.predict(request)
    document = json.loads((request.output_dir / MANIFEST_NAME).read_text())

    assert request_mismatch(document, request, seed=7) is None
    assert request_mismatch(document, request, seed=8) == "the seed differs"
    changed = dataclasses.replace(request, num_steps=4)
    assert request_mismatch(document, changed, seed=7) == (
        "the sampling settings differ"
    )
