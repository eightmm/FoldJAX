"""Unique sequences share the unpaired ticket; the cached bytes do not change.

Twenty sequences used to cost forty serial tickets. The fake server below
answers per query, independently of what else shares the ticket, and separates
query blocks with NUL the way ColabFold's result files do -- so these tests
prove the splitting and renumbering, not the real server's per-query
independence (which rests on ColabFold's and Boltz's own clients batching).
"""

from __future__ import annotations

import io
import json
import tarfile
import urllib.parse
from pathlib import Path

import pytest

from foldjax.search import MsaSearchPipeline
from foldjax.search import msa as search_msa
from foldjax.search.msa import HttpResponse, RemoteMMseqs2Client, SearchError

SEQUENCES = ["MKTAYIAKQR", "GSHMLEDPVA", "QQWERTYLKA"]


def _hits(number: int, sequence: str, database: str) -> str:
    rows = [f">{number}\n{sequence}\n"]
    for index in range(3):
        mutated = sequence[:index] + "A" + sequence[index + 1 :]
        rows.append(f">{database}_{sequence[:3]}_{index}\n{mutated}\n")
    return "".join(rows)


class _ColabFold:
    """Per-query answers, NUL-separated blocks, tickets counted by mode."""

    def __init__(self, fail_pair_for: str | None = None, fail_env: bool = False):
        self.jobs: dict[str, tuple[str, list[str]]] = {}
        self.tickets: list[tuple[str, int]] = []
        self.fail_pair_for = fail_pair_for
        self.fail_env = fail_env

    def __call__(self, method, url, data, headers, timeout) -> HttpResponse:
        path = urllib.parse.urlsplit(url).path.lstrip("/")
        if method == "POST":
            form = urllib.parse.parse_qs(data.decode())
            mode, query = form["mode"][0], form["q"][0]
            sequences = [
                block.split("\n", 1)[1].strip() for block in query.split(">")[1:]
            ]
            self.tickets.append((mode, len(sequences)))
            if mode == "env" and self.fail_env:
                return HttpResponse(200, b'{"status":"ERROR"}')
            if mode != "env" and sequences == [self.fail_pair_for]:
                return HttpResponse(200, b'{"status":"ERROR"}')
            job = f"job{len(self.jobs)}"
            self.jobs[job] = (mode, sequences)
            answer = {"status": "COMPLETE", "id": job}
            return HttpResponse(200, json.dumps(answer).encode())
        job = path.rsplit("/", 1)[-1]
        mode, sequences = self.jobs[job]
        names = (
            ["pair.a3m"]
            if mode != "env"
            else ["uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"]
        )
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for name in names:
                text = "".join(
                    _hits(101 + index, sequence, name.split(".")[0]) + "\x00"
                    for index, sequence in enumerate(sequences)
                )
                raw = text.encode()
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
        return HttpResponse(200, buffer.getvalue())


def _client(server: _ColabFold) -> RemoteMMseqs2Client:
    return RemoteMMseqs2Client(
        "https://msa.invalid", version="v", transport=server, poll_interval=0
    )


def test_a_shared_ticket_gives_each_sequence_its_own_bytes() -> None:
    alone = _ColabFold()
    single = [_client(alone).search(sequence) for sequence in SEQUENCES]
    together = _ColabFold()
    batched = _client(together).search_many(SEQUENCES)

    for one, many in zip(single, batched, strict=True):
        assert many.unpaired == one.unpaired
        assert many.paired == one.paired
    assert alone.tickets.count(("env", 1)) == 3
    assert together.tickets == [("env", 3)] + [("paircomplete", 1)] * 3


def test_a_large_batch_is_split_into_bounded_tickets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(search_msa, "MAX_QUERIES_PER_TICKET", 2)
    server = _ColabFold()
    _client(server).search_many(SEQUENCES)
    assert [ticket for ticket in server.tickets if ticket[0] == "env"] == [
        ("env", 2),
        ("env", 1),
    ]


def test_the_pipeline_batches_misses_and_caches_each(tmp_path: Path) -> None:
    server = _ColabFold()
    pipeline = MsaSearchPipeline(tmp_path, _client(server))
    first = pipeline.search(SEQUENCES + SEQUENCES[:1])
    assert [ticket[0] for ticket in server.tickets].count("env") == 1
    assert first[0] == first[3]

    server.tickets.clear()
    assert pipeline.search(SEQUENCES) == first[:3]
    assert server.tickets == []

    alone = MsaSearchPipeline(tmp_path / "alone", _client(_ColabFold()))
    for sequence, result in zip(SEQUENCES, first, strict=False):
        (single,) = alone.search([sequence])
        for key in ("pairedMsaPath", "unpairedMsaPath"):
            assert Path(single[key]).read_bytes() == Path(result[key]).read_bytes()


def test_one_sequences_failure_leaves_the_others_their_alignments(
    tmp_path: Path,
) -> None:
    server = _ColabFold(fail_pair_for=SEQUENCES[1])
    pipeline = MsaSearchPipeline(tmp_path, _client(server))
    outcomes = pipeline.search_each(SEQUENCES)

    assert isinstance(outcomes[1], SearchError)
    assert Path(outcomes[0]["unpairedMsaPath"]).is_file()
    assert Path(outcomes[2]["unpairedMsaPath"]).is_file()
    with pytest.raises(SearchError):
        pipeline.search(SEQUENCES)


def test_a_failed_shared_ticket_fails_every_sequence_in_it(tmp_path: Path) -> None:
    pipeline = MsaSearchPipeline(tmp_path, _client(_ColabFold(fail_env=True)))
    outcomes = pipeline.search_each(SEQUENCES)
    assert all(isinstance(outcome, SearchError) for outcome in outcomes)
    assert not [path for path in tmp_path.iterdir() if not path.name.startswith(".")]
