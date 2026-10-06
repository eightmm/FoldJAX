"""W15: a ColabFold pairing alignment pairs the chains of a heteromer.

The server's headers (``>UniRef100_<accession>\\t<scores>``) carry no species,
so a model that pairs by species pairs none of them. Protenix and OpenDDE now
take the complex search their upstreams submit and read its rows by number,
as Protenix's ColabFold mode writes them; AlphaFold 3 is pinned at what it
does. Nothing here reaches the network: searches are stubs or a fake transport.
"""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from foldjax import msa_search
from foldjax.input import materialize_native_input, row_species_a3m
from foldjax.models.protenix.data import featurize_json as pj
from foldjax.registry import capabilities
from foldjax.search.msa import (
    ComplexPairPayload,
    HttpResponse,
    MsaPayload,
    MsaSearchPipeline,
    RemoteMMseqs2Client,
)

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQ"
OTHER = "GSHMLEDPVDAFQLGKVLNQ"
#: Headers as the ColabFold server writes them (the store's cached
#: ``pairing.a3m`` for SEQUENCE starts with these).
SCORES = "48\t1.00\t3.260E-05\t0\t21\t22\t0\t21\t37"


def _colabfold_block(query: str, hits: list[str], number: int = 101) -> str:
    rows = [f">{number}\n{query}\n"]
    rows += [
        f">UniRef100_A0A{index:07d}\t{SCORES}\n{hit}\n"
        for index, hit in enumerate(hits)
    ]
    return "".join(rows)


def _pair(a3ms: list[str], queries: list[str]) -> list[dict[str, np.ndarray]]:
    chains = []
    for asym_id, (a3m, query) in enumerate(zip(a3ms, queries, strict=True)):
        features = pj._featurize_a3m(query, a3m, dedup=False)
        chains.append(
            {"asym_id": asym_id, **{f"{k}_all_seq": v for k, v in features.items()}}
        )
    paired = pj._pair_chains_by_species(chains, 8192, set(range(len(chains))), 600)
    return pj._filter_all_gapped_rows(paired, set(range(len(chains))))


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    for name in (
        "FOLDJAX_MSA_COMMAND",
        "FOLDJAX_RNA_MSA_COMMAND",
        "FOLDJAX_MSA_SERVER_URL",
        "FOLDJAX_MSA_SERVER_VERSION",
        "FOLDJAX_MSA_LOCAL_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)


# -- the finding ----------------------------------------------------------------


def test_colabfold_headers_carry_no_species_for_any_species_pairing_port():
    """Per-chain ColabFold alignments pair nothing in Protenix/OpenDDE or AF3."""
    first = _colabfold_block(SEQUENCE, [SEQUENCE, "A" * len(SEQUENCE)])
    second = _colabfold_block(OTHER, [OTHER, "C" * len(OTHER)])

    features = pj._featurize_a3m(SEQUENCE, first, dedup=False)
    assert list(features["species"]) == ["", "", ""]
    (chain, _other) = _pair([first, second], [SEQUENCE, OTHER])
    assert chain["msa_all_seq"].shape[0] == 1  # the query alone

    path = (
        Path(pj.__file__).parents[3]
        / "models/alphafold3/_upstream/alphafold3/data/msa_identifiers.py"
    )
    spec = importlib.util.spec_from_file_location("_af3_msa_identifiers", path)
    assert spec is not None and spec.loader is not None
    identifiers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identifiers)
    header = f"UniRef100_A0A376LE91\t{SCORES}"
    assert identifiers.get_identifiers(header).species_id == ""
    # A UniProt header is what AlphaFold 3 pairs on.
    assert identifiers.get_identifiers("tr|A0A376LE91|A0A376LE91_ECOLX").species_id


# -- the row-number rewrite ------------------------------------------------------


def test_row_species_a3m_pairs_row_i_with_row_i():
    gap = "-" * len(OTHER)
    first = _colabfold_block(SEQUENCE, [SEQUENCE, "A" * len(SEQUENCE), SEQUENCE])
    # Greedy pairing pads a chain with no hit in a row with gaps.
    second = _colabfold_block(OTHER, ["C" * len(OTHER), gap, OTHER], number=102)
    rewritten = [row_species_a3m(first), row_species_a3m(second)]

    assert rewritten[0].splitlines()[0] == ">query"
    assert rewritten[0].splitlines()[2] == f">UniRef100_A0A0000000_1/\t{SCORES}"
    species = pj._featurize_a3m(SEQUENCE, rewritten[0], dedup=False)["species"]
    assert list(species) == ["", "1", "2", "3"]

    a, b = _pair(rewritten, [SEQUENCE, OTHER])
    rows_a = [bytes(row) for row in a["msa_all_seq"].astype(np.int8)]
    rows_b = [bytes(row) for row in b["msa_all_seq"].astype(np.int8)]
    expected_a = pj._featurize_a3m(SEQUENCE, first, dedup=False)["msa"]
    expected_b = pj._featurize_a3m(OTHER, second, dedup=False)["msa"]
    # Query, then rows 1..3 in their server order: each chain's row i is
    # paired with the other chain's row i, the gap row included.
    assert rows_a == [bytes(row) for row in expected_a.astype(np.int8)]
    assert rows_b == [bytes(row) for row in expected_b.astype(np.int8)]


def test_row_species_a3m_keeps_a_repeated_header_on_its_row():
    """Upstream numbers a dict keyed by header; a repeat must not shift rows."""
    block = (
        f">101\n{SEQUENCE}\n>UniRef100_X\t{SCORES}\n{SEQUENCE}\n"
        f">UniRef100_X\t{SCORES}\n{'A' * len(SEQUENCE)}\n"
        f">DUMMY_1\n{'-' * len(SEQUENCE)}\n"
    )
    species = pj._featurize_a3m(SEQUENCE, row_species_a3m(block), dedup=False)[
        "species"
    ]
    assert list(species) == ["", "1", "2", "3"]


# -- the search and the writer ----------------------------------------------------


class _PlainStub:
    """A search that pairs a complex with Protenix's/OpenDDE's modes."""

    name = "stub"
    version = "1"
    complex_pairing_mode = "pairgreedy-env"
    complex_pairing_modes = (
        "pairgreedy-env",
        "paircomplete-env",
        "pairgreedy",
        "paircomplete",
    )

    def __init__(self) -> None:
        self.complex_calls: list[tuple[list[str], str | None]] = []

    def search(self, sequence: str) -> MsaPayload:
        return MsaPayload(
            paired=_colabfold_block(sequence, [sequence]),
            unpaired=f">101\n{sequence}\n>u1\n{sequence}\n",
        )

    def search_complex(self, sequences, mode: str | None = None):
        self.complex_calls.append((list(sequences), mode))
        blocks = []
        for index, sequence in enumerate(sequences):
            second = "-" * len(sequence) if index == 1 else "A" * len(sequence)
            blocks.append(
                _colabfold_block(sequence, [sequence, second], number=101 + index)
            )
        return ComplexPairPayload(tuple(blocks), {"mode": mode})


@pytest.fixture
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _PlainStub:
    backend = _PlainStub()
    monkeypatch.setattr(
        msa_search,
        "_msa_pipeline",
        lambda: MsaSearchPipeline(tmp_path / "msa-cache", backend),
    )
    return backend


def _job(tmp_path: Path, *sequences: str, **extra: Any) -> Path:
    path = tmp_path / "job.json"
    entities = [
        {"type": "protein", "id": chr(65 + index), "sequence": sequence, **extra}
        for index, sequence in enumerate(sequences)
    ]
    path.write_text(json.dumps({"name": "w15", "entities": entities}))
    return path


def _materialize(source: Path, model: str, out: Path, **kwargs: Any) -> Path:
    kwargs.setdefault("msa", "auto")
    return materialize_native_input(source, capabilities(model), out, seed=0, **kwargs)


def _chains(native: Any) -> list[dict[str, Any]]:
    return [entry["proteinChain"] for entry in native[0]["sequences"]]


@pytest.mark.parametrize("model", ["protenix", "opendde"])
def test_model_pairing_pairs_the_heteromer_by_row(tmp_path: Path, stub, model):
    path = _materialize(_job(tmp_path, SEQUENCE, OTHER), model, tmp_path / "out")
    assert stub.complex_calls == [([SEQUENCE, OTHER], "pairgreedy")]
    chains = _chains(json.loads(path.read_text()))
    texts = [Path(chain["pairedMsaPath"]).read_text() for chain in chains]
    for index, chain in enumerate(chains):
        assert (
            Path(chain["pairedMsaPath"]).parent == (tmp_path / "out" / "msa").resolve()
        )
        assert Path(chain["pairedMsaPath"]).name == f"entity_{index:04d}_pairing.a3m"
    a, b = _pair(texts, [SEQUENCE, OTHER])
    # Query plus row 1 paired in both chains, and row 2 (chain B's gap row)
    # kept beside chain A's hit.
    assert a["msa_all_seq"].shape[0] == b["msa_all_seq"].shape[0] == 3
    assert (b["msa_all_seq"][2] == pj._GAP_IDX).all()
    assert msa_search.resolve_pairing(model) == {
        "requested": "model",
        "resolved": "greedy",
        "mode": "pairgreedy",
        "paired_by": "row",
    }


def test_complete_pairing_for_protenix_asks_paircomplete(tmp_path: Path, stub):
    _materialize(
        _job(tmp_path, SEQUENCE, OTHER),
        "protenix",
        tmp_path / "out",
        msa_pairing="complete",
    )
    assert stub.complex_calls == [([SEQUENCE, OTHER], "paircomplete")]


@pytest.mark.parametrize("sequences", [(SEQUENCE,), (SEQUENCE, SEQUENCE)])
def test_monomer_and_homomer_get_no_paired_alignment(tmp_path: Path, stub, sequences):
    """As upstream Protenix's ColabFold mode: no pairing search, no pairing rows."""
    path = _materialize(_job(tmp_path, *sequences), "protenix", tmp_path / "out")
    assert stub.complex_calls == []
    assert all(
        "pairedMsaPath" not in chain for chain in _chains(json.loads(path.read_text()))
    )


def test_a_callers_paired_msa_is_passed_through_untouched(tmp_path: Path, stub):
    paired = tmp_path / "paired.a3m"
    paired.write_text(f">query\n{SEQUENCE}\n>tr|P12345|P12345_HUMAN\n{SEQUENCE}\n")
    unpaired = tmp_path / "unpaired.a3m"
    unpaired.write_text(f">query\n{SEQUENCE}\n")
    path = _materialize(
        _job(
            tmp_path,
            SEQUENCE,
            OTHER,
            paired_msa=str(paired),
            unpaired_msa=str(unpaired),
        ),
        "protenix",
        tmp_path / "out",
    )
    assert stub.complex_calls == []
    for chain in _chains(json.loads(path.read_text())):
        assert chain["pairedMsaPath"] == str(paired)


def test_blocks_of_different_depths_are_refused(tmp_path: Path, stub, monkeypatch):
    def uneven(sequences, mode=None):
        stub.complex_calls.append((list(sequences), mode))
        return ComplexPairPayload(
            (
                _colabfold_block(sequences[0], [sequences[0]]),
                _colabfold_block(sequences[1], [], number=102),
            ),
            {},
        )

    monkeypatch.setattr(stub, "search_complex", uneven)
    with pytest.raises(ValueError, match="different depths"):
        _materialize(_job(tmp_path, SEQUENCE, OTHER), "protenix", tmp_path / "out")


def test_alphafold3_keeps_its_per_chain_alignment_and_refuses_complex(
    tmp_path: Path, stub
):
    assert msa_search.resolve_pairing("alphafold3") == {
        "requested": "model",
        "resolved": "per_chain",
        "mode": "paircomplete",
        "paired_by": "species",
    }
    with pytest.raises(ValueError, match="species"):
        _materialize(
            _job(tmp_path, SEQUENCE, OTHER),
            "alphafold3",
            tmp_path / "o",
            msa_pairing="greedy",
        )


def test_remote_client_submits_the_plain_greedy_mode():
    tickets: list[dict[str, list[str]]] = []
    text = (
        _colabfold_block(SEQUENCE, [SEQUENCE])
        + "\x00"
        + _colabfold_block(OTHER, [OTHER], number=102)
    )

    def transport(method, url, data, _headers, _timeout):
        if method == "POST":
            tickets.append(urllib.parse.parse_qs(data.decode()))
            return HttpResponse(200, b'{"status":"COMPLETE","id":"t-1"}')
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            raw = text.encode()
            info = tarfile.TarInfo("pair.a3m")
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        return HttpResponse(200, buffer.getvalue())

    client = RemoteMMseqs2Client(
        "https://msa.example", version="1", transport=transport, poll_interval=0
    )
    payload = client.search_complex([SEQUENCE, OTHER], mode="pairgreedy")
    assert tickets[0]["mode"] == ["pairgreedy"]
    assert payload.source["mode"] == "pairgreedy"
    assert [block.splitlines()[1] for block in payload.paired] == [SEQUENCE, OTHER]
