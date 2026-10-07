"""W10 workflow features: MSA pairing, alignment statistics, ranking, prefetch,
private template folders, presets and ModelCIF confidence records.

Nothing here reaches the network or a GPU: searches are stub backends or fake
transports, the local search wrapper runs against a fake ``colabfold`` module,
and structures are small synthetic mmCIFs or the replayed fixtures of
``tests/test_output_contract.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

from foldjax import cli, msa_search
from foldjax.input import materialize_native_input
from foldjax.registry import capabilities
from foldjax.schema import PredictionRequest
from foldjax.search.msa import (
    ComplexPairPayload,
    HttpResponse,
    LocalMsaClient,
    MsaPayload,
    MsaSearchPipeline,
    RemoteMMseqs2Client,
)

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQ"
OTHER = "GSHMLEDPVDAFQLGKVLNQ"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never the real store: every cache this file touches is under tmp."""
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    for name in (
        "FOLDJAX_MSA_COMMAND",
        "FOLDJAX_RNA_MSA_COMMAND",
        "FOLDJAX_MSA_SERVER_URL",
        "FOLDJAX_MSA_SERVER_VERSION",
        "FOLDJAX_MSA_LOCAL_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _job(tmp_path: Path, *sequences: str) -> Path:
    return _write(
        tmp_path / "job.json",
        {
            "name": "w10",
            "entities": [
                {"type": "protein", "id": chr(65 + index), "sequence": sequence}
                for index, sequence in enumerate(sequences)
            ],
        },
    )


class _Stub:
    """A per-chain search with complex pairing, recording every call."""

    name = "stub"
    version = "1"
    complex_pairing_mode = "pairgreedy-env"
    complex_pairing_modes = ("pairgreedy-env", "paircomplete-env")

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.complex_calls: list[tuple[list[str], str | None]] = []
        self.unpaired_only_calls = 0

    def search(self, sequence: str) -> MsaPayload:
        self.calls.append(sequence)
        return MsaPayload(
            paired=f">101\n{sequence}\n>p\n{sequence}\n",
            unpaired=f">101\n{sequence}\n>u1\n{sequence}\n>u2\n{'A' * len(sequence)}\n",
        )

    def unpaired_only(self) -> _Stub:
        self.unpaired_only_calls += 1
        return self

    def search_complex(self, sequences, mode: str | None = None):
        self.complex_calls.append((list(sequences), mode))
        # Row 1 pairs every chain; row 2 is all-gap for the second chain, as a
        # greedy pairing leaves a chain with no hit in that taxon.
        blocks = []
        for index, sequence in enumerate(sequences):
            gap = "-" * len(sequence)
            second = gap if index == 1 else sequence
            blocks.append(
                f">{101 + index}\n{sequence}\n>h1\n{sequence}\n>h2\n{second}\n"
            )
        return ComplexPairPayload(tuple(blocks), {"mode": mode})


@pytest.fixture
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Stub:
    backend = _Stub()
    monkeypatch.setattr(
        msa_search,
        "_msa_pipeline",
        lambda: MsaSearchPipeline(tmp_path / "msa-cache", backend),
    )
    return backend


def _materialize(source: Path, model: str, out: Path, **kwargs: Any) -> Path:
    kwargs.setdefault("msa", "auto")
    return materialize_native_input(source, capabilities(model), out, seed=0, **kwargs)


# -- 1. --msa-pairing -----------------------------------------------------------


def test_model_pairing_keeps_the_cache_identities_it_always_had(tmp_path: Path):
    """``model`` must not move a single existing cache entry.

    The per-chain identity below is the one every remote search has written
    since the cache existed; the complex identity is OpenFold3's.
    """
    pipeline = msa_search._msa_pipeline()
    expected = {
        "schema_version": 1,
        "sequence": SEQUENCE,
        "backend": {"name": "remote-mmseqs2", "version": "colabfold-mmseqs2"},
        "options": {"host": "https://api.colabfold.com", "pairing": "paircomplete"},
    }
    canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    assert pipeline._identity(SEQUENCE)[0] == hashlib.sha256(
        canonical.encode()
    ).hexdigest()
    narrowed = pipeline.without_per_chain_pairing()
    assert narrowed.options["pairing"] == "none"
    assert narrowed._identity(SEQUENCE)[0] != pipeline._identity(SEQUENCE)[0]
    assert narrowed.backend.pairs_per_chain is False
    assert pipeline.backend.pairs_per_chain is True


def test_complex_pairing_mode_is_in_the_cache_key(tmp_path: Path):
    backend = _Stub()
    pipeline = MsaSearchPipeline(tmp_path, backend)

    greedy = pipeline.search_complex([SEQUENCE, OTHER])
    again = pipeline.search_complex([SEQUENCE, OTHER], mode="pairgreedy-env")
    complete = pipeline.search_complex([SEQUENCE, OTHER], mode="paircomplete-env")

    # The backend default is called without a mode, so a backend that only
    # knows OpenFold3's strategy keeps working; spelling it hits the same entry.
    assert backend.complex_calls == [
        ([SEQUENCE, OTHER], None),
        ([SEQUENCE, OTHER], "paircomplete-env"),
    ]
    assert again == greedy
    assert Path(complete[0]["pairedMsaPath"]).parent != Path(
        greedy[0]["pairedMsaPath"]
    ).parent
    provenance = json.loads(Path(complete[0]["provenancePath"]).read_text())
    assert provenance["mode"] == "paircomplete-env"


def test_remote_complete_pairing_and_unpaired_only_tickets():
    tickets: list[dict[str, list[str]]] = []

    def archive(member: str, text: str) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            raw = text.encode()
            info = tarfile.TarInfo(member)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        return buffer.getvalue()

    def transport(method, url, data, _headers, _timeout):
        if method == "POST":
            tickets.append(urllib.parse.parse_qs(data.decode()))
            return HttpResponse(200, b'{"status":"COMPLETE","id":"t-1"}')
        mode = tickets[-1]["mode"][0]
        if mode == "env":
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
                for name in ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"):
                    raw = f">101\n{SEQUENCE}\n".encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(raw)
                    tar.addfile(info, io.BytesIO(raw))
            return HttpResponse(200, buffer.getvalue())
        return HttpResponse(
            200, archive("pair.a3m", f">101\n{SEQUENCE}\n\x00>102\n{OTHER}\n")
        )

    client = RemoteMMseqs2Client(
        "https://msa.invalid", version="v", poll_interval=0, transport=transport
    )
    client.search_complex([SEQUENCE, OTHER], mode="paircomplete-env")
    assert tickets[-1]["mode"] == ["paircomplete-env"]

    tickets.clear()
    payload = client.unpaired_only().search(SEQUENCE)
    assert [ticket["mode"] for ticket in tickets] == [["env"]]
    assert payload.paired == f">101\n{SEQUENCE}\n"
    tickets.clear()
    client.search(SEQUENCE)
    assert [ticket["mode"] for ticket in tickets] == [["env"], ["paircomplete"]]


def test_local_wrapper_is_told_when_no_pairing_is_wanted(tmp_path: Path):
    seen: list[dict[str, str] | None] = []

    def runner(command, **kwargs):
        seen.append(kwargs.get("env"))
        output = Path(command[command.index("--output") + 1])
        (output / "pairing.a3m").write_text(f">q\n{SEQUENCE}\n")
        (output / "non_pairing.a3m").write_text(f">q\n{SEQUENCE}\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    client = LocalMsaClient(["wrapper"], version="v", runner=runner)
    client.search(SEQUENCE)
    client.unpaired_only().search(SEQUENCE)
    assert seen[0] is None
    assert seen[1]["FOLDJAX_MSA_PAIRING"] == "none"


def test_none_pairing_delivers_no_paired_alignment(tmp_path: Path, stub: _Stub):
    records: list = []
    native = json.loads(
        _materialize(
            _job(tmp_path, SEQUENCE), "protenix", tmp_path / "out",
            msa_pairing="none", msa_search=records,
        ).read_text()
    )
    chain = native[0]["sequences"][0]["proteinChain"]
    assert "pairedMsaPath" not in chain
    assert Path(chain["unpairedMsaPath"]).is_file()
    assert stub.unpaired_only_calls == 1
    provenance = json.loads(Path(records[0]["provenance"]).read_text())
    assert provenance["options"] == {"pairing": "none"}


def test_model_pairing_is_unchanged_per_model(tmp_path: Path, stub: _Stub):
    native = json.loads(
        _materialize(_job(tmp_path, SEQUENCE), "alphafold3", tmp_path / "p").read_text()
    )
    assert "pairedMsaPath" in native["sequences"][0]["protein"]
    assert stub.complex_calls == []
    # Boltz-2's own choice is upstream's one greedy complex search.
    _materialize(_job(tmp_path, SEQUENCE, OTHER), "boltz2", tmp_path / "b")
    assert stub.unpaired_only_calls == 0
    assert stub.complex_calls == [([SEQUENCE, OTHER], None)]


def test_openfold3_complete_pairing_runs_the_complete_strategy(
    tmp_path: Path, stub: _Stub
):
    native = json.loads(
        _materialize(
            _job(tmp_path, SEQUENCE, OTHER), "openfold3", tmp_path / "out",
            msa_pairing="complete",
        ).read_text()
    )
    assert stub.complex_calls == [([SEQUENCE, OTHER], "paircomplete-env")]
    (query,) = native["queries"].values()
    assert all("paired_msa_file_paths" in chain for chain in query["chains"])


def test_boltz2_greedy_pairing_writes_upstreams_keyed_csv(tmp_path: Path, stub: _Stub):
    import yaml

    records: list = []
    path = _materialize(
        _job(tmp_path, SEQUENCE, OTHER), "boltz2", tmp_path / "out",
        msa_pairing="greedy", msa_search=records,
    )
    native = yaml.safe_load(path.read_text())
    assert stub.complex_calls == [([SEQUENCE, OTHER], None)]
    first, second = (entry["protein"] for entry in native["sequences"])
    assert first["msa"].endswith(".csv") and second["msa"].endswith(".csv")
    lines = Path(second["msa"]).read_text().splitlines()
    # Paired rows keep their row number (the all-gap row 2 is dropped), then
    # the unpaired rows with key -1, minus the query the paired block holds.
    assert lines == [
        "key,sequence",
        f"0,{OTHER}",
        f"1,{OTHER}",
        f"-1,{OTHER}",
        f"-1,{'A' * len(OTHER)}",
    ]
    assert all("paired_msa" in record for record in records)


def test_species_pairing_backend_refuses_a_complex_pairing(tmp_path: Path):
    # Protenix and OpenDDE read a complex search by row (tests/test_w15_msa_pairing.py).
    with pytest.raises(ValueError, match="species"):
        _materialize(_job(tmp_path, SEQUENCE, OTHER), "alphafold3", tmp_path / "o",
                     msa_pairing="greedy")


def test_pairing_needs_a_search_and_a_reader(tmp_path: Path):
    job = _job(tmp_path, SEQUENCE)
    with pytest.raises(ValueError, match="searched alignment"):
        PredictionRequest(model="boltz2", input=job, msa_pairing="none")
    with pytest.raises(ValueError, match="no paired alignment"):
        msa_search.refuse_msa_pairing("esmfold2", "complete")
    assert msa_search.resolve_pairing("openfold3") == {
        "requested": "model", "resolved": "greedy", "mode": "pairgreedy-env",
        "paired_by": "row",
    }
    assert msa_search.resolve_pairing("boltz2")["resolved"] == "greedy"
    assert msa_search.resolve_pairing("esmfold2")["resolved"] == "none"
    assert msa_search.resolve_pairing("alphafold3")["resolved"] == "per_chain"


def test_cli_pairing_reaches_request_and_plan(tmp_path: Path):
    job = _job(tmp_path, SEQUENCE)
    args = cli._parser().parse_args(
        ["plan", "--model", "boltz2", "--input", str(job), "--msa", "auto",
         "--msa-pairing", "none"]
    )
    assert cli._request(args).msa_pairing == "none"


def test_manifest_records_pairing_and_resume_compares_it(tmp_path: Path):
    from foldjax import manifest
    from foldjax.schema import PredictionResult, PredictionSample
    from tests.test_resume_manifest import _file, _request

    request = _request(tmp_path, msa="auto", msa_pairing="none")
    request.output_dir.mkdir()
    structure = _file(request.output_dir / "sample.cif", b"data_mock\n#\n")
    result = PredictionResult(
        model="boltz2",
        samples=(PredictionSample(seed=7, structure_path=structure, scores={}),),
        output_dir=request.output_dir,
    )
    document = manifest.describe_run(
        request, result, directory=request.output_dir, msa_stats=[]
    )
    assert document["msa_pairing"] == {
        "requested": "none", "resolved": "none", "mode": None, "paired_by": None,
    }
    assert document["msa_stats"]["chains"] == []
    assert manifest.matches_request(document, request, seed=7)
    other = PredictionRequest(
        **{
            **{name: getattr(request, name) for name in request.__dataclass_fields__},
            "msa_pairing": "model",
        }
    )
    assert not manifest.matches_request(document, other, seed=7)
    # Written before the field existed: ran at "model".
    old = {key: value for key, value in document.items() if key != "msa_pairing"}
    assert manifest.matches_request(old, other, seed=7)


# -- 2. alignment depth and Neff ------------------------------------------------


def test_neff_counts_clusters_at_eighty_percent_identity():
    from foldjax.msa_stats import neff

    query = "ACDEFGHIKL"
    near = "ACDEFGHIKV"  # 9/10 identical: one cluster with the query
    far = "WWWWWGHIKL"  # 5/10: its own
    value, used = neff([query, near, far, far])
    assert used == 4
    # query and near weigh 1/2 each; the two copies of far 1/2 each.
    assert value == pytest.approx(2.0)
    # Lower-case insertions are not match columns; a row of another length is
    # not this alignment's and is left out.
    value, used = neff([query, "ACDefEFGHIKL", "ACD"])
    assert (value, used) == (pytest.approx(1.0), 2)


def test_alignment_stats_reach_manifest_show_and_rows(tmp_path: Path):
    from foldjax.msa_stats import alignment_stats, job_msa_stats

    alignment = tmp_path / "a.a3m"
    alignment.write_text(">q\nACDEFGHIKL\n>1\nACDEFGHIKV\n>2\nWWWWWGHIKL\n")
    record = alignment_stats(alignment)
    assert record["depth"] == 3 and record["neff"] == pytest.approx(2.0)
    job = {
        "entities": [
            {"type": "protein", "id": "A", "sequence": "ACDEFGHIKL",
             "unpaired_msa": "a.a3m"},
            {"type": "ligand", "id": "L", "ccd": "ATP"},
        ]
    }
    (chain,) = job_msa_stats(job, tmp_path)
    assert chain["chains"] == ["A"] and chain["unpaired_msa"]["depth"] == 3

    from tests.test_output_contract import _run

    _confidence, manifest, _fixture = _run(tmp_path / "run", "boltz2_e9_8reh")
    stats = manifest["msa_stats"]
    assert "80% identity" in stats["definition"]
    (entry,) = stats["chains"]
    assert entry["unpaired_msa"]["depth"] == 1
    assert entry["unpaired_msa"]["neff"] == pytest.approx(1.0)

    from foldjax import report, results

    out = tmp_path / "run" / "out"
    text = report.render_all(report.read_manifests(out))
    assert "alignment A: unpaired 1 rows, Neff 1.0" in text
    (row,) = results.results_table(results.load_results(out))
    assert row["msa_stats"] == [
        {"chains": ["A"], "depth": 1, "neff": 1.0,
         "paired_depth": None, "paired_neff": None}
    ]


# -- 3. show --rank-by -----------------------------------------------------------


def _rows() -> list[dict[str, Any]]:
    return [
        {"status": "ok", "model": "boltz2", "configuration": "c1", "input": "a",
         "input_name": "a", "seed": 1, "sample": 0, "plddt": 70.0, "score.pae": 4.0},
        {"status": "ok", "model": "boltz2", "configuration": "c1", "input": "b",
         "input_name": "b", "seed": 1, "sample": 0, "plddt": 90.0, "score.pae": 2.0},
        {"status": "ok", "model": "protenix", "configuration": "c2", "input": "a",
         "input_name": "a", "seed": 1, "sample": 0, "plddt": 80.0},
        {"status": "ok", "model": "protenix", "configuration": "c2", "input": "b",
         "input_name": "b", "seed": 1, "sample": 0, "plddt": None},
        {"status": "failed", "model": "protenix", "input": "c", "error": "x"},
    ]


def test_rank_by_ranks_within_each_model_only():
    from foldjax.results import rank_rows, render_ranked

    ranked = rank_rows(_rows(), "plddt")
    order = [(row["model"], row["input"], row["rank_within_model"]) for row in ranked]
    assert order == [
        ("boltz2", "b", 1),
        ("boltz2", "a", 2),
        ("protenix", "a", 1),
        ("protenix", "b", None),
    ]
    ascending = rank_rows(_rows(), "score.pae:asc")
    assert [row["input"] for row in ascending if row["model"] == "boltz2"] == ["b", "a"]
    assert "not comparable across models" in render_ranked(ranked, "plddt")
    with pytest.raises(ValueError, match="no sample has a numeric value"):
        rank_rows(_rows(), "nonsense")


def test_show_rank_by_on_a_written_batch(tmp_path: Path, capsys):
    from tests.test_results_loader import _batch

    root = _batch(tmp_path, ("boltz2", "protenix"))
    assert cli.main(["show", str(root), "--rank-by", "plddt", "--format", "json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["rank_within_model"] for row in rows} == {1}
    assert {row["model"] for row in rows} == {"boltz2", "protenix"}
    assert cli.main(["show", str(root), "--rank-by", "ranking"]) == 0
    assert "by ranking" in capsys.readouterr().out
    with pytest.raises(ValueError, match="--rank-by orders samples"):
        cli.main(["show", str(root), "--rank-by", "plddt", "--screen"])


# -- 4. msa prefetch and the reference wrapper --------------------------------


def test_prefetch_caches_and_writes_nothing_else(
    tmp_path: Path, stub: _Stub, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    job = _job(tmp_path / "jobs", SEQUENCE, OTHER)
    assert cli.main(["msa", "prefetch", str(job)]) == 0
    records = json.loads(capsys.readouterr().out)
    assert sorted(record["chain"] for record in records) == ["A", "B"]
    assert all(Path(record["unpaired_msa"]).is_file() for record in records)
    assert stub.calls == [SEQUENCE, OTHER]
    assert not (tmp_path / "foldjax-outputs").exists()

    # predict would now find the cache: a second prefetch searches nothing.
    assert cli.main(["msa", "prefetch", str(job), "--model", "openfold3"]) == 0
    capsys.readouterr()
    assert stub.calls == [SEQUENCE, OTHER]
    assert stub.complex_calls == [([SEQUENCE, OTHER], None)]


def test_prefetch_exits_3_when_a_search_fails(tmp_path: Path, monkeypatch, capsys):
    class _Broken(_Stub):
        def search(self, sequence):
            raise OSError("server down")

    backend = _Broken()
    monkeypatch.setattr(
        msa_search,
        "_msa_pipeline",
        lambda: MsaSearchPipeline(tmp_path / "msa-cache", backend),
    )
    job = _job(tmp_path, SEQUENCE)
    assert cli.main(["msa", "prefetch", str(job)]) == 3
    (record,) = json.loads(capsys.readouterr().out)
    assert "server down" in record["error"]


def test_a_failed_prefetch_does_not_claim_to_fold(
    tmp_path: Path, monkeypatch, capsys, recwarn
):
    # Prefetch folds nothing, so its failure says what was not cached; the
    # predict wording ("folding them from single sequence") misled a batch.
    class _Broken(_Stub):
        def search(self, sequence):
            raise OSError("server down")

    monkeypatch.setattr(
        msa_search,
        "_msa_pipeline",
        lambda: MsaSearchPipeline(tmp_path / "msa-cache", _Broken()),
    )
    job = _job(tmp_path, SEQUENCE)
    assert cli.main(["msa", "prefetch", str(job)]) == 3
    said = capsys.readouterr().err + "".join(str(w.message) for w in recwarn)
    assert "MSA prefetch failed for chain(s) A (server down)" in said
    assert "no alignment was cached for them" in said
    assert "folding" not in said


_FAKE_COLABFOLD = '''
import os
from pathlib import Path

CALLS = Path(os.environ["FAKE_COLABFOLD_LOG"])


def _log(name, **kwargs):
    with CALLS.open("a") as handle:
        pairs = " ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        handle.write(name + " " + pairs + "\\n")


def run_mmseqs(mmseqs, params):
    _log("run_mmseqs", module=params[0])
    query = Path(params[1]).read_text().splitlines()[1]
    Path(params[2]).with_suffix(".query").write_text(query)


def _query(base):
    return (Path(base) / "qdb.query").read_text()


def mmseqs_search_monomer(dbbase, base, **kwargs):
    _log("monomer", gpu=kwargs["gpu"], threads=kwargs["threads"])
    q = _query(base)
    (Path(base) / "0.a3m").write_text(f">101\\n{q}\\n>UniRef100_U\\t1\\n{q}\\n")


def mmseqs_search_pair(dbbase, base, **kwargs):
    _log("pair", pair_env=kwargs["pair_env"], gpu=kwargs["gpu"])
    q = _query(base)
    (Path(base) / "0.paired.a3m").write_text(f">101\\n{q}\\n>UniRef100_P\\t1\\n{q}\\n")
'''


@pytest.fixture
def fake_colabfold(tmp_path: Path, monkeypatch) -> Path:
    package = tmp_path / "fakecf" / "colabfold" / "mmseqs"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "search.py").write_text(_FAKE_COLABFOLD)
    log = tmp_path / "colabfold.log"
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fakecf"))
    monkeypatch.setenv("FAKE_COLABFOLD_LOG", str(log))
    return log


def test_reference_wrapper_meets_the_local_search_contract(
    tmp_path: Path, fake_colabfold: Path, capsys
):
    assert cli.main(["msa", "wrapper"]) == 0
    wrapper = Path(capsys.readouterr().out.strip())
    assert wrapper.name == "colabfold_local.py" and wrapper.is_file()
    # It runs by path under another interpreter: no FoldJAX import allowed.
    assert "foldjax" not in "".join(
        line for line in wrapper.read_text().splitlines()
        if line.startswith(("import", "from"))
    )
    command = [sys.executable, str(wrapper), "--db", str(tmp_path), "--gpu",
               "--mmseqs", sys.executable]
    client = LocalMsaClient(command, version="cf-local")
    payload = client.search(SEQUENCE)
    assert payload.unpaired.startswith(f">101\n{SEQUENCE}\n>UniRef100_U")
    assert payload.paired.startswith(f">101\n{SEQUENCE}\n>UniRef100_P")
    log = fake_colabfold.read_text()
    assert "monomer gpu=1" in log and "pair gpu=True pair_env=False" in log

    fake_colabfold.write_text("")
    unpaired_only = client.unpaired_only().search(SEQUENCE)
    assert unpaired_only.paired == f">101\n{SEQUENCE}\n"
    assert "pair " not in fake_colabfold.read_text()


def test_wrapper_without_colabfold_says_which_interpreter(tmp_path: Path):
    import importlib.util
    import os

    from foldjax.search import colabfold_local

    if importlib.util.find_spec("colabfold") is not None:
        pytest.skip("ColabFold is installed in this interpreter")
    fasta = tmp_path / "q.fasta"
    fasta.write_text(f">q\n{SEQUENCE}\n")
    completed = subprocess.run(
        [sys.executable, colabfold_local.__file__, "--db", str(tmp_path),
         "--input", str(fasta), "--output", str(tmp_path / "o")],
        capture_output=True, text=True, check=False,
        env={**os.environ},
    )
    assert completed.returncode == 2
    assert "pip install colabfold" in completed.stderr


# -- 5. --templates DIR ----------------------------------------------------------


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    from tests.test_template_search import QUERY, _mmcif

    directory = tmp_path / "private"
    directory.mkdir()
    # No release date at all: a private structure usually has none.
    hit = _mmcif("zz01", "2001-01-01", [("A", "X", "GS" + QUERY[2:28] + "PP", {1})])
    lines = hit.splitlines()
    start = lines.index("_pdbx_audit_revision_history.ordinal") - 1
    hit = "\n".join(lines[:start] + lines[start + 5 :])
    assert "audit" not in hit
    (directory / "In-House Model_1.cif").write_text(hit + "\n")
    (directory / "unrelated.cif").write_text(
        _mmcif("zz02", "2001-01-01", [("A", "B", "W" * 30, set())])
    )
    return directory


@pytest.fixture(autouse=True)
def _no_alphafold3_runtime(monkeypatch) -> None:
    """AlphaFold 3's writer filters a template with its compiled runtime."""
    monkeypatch.setattr(
        "foldjax.input._filter_alphafold3_template_mmcif",
        lambda source, chain_id: Path(source).read_text(),
    )


def _no_mmseqs(monkeypatch) -> None:
    import shutil

    real = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda name, *a, **k: None if name == "mmseqs" else real(name)
    )


def _template_job(tmp_path: Path) -> Path:
    from tests.test_template_search import QUERY

    return _write(
        tmp_path / "tjob.json",
        {"name": "t", "entities": [{"type": "protein", "id": "Q", "sequence": QUERY}]},
    )


def test_a_private_folder_feeds_the_template_path_via_kalign(
    tmp_path: Path, folder: Path, monkeypatch
):
    pytest.importorskip("kalign")
    _no_mmseqs(monkeypatch)
    records: list = []
    path = materialize_native_input(
        _template_job(tmp_path), capabilities("alphafold3"), tmp_path / "out",
        seed=1, msa="single", templates="auto", template_dir=folder,
        template_search=records,
    )
    (record,) = records
    assert record["source"]["kind"] == "private folder"
    assert record["source"]["aligner"] == "kalign"
    # The model's 2021-09-30 cutoff would drop a dateless structure.
    assert record["cutoff"]["max_template_date"] is None
    assert [item["pdb_id"] for item in record["templates"]] == ["inhousemodel1"]
    native = json.loads(path.read_text())
    (template,) = native["sequences"][0]["protein"]["templates"]
    assert template["queryIndices"]


def test_a_private_folder_reaches_boltz2_as_files(tmp_path: Path, folder, monkeypatch):
    import yaml

    _no_mmseqs(monkeypatch)
    pytest.importorskip("kalign")
    path = materialize_native_input(
        _template_job(tmp_path), capabilities("boltz2"), tmp_path / "out",
        seed=1, msa="single", templates="auto", template_dir=folder,
    )
    templates = yaml.safe_load(path.read_text())["templates"]
    assert [Path(item["cif"]).name for item in templates] == ["In-House Model_1.cif"]


def test_a_private_folder_uses_mmseqs_when_installed(
    tmp_path: Path, folder: Path, monkeypatch
):
    pytest.importorskip("kalign")
    from tests.test_template_search import QUERY

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "mmseqs"
    # easy-search QUERY TARGETS OUT TMP: report the in-house chain as a hit.
    fake.write_text(
        "#!/bin/sh\n"
        "printf 'query\\tinhousemodel1_X\\t0.9\\t26\\t0\\t0\\t3\\t28\\t3\\t28\\t1e-20"
        "\\t150\\n' > \"$4\"\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    records: list = []
    materialize_native_input(
        _template_job(tmp_path), capabilities("alphafold3"), tmp_path / "out",
        seed=1, msa="single", templates="auto", template_dir=folder,
        template_search=records,
    )
    (record,) = records
    assert record["source"]["aligner"] == "mmseqs"
    assert record["templates"][0]["e_value"] == 1e-20
    assert len(QUERY) > 0


def test_a_private_folder_is_refused_without_an_aligner(
    tmp_path: Path, folder: Path, monkeypatch
):
    import importlib.util

    from foldjax.template_search import refuse_template_search

    _no_mmseqs(monkeypatch)
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a: None if name == "kalign" else real(name, *a),
    )
    with pytest.raises(ValueError, match="Kalign"):
        refuse_template_search("alphafold3", "auto", None, template_dir=folder)
    with pytest.raises(ValueError, match="needs a local aligner"):
        refuse_template_search("boltz2", "auto", None, template_dir=folder)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="holds no"):
        refuse_template_search("boltz2", "auto", None, template_dir=empty)


def test_plan_answers_a_server_template_search_without_kalign_as_predict_does(
    tmp_path: Path, monkeypatch, capsys
):
    """`plan` used to pass silently: every searched hit is realigned with
    Kalign, so `required` fails the run and `auto` folds without templates."""
    import importlib.util

    from foldjax.template_search import refuse_template_search

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a: None if name == "kalign" else real(name, *a),
    )
    with pytest.raises(ValueError, match="realigns each hit with Kalign"):
        refuse_template_search("protenix", "required", {"use_template": True})
    refuse_template_search("protenix", "auto", {"use_template": True})
    # Boltz-2 reads a hit as it comes; nothing to realign.
    refuse_template_search("boltz2", "required", None)

    weights = tmp_path / "w.jax"
    weights.write_bytes(b"w")
    argv = [
        "plan", "--model", "protenix", "--sequence", "MKTAYIAKQRQISFVK",
        "--msa", "single", "--weights", str(weights),
        "--option", "use_template=true", "--templates",
    ]
    with pytest.warns(UserWarning, match="--templates auto will fold without"):
        assert cli.main([*argv, "auto"]) == 0
    with pytest.raises(ValueError, match="realigns each hit with Kalign"):
        cli.main([*argv, "required"])


def test_cli_templates_takes_a_directory(tmp_path: Path, folder: Path):
    job = _template_job(tmp_path)
    args = cli._parser().parse_args(
        ["plan", "--model", "boltz2", "--input", str(job), "--msa", "single",
         "--templates", str(folder)]
    )
    request = cli._request(args)
    assert (request.templates, request.template_dir) == ("auto", folder)
    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            ["plan", "--model", "boltz2", "--input", str(job),
             "--templates", str(tmp_path / "missing")]
        )
    with pytest.raises(ValueError, match="template_dir"):
        PredictionRequest(model="boltz2", input=job, template_dir=folder)


def test_manifest_hashes_the_folder_and_resume_sees_a_change(
    tmp_path: Path, folder: Path
):
    from foldjax import manifest
    from foldjax.schema import PredictionResult, PredictionSample
    from tests.test_resume_manifest import _file, _request

    request = _request(tmp_path, templates="auto", template_dir=folder)
    request.output_dir.mkdir()
    structure = _file(request.output_dir / "sample.cif", b"data_mock\n#\n")
    result = PredictionResult(
        model="boltz2",
        samples=(PredictionSample(seed=7, structure_path=structure, scores={}),),
        output_dir=request.output_dir,
    )
    document = manifest.describe_run(request, result, directory=request.output_dir)
    assert document["template_dir"]["files"] == 2
    assert manifest.matches_request(document, request, seed=7)
    (folder / "unrelated.cif").write_text("data_changed\n")
    assert not manifest.matches_request(document, request, seed=7)


# -- 6. --preset fast -------------------------------------------------------------


def test_fast_is_published_only_for_protenix_mini():
    from foldjax.presets import resolve_preset

    sampling, source = resolve_preset("fast", "protenix", "mini-esm-v0.5.0")
    assert sampling == {"num_steps": 5, "num_recycles": 4}
    assert "supported_models.md" in source
    with pytest.raises(ValueError, match="mini-esm-v0.5.0"):
        resolve_preset("fast", "protenix", "released")
    for model in ("alphafold3", "boltz2", "esmfold2", "opendde", "openfold3"):
        with pytest.raises(ValueError, match=f"not available for {model}"):
            resolve_preset("fast", model, "released")


@pytest.mark.parametrize(
    "profile",
    [
        "mini-default-v0.5.0",
        "mini-esm-v0.5.0",
        "mini-ism-v0.5.0",
        "tiny-default-v0.5.0",
    ],
)
def test_fast_matches_the_mini_checkpoints_own_schedule(profile: str):
    from foldjax.backends.protenix import _PROFILE_MODEL_NAMES
    from foldjax.models.protenix.runtime_policy import model_inference_defaults
    from foldjax.presets import resolve_preset

    sampling, _ = resolve_preset("fast", "protenix", profile)
    defaults = model_inference_defaults(_PROFILE_MODEL_NAMES[profile])
    assert sampling == {key: defaults[key] for key in sampling}


def test_preset_refuses_a_conflicting_knob_and_is_recorded(tmp_path: Path):
    from types import SimpleNamespace

    from foldjax.presets import preset_record, preset_updates

    request = SimpleNamespace(preset="fast", num_steps=200, num_recycles=None)
    with pytest.raises(ValueError, match="--num-steps 200"):
        preset_updates(request, model="protenix", profile="mini-esm-v0.5.0")
    request.num_steps = 5
    assert preset_updates(request, model="protenix", profile="mini-esm-v0.5.0") == {
        "num_steps": 5, "num_recycles": 4,
    }
    recorded = preset_record(
        SimpleNamespace(preset="fast", model="protenix", profile="mini-esm-v0.5.0")
    )
    assert recorded["name"] == "fast" and recorded["sampling"]["num_steps"] == 5
    job = _job(tmp_path, SEQUENCE)
    args = cli._parser().parse_args(
        ["predict", "--model", "protenix", "--input", str(job), "--preset", "fast"]
    )
    assert cli._request(args).preset == "fast"


def test_resolve_request_applies_the_preset(tmp_path: Path, monkeypatch):
    from foldjax import api

    job = _job(tmp_path, SEQUENCE)
    weights = tmp_path / "w.jax"
    weights.write_bytes(b"x")
    request = PredictionRequest(
        model="boltz2", input=job, weights=weights, preset="fast", msa="single"
    )
    with pytest.raises(ValueError, match="Boltz-2 publishes only"):
        api.resolve_request(request, draw_seeds=False)


# -- 7. ModelCIF QA metadata ----------------------------------------------------


_PLAIN_CIF = """data_x
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 N N ALA A 1 0 0 0 80.0 1
ATOM 2 C CA ALA A 1 1 0 0 90.0 1
ATOM 3 N N GLY A 2 2 0 0 50.0 1
HETATM 4 C C1 LIG B . 3 0 0 40.0 1
"""


def _categories(path: Path) -> dict[str, list[list[str]]]:
    from gemmi import cif

    block = cif.read(str(path)).sole_block()
    out = {}
    for category, tags in {
        "_ma_qa_metric.": ["id", "mode", "software_group_id"],
        "_ma_qa_metric_global.": ["metric_id", "metric_value"],
        "_ma_qa_metric_local.": [
            "label_asym_id", "label_seq_id", "metric_id", "metric_value",
        ],
        "_software.": ["pdbx_ordinal", "name", "version"],
    }.items():
        out[category] = [
            [
                row[i] if row[i] in (".", "?") else cif.as_string(row[i])
                for i in range(len(tags))
            ]
            for row in block.find(category, tags)
        ]
    return out


def test_qa_metrics_and_software_are_added_to_a_plain_cif(tmp_path: Path):
    from foldjax import __version__
    from foldjax.output import _normalize_cif

    path = tmp_path / "s.cif"
    path.write_text(_PLAIN_CIF)
    _normalize_cif(path, job="j", model="protenix", seed=1, index=0, plddt=71.234)
    found = _categories(path)
    assert [row[1] for row in found["_ma_qa_metric."]] == ["global", "local"]
    assert found["_ma_qa_metric_global."] == [["1", "71.23"]]
    assert found["_ma_qa_metric_local."] == [
        ["A", "1", "2", "85.00"],
        ["A", "2", "2", "50.00"],
        ["B", ".", "2", "40.00"],
    ]
    assert found["_software."] == [
        ["1", "FoldJAX", __version__],
        ["2", "Protenix", "2.0.0"],
    ]
    # Idempotent: a second pass adds nothing.
    _normalize_cif(path, job="j", model="protenix", seed=1, index=0, plddt=71.234)
    assert _categories(path) == found


def test_existing_qa_categories_are_kept_and_extended(tmp_path: Path):
    from foldjax.output import _normalize_cif

    path = tmp_path / "s.cif"
    path.write_text(
        _PLAIN_CIF
        + "_software.pdbx_ordinal 1\n_software.name AlphaFold\n"
        + "_software.version 'AlphaFold-beta-20231127'\n"
        + "loop_\n_ma_qa_metric.id\n_ma_qa_metric.name\n_ma_qa_metric.type\n"
        + "_ma_qa_metric.mode\n_ma_qa_metric.software_group_id\n"
        + "1 pLDDT pLDDT local 1\n"
        + "loop_\n_ma_qa_metric_local.ordinal_id\n_ma_qa_metric_local.model_id\n"
        + "_ma_qa_metric_local.label_asym_id\n_ma_qa_metric_local.label_seq_id\n"
        + "_ma_qa_metric_local.label_comp_id\n_ma_qa_metric_local.metric_id\n"
        + "_ma_qa_metric_local.metric_value\n1 1 A 1 ALA 1 33.00\n"
    )
    _normalize_cif(path, job="j", model="alphafold3", seed=1, index=0)
    found = _categories(path)
    # The writer's local metric is untouched; a global one is added after it,
    # valued at the atom mean (no summary value was given).
    metrics = [row[:2] for row in found["_ma_qa_metric."]]
    assert metrics == [["1", "local"], ["2", "global"]]
    assert found["_ma_qa_metric_local."] == [["A", "1", "1", "33.00"]]
    assert found["_ma_qa_metric_global."] == [["2", "65.00"]]
    names = [row[1] for row in found["_software."]]
    assert names == ["AlphaFold", "FoldJAX"]


@pytest.mark.parametrize(
    "case_name",
    [
        "alphafold3_e9_8reh",
        "boltz2_e9_8reh",
        "esmfold2_e9_8reh",
        "opendde_e9_8reh",
        "openfold3_e9_8reh",
        "protenix_e9_8reh",
    ],
)
def test_every_model_writes_modelcif_qa_and_software(tmp_path: Path, case_name):
    from tests.test_output_contract import _run

    _confidence, manifest, _fixture = _run(tmp_path, case_name)
    structure = tmp_path / "out" / manifest["samples"][0]["structure_path"]
    found = _categories(structure)
    modes = {row[1] for row in found["_ma_qa_metric."]}
    assert {"global", "local"} <= modes
    assert found["_ma_qa_metric_global."] and found["_ma_qa_metric_local."]
    assert all(0.0 <= float(row[3]) <= 100.0 for row in found["_ma_qa_metric_local."])
    names = " ".join(row[1] for row in found["_software."]).lower()
    assert "foldjax" in names
    upstream = {
        "alphafold3": "alphafold", "boltz2": "boltz", "esmfold2": "esmfold2",
        "opendde": "opendde", "openfold3": "openfold3", "protenix": "protenix",
    }[manifest["model"]]
    assert upstream in names
