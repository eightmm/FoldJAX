"""Sequence normalization, chain ids, field suggestions, and alignment policy.

These cover the boundary where a person's document becomes a model's input --
the place where a mistake used to be reported by a featurizer several layers
down, or not reported at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foldjax.input import (
    assign_chain_ids,
    common_schema_features,
    compatibility,
    materialize_native_input,
    native_only_features,
    read_job_document,
)
from foldjax.job import Job, parse_fasta
from foldjax.registry import capabilities

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQ"


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _materialize(source: Path, model: str, out: Path, **kwargs: Any) -> Path:
    # These tests are about translation, not the missing-alignment policy.
    kwargs.setdefault("msa", "single")
    return materialize_native_input(source, capabilities(model), out, seed=0, **kwargs)


def test_a_pasted_block_scalar_sequence_loses_its_whitespace(tmp_path: Path) -> None:
    """`sequence: |` is how people paste, and its newlines used to survive."""
    source = tmp_path / "job.yaml"
    source.write_text(
        "name: demo\n"
        "entities:\n"
        "  - type: protein\n"
        "    id: A\n"
        "    sequence: |\n"
        "      MKTAYIAKQRQ\n"
        "      ISFVKSHFSRQ\n",
        encoding="utf-8",
    )

    native = json.loads(_materialize(source, "protenix", tmp_path / "out").read_text())

    assert native[0]["sequences"][0]["proteinChain"]["sequence"] == SEQUENCE


def test_a_lowercase_sequence_is_upper_cased(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE.lower()}]},
    )

    native = json.loads(_materialize(source, "protenix", tmp_path / "out").read_text())

    assert native[0]["sequences"][0]["proteinChain"]["sequence"] == SEQUENCE


def test_a_non_letter_in_a_sequence_points_at_the_position(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": "A", "sequence": "MKTA-YIAK"}]},
    )

    with pytest.raises(ValueError, match="position 5"):
        _materialize(source, "protenix", tmp_path / "out")


def test_a_protein_pasted_into_a_dna_entity_is_refused(tmp_path: Path) -> None:
    """`ACGT` is a valid protein, so only the reverse mistake is detectable."""
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "dna", "id": "A", "sequence": "ACGTQEIL"}]},
    )

    with pytest.raises(ValueError, match="unsupported residue 'Q'"):
        _materialize(source, "protenix", tmp_path / "out")

    source = _write(
        tmp_path / "dna.json",
        {"entities": [{"type": "dna", "id": "A", "sequence": "ACGTNRY"}]},
    )
    assert _materialize(source, "protenix", tmp_path / "ok").is_file()


def test_absent_chain_ids_are_assigned_around_the_explicit_ones() -> None:
    entities = [
        {"type": "protein", "sequence": SEQUENCE},
        {"type": "protein", "id": "A", "sequence": SEQUENCE},
        {"type": "ligand", "id": ["C", "D"]},
        {"type": "protein", "sequence": SEQUENCE},
    ]

    assign_chain_ids(entities)

    assert [entity["id"] for entity in entities] == ["B", "A", ["C", "D"], "E"]


def test_chain_ids_continue_past_the_alphabet() -> None:
    entities = [{"type": "protein", "sequence": SEQUENCE} for _ in range(27)]

    assign_chain_ids(entities)

    assert entities[25]["id"] == "Z"
    assert entities[26]["id"] == "AA"


def test_a_misspelled_field_names_the_one_that_was_meant(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "job.json",
        {
            "entities": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": SEQUENCE,
                    "unpared_msa": "x.a3m",
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="did you mean 'unpaired_msa'"):
        _materialize(source, "protenix", tmp_path / "out")


def test_an_unrecognizable_field_is_still_refused_without_a_guess(
    tmp_path: Path,
) -> None:
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}], "zz": 1},
    )

    with pytest.raises(ValueError, match=r"unsupported top-level fields: 'zz'$"):
        _materialize(source, "protenix", tmp_path / "out")


def test_folding_without_an_alignment_says_so(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}]},
    )

    with pytest.warns(UserWarning, match="single sequence"):
        _materialize(source, "protenix", tmp_path / "out", msa="single")


def test_a_chain_with_an_alignment_is_not_warned_about(tmp_path: Path) -> None:
    alignment = tmp_path / "a.a3m"
    alignment.write_text(f">query\n{SEQUENCE}\n", encoding="utf-8")
    source = _write(
        tmp_path / "job.json",
        {
            "entities": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": SEQUENCE,
                    "unpaired_msa": str(alignment),
                }
            ]
        },
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _materialize(source, "protenix", tmp_path / "out")


class _StubSearch:
    """A search backend that answers without a network."""

    name = "stub"
    version = "1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, sequence: str):
        from foldjax.search.msa import MsaPayload

        self.calls.append(sequence)
        return MsaPayload(
            paired=f">query\n{sequence}\n>b\n{sequence}\n",
            unpaired=f">query\n{sequence}\n>c\n{sequence}\n",
            source={"stub": True},
        )


def _stub_pipeline(tmp_path: Path, backend: _StubSearch):
    from foldjax.search.msa import MsaSearchPipeline

    return MsaSearchPipeline(tmp_path / "msa-cache", backend)


def test_msa_auto_fills_in_the_alignment_and_reaches_the_dialect(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _StubSearch()
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
    )
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}]},
    )

    written = _materialize(source, "protenix", tmp_path / "out", msa="auto")
    native = json.loads(written.read_text())

    chain = native[0]["sequences"][0]["proteinChain"]
    assert Path(chain["unpairedMsaPath"]).is_file()
    # A monomer gets no pairing alignment, as upstream Protenix's ColabFold
    # mode writes it a query-only pairing.a3m.
    assert "pairedMsaPath" not in chain
    assert backend.calls == [SEQUENCE]
    searched = json.loads((tmp_path / "out" / "msa_search.json").read_text())
    assert searched[0]["chain"] == "A"


def test_a_searched_alignment_is_reused_for_the_next_model(
    tmp_path: Path, monkeypatch
) -> None:
    """The cache is keyed by sequence, not by model: three backends, one search."""
    backend = _StubSearch()
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
    )
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}]},
    )

    _materialize(source, "protenix", tmp_path / "one", msa="auto")
    _materialize(source, "boltz2", tmp_path / "two", msa="auto")

    assert backend.calls == [SEQUENCE]


def test_boltz_takes_the_unpaired_alignment_and_never_a_paired_one(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _StubSearch()
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
    )
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}]},
    )

    native = json.loads(
        _materialize(source, "boltz2", tmp_path / "out", msa="auto").read_text()
    )

    chain = native["sequences"][0]["protein"]
    assert chain["msa"].endswith(".a3m")


def test_boltz_server_csv_is_upstreams_compute_msa() -> None:
    """Row for row what `boltz/main.py` `compute_msa` writes for one entity."""
    from foldjax.input import boltz_server_msa_csv

    paired = ">101\nAAAA\n>p1\n----\n>p2\nAC-A\n"
    unpaired = ">101\nAAAA\n>u1\nACAA\n>u2\nAAcAA\n"
    assert boltz_server_msa_csv(paired, unpaired).splitlines() == [
        "key,sequence",
        # Paired rows keep their row number as the key; the all-gap row goes.
        "0,AAAA",
        "2,AC-A",
        # The query is already the first paired row, so the unpaired one goes.
        "-1,ACAA",
        "-1,AAcAA",
    ]
    # No paired block (a single entity): the unpaired rows alone, query kept.
    assert boltz_server_msa_csv("", unpaired).splitlines() == [
        "key,sequence",
        "-1,AAAA",
        "-1,ACAA",
        "-1,AAcAA",
    ]


def _boltz_accepts_the_msa_fields(native: dict[str, Any]) -> None:
    """Run Boltz's own schema parser far enough to check the MSA fields.

    Its same-sequence/same-MSA check runs before any chemistry is looked up;
    with no CCD the parse then stops at the first residue lookup, which is not
    what this asserts.
    """
    from foldjax.models.boltz2.data.parse.schema import parse_boltz_schema

    try:
        parse_boltz_schema("job", native, {}, None, boltz_2=True)
    except ValueError as error:
        assert "share the same MSA" not in str(error), error
    except (KeyError, AttributeError, TypeError, FileNotFoundError):
        pass


def test_boltz_pairs_a_heteromer_in_one_complex_search(
    tmp_path: Path, monkeypatch
) -> None:
    """`--msa auto` on a heteromer pairs it, as Boltz-2's server search does.

    Boltz submits its protein entities together as one `pairgreedy-env` job
    and writes each a CSV whose paired rows share a key (`main.py`
    `compute_msa`). The unpaired search alone left the complex unpaired.
    """
    from foldjax.models.boltz2.data.parse.csv import parse_csv

    backend = _ComplexStubSearch()
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
    )

    native = json.loads(
        _materialize(_heteromer(tmp_path), "boltz2", tmp_path / "out", msa="auto")
        .read_text()
    )

    assert backend.complex_calls == [[SEQUENCE, OTHER_SEQUENCE]]
    paths = [entry["protein"]["msa"] for entry in native["sequences"]]
    assert all(path.endswith(".csv") for path in paths)
    # One CSV per sequence: Boltz refuses two chains of one sequence naming two
    # alignments, and its own parser is what says so.
    assert paths[0] == paths[2] != paths[1]
    _boltz_accepts_the_msa_fields(native)
    for path, sequence in zip(paths, (SEQUENCE, OTHER_SEQUENCE, SEQUENCE), strict=True):
        lines = Path(path).read_text().splitlines()
        hit = "A" * len(sequence)
        assert lines[:3] == ["key,sequence", f"0,{sequence}", f"1,{hit}"]
        assert all(line.startswith("-1,") for line in lines[3:])
        # Boltz reads it as a paired alignment: the paired rows carry their key.
        msa = parse_csv(Path(path))
        assert [int(row["taxonomy"]) for row in msa.sequences[:2]] == [0, 1]


def test_boltz_monomer_and_homomer_stay_unpaired(tmp_path: Path, monkeypatch) -> None:
    """Upstream pairs only two or more protein entities; one gets none."""
    backend = _ComplexStubSearch()
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
    )
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": ["A", "B"], "sequence": SEQUENCE}]},
    )

    native = json.loads(
        _materialize(source, "boltz2", tmp_path / "out", msa="auto").read_text()
    )

    assert backend.complex_calls == []
    assert native["sequences"][0]["protein"]["msa"].endswith(".a3m")


def test_a_common_job_still_cannot_name_a_boltz_csv(tmp_path: Path) -> None:
    alignment = tmp_path / "pairs.csv"
    alignment.write_text("key,sequence\n0,MKT\n")
    source = _write(
        tmp_path / "job.json",
        {
            "entities": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": "MKT",
                    "unpaired_msa": str(alignment),
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="paired-alignment format"):
        _materialize(source, "boltz2", tmp_path / "out")


def test_openfold3_links_a_searched_alignment_under_a_stem_it_reads(
    tmp_path: Path, monkeypatch
) -> None:
    """OpenFold3 selects a database by file *stem* and ignores anything else.

    A searched alignment arrives named after its cache key, so it has to be
    linked under an accepted stem or the run would silently have no MSA -- the
    exact failure `--msa auto` exists to remove.
    """
    backend = _StubSearch()
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
    )
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}]},
    )

    native = json.loads(
        _materialize(source, "openfold3", tmp_path / "out", msa="auto").read_text()
    )

    (query,) = native["queries"].values()
    chain = query["chains"][0]
    main = Path(chain["main_msa_file_paths"][0])
    assert main.stem == "colabfold_main"
    assert main.is_file()
    # A monomer is never paired: upstream pairs only more than one distinct
    # protein sequence (colabfold_msa_server.py:642-648).
    assert "paired_msa_file_paths" not in chain


OTHER_SEQUENCE = "GSHMLEDPVDAFQLGKVLNQ"


class _ComplexStubSearch(_StubSearch):
    """The stub, plus one-job complex pairing like the remote client."""

    def __init__(self) -> None:
        super().__init__()
        self.complex_calls: list[list[str]] = []

    def search_complex(self, sequences):
        from foldjax.search.msa import ComplexPairPayload

        self.complex_calls.append(list(sequences))
        return ComplexPairPayload(
            tuple(
                f">{101 + i}\n{s}\n>hit\n{'A' * len(s)}\n"
                for i, s in enumerate(sequences)
            )
        )


def _heteromer(tmp_path: Path, **extra: Any) -> Path:
    return _write(
        tmp_path / "job.json",
        {
            "entities": [
                {"type": "protein", "id": "A", "sequence": SEQUENCE, **extra},
                {"type": "protein", "id": "B", "sequence": OTHER_SEQUENCE},
                {"type": "protein", "id": "C", "sequence": SEQUENCE},
            ]
        },
    )


def test_openfold3_pairs_a_heteromer_in_one_complex_search(
    tmp_path: Path, monkeypatch
) -> None:
    """One pairing job over the distinct sequences, as OpenFold3 v0.5.0 submits.

    Per-chain pair jobs are not row-aligned across chains, and v0.5.0 refuses
    paired blocks of different depth; the complex job is what upstream sends.
    """
    backend = _ComplexStubSearch()
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
    )

    out = tmp_path / "out"
    native = json.loads(
        _materialize(_heteromer(tmp_path), "openfold3", out, msa="auto").read_text()
    )

    assert backend.complex_calls == [[SEQUENCE, OTHER_SEQUENCE]]
    assert backend.calls == [SEQUENCE, OTHER_SEQUENCE]
    (query,) = native["queries"].values()
    blocks = []
    for chain, sequence in zip(
        query["chains"], (SEQUENCE, OTHER_SEQUENCE, SEQUENCE), strict=True
    ):
        paired = Path(chain["paired_msa_file_paths"][0])
        assert paired.stem == "colabfold_paired"
        text = paired.read_text()
        assert text.splitlines()[1] == sequence
        blocks.append(text)
    assert blocks[0] == blocks[2]
    searched = json.loads((out / "msa_search.json").read_text())
    assert all("paired_msa" in record for record in searched)


def test_alphafold3_keeps_per_chain_pairing(tmp_path: Path, monkeypatch) -> None:
    # Protenix and OpenDDE pair the complex in one search
    # (tests/test_w15_msa_pairing.py).
    backend = _ComplexStubSearch()
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
    )

    native = json.loads(
        _materialize(
            _heteromer(tmp_path), "alphafold3", tmp_path / "out", msa="auto"
        ).read_text()
    )

    assert backend.complex_calls == []
    chain = native["sequences"][0]["protein"]
    assert Path(chain["pairedMsaPath"]).name == "pairing.a3m"


def test_openfold3_does_not_pair_without_a_complex_capable_search(
    tmp_path: Path, monkeypatch
) -> None:
    """A local wrapper searches one sequence at a time and cannot pair a complex."""
    backend = _StubSearch()
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
    )

    with pytest.warns(UserWarning, match="cannot pair a complex"):
        path = _materialize(
            _heteromer(tmp_path), "openfold3", tmp_path / "out", msa="auto"
        )

    (query,) = json.loads(path.read_text())["queries"].values()
    assert all("paired_msa_file_paths" not in chain for chain in query["chains"])
    assert all(chain["main_msa_file_paths"] for chain in query["chains"])


def test_openfold3_does_not_pair_around_a_supplied_alignment(
    tmp_path: Path, monkeypatch
) -> None:
    """Pairing would send a chain the caller chose to align locally."""
    backend = _ComplexStubSearch()
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
    )
    alignment = tmp_path / "mine.a3m"
    alignment.write_text(f">query\n{SEQUENCE}\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="not pairing the complex"):
        _materialize(
            _heteromer(tmp_path, unpaired_msa=str(alignment)),
            "openfold3",
            tmp_path / "out",
            msa="auto",
        )

    assert backend.complex_calls == []


def test_openfold3_never_replaces_a_supplied_paired_alignment(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _ComplexStubSearch()
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
    )
    alignment = tmp_path / "colabfold_paired.a3m"
    alignment.write_text(f">query\n{SEQUENCE}\n>mine\n{SEQUENCE}\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="not pairing the complex"):
        path = _materialize(
            _heteromer(tmp_path, paired_msa=str(alignment)),
            "openfold3",
            tmp_path / "out",
            msa="auto",
        )

    assert backend.complex_calls == []
    (query,) = json.loads(path.read_text())["queries"].values()
    (paired,) = query["chains"][0]["paired_msa_file_paths"]
    assert Path(paired).read_text() == alignment.read_text()


def test_a_failed_search_falls_back_under_auto_and_fails_under_required(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.search.msa import SearchError

    class _Broken(_StubSearch):
        def search(self, sequence: str):
            raise SearchError("server is down")

    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, _Broken())
    )
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}]},
    )

    with pytest.warns(UserWarning, match="MSA search failed"):
        assert _materialize(source, "protenix", tmp_path / "auto", msa="auto").is_file()

    with pytest.raises(ValueError, match="msa='required'"):
        _materialize(source, "protenix", tmp_path / "req", msa="required")


def test_one_chains_failed_search_keeps_the_others_and_is_recorded(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.search.msa import SearchError

    class _HalfBroken(_StubSearch):
        def search(self, sequence: str):
            if sequence == OTHER_SEQUENCE:
                raise SearchError("server lost this one")
            return super().search(sequence)

    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline",
        lambda: _stub_pipeline(tmp_path, _HalfBroken()),
    )
    source = _write(
        tmp_path / "job.json",
        {
            "entities": [
                {"type": "protein", "id": "A", "sequence": SEQUENCE},
                {"type": "protein", "id": "B", "sequence": OTHER_SEQUENCE},
            ]
        },
    )
    records: list[dict[str, Any]] = []
    with pytest.warns(UserWarning, match=r"chain\(s\) B .*server lost this one"):
        _materialize(
            source, "protenix", tmp_path / "out", msa="auto", msa_search=records
        )

    by_chain = {record["chain"]: record for record in records}
    assert "unpaired_msa" in by_chain["A"]
    assert by_chain["B"] == {"chain": "B", "error": "server lost this one"}
    written = json.loads((tmp_path / "out" / "msa_search.json").read_text())
    assert written == records


def test_a_failed_search_is_a_progress_line_for_every_input(
    tmp_path: Path, monkeypatch
) -> None:
    import io
    import warnings

    from foldjax import progress
    from foldjax.search.msa import SearchError

    class _Broken(_StubSearch):
        def search(self, sequence: str):
            raise SearchError("server is down")

    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, _Broken())
    )
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}]},
    )
    stream = io.StringIO()
    progress.enable(stream)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for name in ("first", "second"):
                _materialize(source, "protenix", tmp_path / name, msa="auto")
    finally:
        progress.disable()
    assert stream.getvalue().count("MSA search failed for chain(s) A") == 2


def test_the_manifest_records_msa_search_and_old_manifests_still_resume(
    tmp_path: Path,
) -> None:
    import dataclasses

    from foldjax import manifest
    from foldjax.schema import PredictionResult, PredictionSample
    from tests.test_resume_manifest import _file, _request

    request = dataclasses.replace(_request(tmp_path), msa="auto")
    request.output_dir.mkdir()
    structure = _file(request.output_dir / "sample.cif", b"data_mock\n#\n")
    result = PredictionResult(
        model="boltz2",
        samples=(PredictionSample(seed=7, structure_path=structure, scores={}),),
        output_dir=request.output_dir,
    )
    record = [{"chain": "A", "error": "server is down"}]
    document = manifest.describe_run(
        request, result, directory=request.output_dir, msa_search=record
    )
    assert document["msa_search"] == record
    assert manifest.matches_request(document, request, seed=7)
    old = {key: value for key, value in document.items() if key != "msa_search"}
    assert manifest.matches_request(old, request, seed=7)


def test_an_explicit_alignment_is_never_replaced_by_a_search(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _StubSearch()
    monkeypatch.setattr(
        "foldjax.msa_search._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
    )
    alignment = tmp_path / "mine.a3m"
    alignment.write_text(f">query\n{SEQUENCE}\n", encoding="utf-8")
    source = _write(
        tmp_path / "job.json",
        {
            "entities": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": SEQUENCE,
                    "unpaired_msa": str(alignment),
                }
            ]
        },
    )

    native = json.loads(
        _materialize(source, "protenix", tmp_path / "out", msa="auto").read_text()
    )

    assert native[0]["sequences"][0]["proteinChain"]["unpairedMsaPath"] == str(
        alignment
    )
    assert backend.calls == []


def test_an_unknown_msa_policy_is_refused(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}]},
    )

    with pytest.raises(ValueError, match="msa must be one of"):
        _materialize(source, "protenix", tmp_path / "out", msa="yes-please")


def test_compatibility_accepts_esmfold2_all_biomolecule_input() -> None:
    document = {
        "entities": [
            {"type": "protein", "id": "A", "sequence": SEQUENCE},
            {"type": "ligand", "id": "L", "ccd": "ATP"},
        ]
    }

    assert compatibility(document, "boltz2") is None
    assert compatibility(document, "esmfold2") is None
    # The check must not consume the caller's document.
    assert document["entities"][0]["sequence"] == SEQUENCE


def test_compatibility_reports_a_field_the_backend_cannot_express() -> None:
    document = {
        "entities": [
            {"type": "protein", "id": "A", "sequence": SEQUENCE},
            {"type": "ligand", "id": "L", "ccd": "ATP"},
        ],
        # Residue 3 is a threonine, whose hydroxyl is OG1. With a CCD
        # installed, an atom it lacks is refused before `bonds` is weighed.
        "bonds": [[["A", 3, "OG1"], ["L", 1, "PA"]]],
    }

    assert compatibility(document, "protenix") is None
    assert "bonds" in (compatibility(document, "openfold3") or "")


def test_capabilities_name_what_the_common_schema_cannot_reach() -> None:
    """The two lists describe the schema's reach, not the model's abilities."""
    protenix = capabilities("protenix")

    assert "unpaired_msa" in protenix.common_schema_features
    assert "templates" in protenix.common_schema_features
    # ESMFold2 has no template head at all, so there is nothing to report.
    assert "templates" not in native_only_features("esmfold2", capabilities("esmfold2"))
    # Boltz derives pairing from one per-chain a3m; a paired MSA has nowhere
    # to go and the document is refused rather than quietly halved.
    assert "paired_msa" not in common_schema_features("boltz2")
    assert "templates_unmapped" in common_schema_features("boltz2")

    # OpenDDE's released default leaves templates off, but the v1.1.1 route is
    # reachable through the explicit use_template option. Capabilities describe
    # what can be selected; without the opt-in a template is dropped, as
    # upstream drops it (compatibility below).
    opendde = capabilities("opendde")
    assert opendde.supports_templates is True
    assert "templates" in opendde.common_schema_features
    assert "templates" not in opendde.native_only_features


@pytest.mark.parametrize(
    "entity",
    [
        {
            "type": "rna",
            "id": "R",
            "sequence": "ACGU",
            "unpaired_msa": "rna.a3m",
        },
        {
            "type": "protein",
            "id": "A",
            "sequence": SEQUENCE,
            "templates": [
                {
                    "mmcif": "template.cif",
                    "query_indices": [1],
                    "template_indices": [1],
                }
            ],
        },
    ],
    ids=["rna_msa", "template"],
)
def test_opendde_compatibility_accepts_inputs_it_drops_as_upstream(
    entity: dict,
) -> None:
    """The run proceeds, dropping the input with a warning, so the job is runnable.

    `compatibility` answers with `_validate` at its defaults; the refusal that
    ``ignore_nucleic_msa=false`` or ``ignore_templates=false`` asks for is
    covered in tests/test_nucleic_msa.py and tests/test_template_gate.py.
    """
    assert compatibility({"entities": [entity]}, "opendde") is None


def test_fasta_records_become_chains() -> None:
    records = parse_fasta(">one\nMKTA\nYIAK\n\n>two\nGGSG\n")

    assert records == [("one", "MKTAYIAK"), ("two", "GGSG")]


def test_a_fasta_job_names_its_chains_and_keeps_the_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "target.fasta"
    path.write_text(">A\nMKTAYIAK\n>sp|P69905|HBA_HUMAN\nGGSGGS\n", encoding="utf-8")

    job = Job.from_fasta(path)

    assert job.name == "target"
    # A short header is the writer naming the chain; a UniProt description is not.
    assert [entity.id for entity in job.entities] == ["A", "B"]


def test_a_byte_order_mark_does_not_hide_the_first_fasta_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bom.fasta"
    path.write_bytes(b"\xef\xbb\xbf>A\nMKTAYIAK\n")

    job = Job.from_fasta(path)

    assert [(entity.id, entity.sequence) for entity in job.entities] == [
        ("A", "MKTAYIAK")
    ]


def test_a_byte_order_mark_does_not_make_a_json_job_unreadable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bom.json"
    document = {"entities": [{"type": "protein", "id": "A", "sequence": "MKTA"}]}
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(document).encode())

    assert read_job_document(path) == document


def test_a_fasta_in_another_encoding_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "latin1.fasta"
    path.write_bytes(">A caf\xe9\nMKTAYIAK\n".encode("latin-1"))

    with pytest.raises(ValueError, match="latin1.fasta is not UTF-8 text") as error:
        Job.from_fasta(path)
    assert not isinstance(error.value, UnicodeDecodeError)


def test_a_trailing_stop_codon_is_dropped_from_a_fasta_protein(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stop.fasta"
    path.write_text(">A\nMKTAYIAK*\n>B\nGGSG\n", encoding="utf-8")

    job = Job.from_fasta(path)

    assert [entity.sequence for entity in job.entities] == ["MKTAYIAK", "GGSG"]


def test_an_internal_stop_codon_is_still_refused(tmp_path: Path) -> None:
    path = tmp_path / "internal.fasta"
    path.write_text(">A\nMKTA*YIAK\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unsupported residue '\*' at position 5"):
        _materialize(
            Job.from_fasta(path).write(tmp_path / "job.json"), "protenix", tmp_path
        )


def test_a_stop_codon_alone_is_not_a_sequence(tmp_path: Path) -> None:
    path = tmp_path / "stop_only.fasta"
    path.write_text(">A\n*\n", encoding="utf-8")

    with pytest.raises(ValueError, match="has no sequence"):
        Job.from_fasta(path)


def test_a_fasta_without_a_header_is_refused() -> None:
    with pytest.raises(ValueError, match="must start with a '>' header"):
        parse_fasta("MKTAYIAK\n")


def test_a_job_from_bare_sequences_names_its_own_chains() -> None:
    job = Job.from_sequences(
        [SEQUENCE, SEQUENCE], ligand_ccd=["ATP"], ligand_smiles=["CCO"], name="demo"
    )
    document = job.to_document()

    assert [entity["id"] for entity in document["entities"]] == ["A", "B", "C", "D"]
    assert document["entities"][2]["ccd"] == "ATP"
    assert document["entities"][3]["smiles"] == "CCO"


def test_a_job_needs_something_in_it() -> None:
    with pytest.raises(ValueError, match="at least one sequence or ligand"):
        Job.from_sequences()


def _template_cif(tmp_path: Path) -> Path:
    path = tmp_path / "template.cif"
    path.write_text("data_template\n_entry.id template\n", encoding="utf-8")
    return path


#: (author chain, label chain, entity, full sequence, resolved 0-based positions)
_TEMPLATE_CHAINS = (
    ("A", "A", "1", ("ALA", "GLY", "SER", "THR", "VAL", "TRP"), (0, 2, 3, 4, 5)),
    ("B", "B", "2", ("LEU", "ILE", "LYS", "MET", "PHE", "TYR"), (1, 3, 4, 5)),
)


def _structure_cif(
    tmp_path: Path, chains=_TEMPLATE_CHAINS, ligand=True, atom_entities=True
) -> Path:
    """A small mmCIF with unresolved residues and a ligand in the last chain.

    Without ``atom_entities`` the atoms carry no ``label_entity_id``, and the
    entities are declared through ``_struct_asym`` alone.
    """
    lines = ["data_template", "_entry.id template", "#", "loop_"]
    lines += [f"_entity_poly_seq.{key}" for key in ("entity_id", "num", "mon_id")]
    for _, _, entity, sequence, _ in chains:
        lines += [f"{entity} {num} {name}" for num, name in enumerate(sequence, 1)]
    if not atom_entities:
        lines += ["#", "loop_", "_struct_asym.id", "_struct_asym.entity_id"]
        lines += [f"{label} {entity}" for _, label, entity, _, _ in chains]
    keys = (
        "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id "
        "label_asym_id label_entity_id label_seq_id Cartn_x Cartn_y Cartn_z "
        "occupancy B_iso_or_equiv auth_seq_id auth_asym_id pdbx_PDB_model_num"
    ).split()
    if not atom_entities:
        keys.remove("label_entity_id")
    lines += ["#", "loop_"] + [f"_atom_site.{key}" for key in keys]
    serial = 0
    for index, (auth, label, entity, sequence, resolved) in enumerate(chains):
        for position in resolved:
            for offset, atom in enumerate(("N", "CA", "C", "O")):
                serial += 1
                # Distinct coordinates per chain, residue and atom.
                x, y, z = 20.0 * index + position, 1.5 * position, 0.7 * offset
                column = f"{entity} " if atom_entities else ""
                lines.append(
                    f"ATOM {serial} {atom[0]} {atom} . {sequence[position]} "
                    f"{label} {column}{position + 1} {x:.3f} {y:.3f} {z:.3f} "
                    f"1.0 10.0 {position + 1} {auth} 1"
                )
    if ligand and not atom_entities:
        raise ValueError("the ligand row is written with an entity column")
    if ligand:
        auth, label = chains[-1][0], chains[-1][1] + "L"
        lines.append(
            f"HETATM {serial + 1} C C1 . ATP {label} 9 . 1.0 2.0 3.0 1.0 10.0 "
            f"101 {auth} 1"
        )
    path = tmp_path / "structure.cif"
    path.write_text("\n".join([*lines, "#", ""]), encoding="utf-8")
    return path


def _ca(path: Path, chain: str, position: int) -> list[float]:
    """The CA a 0-based full-sequence position has in the file."""
    import gemmi

    structure = gemmi.read_structure(str(path))
    for residue in structure[0][chain]:
        if residue.label_seq == position + 1:
            atom = residue["CA"][0].pos
            return [atom.x, atom.y, atom.z]
    raise AssertionError(f"no residue {position} in chain {chain}")


def _protenix_payload(source: Path, out: Path, model: str = "protenix") -> list:
    written = _materialize(source, model, out, options={"use_template": True})
    native = json.loads(written.read_text())
    sidecar = Path(native[0]["sequences"][0]["proteinChain"]["templatesPath"])
    return json.loads(sidecar.read_text())


def _mapped_job(tmp_path: Path, template: Path, **fields: Any) -> Path:
    return _write(
        tmp_path / "job.json",
        {
            "entities": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": SEQUENCE,
                    "templates": [{"mmcif": str(template), **fields}],
                }
            ]
        },
    )


def test_a_mapped_template_reaches_alphafold3_and_protenix(tmp_path: Path) -> None:
    """Three of the dialects need the residue map, and each spells it its own way.

    The common indices are AlphaFold 3's -- 0-based, the template's over its
    chain's full `_entity_poly_seq` -- and reach it verbatim. Protenix and
    OpenDDE count the first chain's resolved residues instead, so the writer
    restates the map in those: a behaviour change from passing it verbatim.
    """
    template = _structure_cif(tmp_path)
    # Chain A resolves positions 0, 2, 3, 4, 5: position 1 (GLY) is missing.
    source = _mapped_job(
        tmp_path, template, query_indices=[1, 2, 3, 4], template_indices=[0, 1, 2, 5]
    )

    written = _materialize(source, "alphafold3", tmp_path / "af3")
    native = json.loads(written.read_text())
    chain = native["sequences"][0]["protein"]
    assert chain["templates"][0]["mmcifPath"] == str(template)
    assert chain["templates"][0]["queryIndices"] == [1, 2, 3, 4]
    assert chain["templates"][0]["templateIndices"] == [0, 1, 2, 5]

    # Protenix reads templates only under use_template, released false.
    for model in ("protenix", "opendde"):
        payload = _protenix_payload(source, tmp_path / model, model)
        # Protenix reads the structure's contents, not a path, so the mmCIF is
        # inlined into a file of its own rather than into the job document; A
        # is already the first chain, so the file is the caller's, verbatim.
        assert payload[0]["mmcif"] == template.read_text()
        # The unresolved GLY has no observed ordinal and leaves the map.
        assert payload[0]["queryIndices"] == [1, 3, 4]
        assert payload[0]["templateIndices"] == [0, 1, 4]


def test_protenix_reads_the_named_template_chain_at_alphafold3s_indices(
    tmp_path: Path,
) -> None:
    """What Protenix featurizes is what AlphaFold 3 is told: the same CAs."""
    import numpy as np

    from foldjax.models.protenix.data.template_features import (
        _single_json_template,
    )

    template = _structure_cif(tmp_path)
    # Chain B is second in the file and resolves positions 1, 3, 4, 5.
    query, positions = [0, 1, 2, 3], [1, 2, 3, 5]
    source = _mapped_job(
        tmp_path,
        template,
        chain_id="B",
        query_indices=query,
        template_indices=positions,
    )
    (entry,) = _protenix_payload(source, tmp_path / "px")
    features = _single_json_template(entry, len(SEQUENCE), observed_residues=True)

    ca = 1  # atom37 index of CA
    coords = features["template_all_atom_positions"][:, ca]
    masks = features["template_all_atom_masks"][:, ca]
    # Position 2 (LYS) is unresolved: no coordinates, as under AlphaFold 3.
    assert masks[[0, 1, 2, 3]].tolist() == [1.0, 0.0, 1.0, 1.0]
    # The template is zero-centred, so compare displacements between residues.
    expected = np.subtract(_ca(template, "B", 5), _ca(template, "B", 1))
    np.testing.assert_allclose(coords[3] - coords[0], expected, atol=1e-3)
    # ILE and TYR, chain B's residues at positions 1 and 5; chain A has
    # neither.
    aatype = features["template_aatype"]
    assert aatype[0] != aatype[3]
    assert aatype[1] == aatype[len(SEQUENCE) - 1]  # an unmapped query: a gap


def test_entities_declared_only_in_struct_asym_still_count_the_sequence(
    tmp_path: Path,
) -> None:
    """No label_entity_id on the atoms: the entity comes from _struct_asym.

    Counting resolved residues instead would shift every index past chain B's
    unresolved position 0 by one residue.
    """
    template = _structure_cif(tmp_path, ligand=False, atom_entities=False)
    source = _mapped_job(
        tmp_path,
        template,
        chain_id="B",
        query_indices=[0, 1, 2],
        template_indices=[1, 3, 5],
    )
    (entry,) = _protenix_payload(source, tmp_path / "px")
    # Chain B resolves positions 1, 3, 4, 5: ordinals 0, 1, 3.
    assert (entry["queryIndices"], entry["templateIndices"]) == ([0, 1, 2], [0, 1, 3])


def test_a_sequence_declared_for_other_entities_only_is_refused(
    tmp_path: Path,
) -> None:
    """Rows exist, none for this chain: there is no full sequence to count."""
    chains = (
        ("A", "A", "1", ("ALA", "GLY", "SER"), (0, 1, 2)),
        ("B", "B", "2", ("LEU", "ILE"), (0, 1)),
    )
    template = _structure_cif(tmp_path, chains, ligand=False, atom_entities=False)
    text = template.read_text()
    # Chain B's struct_asym row names an entity with no _entity_poly_seq rows.
    template.write_text(text.replace("\nB 2\n", "\nB 7\n"))
    source = _mapped_job(
        tmp_path, template, chain_id="B", query_indices=[0], template_indices=[0]
    )
    with pytest.raises(ValueError, match="no _entity_poly_seq rows"):
        _protenix_payload(source, tmp_path / "px")


def test_an_observed_chain_file_keeps_its_map(tmp_path: Path) -> None:
    """A single fully resolved chain means the same under both readings."""
    chains = (("Q", "A", "1", ("ALA", "GLY", "SER"), (0, 1, 2)),)
    template = _structure_cif(tmp_path, chains, ligand=False)
    source = _mapped_job(
        tmp_path, template, chain_id="Q", query_indices=[4, 5], template_indices=[0, 2]
    )
    (entry,) = _protenix_payload(source, tmp_path / "px")
    assert entry["mmcif"] == template.read_text()
    assert (entry["queryIndices"], entry["templateIndices"]) == ([4, 5], [0, 2])


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"chain_id": "Z"}, "'Z' is not an author chain"),
        ({"chain_id": "B", "template_indices": [6]}, "template index 6 is outside"),
    ],
)
def test_protenix_refuses_a_template_map_it_cannot_restate(
    tmp_path: Path, fields: dict, message: str
) -> None:
    template = _structure_cif(tmp_path)
    source = _mapped_job(
        tmp_path, template, **{"query_indices": [0], "template_indices": [0], **fields}
    )
    with pytest.raises(ValueError, match=message):
        _protenix_payload(source, tmp_path / "px")


def test_alphafold3_filters_a_named_template_chain_to_a_sidecar(
    tmp_path: Path, monkeypatch
) -> None:
    template = _template_cif(tmp_path)
    source = _write(
        tmp_path / "job.json",
        {
            "entities": [
                {
                    "type": "protein",
                    "id": ["A", "B"],
                    "sequence": SEQUENCE,
                    "templates": [
                        {
                            "mmcif": str(template),
                            "chain_id": "Q",
                            "query_indices": [1],
                            "template_indices": [2],
                        }
                    ],
                }
            ]
        },
    )
    seen: list[tuple[Path, str]] = []

    def filter_template(path: Path, chain_id: str) -> str:
        seen.append((path, chain_id))
        return "data_filtered\n_entry.id filtered\n"

    monkeypatch.setattr(
        "foldjax.input._filter_alphafold3_template_mmcif", filter_template
    )

    written = _materialize(source, "alphafold3", tmp_path / "af3")
    native = json.loads(written.read_text())
    sidecar = Path(native["sequences"][0]["protein"]["templates"][0]["mmcifPath"])

    assert seen == [(template, "Q")]
    assert sidecar == tmp_path / "af3/templates/entity_0000_template_0000.cif"
    assert sidecar.read_text() == "data_filtered\n_entry.id filtered\n"


def test_an_unmapped_template_reaches_boltz_and_openfold3_and_is_refused_elsewhere(
    tmp_path: Path,
) -> None:
    template = _template_cif(tmp_path)
    source = _write(
        tmp_path / "job.json",
        {
            "entities": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": SEQUENCE,
                    "templates": [{"mmcif": str(template), "chain_id": "Q"}],
                }
            ]
        },
    )

    native = json.loads(_materialize(source, "boltz2", tmp_path / "b").read_text())
    assert native["templates"] == [
        {"cif": str(template), "chain_id": ["A"], "template_id": ["Q"]}
    ]

    # OpenFold3 aligns a bare file itself (upstream's CIF-direct mode). This
    # stub names no chain, so the author id cannot be resolved to a label id
    # and is passed on for OpenFold3 to report.
    query = json.loads(_materialize(source, "openfold3", tmp_path / "of3").read_text())
    (chain,) = query["queries"]["query"]["chains"]
    assert chain["template_cif_paths"] == [str(template)]
    assert chain["template_cif_chain_ids"] == ["Q"]

    with pytest.raises(ValueError, match="requires query_indices"):
        _materialize(source, "protenix", tmp_path / "px")
    with pytest.raises(ValueError, match="no per-job template field"):
        _materialize(source, "esmfold2", tmp_path / "esm")


def test_a_mapped_template_is_refused_by_the_model_that_aligns_itself(
    tmp_path: Path,
) -> None:
    source = _write(
        tmp_path / "job.json",
        {
            "entities": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": SEQUENCE,
                    "templates": [
                        {
                            "mmcif": str(_template_cif(tmp_path)),
                            "query_indices": [1],
                            "template_indices": [1],
                        }
                    ],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="aligns the mmCIF itself"):
        _materialize(source, "boltz2", tmp_path / "b")


def test_mismatched_template_indices_are_refused(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "job.json",
        {
            "entities": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": SEQUENCE,
                    "templates": [
                        {
                            "mmcif": str(_template_cif(tmp_path)),
                            "query_indices": [1, 2],
                            "template_indices": [1],
                        }
                    ],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="differ in length"):
        _materialize(source, "alphafold3", tmp_path / "af3")


def test_affinity_reaches_boltz_and_no_other_model(tmp_path: Path) -> None:
    document = {
        "entities": [
            {"type": "protein", "id": "A", "sequence": SEQUENCE},
            {"type": "ligand", "id": "L", "ccd": "ATP"},
        ],
        "properties": [{"affinity": {"binder": "L"}}],
    }
    source = _write(tmp_path / "job.json", document)

    native = json.loads(_materialize(source, "boltz2", tmp_path / "b").read_text())
    assert native["properties"] == [{"affinity": {"binder": "L"}}]

    with pytest.raises(ValueError, match="binding affinity"):
        _materialize(source, "protenix", tmp_path / "px")


def test_an_affinity_binder_must_name_a_chain(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "job.json",
        {
            "entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}],
            "properties": [{"affinity": {"binder": "Z"}}],
        },
    )

    with pytest.raises(ValueError, match="not a chain in this job"):
        _materialize(source, "boltz2", tmp_path / "b")


def test_capabilities_stop_claiming_templates_the_schema_now_carries() -> None:
    assert "templates" not in capabilities("protenix").native_only_features
    assert "affinity" not in capabilities("boltz2").native_only_features
    # OpenFold3's common templates are translated into its native query.
    assert "templates" not in capabilities("openfold3").native_only_features


def test_a_structure_becomes_a_job_of_its_chains_and_ligands(tmp_path: Path) -> None:
    from foldjax.job import Job

    path = tmp_path / "1abc.pdb"
    path.write_text(
        "ATOM      1  CA  MET A   1      10.000  10.000  10.000  1.00 20.00  \n"
        "ATOM      2  CA  LYS A   2      11.000  10.000  10.000  1.00 20.00  \n"
        "ATOM      3  CA  THR A   3      12.000  10.000  10.000  1.00 20.00  \n"
        "HETATM    4  PA  ATP B 101      20.000  20.000  20.000  1.00 20.00  \n"
        "HETATM    5  O   HOH C 201      30.000  30.000  30.000  1.00 20.00  \n"
        "END\n",
        encoding="utf-8",
    )

    document = Job.from_structure(path).to_document()

    assert document["name"] == "1abc"
    assert document["entities"][0] == {
        "type": "protein",
        "id": "A",
        "sequence": "MKT",
    }
    assert document["entities"][1] == {"type": "ligand", "id": "B", "ccd": "ATP"}
    # Solvent is not chemistry worth folding.
    assert len(document["entities"]) == 2


def test_a_structure_with_nothing_foldable_says_so(tmp_path: Path) -> None:
    from foldjax.job import Job

    path = tmp_path / "water.pdb"
    path.write_text(
        "HETATM    1  O   HOH C 201      30.000  30.000  30.000  1.00 20.00  \nEND\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no foldable chains"):
        Job.from_structure(path)
