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
)
from foldjax.job import Job, parse_fasta
from foldjax.registry import capabilities

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQ"


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _materialize(source: Path, model: str, out: Path, **kwargs: Any) -> Path:
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
        _materialize(source, "protenix", tmp_path / "out")


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
        "foldjax.input._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
    )
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}]},
    )

    written = _materialize(source, "protenix", tmp_path / "out", msa="auto")
    native = json.loads(written.read_text())

    chain = native[0]["sequences"][0]["proteinChain"]
    assert Path(chain["unpairedMsaPath"]).is_file()
    assert Path(chain["pairedMsaPath"]).is_file()
    assert backend.calls == [SEQUENCE]
    searched = json.loads((tmp_path / "out" / "msa_search.json").read_text())
    assert searched[0]["chain"] == "A"


def test_a_searched_alignment_is_reused_for_the_next_model(
    tmp_path: Path, monkeypatch
) -> None:
    """The cache is keyed by sequence, not by model: three backends, one search."""
    backend = _StubSearch()
    monkeypatch.setattr(
        "foldjax.input._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
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
        "foldjax.input._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
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
        "foldjax.input._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
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
    paired = Path(chain["paired_msa_file_paths"][0])
    assert main.stem == "colabfold_main"
    assert paired.stem == "colabfold_paired"
    assert main.is_file() and paired.is_file()


def test_a_failed_search_falls_back_under_auto_and_fails_under_required(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.search.msa import SearchError

    class _Broken(_StubSearch):
        def search(self, sequence: str):
            raise SearchError("server is down")

    monkeypatch.setattr(
        "foldjax.input._msa_pipeline", lambda: _stub_pipeline(tmp_path, _Broken())
    )
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": "A", "sequence": SEQUENCE}]},
    )

    with pytest.warns(UserWarning, match="MSA search failed"):
        assert _materialize(source, "protenix", tmp_path / "auto", msa="auto").is_file()

    with pytest.raises(ValueError, match="msa='required'"):
        _materialize(source, "protenix", tmp_path / "req", msa="required")


def test_an_explicit_alignment_is_never_replaced_by_a_search(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _StubSearch()
    monkeypatch.setattr(
        "foldjax.input._msa_pipeline", lambda: _stub_pipeline(tmp_path, backend)
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
        "bonds": [[["A", 3, "OG"], ["L", 1, "PA"]]],
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
    # what can be selected, while compatibility below enforces the opt-in.
    opendde = capabilities("opendde")
    assert opendde.supports_templates is True
    assert "templates" in opendde.common_schema_features
    assert "templates" not in opendde.native_only_features


@pytest.mark.parametrize(
    ("entity", "message"),
    [
        (
            {
                "type": "rna",
                "id": "R",
                "sequence": "ACGU",
                "unpaired_msa": "rna.a3m",
            },
            "use_rna_msa=true",
        ),
        (
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
            "use_template=true",
        ),
    ],
)
def test_opendde_compatibility_rejects_common_inputs_it_would_drop(
    entity: dict, message: str
) -> None:
    reason = compatibility({"entities": [entity]}, "opendde")

    assert reason is not None
    assert message in reason


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


def test_a_mapped_template_reaches_alphafold3_and_protenix(tmp_path: Path) -> None:
    """Three of the dialects need the residue map, and each spells it its own way."""
    template = _template_cif(tmp_path)
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
                            "mmcif": str(template),
                            "query_indices": [1, 2, 3],
                            "template_indices": [5, 6, 7],
                        }
                    ],
                }
            ]
        },
    )

    written = _materialize(source, "alphafold3", tmp_path / "af3")
    native = json.loads(written.read_text())
    chain = native["sequences"][0]["protein"]
    assert chain["templates"][0]["mmcifPath"] == str(template)
    assert chain["templates"][0]["queryIndices"] == [1, 2, 3]
    assert chain["templates"][0]["templateIndices"] == [5, 6, 7]

    written = _materialize(source, "protenix", tmp_path / "px")
    native = json.loads(written.read_text())
    sidecar = Path(native[0]["sequences"][0]["proteinChain"]["templatesPath"])
    # Protenix reads the structure's contents, not a path, so the mmCIF is
    # inlined into a file of its own rather than into the job document.
    payload = json.loads(sidecar.read_text())
    assert payload[0]["mmcif"].startswith("data_template")
    assert payload[0]["queryIndices"] == [1, 2, 3]


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


def test_an_unmapped_template_reaches_boltz_and_is_refused_elsewhere(
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

    with pytest.raises(ValueError, match="requires query_indices"):
        _materialize(source, "protenix", tmp_path / "px")
    with pytest.raises(ValueError, match="no per-job template field"):
        _materialize(source, "openfold3", tmp_path / "of3")


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
    # OpenFold3 builds templates from its own pipeline and has no job field.
    assert "templates" in capabilities("openfold3").native_only_features


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
