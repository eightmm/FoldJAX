import json
from pathlib import Path

import pytest
import yaml

from foldjax.input import _boltz, materialize_native_input
from foldjax.registry import capabilities


def _materialize(source: Path, model: str, output_dir: Path, *, seed: int = 9) -> Path:
    return materialize_native_input(source, capabilities(model), output_dir, seed=seed)


def _write(path: Path, job: dict) -> Path:
    path.write_text(json.dumps(job))
    return path


@pytest.fixture
def job() -> dict:
    return {
        "name": "complex",
        "entities": [
            {
                "type": "protein",
                "id": ["A"],
                "sequence": "ACD",
                "unpaired_msa": "msa.a3m",
                "modifications": [{"ccd": "SEP", "position": 2}],
            },
            {"type": "dna", "id": ["B"], "sequence": "ACGT"},
            {"type": "ligand", "id": ["L"], "ccd": "ATP"},
        ],
    }


@pytest.fixture
def common_job(tmp_path: Path, job: dict) -> Path:
    return _write(tmp_path / "job.json", job)


def test_materializes_alphafold3_input(common_job: Path, tmp_path: Path) -> None:
    native = json.loads(_materialize(common_job, "alphafold3", tmp_path).read_text())
    assert native["dialect"] == "alphafold3"
    assert native["version"] == 4
    assert native["modelSeeds"] == [9]
    protein = native["sequences"][0]["protein"]
    assert protein["unpairedMsaPath"] == str(common_job.parent / "msa.a3m")
    assert protein["modifications"] == [{"ptmType": "SEP", "ptmPosition": 2}]
    assert native["sequences"][2]["ligand"]["ccdCodes"] == ["ATP"]


def test_materializes_boltz_input_as_parseable_yaml(
    common_job: Path, tmp_path: Path
) -> None:
    path = _materialize(common_job, "boltz2", tmp_path)
    # The Boltz-2 port dispatches its parser on the suffix, rejecting .json outright.
    assert path.suffix == ".yaml"
    native = yaml.safe_load(path.read_text())
    assert native["version"] == 1
    protein = native["sequences"][0]["protein"]
    assert protein["msa"] == str(common_job.parent / "msa.a3m")
    assert protein["modifications"] == [{"ccd": "SEP", "position": 2}]
    assert native["sequences"][2]["ligand"]["ccd"] == "ATP"


def test_materializes_protenix_input(common_job: Path, tmp_path: Path) -> None:
    natives = json.loads(_materialize(common_job, "protenix", tmp_path).read_text())
    chain = natives[0]["sequences"][0]["proteinChain"]
    assert natives[0]["modelSeeds"] == [9]
    assert chain["count"] == 1
    # Protenix requires the CCD_ prefix that AlphaFold 3 omits.
    assert chain["modifications"] == [{"ptmType": "CCD_SEP", "ptmPosition": 2}]
    assert natives[0]["sequences"][1]["dnaSequence"]["id"] == ["B"]
    assert natives[0]["sequences"][2]["ligand"]["ligand"] == "CCD_ATP"


def test_materializes_opendde_input_as_a_native_job_list(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "job.json",
        {
            "name": "dde",
            "entities": [
                {"type": "protein", "id": ["A", "B"], "sequence": "ACD"},
                {"type": "dna", "id": ["D"], "sequence": "ACGT"},
                {"type": "ligand", "id": ["L"], "ccd": "ATP"},
            ],
        },
    )

    path = _materialize(source, "opendde", tmp_path, seed=17)
    native = json.loads(path.read_text())

    assert path.name == "opendde_input.json"
    assert isinstance(native, list) and len(native) == 1
    assert native[0]["name"] == "dde"
    assert native[0]["modelSeeds"] == [17]
    assert native[0]["sequences"][0]["proteinChain"] == {
        "id": ["A", "B"],
        "count": 2,
        "sequence": "ACD",
        "modifications": [],
    }
    assert native[0]["sequences"][1]["dnaSequence"]["id"] == ["D"]
    assert native[0]["sequences"][2]["ligand"]["ligand"] == "CCD_ATP"


def test_materializes_opendde_modifications_and_covalent_bonds(
    tmp_path: Path,
) -> None:
    source = _write(
        tmp_path / "modified_bonded.json",
        {
            "name": "modified_bonded",
            "entities": [
                {
                    "type": "protein",
                    "id": ["A"],
                    "sequence": "ACD",
                    "modifications": [{"ccd": "SEP", "position": 2}],
                },
                {"type": "ligand", "id": ["L"], "ccd": "ATP"},
            ],
            "bonds": [[["A", 2, "OG"], ["L", 1, "PA"]]],
        },
    )
    native = json.loads(_materialize(source, "opendde", tmp_path).read_text())[0]

    assert native["sequences"][0]["proteinChain"]["modifications"] == [
        {"ptmType": "CCD_SEP", "ptmPosition": 2}
    ]
    assert native["covalent_bonds"] == [
        {
            "entity1": 1,
            "copy1": 1,
            "position1": 2,
            "atom1": "OG",
            "entity2": 2,
            "copy2": 1,
            "position2": 1,
            "atom2": "PA",
        }
    ]


def test_materializes_chai_fasta(tmp_path: Path, job: dict) -> None:
    job["entities"][0].pop("unpaired_msa")
    job["entities"][0].pop("modifications")
    job["entities"][2] = {"type": "ligand", "id": ["L"], "smiles": "CCO"}
    source = _write(tmp_path / "job.json", job)
    path = _materialize(source, "chai", tmp_path)
    assert path.suffix == ".fasta"
    assert path.read_text() == (
        ">protein|name=A\nACD\n>dna|name=B\nACGT\n>ligand|name=L\nCCO\n"
    )


def test_chai_expands_one_record_per_chain(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "job.json",
        {"entities": [{"type": "protein", "id": ["A", "B"], "sequence": "AC"}]},
    )
    path = _materialize(source, "chai", tmp_path)
    assert path.read_text() == ">protein|name=A\nAC\n>protein|name=B\nAC\n"


def test_bonds_translate_to_each_native_representation(
    tmp_path: Path, job: dict
) -> None:
    job["bonds"] = [[["A", 2, "OG"], ["L", 1, "PA"]]]
    source = _write(tmp_path / "job.json", job)

    af3 = json.loads(_materialize(source, "alphafold3", tmp_path).read_text())
    assert af3["bondedAtomPairs"] == [[["A", 2, "OG"], ["L", 1, "PA"]]]

    boltz = yaml.safe_load(_materialize(source, "boltz2", tmp_path).read_text())
    assert boltz["constraints"] == [
        {"bond": {"atom1": ["A", 2, "OG"], "atom2": ["L", 1, "PA"]}}
    ]

    # Protenix addresses bonds by 1-based entity number and copy index.
    protenix = json.loads(_materialize(source, "protenix", tmp_path).read_text())
    assert protenix[0]["covalent_bonds"] == [
        {
            "entity1": 1,
            "copy1": 1,
            "position1": 2,
            "atom1": "OG",
            "entity2": 3,
            "copy2": 1,
            "position2": 1,
            "atom2": "PA",
        }
    ]


def test_protenix_bond_copy_index_follows_chain_order(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "job.json",
        {
            "entities": [{"type": "protein", "id": ["A", "B"], "sequence": "ACD"}],
            "bonds": [[["B", 1, "N"], ["B", 3, "C"]]],
        },
    )
    native = json.loads(_materialize(source, "protenix", tmp_path).read_text())
    bond = native[0]["covalent_bonds"][0]
    assert (bond["entity1"], bond["copy1"]) == (1, 2)
    assert (bond["entity2"], bond["copy2"]) == (1, 2)


@pytest.mark.parametrize(
    ("model", "mutate", "message"),
    [
        # Boltz derives pairing from one a3m, so a paired MSA has nowhere to go.
        (
            "boltz2",
            lambda job: job["entities"][0].update({"paired_msa": "paired.a3m"}),
            "cannot express paired_msa",
        ),
        # Chai addresses ligands by SMILES only and takes a job-level MSA dir.
        (
            "chai",
            lambda job: (
                job["entities"][0].clear()
                or job["entities"][0].update(
                    {"type": "protein", "id": ["A"], "sequence": "ACD"}
                )
            ),
            "cannot express ligand_ccd",
        ),
        # Chai has no paired-MSA concept; `unpaired_msa` it can express, by
        # writing its own aligned-Parquet files beside the FASTA.
        (
            "chai",
            lambda job: job["entities"].__setitem__(
                slice(None),
                [
                    {
                        "type": "protein",
                        "id": ["A"],
                        "sequence": "ACD",
                        "paired_msa": "paired.a3m",
                    }
                ],
            ),
            "cannot express paired_msa",
        ),
    ],
)
def test_unsupported_features_fail_instead_of_being_dropped(
    tmp_path: Path, job: dict, model: str, mutate, message: str
) -> None:
    mutate(job)
    source = _write(tmp_path / "job.json", job)
    with pytest.raises(ValueError, match=message):
        _materialize(source, model, tmp_path)


def test_rejects_unsupported_common_entity(common_job: Path, tmp_path: Path) -> None:
    job = json.loads(common_job.read_text())
    job["entities"].append({"type": "glycan", "id": ["G"]})
    common_job.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="unsupported entity type"):
        _materialize(common_job, "boltz2", tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda job: job.update({"unknown": 1}), "unsupported top-level fields"),
        (
            lambda job: job["entities"][0].update({"template": "x"}),
            "unsupported protein entity fields",
        ),
        (
            lambda job: job["entities"].append(
                {"type": "protein", "id": ["A"], "sequence": "AC"}
            ),
            "duplicate chain id",
        ),
        (
            lambda job: job["entities"][2].update({"smiles": "CCO"}),
            "either ccd or smiles",
        ),
        (
            lambda job: job["entities"][0].update(
                {"modifications": [{"ccd": "SEP", "position": 99}]}
            ),
            "outside sequence length",
        ),
        (
            lambda job: job["entities"][0].update(
                {"modifications": [{"ccd": "SEP", "pos": 1}]}
            ),
            "exactly a ccd and a position",
        ),
        (
            lambda job: job.update({"bonds": [[["Z", 1, "N"], ["A", 1, "C"]]]}),
            "unknown chain id",
        ),
        (
            lambda job: job.update({"bonds": [[["A", 1, "N"]]]}),
            "pair of atom references",
        ),
        (lambda job: job["entities"].clear(), "non-empty entities list"),
        (
            lambda job: job["entities"][0].pop("sequence"),
            "protein entity requires sequence",
        ),
        (lambda job: job["entities"][1].pop("id"), "non-empty id"),
    ],
)
def test_common_schema_validation(
    tmp_path: Path, job: dict, mutate, message: str
) -> None:
    mutate(job)
    source = _write(tmp_path / "job.json", job)
    with pytest.raises(ValueError, match=message):
        _materialize(source, "protenix", tmp_path)


def test_rejects_non_object_document(tmp_path: Path) -> None:
    source = tmp_path / "job.json"
    source.write_text("[]")
    with pytest.raises(ValueError, match="must be a JSON or YAML mapping"):
        _materialize(source, "protenix", tmp_path)


def test_boltz_protein_without_an_alignment_asks_for_single_sequence() -> None:
    """Boltz refuses a protein chain that neither has an MSA nor opts out.

    Its own message names the fix: "Use `msa: empty` per protein chain". Omitting
    the key is not equivalent, so every alignment-free job written here failed at
    the backend until the writer said so explicitly.
    """
    document = _boltz(
        {
            "name": "t",
            "entities": [
                {"type": "protein", "id": ["A"], "sequence": "GSHM"},
            ],
        },
        Path("."),
    )
    assert document["sequences"][0]["protein"]["msa"] == "empty"


def test_boltz_nucleic_acid_chains_carry_no_msa_key() -> None:
    """Only proteins take an alignment; a `msa` key elsewhere is not valid input."""
    document = _boltz(
        {
            "name": "t",
            "entities": [
                {"type": "dna", "id": ["B"], "sequence": "ACGT"},
                {"type": "rna", "id": ["C"], "sequence": "ACGU"},
            ],
        },
        Path("."),
    )
    for record in document["sequences"]:
        (body,) = record.values()
        assert "msa" not in body


def test_boltz_keeps_a_supplied_alignment(tmp_path) -> None:
    """An explicit alignment must win over the single-sequence opt-out."""
    alignment = tmp_path / "hits.a3m"
    alignment.write_text(">q\nGSHM\n")
    document = _boltz(
        {
            "name": "t",
            "entities": [
                {
                    "type": "protein",
                    "id": ["A"],
                    "sequence": "GSHM",
                    "unpaired_msa": str(alignment),
                }
            ],
        },
        tmp_path,
    )
    assert document["sequences"][0]["protein"]["msa"].endswith("hits.a3m")


def _chai_job(tmp_path: Path, sequence: str, a3m_query: str) -> Path:
    alignment = tmp_path / "hits.a3m"
    alignment.write_text(f">query\n{a3m_query}\n>hit1\n{'A' * len(a3m_query)}\n")
    return _write(
        tmp_path / "job.json",
        {
            "name": "t",
            "entities": [
                {
                    "type": "protein",
                    "id": ["A"],
                    "sequence": sequence,
                    "unpaired_msa": str(alignment),
                }
            ],
        },
    )


def test_chai_writes_the_alignment_under_the_sequence_it_will_be_looked_up_by(
    tmp_path,
) -> None:
    """Chai addresses alignments by a hash of the query sequence.

    The reader looks the file up by the *job's* sequence, so that is what has
    to name it.
    """
    from foldjax.input import CHAI_MSA_DIRNAME
    from foldjax.models.chai.data.msa import expected_aligned_pqt_basename

    out = tmp_path / "out"
    _materialize(_chai_job(tmp_path, "GSHM", "GSHM"), "chai", out)

    written = sorted((out / CHAI_MSA_DIRNAME).glob("*.aligned.pqt"))
    assert [path.name for path in written] == [expected_aligned_pqt_basename("GSHM")]


def test_chai_refuses_an_alignment_built_for_a_different_sequence(tmp_path) -> None:
    """A query row that is not the folded sequence means the wrong alignment.

    The filename used to come from the A3M's own query row while the reader
    looked it up by the job's sequence, so this case wrote a file under a hash
    nothing would ask for. Chai answers a missing alignment with a warning and
    a single-sequence run, so the prediction reported success with the MSA
    silently gone -- the one failure mode an alignment is supposed to prevent.
    It is now refused outright, and nothing is written.
    """
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="different protein"):
        _materialize(_chai_job(tmp_path, "GSHM", "WWWW"), "chai", out)

    from foldjax.input import CHAI_MSA_DIRNAME

    assert not list((out / CHAI_MSA_DIRNAME).glob("*.aligned.pqt"))
