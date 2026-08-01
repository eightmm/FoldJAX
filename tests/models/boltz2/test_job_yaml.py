import pytest
import yaml

from foldjax.models.boltz2.data.job_yaml import build_job_yaml


def test_build_job_yaml_preserves_all_supported_entity_types() -> None:
    parsed = yaml.safe_load(
        build_job_yaml(
            ["ACD"],
            ["ATP"],
            dna=["ACGT"],
            rna=["ACGU"],
            ligands_smiles=["CC(=O)O"],
        )
    )

    assert parsed["version"] == 1
    assert [next(iter(item)) for item in parsed["sequences"]] == [
        "protein",
        "dna",
        "rna",
        "ligand",
        "ligand",
    ]
    assert [next(iter(item.values()))["id"] for item in parsed["sequences"]] == [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]
    assert parsed["sequences"][0]["protein"]["msa"] == "empty"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "at least one"),
        ({"proteins": ["AC-D"]}, "protein sequence"),
        ({"dna": ["ACGU"]}, "DNA sequence"),
        ({"rna": ["ACGT"]}, "RNA sequence"),
        ({"ligands_ccd": [""]}, "CCD"),
        ({"ligands_smiles": [""]}, "SMILES"),
    ],
)
def test_build_job_yaml_rejects_invalid_entities(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_job_yaml(**kwargs)
