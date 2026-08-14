import json
import pickle
from pathlib import Path

import pytest

from foldjax.input import materialize_native_input
from foldjax.models.boltz2.data.identifiers import (
    resolve_molecule_pickle,
    validate_ccd_identifier,
)
from foldjax.models.boltz2.data.mol import load_molecules
from foldjax.models.boltz2.data.parse.schema import parse_boltz_schema
from foldjax.registry import capabilities


@pytest.mark.parametrize("code", ["A", "ATP", "6MA", "A1BFJ", "12345"])
def test_valid_alphanumeric_ccd_identifiers_are_preserved(code: str) -> None:
    assert validate_ccd_identifier(code) == code


@pytest.mark.parametrize(
    "code",
    [
        "../escape",
        r"..\escape",
        "ATP/../SEP",
        "ATP.pkl",
        ".",
        "..",
        "ATP%2fSEP",
        "ＡＴＰ",
        "atp",
        "A" * 17,
    ],
)
def test_ccd_identifiers_reject_path_spellings_and_noncanonical_forms(
    code: str,
) -> None:
    with pytest.raises(ValueError, match="uppercase ASCII"):
        validate_ccd_identifier(code)


def test_molecule_loader_rejects_relative_and_absolute_escape_paths(
    tmp_path: Path,
) -> None:
    molecule_dir = tmp_path / "mols"
    molecule_dir.mkdir()
    outside = tmp_path / "outside.pkl"
    with outside.open("wb") as file:
        pickle.dump({"must_not_load": True}, file)

    for identifier in ("../outside", str(outside.with_suffix(""))):
        with pytest.raises(ValueError, match="uppercase ASCII"):
            load_molecules(molecule_dir, [identifier])


def test_molecule_loader_rejects_symlink_outside_configured_root(
    tmp_path: Path,
) -> None:
    molecule_dir = tmp_path / "mols"
    molecule_dir.mkdir()
    outside = tmp_path / "outside.pkl"
    with outside.open("wb") as file:
        pickle.dump({"must_not_load": True}, file)
    (molecule_dir / "ATP.pkl").symlink_to(outside)

    with pytest.raises(ValueError, match="outside the configured molecule directory"):
        load_molecules(molecule_dir, ["ATP"])


def test_molecule_loader_accepts_regular_file_inside_configured_root(
    tmp_path: Path,
) -> None:
    molecule_dir = tmp_path / "mols"
    molecule_dir.mkdir()
    expected = {"safe": True}
    with (molecule_dir / "A1BFJ.pkl").open("wb") as file:
        pickle.dump(expected, file)

    assert resolve_molecule_pickle(molecule_dir, "A1BFJ").parent == molecule_dir
    assert load_molecules(molecule_dir, ["A1BFJ"]) == {"A1BFJ": expected}


@pytest.mark.parametrize("code", ["../outside", r"..\outside", "/tmp/outside"])
def test_common_boltz_ligand_ccd_rejects_path_selectors(
    tmp_path: Path, code: str
) -> None:
    source = tmp_path / "job.json"
    source.write_text(
        json.dumps(
            {
                "entities": [
                    {"type": "protein", "id": "A", "sequence": "A"},
                    {"type": "ligand", "id": "L", "ccd": code},
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="ligand CCD code.*uppercase ASCII"):
        materialize_native_input(
            source, capabilities("boltz2"), tmp_path / "out", seed=0
        )


def test_common_boltz_modification_ccd_rejects_path_selector(
    tmp_path: Path,
) -> None:
    source = tmp_path / "job.json"
    source.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "type": "protein",
                        "id": "A",
                        "sequence": "A",
                        "modifications": [{"ccd": "../outside", "position": 1}],
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="modification CCD code.*uppercase ASCII"):
        materialize_native_input(
            source, capabilities("boltz2"), tmp_path / "out", seed=0
        )


@pytest.mark.parametrize(
    ("entity", "message"),
    [
        ({"ligand": {"id": "L", "ccd": "../outside"}}, "ligand CCD code"),
        (
            {
                "protein": {
                    "id": "A",
                    "sequence": "A",
                    "modifications": [{"ccd": r"..\outside", "position": 1}],
                }
            },
            "modification CCD code",
        ),
    ],
)
def test_native_boltz_schema_rejects_untrusted_pickle_selectors(
    tmp_path: Path, entity: dict, message: str
) -> None:
    molecule_dir = tmp_path / "mols"
    molecule_dir.mkdir()
    schema = {"version": 1, "sequences": [entity]}

    with pytest.raises(ValueError, match=message):
        parse_boltz_schema("unsafe", schema, {}, molecule_dir, boltz_2=True)
