"""Rigid structure comparison is lazy, deterministic, and chemistry-aware."""

from __future__ import annotations

import json
from pathlib import Path

import gemmi
import numpy as np
import pytest

import foldjax
from foldjax.alignment import (
    AlignmentReport,
    RigidTransform,
    StructureAlignmentError,
    align_predictions,
    align_structures,
)
from foldjax.schema import PredictionResult, PredictionSample


def test_alignment_api_is_available_from_the_root_package() -> None:
    assert foldjax.align_predictions is align_predictions
    assert foldjax.align_structures is align_structures


def _write_structure(
    path: Path,
    chains: dict[str, tuple[str, list[tuple[str, tuple[float, float, float]]]]],
    *,
    category: bool = False,
) -> Path:
    structure = gemmi.Structure()
    structure.name = path.stem
    model = gemmi.Model("1")
    for chain_name, (kind, entries) in chains.items():
        chain = gemmi.Chain(chain_name)
        for index, (residue_name, xyz) in enumerate(entries, start=1):
            residue = gemmi.Residue()
            residue.name = residue_name
            residue.seqid = gemmi.SeqId(index, " ")
            residue.het_flag = "H" if kind == "ligand" else "A"
            names = (
                ("C1", "C2", "O1", "N1")
                if kind == "ligand"
                else (("C4'",) if kind == "nucleic" else ("CA",))
            )
            ligand_offsets = (
                (0.0, 0.0, 0.0),
                (0.8, 0.0, 0.0),
                (0.0, 0.9, 0.0),
                (0.1, 0.2, 0.7),
            )
            for offset, atom_name in enumerate(names):
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(atom_name[0])
                delta = ligand_offsets[offset] if kind == "ligand" else (0, 0, 0)
                atom.pos = gemmi.Position(
                    xyz[0] + delta[0],
                    xyz[1] + delta[1],
                    xyz[2] + delta[2],
                )
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    if path.suffix == ".pdb":
        structure.write_pdb(str(path))
        return path
    document = structure.make_mmcif_document()
    if category:
        document.sole_block().set_pair("_foldjax_test.note", "preserve-me")
    document.write_file(str(path))
    return path


def _transform_entries(
    entries: list[tuple[str, tuple[float, float, float]]],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> list[tuple[str, tuple[float, float, float]]]:
    return [
        (name, tuple(np.asarray(xyz) @ rotation + translation)) for name, xyz in entries
    ]


def _positions(cif_text: str, tmp_path: Path) -> np.ndarray:
    path = tmp_path / "read-back.cif"
    path.write_text(cif_text)
    structure = gemmi.read_structure(str(path))
    return np.asarray(
        [
            (atom.pos.x, atom.pos.y, atom.pos.z)
            for chain in structure[0]
            for residue in chain
            for atom in residue
        ]
    )


def test_alignment_is_lazy_and_recovers_one_complex_frame(tmp_path: Path) -> None:
    protein = [
        ("ALA", (0.0, 0.0, 0.0)),
        ("GLY", (2.0, 0.0, 0.0)),
        ("SER", (2.0, 2.0, 0.0)),
        ("THR", (2.0, 2.0, 2.0)),
    ]
    rna = [
        ("A", (5.0, 1.0, 0.0)),
        ("C", (6.0, 2.0, 0.0)),
        ("G", (7.0, 2.0, 1.0)),
        ("U", (8.0, 3.0, 2.0)),
    ]
    rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    translation = np.asarray((11.0, -7.0, 3.0))
    reference = _write_structure(
        tmp_path / "reference.cif",
        {"A": ("protein", protein), "B": ("nucleic", rna)},
    )
    mobile = _write_structure(
        tmp_path / "mobile.cif",
        {
            "A": ("protein", _transform_entries(protein, rotation, translation)),
            "B": ("nucleic", _transform_entries(rna, rotation, translation)),
        },
    )

    before = set(tmp_path.iterdir())
    report = align_structures(
        {"reference": reference, "mobile": mobile}, reference="reference"
    )

    assert set(tmp_path.iterdir()) == before
    assert report["mobile"].matched_atoms == 8
    assert report["mobile"].matched_residues == 8
    assert report["mobile"].coverage == 1.0
    assert report["mobile"].rmsd < 1e-6
    np.testing.assert_allclose(
        _positions(report.aligned_cif("mobile"), tmp_path),
        _positions(reference.read_text(), tmp_path),
        atol=1e-3,
    )


def test_external_reference_matches_unique_sequence_under_another_chain_id(
    tmp_path: Path,
) -> None:
    entries = [
        ("ALA", (0.0, 0.0, 0.0)),
        ("GLY", (1.0, 0.0, 0.0)),
        ("SER", (1.0, 1.0, 0.0)),
    ]
    source = _write_structure(tmp_path / "source.cif", {"A": ("protein", entries)})
    reference = _write_structure(tmp_path / "target.pdb", {"X": ("protein", entries)})

    report = align_structures({"prediction": source}, reference=reference)

    assert report.reference_key is None
    assert dict(report["prediction"].chain_map) == {"A": "X"}


def test_coverage_uses_all_selected_reference_atoms(tmp_path: Path) -> None:
    source_entries = [
        ("ALA", (0.0, 0.0, 0.0)),
        ("GLY", (1.0, 0.0, 0.0)),
        ("SER", (1.0, 1.0, 0.0)),
    ]
    reference_entries = [
        *source_entries,
        ("THR", (1.0, 1.0, 1.0)),
    ]
    source = _write_structure(
        tmp_path / "source.cif", {"A": ("protein", source_entries)}
    )
    reference = _write_structure(
        tmp_path / "reference.cif", {"A": ("protein", reference_entries)}
    )

    report = align_structures({"source": source}, reference=reference)

    assert report["source"].matched_atoms == 3
    assert report["source"].reference_atoms == 4
    assert report["source"].coverage == pytest.approx(0.75)


def test_homomer_permutation_is_not_guessed(tmp_path: Path) -> None:
    entries = [
        ("ALA", (0.0, 0.0, 0.0)),
        ("GLY", (1.0, 0.0, 0.0)),
        ("SER", (1.0, 1.0, 0.0)),
    ]
    source = _write_structure(
        tmp_path / "source.cif",
        {"A": ("protein", entries), "B": ("protein", entries)},
    )
    reference = _write_structure(
        tmp_path / "reference.cif",
        {"X": ("protein", entries), "Y": ("protein", entries)},
    )

    with pytest.raises(StructureAlignmentError, match="ambiguous"):
        align_structures({"prediction": source}, reference=reference)

    report = align_structures(
        {"prediction": source},
        reference=reference,
        chain_map={"A": "X", "B": "Y"},
    )
    assert dict(report["prediction"].chain_map) == {"A": "X", "B": "Y"}


def test_ligand_only_alignment_uses_named_heavy_atoms(tmp_path: Path) -> None:
    entries = [("ATP", (0.0, 0.0, 0.0))]
    source = _write_structure(tmp_path / "source.cif", {"L": ("ligand", entries)})
    reference = _write_structure(tmp_path / "reference.cif", {"X": ("ligand", entries)})

    report = align_structures({"ligand": source}, reference=reference)

    assert report["ligand"].matched_atoms == 4
    assert report["ligand"].rmsd < 1e-9


def test_materialization_preserves_cif_categories_and_checks_digest(
    tmp_path: Path,
) -> None:
    entries = [
        ("ALA", (0.0, 0.0, 0.0)),
        ("GLY", (1.0, 0.0, 0.0)),
        ("SER", (1.0, 1.0, 0.0)),
    ]
    source = _write_structure(
        tmp_path / "source.cif", {"A": ("protein", entries)}, category=True
    )
    report = align_structures({"source": source}, reference="source")

    aligned = report.aligned_cif("source")
    assert "_foldjax_test.note preserve-me" in aligned
    output = report.write_aligned("source", tmp_path / "aligned.cif")
    assert output.is_file()
    with pytest.raises(FileExistsError):
        report.write_aligned("source", output)

    source.write_text(source.read_text() + "\n# changed\n")
    with pytest.raises(StructureAlignmentError, match="changed after alignment"):
        report.aligned_cif("source")
    source.unlink()
    with pytest.raises(StructureAlignmentError, match="cannot verify"):
        report.aligned_cif("source")


def test_report_is_small_json_and_prediction_keys_are_stable(tmp_path: Path) -> None:
    entries = [
        ("ALA", (0.0, 0.0, 0.0)),
        ("GLY", (1.0, 0.0, 0.0)),
        ("SER", (1.0, 1.0, 0.0)),
    ]
    path = _write_structure(tmp_path / "source.cif", {"A": ("protein", entries)})
    result = PredictionResult(
        model="openfold3",
        samples=(
            PredictionSample(
                seed=7,
                structure_path=path,
                metadata={"sample": 2},
            ),
        ),
    )
    key = "openfold3/seed-7/sample-02"

    report = align_predictions((result,), reference=key)
    payload = json.loads(report.to_json())

    assert report.keys() == (key,)
    assert payload["schema"] == "foldjax-structure-alignment-v1"
    assert len(report.to_json().encode()) < 4_000
    target = report.write_json(tmp_path / "alignment.json")
    assert target.is_file()
    restored = AlignmentReport.read_json(target)
    assert restored.summary() == report.summary()


def test_transform_rejects_invalid_shapes_and_collinear_anchors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="xyz axis"):
        RigidTransform.identity().apply(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="reflect"):
        RigidTransform(
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
            translation=(0.0, 0.0, 0.0),
        )

    entries = [
        ("ALA", (0.0, 0.0, 0.0)),
        ("GLY", (1.0, 0.0, 0.0)),
        ("SER", (2.0, 0.0, 0.0)),
    ]
    source = _write_structure(tmp_path / "source.cif", {"A": ("protein", entries)})
    with pytest.raises(StructureAlignmentError, match="collinear"):
        align_structures({"source": source}, reference="source")
