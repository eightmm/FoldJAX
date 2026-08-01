from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from .scripts.compare_prediction_outputs import compare_prediction_outputs

_ATOMS = (
    ("A", "1", "1", "ALA", "CA"),
    ("A", "1", "2", "GLY", "CA"),
    ("A", "1", "3", "SER", "CA"),
    ("B", "2", "1", "DA", "C4'"),
    ("B", "2", "2", "DC", "C4'"),
    ("B", "2", "3", "DG", "C4'"),
)


def _write_cif(
    path: Path,
    coordinates: np.ndarray,
    *,
    reverse_rows: bool = False,
    missing_marker: str = ".",
) -> None:
    rows = []
    for index, ((chain, entity, sequence, component, atom), coordinate) in enumerate(
        zip(_ATOMS, coordinates, strict=True), start=1
    ):
        atom_token = f'"{atom}"' if "'" in atom else atom
        rows.append(
            " ".join(
                (
                    "ATOM",
                    str(index),
                    "C",
                    atom_token,
                    missing_marker,
                    component,
                    chain,
                    entity,
                    sequence,
                    missing_marker,
                    *(f"{value:.6f}" for value in coordinate),
                    "1",
                )
            )
        )
    if reverse_rows:
        rows.reverse()
    path.write_text(
        """data_toy
loop_
_entity_poly.entity_id
_entity_poly.type
1 polypeptide(L)
2 polydeoxyribonucleotide

loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
"""
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def _write_summaries(reference: Path, candidate: Path) -> None:
    reference.write_text(
        json.dumps(
            {
                "plddt": 80.0,
                "ptm": 0.5,
                "chain_ptm": [0.6, 0.4],
                "chain_pair_iptm": [[0.0, 0.2], [0.2, 0.0]],
                "has_clash": True,
                "num_recycles": 1,
            }
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            {
                "plddt": 81.0,
                "ptm": 0.45,
                "chain_ptm": [0.5, 0.5],
                "chain_pair_iptm": [[0.0, 0.3], [0.3, 0.0]],
                "has_clash": True,
                "num_recycles": 1,
            }
        ),
        encoding="utf-8",
    )


def test_compare_prediction_outputs_matches_keys_and_reports_metrics(
    tmp_path: Path,
) -> None:
    reference_cif = tmp_path / "reference.cif"
    candidate_cif = tmp_path / "candidate.cif"
    reference_summary = tmp_path / "reference.json"
    candidate_summary = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    reference_coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.0],
            [0.1, 0.0, 0.2],
            [0.1, 0.0, 0.7],
            [0.6, 0.0, 0.2],
        ]
    )
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    candidate_coordinates = reference_coordinates @ rotation.T + np.asarray(
        [5.0, -2.0, 1.0]
    )
    candidate_coordinates[3:, 0] += 0.1
    _write_cif(reference_cif, reference_coordinates)
    _write_cif(
        candidate_cif,
        candidate_coordinates,
        reverse_rows=True,
        missing_marker="?",
    )
    _write_summaries(reference_summary, candidate_summary)

    report = compare_prediction_outputs(
        reference_cif=reference_cif,
        candidate_cif=candidate_cif,
        reference_summary=reference_summary,
        candidate_summary=candidate_summary,
        out=output,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["atom_identity"]["atom_count"] == 6
    assert report["atom_identity"]["candidate_rows_reordered"] is True
    assert report["counts"]["reference"]["per_chain"]["A"]["atom_count"] == 3
    assert report["rmsd_angstrom"]["all_atom"]["kabsch_rmsd"] > 0.0
    assert report["rmsd_angstrom"]["per_chain"]["A"]["independent_kabsch_rmsd"] < 1e-12
    assert report["rmsd_angstrom"]["markers"]["A:CA"]["atom_count"] == 3
    assert report["rmsd_angstrom"]["markers"]["B:C4'"]["atom_count"] == 3
    assert report["scores"]["scalar"]["plddt"]["delta"] == 1.0
    assert report["scores"]["array"]["chain_ptm"]["rmse"] == pytest.approx(0.1)
    assert report["scores"]["array"]["chain_ptm"]["max_abs"] == pytest.approx(0.1)
    assert report["clash"]["reference"]["chain_pairs"]["A:B"]["close_pair_count"] == 9
    assert report["clash"]["reference"]["computed_has_clash"] is True
    assert report["checks"]["candidate_summary_has_clash_consistent"] is True
    assert "shared random tape" in report["interpretation"]["rmsd_rng_caveat"]


def test_compare_prediction_outputs_rejects_different_atom_key_sets(
    tmp_path: Path,
) -> None:
    reference_cif = tmp_path / "reference.cif"
    candidate_cif = tmp_path / "candidate.cif"
    reference_summary = tmp_path / "reference.json"
    candidate_summary = tmp_path / "candidate.json"
    coordinates = np.zeros((6, 3), dtype=np.float64)
    _write_cif(reference_cif, coordinates)
    _write_cif(candidate_cif, coordinates)
    candidate_cif.write_text(
        candidate_cif.read_text(encoding="utf-8").replace('"C4\'"', "P", 1),
        encoding="utf-8",
    )
    _write_summaries(reference_summary, candidate_summary)

    with pytest.raises(ValueError, match="atom key sets differ"):
        compare_prediction_outputs(
            reference_cif=reference_cif,
            candidate_cif=candidate_cif,
            reference_summary=reference_summary,
            candidate_summary=candidate_summary,
            out=tmp_path / "comparison.json",
        )
