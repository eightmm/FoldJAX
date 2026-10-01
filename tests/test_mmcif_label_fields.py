"""Every writer's mmCIF names chains and numbers residues in the label columns.

Structural comparison reads `label_asym_id`/`label_seq_id` and the entity
sequence, so a writer that leaves them to gemmi's defaults (`Axp`, `.`, no
`_entity_poly_seq`) hands every downstream tool a different file from the
other models'. Each case below drives a writer directly on a small structure,
on CPU and without weights, and checks the label columns against the job's
own chains: `label_asym_id` is the chain id (equal to `auth_asym_id`), polymer
residues are numbered 1..n, non-polymers carry `.`, and each polymer entity
lists its sequence.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

gemmi = pytest.importorskip("gemmi")


def _assert_labels(text: str, chains: dict[str, int | None]) -> None:
    """``chains`` maps each job chain id to its polymer length, None for a ligand."""
    block = gemmi.cif.read_string(text).sole_block()
    site = block.get_mmcif_category("_atom_site.")
    assert site, "no _atom_site rows"
    label_asym = site["label_asym_id"]
    auth_asym = site["auth_asym_id"]
    assert label_asym == auth_asym
    assert set(label_asym) == set(chains)

    numbers: dict[str, set] = defaultdict(set)
    entity_of: dict[str, set] = defaultdict(set)
    for asym, seq, entity in zip(
        label_asym, site["label_seq_id"], site["label_entity_id"], strict=True
    ):
        numbers[asym].add(seq)
        entity_of[asym].add(entity)
    for chain, length in chains.items():
        assert len(entity_of[chain]) == 1, chain
        if length is None:
            assert numbers[chain] == {False}, chain  # "." in the file
        else:
            assert {int(value) for value in numbers[chain]} == set(
                range(1, length + 1)
            ), chain

    struct_asym = block.get_mmcif_category("_struct_asym.")
    assert set(struct_asym["id"]) == set(chains)
    for asym, entity in zip(struct_asym["id"], struct_asym["entity_id"], strict=True):
        assert entity_of[asym] == {entity}

    poly_seq = block.get_mmcif_category("_entity_poly_seq.")
    polymers = {chain: n for chain, n in chains.items() if n is not None}
    if polymers:
        assert poly_seq, "no _entity_poly_seq"
        lengths: dict[str, int] = defaultdict(int)
        for entity in poly_seq["entity_id"]:
            lengths[entity] += 1
        for chain, length in polymers.items():
            (entity,) = entity_of[chain]
            assert lengths[entity] == length, chain


# --- ESMFold2 -----------------------------------------------------------------


def test_esmfold2_protein_writer_labels() -> None:
    from foldjax.models.esmfold2.data import features, pdb

    peptide = "ACDEFGHIK"
    built = features.build_features([(peptide, "A", 0, 0), (peptide, "B", 0, 1)])
    n_atoms = built["ref_pos"].shape[1]
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    text = pdb.to_mmcif(coords, built, name="pair")
    _assert_labels(text, {"A": len(peptide), "B": len(peptide)})

    # Identical chains share one entity, as the other writers number them.
    block = gemmi.cif.read_string(text).sole_block()
    assert block.get_mmcif_category("_entity.")["id"] == ["1"]


def test_esmfold2_writer_keeps_auth_numbering() -> None:
    """The bench reads ESMFold2 residues by auth_seq_id; that column is unchanged."""
    from foldjax.models.esmfold2.data import features, pdb

    built = features.build_features([("ACD", "A", 0, 0)])
    coords = np.zeros((built["ref_pos"].shape[1], 3), dtype=np.float32)
    site = (
        gemmi.cif.read_string(pdb.to_mmcif(coords, built))
        .sole_block()
        .get_mmcif_category("_atom_site.")
    )
    assert site["auth_seq_id"] == site["label_seq_id"]
    assert sorted({int(value) for value in site["auth_seq_id"]}) == [1, 2, 3]


def test_esmfold2_all_biomolecule_writer_labels(monkeypatch) -> None:
    from foldjax.models.esmfold2.data import all_atom, pdb
    from tests.models.esmfold2.test_all_atom_features import (
        _FakeCCD,
        _mixed_document,
    )

    monkeypatch.setattr(all_atom, "get_ccd_store", _FakeCCD)
    built = all_atom.build_job_features(
        _mixed_document(), base_dir=".", ccd_path="unused.pkl", seed=7
    )
    text = pdb.to_mmcif(built["ref_pos"][0], built, name="mixed")
    _assert_labels(
        text,
        {"PROT": 3, "DNA": 2, "RNA": 2, "ATPCHAIN": None, "SMILES": None},
    )


# --- Protenix and OpenDDE (one writer) ----------------------------------------


def test_protenix_writer_labels(tmp_path: Path) -> None:
    from foldjax.models.protenix.data.featurize_json import featurize_protein_json
    from foldjax.models.protenix.data.output import write_protenix_outputs

    features = featurize_protein_json(
        {
            "name": "labels",
            "sequences": [
                {"proteinChain": {"sequence": "CAG", "count": 2, "id": ["P", "Q"]}},
                {"ion": {"ion": "MG", "count": 1, "id": ["M"]}},
            ],
        },
        n_queries=2,
        n_keys=2,
    )
    n_atom = len(features["atom_to_token_idx"])
    cif_path = write_protenix_outputs(
        tmp_path,
        job_name="labels",
        seed=1,
        output={
            "coordinate": np.zeros((1, n_atom, 3), dtype=np.float32),
            "atom_plddt": np.full((1, n_atom), 0.75, dtype=np.float32),
            "summary_ranking_score": np.ones((1,), dtype=np.float32),
        },
        features=features,
    )[0]
    _assert_labels(Path(cif_path).read_text(), {"P": 3, "Q": 3, "M": None})


# --- OpenFold3 ------------------------------------------------------------------


def test_openfold3_writer_labels(tmp_path: Path) -> None:
    from foldjax.models.openfold3.data import OutputMetadata
    from foldjax.models.openfold3.inference import Prediction
    from foldjax.models.openfold3.output import write_prediction_outputs
    from tests.models.openfold3.feature_fixture import minimal_features

    # Two protein residues of chain P (one token each), one ligand atom in L.
    metadata = OutputMetadata(
        atom_name=np.asarray(["N", "CA", "N", "CA", "C1"]),
        element=np.asarray(["N", "C", "N", "C", "C"]),
        residue_name=np.asarray(["GLY", "GLY", "ALA", "ALA", "LIG"]),
        residue_id=np.asarray([1, 1, 2, 2, 1], dtype=np.int64),
        chain_id=np.asarray(["P", "P", "P", "P", "L"]),
        entity_id=np.asarray([1, 1, 1, 1, 2], dtype=np.int64),
        molecule_type_id=np.asarray([0, 0, 0, 0, 3], dtype=np.int8),
        bonds=np.zeros((0, 2), dtype=np.int64),
        bond_type=np.asarray([], dtype="<U6"),
    )
    features = minimal_features(tokens=3, atoms=5)
    features["atom_to_token_index"] = np.asarray([[0, 0, 1, 1, 2]], dtype=np.int32)
    features["num_atoms_per_token"] = np.asarray([[2, 2, 1]], dtype=np.int32)
    features["residue_index"] = np.asarray([[1, 2, 1]], dtype=np.int32)
    features["asym_id"] = np.asarray([[1, 1, 2]], dtype=np.int32)
    features["entity_id"] = np.asarray([[1, 1, 2]], dtype=np.int32)
    features["is_protein"] = np.asarray([[1, 1, 0]], dtype=np.int32)
    features["is_atomized"] = np.asarray([[0, 0, 1]], dtype=np.int32)
    names = np.zeros((1, 5, 4, 64), dtype=np.int32)
    for atom_index, atom_name in enumerate(metadata.atom_name):
        for position, character in enumerate(str(atom_name).ljust(4)):
            names[0, atom_index, position, ord(character) - 32] = 1
    features["ref_atom_name_chars"] = names
    prediction = Prediction(
        coordinates=np.zeros((1, 5, 3), dtype=np.float32),
        plddt=np.full((1, 5), 0.9, dtype=np.float32),
        ptm=np.asarray([0.7], dtype=np.float32),
        iptm=np.asarray([0.6], dtype=np.float32),
        chain_pair_iptm=None,
        pae_logits=None,
        pde_logits=None,
        distogram_logits=None,
    )
    written = write_prediction_outputs(
        prediction, features, tmp_path, output_metadata=metadata
    )
    _assert_labels(Path(written["structures"][0]).read_text(), {"P": 2, "L": None})


# --- Boltz-2 --------------------------------------------------------------------


def _boltz2_mols() -> Path | None:
    from foldjax.paths import weights_dir

    candidate = weights_dir() / "boltz2" / "mols"
    return candidate if candidate.is_dir() else None


def test_boltz2_writer_labels(tmp_path: Path) -> None:
    mols = _boltz2_mols()
    if mols is None:
        pytest.skip("Boltz-2 CCD molecule directory is not in the weight store")
    from foldjax.models.boltz2.api import featurize
    from foldjax.models.boltz2.data.write.structure import write_prediction

    features, record_id, struct_dir = featurize(
        seq=["MKV", "MKV"], ligand_ccd=["ZN"], mols=mols, out_dir=tmp_path / "in"
    )
    atom_mask = np.asarray(features["atom_pad_mask"]).reshape(-1)
    path = write_prediction(
        structure_npz=struct_dir / f"{record_id}.npz",
        coords=np.zeros((atom_mask.size, 3), dtype=np.float32),
        atom_pad_mask=atom_mask,
        out_path=tmp_path / "out.cif",
        fmt="cif",
    )
    _assert_labels(Path(path).read_text(), {"A": 3, "B": 3, "C": None})


# --- AlphaFold 3 ----------------------------------------------------------------


def test_alphafold3_writer_labels() -> None:
    try:
        from foldjax.models.alphafold3 import build

        build.register_runtime()
        from alphafold3 import structure
        from alphafold3.constants import chemical_components

        ccd = chemical_components.Ccd()
    except Exception as error:  # noqa: BLE001 - any missing native piece skips
        pytest.skip(f"AlphaFold 3 native runtime is not available: {error}")

    built = structure.from_sequences_and_bonds(
        sequences=["MKV", "MKV", "ZN"],
        chain_types=["polypeptide(L)", "polypeptide(L)", "non-polymer"],
        sequence_formats=[
            structure.SequenceFormat.FASTA,
            structure.SequenceFormat.FASTA,
            structure.SequenceFormat.CCD_CODES,
        ],
        bonded_atom_pairs=None,
        ccd=ccd,
    )
    _assert_labels(built.to_mmcif(), {"A": 3, "B": 3, "C": None})
