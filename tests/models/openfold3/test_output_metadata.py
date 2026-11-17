"""Exact OpenFold3 atom identity and connectivity across feature archives.

Also the paths on either side of them: which file ``save_features`` actually
wrote, and which names the output writer will accept.
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pytest

from foldjax.models.openfold3.data import (
    MODEL_FEATURES,
    OUTPUT_PREFIX,
    OutputMetadata,
    featurize_query,
    load_feature_archive,
    load_features,
    pad_features,
    save_features,
)
from foldjax.models.openfold3.inference import Prediction
from foldjax.models.openfold3.models.representative_atoms import (
    RepresentativeAtomTable,
)
from foldjax.models.openfold3.output import (
    _output_path,
    _safe_output_name,
    atom_metadata,
    write_prediction_outputs,
)

from .feature_fixture import minimal_features
from .real_targets import requires_real_targets


def _table() -> RepresentativeAtomTable:
    return RepresentativeAtomTable(
        *(np.zeros(32, dtype=np.float32) for _ in RepresentativeAtomTable._fields)
    )


def _metadata() -> OutputMetadata:
    return OutputMetadata(
        atom_name=np.asarray(["N", "CA", "C1", "O1"]),
        element=np.asarray(["N", "C", "C", "O"]),
        residue_name=np.asarray(["SEP", "SEP", "LIG-X", "LIG-X"]),
        residue_id=np.asarray([7, 7, 1, 1], dtype=np.int64),
        chain_id=np.asarray(["P", "P", "L", "L"]),
        entity_id=np.asarray([1, 1, 2, 2], dtype=np.int64),
        molecule_type_id=np.asarray([0, 0, 3, 3], dtype=np.int8),
        bonds=np.asarray([[0, 1], [2, 3], [1, 2]], dtype=np.int64),
        bond_type=np.asarray(["SINGLE", "DOUBLE", "SINGLE"]),
    )


def _features() -> dict[str, np.ndarray]:
    features = minimal_features(tokens=2, atoms=4)
    features["residue_index"] = np.asarray([[7, 1]], dtype=np.int32)
    features["asym_id"] = np.asarray([[1, 2]], dtype=np.int32)
    features["entity_id"] = np.asarray([[1, 2]], dtype=np.int32)
    features["is_protein"] = np.asarray([[1, 0]], dtype=np.int32)
    features["is_atomized"] = np.asarray([[0, 1]], dtype=np.int32)
    names = np.zeros((1, 4, 4, 64), dtype=np.int32)
    for atom_index, atom_name in enumerate(("N", "CA", "C1", "O1")):
        for position, character in enumerate(atom_name.ljust(4)):
            names[0, atom_index, position, ord(character) - 32] = 1
    features["ref_atom_name_chars"] = names
    return features


def _prediction() -> Prediction:
    return Prediction(
        coordinates=np.zeros((1, 4, 3), dtype=np.float32),
        plddt=np.full((1, 4), 0.9, dtype=np.float32),
        ptm=np.asarray([0.7], dtype=np.float32),
        iptm=np.asarray([0.6], dtype=np.float32),
        chain_pair_iptm=None,
        pae_logits=None,
        pde_logits=None,
        distogram_logits=None,
    )


def test_archive_round_trip_preserves_ligand_identity_and_bonds(
    tmp_path: Path,
) -> None:
    features = _features()
    archive = save_features(
        features,
        tmp_path / "ligand.npz",
        representative_atoms=_table(),
        output_metadata=_metadata(),
    )

    compatible_features, _table_value = load_features(archive)
    assert set(compatible_features) == set(features)
    assert not any(name.startswith(OUTPUT_PREFIX) for name in compatible_features)

    loaded, _table_value, metadata = load_feature_archive(archive)
    assert metadata is not None
    np.testing.assert_array_equal(
        metadata.residue_name, ["SEP", "SEP", "LIG-X", "LIG-X"]
    )
    np.testing.assert_array_equal(metadata.bonds, [[0, 1], [2, 3], [1, 2]])

    written = write_prediction_outputs(
        _prediction(),
        loaded,
        tmp_path / "out",
        output_metadata=metadata,
    )
    block = gemmi.cif.read_file(str(written["structures"][0])).sole_block()
    atom_site = block.get_mmcif_category("_atom_site.")
    assert atom_site["label_comp_id"] == ["SEP", "SEP", "LIG-X", "LIG-X"]

    component_bonds = block.get_mmcif_category("_chem_comp_bond.")
    assert set(component_bonds["comp_id"]) == {"SEP", "LIG-X"}
    ligand_index = component_bonds["comp_id"].index("LIG-X")
    assert component_bonds["value_order"][ligand_index] == "DOUB"

    connection = block.get_mmcif_category("_struct_conn.")
    assert connection["conn_type_id"] == ["covale"]
    assert connection["ptnr1_label_comp_id"] == ["SEP"]
    assert connection["ptnr2_label_comp_id"] == ["LIG-X"]
    assert connection["ptnr2_label_seq_id"] == [False]


@requires_real_targets
def test_real_ligand_metadata_survives_the_archive_and_mmcif(
    tmp_path: Path,
) -> None:
    """Exercise the exact sidecar on an actual CCD-backed OpenFold3 job."""
    from foldjax.models.openfold3.data import featurize_query_with_metadata
    from tests._foldbench import load_target, openfold3_query

    target = load_target("1CBS")
    features, metadata = featurize_query_with_metadata(
        openfold3_query(target, include_ligands=True, drop_additives=True)
    )
    ligand = metadata.molecule_type_id == 3
    assert ligand.any()
    assert set(metadata.residue_name[ligand]) == set(target.biological_ligands)
    assert np.any(ligand[metadata.bonds])

    archive = save_features(
        features,
        tmp_path / "1cbs.npz",
        representative_atoms=_table(),
        output_metadata=metadata,
    )
    loaded, _table_value, restored = load_feature_archive(archive)
    assert restored is not None
    np.testing.assert_array_equal(restored.residue_name, metadata.residue_name)
    np.testing.assert_array_equal(restored.bonds, metadata.bonds)

    n_atom = features["atom_mask"].shape[-1]
    prediction = _prediction()._replace(
        coordinates=np.zeros((1, n_atom, 3), dtype=np.float32),
        plddt=np.full((1, n_atom), 0.9, dtype=np.float32),
    )
    written = write_prediction_outputs(
        prediction,
        loaded,
        tmp_path / "real-output",
        name="1CBS",
        output_metadata=restored,
    )
    block = gemmi.cif.read_file(str(written["structures"][0])).sole_block()
    assert set(block.get_mmcif_category("_atom_site.")["label_comp_id"]) >= set(
        target.biological_ligands
    )
    assert block.get_mmcif_category("_chem_comp_bond.")["comp_id"]


def test_malformed_output_metadata_is_rejected_on_save_and_load(
    tmp_path: Path,
) -> None:
    features = _features()
    malformed = _metadata()._replace(
        bonds=np.asarray([[0, 9]], dtype=np.int64),
        bond_type=np.asarray(["SINGLE"]),
    )
    with pytest.raises(ValueError, match="out-of-range bond endpoint"):
        save_features(
            features,
            tmp_path / "bad.npz",
            representative_atoms=_table(),
            output_metadata=malformed,
        )

    archive = save_features(
        features,
        tmp_path / "incomplete.npz",
        representative_atoms=_table(),
        output_metadata=_metadata(),
    )
    with np.load(archive, allow_pickle=False) as loaded:
        payload = {
            name: loaded[name]
            for name in loaded.files
            if name != f"{OUTPUT_PREFIX}bond_type"
        }
    np.savez(archive, **payload)
    with pytest.raises(ValueError, match="output metadata is incomplete"):
        load_feature_archive(archive)


def test_legacy_archive_fallback_warns_about_lost_chemistry(tmp_path: Path) -> None:
    features = _features()
    with pytest.warns(RuntimeWarning, match="no exact output metadata"):
        written = write_prediction_outputs(_prediction(), features, tmp_path)
    block = gemmi.cif.read_file(str(written["structures"][0])).sole_block()
    assert block.get_mmcif_category("_struct_conn.") == {}


def test_released_atom_slot_width_is_fixed() -> None:
    features = _features()
    with pytest.raises(ValueError, match="requires max_atoms_per_token=23"):
        pad_features(features, n_token=2, n_atom=4, max_atoms_per_token=24)
    with pytest.raises(ValueError, match="requires max_atoms_per_token=23"):
        featurize_query({}, max_atoms_per_token=24)


def test_output_archive_modules_import_without_torch() -> None:
    import subprocess
    import sys

    script = r"""
import builtins
import sys

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"output archive closure imported {name}")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from foldjax.models.openfold3.data import load_feature_archive
from foldjax.models.openfold3.output import atom_metadata, write_structure

assert callable(load_feature_archive)
assert callable(atom_metadata) and callable(write_structure)
assert "torch" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout


def test_atom_metadata_ignores_padded_chain_ids() -> None:
    """Chain letters are assigned over the real chains, in their own order.

    Padding's zero sorts before a real sparse ``asym_id``. Ranking over the
    padded column instead used to shift the only emitted chain to B.
    """
    features = {
        "ref_atom_name_chars": np.zeros((1, 2, 4, 64), dtype=np.float32),
        "ref_element": np.zeros((1, 2, 119), dtype=np.float32),
        "atom_mask": np.asarray([[1, 0]], dtype=np.float32),
        "atom_to_token_index": np.asarray([[0, 1]], dtype=np.int64),
        "restype": np.zeros((1, 2, 32), dtype=np.float32),
        "residue_index": np.asarray([[1, 0]], dtype=np.int64),
        "asym_id": np.asarray([[8, 0]], dtype=np.int64),
    }

    assert atom_metadata(features).chain_id.tolist() == ["A"]


def test_save_and_load_features_use_the_path_they_report(tmp_path: Path) -> None:
    """The returned path is the one written, suffix normalization included."""
    path = save_features(
        minimal_features(tokens=3, atoms=5),
        tmp_path / "FEATURES.NPZ",
        representative_atoms=_table(),
    )

    assert path == tmp_path / "FEATURES.npz"
    assert path.is_file()
    loaded, table = load_features(path)
    assert set(loaded) == set(MODEL_FEATURES)
    assert isinstance(table, RepresentativeAtomTable)


def test_saving_features_replaces_a_symlink_without_touching_its_target(
    tmp_path: Path,
) -> None:
    """Writing through a symlink would overwrite a file the user owns."""
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"owned by user")
    target = tmp_path / "features.npz"
    target.symlink_to(outside)

    path = save_features(
        minimal_features(tokens=3, atoms=5),
        target,
        representative_atoms=_table(),
    )

    assert path == target and path.is_file() and not path.is_symlink()
    assert outside.read_bytes() == b"owned by user"
    assert set(load_features(path)[0]) == set(MODEL_FEATURES)


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../escape", "/tmp/escape", "a/b", "a\\b", "a\nb"]
)
def test_output_names_are_single_safe_filename_components(name: str) -> None:
    """The prediction name reaches a path, so it cannot carry a separator."""
    safe = _safe_output_name(name)
    assert safe and safe not in {".", ".."}
    assert "/" not in safe and "\\" not in safe
    assert all(32 <= ord(character) != 127 for character in safe)


@pytest.mark.parametrize("name", ["1ubq", "job_A", "complex-2.v1", "A.B-c_d"])
def test_ordinary_output_names_are_kept_as_written(name: str) -> None:
    from foldjax.models.protenix.data.output import sanitize_job_name
    from foldjax.output import safe_job_name

    assert _safe_output_name(name) == sanitize_job_name(name) == name
    assert safe_job_name(name) == name


def test_existing_output_symlinks_are_refused(tmp_path: Path) -> None:
    """An output slot pre-pointed outside the directory must not be followed."""
    root = tmp_path / "out"
    root.mkdir()
    outside = tmp_path / "outside.npz"
    outside.touch()
    (root / "prediction_raw.npz").symlink_to(outside)

    with pytest.raises(ValueError, match="escapes its output directory"):
        _output_path(root, "prediction_raw.npz")


def test_sidecarless_atom_metadata_orders_chains_by_first_appearance() -> None:
    """Sidecar-less archives label chains A, B, ... by atom-stream appearance."""
    from foldjax.models.openfold3.data import normalize_asym_ids

    features = {
        "ref_atom_name_chars": np.zeros((1, 5, 4, 64), dtype=np.float32),
        "ref_element": np.zeros((1, 5, 119), dtype=np.float32),
        "atom_mask": np.asarray([[1, 1, 1, 1, 0]], dtype=np.float32),
        "atom_to_token_index": np.asarray([[0, 0, 1, 1, 2]], dtype=np.int64),
        "token_mask": np.asarray([[1, 1, 0]], dtype=np.float32),
        "restype": np.zeros((1, 3, 32), dtype=np.float32),
        "residue_index": np.asarray([[1, 1, 0]], dtype=np.int64),
        "asym_id": np.asarray([[5, 2, 0]], dtype=np.int64),
    }
    assert atom_metadata(features).chain_id.tolist() == ["A", "A", "B", "B"]

    # After normalize_asym_ids, [5, 2, 0] -> [1, 0, 0]; first appearance stays A, B.
    normalized, _ = normalize_asym_ids(features)
    assert atom_metadata(normalized).chain_id.tolist() == ["A", "A", "B", "B"]

    # Edge cases: zero emitted atoms -> empty chain_id; 27 chains -> A..Z, AA.
    all_masked = {**features, "atom_mask": np.zeros((1, 5), dtype=np.float32)}
    assert atom_metadata(all_masked).chain_id.tolist() == []
    many = {
        "ref_atom_name_chars": np.zeros((1, 27, 4, 64), dtype=np.float32),
        "ref_element": np.zeros((1, 27, 119), dtype=np.float32),
        "atom_mask": np.ones((1, 27), dtype=np.float32),
        "atom_to_token_index": np.arange(27, dtype=np.int64)[None],
        "restype": np.zeros((1, 27, 32), dtype=np.float32),
        "residue_index": np.ones((1, 27), dtype=np.int64),
        "asym_id": np.arange(27, 0, -1, dtype=np.int64)[None],
    }
    labels = atom_metadata(many).chain_id.tolist()
    assert labels[0] == "A" and labels[25] == "Z" and labels[26] == "AA"


def test_masked_nonpolymer_atoms_do_not_disqualify_polymer_chain_clash_veto() -> None:
    """Masked atoms sharing ``asym_id`` with a polymer chain keep ``has_clash``."""
    import jax.numpy as jnp

    from foldjax.models.openfold3.models.clash import compute_has_clash
    from foldjax.models.openfold3.output import _has_clash_blocked, confidence_summary

    # Two 2-atom polymer chains (asym_id 0 and 1) overlapping within 0.1 A, plus
    # one masked non-polymer padding atom assigned to asym_id 0.
    base = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    zero_atom = np.zeros((1, 3), dtype=np.float32)
    coords = np.concatenate((base, base + 0.1, zero_atom))[None]
    atom_asym = np.asarray([0, 0, 1, 1, 0], dtype=np.int32)
    atom_mask = np.asarray([True, True, True, True, False])
    is_polymer = np.asarray([True, True, True, True, False])

    jax_clash = compute_has_clash(
        jnp.asarray(atom_asym),
        jnp.asarray(coords),
        jnp.asarray(atom_mask),
        jnp.asarray(is_polymer),
        n_chain=2,
    )
    blocked_clash = _has_clash_blocked(coords, atom_mask, atom_asym, is_polymer)
    np.testing.assert_array_equal(np.asarray(jax_clash), [1.0])
    np.testing.assert_array_equal(blocked_clash, [1.0])

    # Through confidence_summary with a padded non-polymer token at index 2 (asym_id 0).
    ranking_features = {
        "atom_mask": atom_mask.astype(np.float32)[None],
        "atom_to_token_index": np.asarray([[0, 0, 1, 1, 2]], dtype=np.int32),
        "token_mask": np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
        "asym_id": np.asarray([[0, 1, 0]], dtype=np.int32),
        "is_protein": np.zeros((1, 3), dtype=np.int32),
        "is_rna": np.asarray([[1, 1, 0]], dtype=np.int32),
        "is_dna": np.zeros((1, 3), dtype=np.int32),
    }
    pred = Prediction(
        coordinates=coords,
        plddt=np.full((1, 5), 0.9, dtype=np.float32),
        ptm=np.asarray([0.8], dtype=np.float32),
        iptm=np.asarray([0.7], dtype=np.float32),
        chain_pair_iptm=None,
        pae_logits=None,
        pde_logits=None,
        distogram_logits=None,
    )
    summary = confidence_summary(pred, ranking_features)
    assert summary["samples"][0]["has_clash"] == 1.0
    assert summary["samples"][0]["sample_ranking_score"] < -90.0

    # Boundary and empty-mask checks: all-masked -> 0.0; distance == threshold -> 0.0.
    zero_mask = np.zeros(5, dtype=bool)
    np.testing.assert_array_equal(
        _has_clash_blocked(coords, zero_mask, atom_asym, is_polymer), [0.0]
    )
    np.testing.assert_array_equal(
        np.asarray(
            compute_has_clash(
                jnp.asarray(atom_asym),
                jnp.asarray(coords),
                jnp.asarray(zero_mask),
                jnp.asarray(is_polymer),
                n_chain=2,
            )
        ),
        [0.0],
    )
    at_limit = np.concatenate(
        (base, base + np.asarray([[1.1, 0.0, 0.0]]), np.zeros((1, 3), dtype=np.float32))
    )[None]
    np.testing.assert_array_equal(
        _has_clash_blocked(at_limit, atom_mask, atom_asym, is_polymer, threshold=1.1),
        [0.0],
    )
