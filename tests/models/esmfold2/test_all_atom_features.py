"""Torch-free coverage for ESMFold2's all-biomolecule input adapter."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.data import all_atom, ccd, features, pdb
from foldjax.models.esmfold2.data.all_atom_constants import (
    MOL_TYPE_DNA,
    MOL_TYPE_NONPOLYMER,
    MOL_TYPE_PROTEIN,
    MOL_TYPE_RNA,
)
from foldjax.models.esmfold2.data.ccd import CCDStore
from foldjax.models.esmfold2.models import esmc
from foldjax.models.esmfold2.models import model as structure_model
from foldjax.paths import weights_dir


class _FakeCCD:
    """Small deterministic chemistry fixture with the CCDStore interface."""

    _atoms = {
        "SEP": [
            ("N", "N", 0),
            ("CA", "C", 0),
            ("C", "C", 0),
            ("O", "O", 0),
            ("CB", "C", 0),
            ("OG", "O", 0),
            ("P", "P", 0),
            ("OXT", "O", -1),
        ],
        "ATP": [("PA", "P", 0), ("O1A", "O", -1), ("O2A", "O", -1)],
    }
    _bonds = {
        "SEP": [("N", "CA"), ("CA", "C"), ("CA", "CB"), ("CB", "OG")],
        "ATP": [("PA", "O1A"), ("PA", "O2A")],
    }

    def __init__(self, _path: str | Path) -> None:
        pass

    def atoms(self, component: str):
        return self._atoms.get(component)

    def bonds(self, component: str):
        return self._bonds.get(component)

    def leaving_atoms(self, component: str) -> set[str]:
        if component == "SEP":
            return {"OXT"}
        if component == "ATP":
            return {"O2A"}
        return set()

    def conformer(self, component: str):
        names = {
            "ALA": all_atom.PROTEIN_HEAVY_ATOMS["ALA"],
            "GLY": all_atom.PROTEIN_HEAVY_ATOMS["GLY"],
            "SER": all_atom.PROTEIN_HEAVY_ATOMS["SER"],
            "DA": all_atom.DNA_HEAVY_ATOMS["DA"],
            "DT": all_atom.DNA_HEAVY_ATOMS["DT"],
            "A": all_atom.RNA_HEAVY_ATOMS["A"],
            "U": all_atom.RNA_HEAVY_ATOMS["U"],
        }.get(component)
        if names is None:
            records = self._atoms.get(component)
            names = [] if records is None else [record[0] for record in records]
        return {
            name: np.asarray((index, index + 1, index + 2), dtype=np.float32)
            for index, name in enumerate(names)
        }

    def idealized_position(self, component: str, atom_name: str):
        return self.conformer(component).get(atom_name)


@pytest.fixture
def fake_ccd(monkeypatch) -> None:
    monkeypatch.setattr(all_atom, "get_ccd_store", _FakeCCD)


def _mixed_document() -> dict:
    return {
        "name": "mixed",
        "entities": [
            {
                "type": "protein",
                "id": "PROT",
                "sequence": "ASA",
                "modifications": [{"ccd": "SEP", "position": 2}],
            },
            {"type": "dna", "id": "DNA", "sequence": "AT"},
            {"type": "rna", "id": "RNA", "sequence": "AU"},
            {"type": "ligand", "id": "ATPCHAIN", "ccd": "ATP"},
            {"type": "ligand", "id": "SMILES", "smiles": "CCO"},
        ],
        "bonds": [[['PROT', 2, 'OG'], ['ATPCHAIN', 1, 'PA']]],
    }


def _decode_rows(array: np.ndarray) -> list[str]:
    return [
        bytes(int(value) for value in row if value).decode("ascii")
        for row in array
    ]


def test_mixed_biomolecules_reach_one_model_feature_dictionary(fake_ccd) -> None:
    built = all_atom.build_job_features(
        _mixed_document(), base_dir=".", ccd_path="unused.pkl", seed=7
    )
    token_mask = built["token_attention_mask"][0]
    atom_mask = built["atom_attention_mask"][0]
    n_tokens = int(token_mask.sum())

    assert set(built["mol_type"][0].tolist()) == {
        MOL_TYPE_PROTEIN,
        MOL_TYPE_DNA,
        MOL_TYPE_RNA,
        MOL_TYPE_NONPOLYMER,
    }
    assert built["token_bonds"].shape == (1, n_tokens, n_tokens, 1)
    assert np.array_equal(
        built["token_bonds"], built["token_bonds"].swapaxes(1, 2)
    )
    assert built["token_bonds"].sum() > 0
    assert built["ref_pos"].shape[1] % all_atom.ATOM_BLOCK == 0
    assert np.all(atom_mask[: int(atom_mask.sum())])
    assert not np.any(atom_mask[int(atom_mask.sum()) :])
    assert set(_decode_rows(built["token_chain_id_chars"][0])) == {
        "PROT",
        "DNA",
        "RNA",
        "ATPCHAIN",
        "SMILES",
    }
    residue_names = _decode_rows(built["token_residue_name_chars"][0])
    assert {"SEP", "DA", "DT", "A", "U", "ATP", "LIG"} <= set(
        residue_names
    )

    # The explicit protein-ligand bond resolves by chain/residue/atom name.
    atom_names = [pdb.decode_atom_name(row) for row in built["ref_atom_name_chars"][0]]
    owners = built["atom_to_token"][0]
    chains = _decode_rows(built["token_chain_id_chars"][0])
    residues = built["residue_index"][0]
    endpoints = {}
    for atom_index in np.flatnonzero(atom_mask):
        token = int(owners[atom_index])
        key = (chains[token], int(residues[token]) + 1, atom_names[atom_index])
        endpoints[key] = token
    left = endpoints[("PROT", 2, "OG")]
    right = endpoints[("ATPCHAIN", 1, "PA")]
    assert built["token_bonds"][0, left, right, 0] == 1

    packed, _sequence_id, expand = esmc.pack_lm_inputs(
        built["input_ids"],
        built["asym_id"],
        built["residue_index"],
        built["mol_type"],
        built["token_attention_mask"],
    )
    assert packed.shape[1] == len("ASA") + 2
    assert np.all(expand[built["mol_type"] != MOL_TYPE_PROTEIN] == -1)
    modified = (
        (built["asym_id"] == 0)
        & (built["residue_index"] == 1)
        & (built["mol_type"] == MOL_TYPE_PROTEIN)
    )
    assert np.unique(expand[modified]).size == 1


def test_all_atom_dictionary_has_the_model_feature_contract(fake_ccd) -> None:
    legacy = features.build_features([("AG", "P", 0, 0)])
    mixed = all_atom.build_job_features(
        {
            "entities": [
                {"type": "protein", "id": "P", "sequence": "AG"},
                {"type": "dna", "id": "D", "sequence": "A"},
            ]
        },
        base_dir=".",
        ccd_path="unused.pkl",
        seed=0,
    )
    assert set(mixed) - all_atom.OUTPUT_METADATA_FEATURES == set(legacy)
    assert mixed["res_type"][0, :2].tolist() == legacy["res_type"][0].tolist()
    assert mixed["input_ids"][0, :2].tolist() == legacy["input_ids"][0].tolist()
    assert mixed["mol_type"][0, :2].tolist() == [MOL_TYPE_PROTEIN] * 2
    assert mixed["distogram_atom_idx"][0, :2].tolist() == [4, 6]
    assert np.array_equal(mixed["ref_pos"][0, 0], [0, 1, 2])


def test_all_biomolecule_mmcif_keeps_identity(fake_ccd) -> None:
    gemmi = pytest.importorskip("gemmi")
    built = all_atom.build_job_features(
        _mixed_document(), base_dir=".", ccd_path="unused.pkl", seed=7
    )
    text = pdb.to_mmcif(built["ref_pos"][0], built, name="mixed")
    structure = gemmi.make_structure_from_block(
        gemmi.cif.read_string(text).sole_block()
    )

    assert {chain.name for chain in structure[0]} == {
        "PROT",
        "DNA",
        "RNA",
        "ATPCHAIN",
        "SMILES",
    }
    assert {residue.name for chain in structure[0] for residue in chain} >= {
        "SEP",
        "DA",
        "DT",
        "A",
        "U",
        "ATP",
        "LIG",
    }
    assert sum(len(residue) for chain in structure[0] for residue in chain) == int(
        built["atom_attention_mask"].sum()
    )


def test_writer_metadata_follows_serving_padding(fake_ccd) -> None:
    built = all_atom.build_job_features(
        {"entities": [{"type": "dna", "id": "DNA", "sequence": "AT"}]},
        base_dir=".",
        ccd_path="unused.pkl",
        seed=0,
    )
    padded = features.pad_features(built, n_token=5, n_atom=64, n_msa=1)

    assert padded["token_chain_id_chars"].shape == (1, 5, 3)
    assert padded["token_residue_name_chars"].shape == (1, 5, 2)
    assert not np.any(padded["token_chain_id_chars"][0, 2:])
    assert not np.any(padded["token_residue_name_chars"][0, 2:])


def test_writer_metadata_never_enters_the_jax_model_tree(fake_ccd, monkeypatch) -> None:
    built = all_atom.build_job_features(
        {"entities": [{"type": "dna", "id": "DNA", "sequence": "AT"}]},
        base_dir=".",
        ccd_path="unused.pkl",
        seed=0,
    )
    seen = {}

    def run(_key, arrays, *_args):
        seen.update(arrays)
        return {"ok": jnp.asarray(1)}

    monkeypatch.setattr(inference, "_run", run)
    result = inference.predict(
        jnp.asarray([0, 0], dtype=jnp.uint32),
        built,
        inference.LoadedModel(
            parameters={"token_bonds.weight": jnp.zeros((2, 1))},
            settings=structure_model.ModelSettings(d_pair=2),
        ),
        precomputed_lm_states=jnp.zeros((1, 2, 1, 1)),
        compile_it=False,
        stop_after_trunk=True,
    )

    assert int(result["ok"]) == 1
    assert not (all_atom.OUTPUT_METADATA_FEATURES & seen.keys())
    assert set(seen) == set(built) - all_atom.OUTPUT_METADATA_FEATURES


def test_ccd_pickle_is_rejected_before_deserialization_when_unverified(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ccd.pkl"
    path.write_bytes(b"not a trusted pickle")

    with pytest.raises(ValueError, match="registered Biohub asset.*size"):
        CCDStore(path).atoms("ATP")


def test_ccd_derived_component_caches_are_lru_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Atom:
        def __init__(self, name: str) -> None:
            self.name = name

        def HasProp(self, key: str) -> bool:  # noqa: N802 - RDKit-compatible fake
            return key in {"leaving_atom", "name"}

        def GetProp(self, key: str) -> str:  # noqa: N802 - RDKit-compatible fake
            return "1" if key == "leaving_atom" else self.name

    class _Molecule:
        def __init__(self, name: str) -> None:
            self.name = name
            self.calls = 0

        def GetAtoms(self) -> list[_Atom]:  # noqa: N802 - RDKit-compatible fake
            self.calls += 1
            return [_Atom(self.name)]

    molecules = {name: _Molecule(name) for name in ("AAA", "BBB", "CCC", "DDD")}
    store = CCDStore(tmp_path / "unused.pkl")
    store._molecules = molecules  # noqa: SLF001 - focused cache contract
    monkeypatch.setattr(ccd, "_DERIVED_COMPONENT_CACHE_LIMIT", 3)

    for name in ("AAA", "BBB", "CCC", "AAA", "DDD"):
        assert store.leaving_atoms(name) == {name}

    assert molecules["AAA"].calls == 1
    assert list(store._leaving_cache) == ["CCC", "AAA", "DDD"]  # noqa: SLF001
    assert store.leaving_atoms("BBB") == {"BBB"}
    assert molecules["BBB"].calls == 2


def test_ccd_store_cache_releases_a_replaced_file(tmp_path: Path) -> None:
    first_path = tmp_path / "first.pkl"
    second_path = tmp_path / "second.pkl"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    ccd._cached_store.cache_clear()  # noqa: SLF001 - process-cache contract
    try:
        first = ccd.get_ccd_store(first_path)
        same = ccd.get_ccd_store(first_path)
        assert same is first
        reference = weakref.ref(first)
        del first, same

        ccd.get_ccd_store(second_path)
        gc.collect()

        assert reference() is None
        assert ccd._cached_store.cache_info().currsize == 1  # noqa: SLF001
    finally:
        ccd._cached_store.cache_clear()  # noqa: SLF001


@pytest.mark.slow
def test_registered_ccd_matches_publisher_tokenization_counts() -> None:
    """Pinned facts independently checked against Biohub's tokenizers."""

    ccd_path = weights_dir("esmfold2") / "ccd.pkl"
    if not ccd_path.is_file():
        pytest.skip("the registered ESMFold2 CCD is not in the weight store")
    document = _mixed_document()
    built = all_atom.build_job_features(
        document, base_dir=".", ccd_path=ccd_path, seed=7
    )

    # Biohub commit d0207ea3: ASA+SEP=20, DA/DT=41, A/U=42,
    # covalently-bound ATP=31, and ethanol=3 heavy atoms.
    assert int(built["atom_attention_mask"].sum()) == 137
    assert int(built["token_attention_mask"].sum()) == 50
    np.testing.assert_array_equal(
        built["ref_pos"][0, 0].view(np.uint32),
        np.asarray([3211403489, 3215485202, 3175790894], dtype=np.uint32),
    )
