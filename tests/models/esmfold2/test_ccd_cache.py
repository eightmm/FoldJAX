from __future__ import annotations

import gc
import io
import os
import pickle
import weakref

import pytest

from foldjax.models.esmfold2.data import ccd


def test_ccd_cache_reuses_one_identity_and_releases_the_previous_generation(
    tmp_path, monkeypatch
) -> None:
    first_path = tmp_path / "first.pkl"
    second_path = tmp_path / "second.pkl"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    constructed = []

    class FakeStore:
        def __init__(self, path):
            self.path = path
            constructed.append(path)

    ccd._cached_store.cache_clear()
    monkeypatch.setattr(ccd, "CCDStore", FakeStore)
    try:
        first = ccd.get_ccd_store(first_path)
        assert ccd.get_ccd_store(first_path) is first
        first_ref = weakref.ref(first)

        second = ccd.get_ccd_store(second_path)
        del first
        gc.collect()

        assert first_ref() is None
        assert ccd.get_ccd_store(second_path) is second
        assert constructed == [str(first_path.resolve()), str(second_path.resolve())]
        assert ccd._cached_store.cache_info().currsize == 1
    finally:
        ccd._cached_store.cache_clear()


def test_release_helper_reports_only_a_loaded_cached_dictionary(tmp_path) -> None:
    path = tmp_path / "ccd.pkl"
    replacement_path = tmp_path / "replacement.pkl"
    path.write_bytes(b"identity")
    replacement_path.write_bytes(b"replacement")
    ccd._cached_store.cache_clear()
    try:
        unloaded = ccd.get_ccd_store(path)
        assert ccd._release_ccd_cache() is False
        del unloaded
        gc.collect()

        loaded = ccd.get_ccd_store(path)
        loaded._molecules = {}  # noqa: SLF001 - loaded-state contract
        assert ccd._release_ccd_cache() is True
        assert ccd._cached_store.cache_info().currsize == 0

        evicted_but_held = ccd.get_ccd_store(path)
        evicted_but_held._molecules = {}  # noqa: SLF001
        current_unloaded = ccd.get_ccd_store(replacement_path)
        assert ccd._release_ccd_cache() is False
        assert evicted_but_held._molecules == {}  # noqa: SLF001
        assert current_unloaded._molecules is None  # noqa: SLF001
    finally:
        ccd._cached_store.cache_clear()


def _read(payload: bytes, tmp_path):
    return ccd._read_deferred(io.BytesIO(payload), tmp_path / "ccd.pkl")


def test_deferred_molecules_replay_exactly_what_pickle_builds(tmp_path) -> None:
    chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    serine = chem.AddHs(chem.MolFromSmiles("N[C@@H](CO)C(=O)O"))
    for atom in serine.GetAtoms():
        atom.SetProp("name", f"A{atom.GetIdx()}")
        atom.SetProp("leaving_atom", "1" if atom.GetIdx() == 6 else "0")
    AllChem.EmbedMolecule(serine, randomSeed=7)
    serine.GetConformer().SetProp("name", "Ideal")
    shared = chem.MolFromSmiles("CCO")
    # Biohub's file carries atom and conformer names; RDKit pickles them only
    # when asked to.
    previous = chem.GetDefaultPickleProperties()
    chem.SetDefaultPickleProperties(chem.PropertyPickleOptions.AllProps)
    try:
        payload = pickle.dumps({"SER": serine, "EOH": shared, "ALIAS": shared})
    finally:
        chem.SetDefaultPickleProperties(previous)

    eager = pickle.loads(payload)  # noqa: S301 - a payload this test built
    deferred = _read(payload, tmp_path)

    assert len(deferred) == 3
    assert "SER" in deferred and "NOPE" not in deferred
    assert deferred.get("NOPE") is None
    options = (
        chem.PropertyPickleOptions.AllProps
        | chem.PropertyPickleOptions.CoordsAsDouble
    )
    for component, molecule in eager.items():
        replayed = deferred.get(component)
        assert type(replayed) is type(molecule)
        assert replayed.ToBinary(options) == molecule.ToBinary(options)
        assert deferred.get(component) is replayed
    # One pickled object stays one object, as `pickle.loads` keeps it.
    assert eager["EOH"] is eager["ALIAS"]
    assert deferred.get("EOH") is deferred.get("ALIAS")

    store = ccd.CCDStore(tmp_path / "unused.pkl")
    store._molecules = deferred  # noqa: SLF001 - the loaded state
    reference = ccd.CCDStore(tmp_path / "unused.pkl")
    reference._molecules = eager  # noqa: SLF001
    assert store.atoms("SER") == reference.atoms("SER")
    assert store.bonds("SER") == reference.bonds("SER")
    assert store.leaving_atoms("SER") == reference.leaving_atoms("SER") == {"A6"}
    replayed_conformer = store.conformer("SER")
    for name, position in reference.conformer("SER").items():
        assert replayed_conformer[name].tobytes() == position.tobytes()


def test_deferred_reader_admits_only_rdkit_molecules(tmp_path) -> None:
    with pytest.raises(pickle.UnpicklingError, match="unexpected global"):
        _read(pickle.dumps({"X": os.getcwd}), tmp_path)
    with pytest.raises(ValueError, match="not a pickled RDKit Mol"):
        _read(pickle.dumps({"X": 1}), tmp_path)
    with pytest.raises(ValueError, match="not a component dictionary"):
        _read(pickle.dumps([]), tmp_path)
