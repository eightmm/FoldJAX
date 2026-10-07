"""Each Boltz-2 preprocessing run owns its ``processed/`` tree, pickle-free.

Every common job is written as ``boltz2_input.yaml``, so every run's record id
is the same. Preprocessing used to keep an existing
``processed/msa/{id}_{idx}.npz``, which answered a second job into the same
output directory -- or the same job after its alignment was edited -- with the
first run's MSA. The processed arrays were also loaded with pickle enabled.
"""

from __future__ import annotations

import io
import os
import pickle
import pickletools
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.boltz2.data.featurize import featurize_yaml
from foldjax.models.boltz2.data.mol import load_mol_pickle
from foldjax.models.boltz2.data.types import MSA, StructureV2
from foldjax.paths import weights_dir


def _mols() -> Path:
    mols = weights_dir("boltz2") / "mols"
    if not mols.is_dir():
        pytest.skip("boltz2 molecule directory not fetched")
    return mols


def _job(directory: Path, a3m_rows: list[str], *, name: str = "boltz2_input") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    a3m = directory / "query.a3m"
    a3m.write_text("".join(f">s{i}\n{row}\n" for i, row in enumerate(a3m_rows)))
    job = directory / f"{name}.yaml"
    job.write_text(
        "version: 1\nsequences:\n  - protein:\n      id: A\n"
        f"      sequence: ACDEFG\n      msa: {a3m}\n"
    )
    return job


@pytest.mark.real_store
def test_a_second_run_into_one_output_reads_its_own_edited_alignment(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    inputs = tmp_path / "inputs"
    first, _, _ = featurize_yaml(_job(inputs, ["ACDEFG", "ACDEYG"]), out, _mols())
    # The same job file, its alignment edited: one more homolog.
    second, _, _ = featurize_yaml(
        _job(inputs, ["ACDEFG", "ACDEYG", "ACKEFG"]), out, _mols()
    )

    assert first["msa"].shape[1] == 2
    assert second["msa"].shape[1] == 3, "the first run's processed MSA was reused"
    msa = MSA.load(out / "processed" / "msa" / "boltz2_input_0.npz")
    assert len(msa.sequences) == 3


@pytest.mark.real_store
def test_another_jobs_record_never_joins_the_manifest(tmp_path: Path) -> None:
    out = tmp_path / "out"
    featurize_yaml(_job(tmp_path / "a", ["ACDEFG"], name="alpha"), out, _mols())
    _, manifest, _ = featurize_yaml(
        _job(tmp_path / "b", ["ACDEFG"], name="beta"), out, _mols()
    )

    assert [record.id for record in manifest.records] == ["beta"]
    assert sorted(p.name for p in (out / "processed" / "records").iterdir()) == [
        "beta.json"
    ]
    assert not any(p.name.startswith(".processed-") for p in out.iterdir())


@pytest.mark.real_store
def test_a_planted_processed_symlink_is_replaced_not_followed(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (out / "processed").symlink_to(elsewhere, target_is_directory=True)

    featurize_yaml(_job(tmp_path / "in", ["ACDEFG"]), out, _mols())

    assert list(elsewhere.iterdir()) == []
    assert not (out / "processed").is_symlink()
    assert (out / "processed" / "manifest.json").is_file()


@pytest.mark.real_store
def test_processed_arrays_load_without_pickle(tmp_path: Path) -> None:
    out = tmp_path / "out"
    featurize_yaml(_job(tmp_path / "in", ["ACDEFG", "ACDEYG"]), out, _mols())

    arrays = sorted((out / "processed").rglob("*.npz"))
    assert {p.parent.name for p in arrays} >= {"structures", "msa", "constraints"}
    for path in arrays:
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                assert data[name].dtype != object, (path, name)
    with np.load(out / "processed" / "structures" / "boltz2_input.npz") as data:
        assert "pocket" not in data.files
    loaded = StructureV2.load(out / "processed/structures/boltz2_input.npz")
    assert loaded.pocket is None


def test_a_pickled_object_field_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "msa.npz"
    np.savez(
        path,
        sequences=np.zeros(0),
        deletions=np.zeros(0),
        residues=np.array([None], dtype=object),
    )
    with pytest.raises(ValueError, match="allow_pickle"):
        MSA.load(path)


@pytest.mark.real_store
def test_extra_molecules_round_trip_through_the_restricted_unpickler(
    tmp_path: Path,
) -> None:
    job = tmp_path / "ligand.yaml"
    job.write_text(
        "version: 1\nsequences:\n  - protein:\n      id: A\n"
        "      sequence: ACDEFG\n      msa: empty\n"
        "  - ligand:\n      id: B\n      smiles: CC(=O)O\n"
    )
    out = tmp_path / "out"
    featurize_yaml(job, out, _mols())

    raw = (out / "processed" / "mols" / "ligand.pkl").read_bytes()
    names = [arg for op, arg, _ in pickletools.genops(raw) if "UNICODE" in op.name]
    assert ("rdkit.Chem.rdchem", "Mol") in set(zip(names, names[1:]))
    molecules = load_mol_pickle(io.BytesIO(raw))
    assert molecules and all(
        type(mol).__name__ == "Mol" for mol in molecules.values()
    )


class _Payload:
    def __reduce__(self):
        return (os.getcwd, ())


def test_a_molecule_pickle_naming_another_global_is_refused() -> None:
    with pytest.raises(pickle.UnpicklingError, match="posix.getcwd|os.getcwd"):
        load_mol_pickle(io.BytesIO(pickle.dumps({"LIG": _Payload()})))
