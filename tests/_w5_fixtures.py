"""Small synthetic outputs for the analysis commands (report, interfaces, compare).

A replay backend hands a generated two-chain complex back through the real
`foldjax.predict` pipeline -- input translation, `foldjax.output.normalize`,
`confidence_arrays.place` and the manifest writer -- so the directories these
tests read are laid out exactly as a run lays them out. Nothing needs a GPU or
weights.

The complex: chain A (30 residues) and chain B (24 residues), two parallel
helices whose facing CB atoms sit 6-8 A apart, plus an optional ligand chain L
(a six-membered carbon ring). Every coordinate is a closed-form function of
the residue index, so a test can perturb a copy and know the answer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

import foldjax
from foldjax import confidence_arrays
from foldjax.registry import backend_override, get_backend
from foldjax.schema import PredictionRequest, PredictionResult, PredictionSample

SEQUENCE_A = "MKTAYIAKQRGISFVKSHFSRQLEERLGLA"
SEQUENCE_B = "GSHMAELKAKLEEALKKAGIEVKA"
_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}  # fmt: skip


def _helix(
    length: int, origin: np.ndarray, phase: float
) -> list[dict[str, np.ndarray]]:
    """Backbone and CB positions of an ideal-ish alpha helix along z."""
    residues = []
    for index in range(length):
        angle = math.radians(100.0 * index) + phase
        rise = 1.5 * index

        def at(radius: float, offset: float, dz: float) -> np.ndarray:
            return origin + np.array(
                [
                    radius * math.cos(angle + offset),
                    radius * math.sin(angle + offset),
                    rise + dz,
                ]
            )

        residues.append(
            {
                "N": at(1.55, -0.45, -0.5),
                "CA": at(2.3, 0.0, 0.0),
                "C": at(1.65, 0.45, 0.6),
                "O": at(1.9, 0.75, 1.6),
                "CB": at(3.4, 0.1, -0.3),
            }
        )
    return residues


def complex_coordinates(*, ligand: bool = False) -> dict[str, list]:
    """chain -> list of (residue name, {atom name: xyz})."""
    chains: dict[str, list] = {}
    for name, sequence, origin, phase in (
        ("A", SEQUENCE_A, np.zeros(3), 0.0),
        ("B", SEQUENCE_B, np.array([9.0, 0.0, 3.0]), math.pi),
    ):
        backbone = _helix(len(sequence), origin, phase)
        chains[name] = []
        for letter, atoms in zip(sequence, backbone, strict=True):
            residue = _THREE[letter]
            kept = {key: value for key, value in atoms.items()}
            if residue == "GLY":
                kept.pop("CB")
            chains[name].append((residue, kept))
    if ligand:
        centre = np.array([4.5, 6.0, 20.0])
        ring = {
            f"C{k + 1}": centre
            + 1.39 * np.array([math.cos(k * math.pi / 3), math.sin(k * math.pi / 3), 0])
            for k in range(6)
        }
        chains["L"] = [("BNZ", ring)]
    return chains


def write_cif(
    path: Path,
    chains: Mapping[str, list],
    *,
    bfactor: Any = 80.0,
    transform: Any = None,
) -> Path:
    """Write ``chains`` as an mmCIF; ``bfactor(chain, index, atom)`` or a constant."""
    structure = gemmi.Structure()
    structure.name = path.stem
    model = gemmi.Model("1")
    serial = 0
    for chain_name, residues in chains.items():
        chain = gemmi.Chain(chain_name)
        for index, (residue_name, atoms) in enumerate(residues, start=1):
            residue = gemmi.Residue()
            residue.name = residue_name
            residue.seqid = gemmi.SeqId(index, " ")
            polymer = residue_name != "BNZ"
            residue.het_flag = "A" if polymer else "H"
            residue.entity_type = (
                gemmi.EntityType.Polymer if polymer else gemmi.EntityType.NonPolymer
            )
            for atom_name, xyz in atoms.items():
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(atom_name[0])
                position = np.asarray(xyz, dtype=float)
                if transform is not None:
                    position = transform(chain_name, index, atom_name, position)
                atom.pos = gemmi.Position(*map(float, position))
                atom.occ = 1.0
                atom.b_iso = float(
                    bfactor(chain_name, index, atom_name)
                    if callable(bfactor)
                    else bfactor
                )
                serial += 1
                atom.serial = serial
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.assign_label_seq_id(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.make_mmcif_document().write_file(str(path))
    return path


def write_job(directory: Path, name: str, *, ligand: bool = False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "a.a3m").write_text(f">query\n{SEQUENCE_A}\n")
    (directory / "b.a3m").write_text(f">query\n{SEQUENCE_B}\n")
    entities: list[dict[str, Any]] = [
        {"type": "protein", "id": "A", "sequence": SEQUENCE_A, "unpaired_msa": "a.a3m"},
        {"type": "protein", "id": "B", "sequence": SEQUENCE_B, "unpaired_msa": "b.a3m"},
    ]
    if ligand:
        entities.append({"type": "ligand", "id": "L", "smiles": "c1ccccc1"})
    path = directory / f"{name}.json"
    path.write_text(json.dumps({"name": name, "entities": entities}))
    return path


def token_maps(chains: Mapping[str, list]) -> dict[str, np.ndarray]:
    """One token per polymer residue, one per ligand atom (the models' rule)."""
    chain_ids, residue_index = [], []
    for name, residues in chains.items():
        for index, (residue_name, atoms) in enumerate(residues, start=1):
            count = len(atoms) if residue_name == "BNZ" else 1
            chain_ids += [name] * count
            residue_index += [index] * count
    return {
        "token_chain_id": np.asarray(chain_ids),
        "token_residue_index": np.asarray(residue_index, dtype=np.int32),
    }


def synthetic_pae(chain_ids: np.ndarray, *, inter: float = 6.0) -> np.ndarray:
    """2 A within a chain; ``inter`` A between chains, rising away from the diagonal."""
    n = len(chain_ids)
    same = chain_ids[:, None] == chain_ids[None, :]
    ramp = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) / max(n, 1)
    return np.where(same, 2.0, inter + 8.0 * ramp).astype(np.float32)


def replay_backend(
    model: str = "boltz2",
    *,
    name: str = "pair",
    ligand: bool = False,
    samples: int = 2,
    with_pae: bool = True,
):
    """A backend class for ``model`` whose predict writes the synthetic complex."""
    chains = complex_coordinates(ligand=ligand)
    base = type(get_backend(model))

    class Replay(base):  # type: ignore[misc, valid-type]
        def predict(self, request: PredictionRequest) -> PredictionResult:
            maps = token_maps(chains)
            job = name
            built = []
            for index in range(samples):

                def jitter(chain, residue, atom, xyz, index=index):
                    return xyz + 0.05 * index * np.array([1.0, -1.0, 0.5])

                structure = write_cif(
                    request.output_dir / f"{job}_model_{index}.cif",
                    chains,
                    bfactor=lambda c, r, a: 90.0 - r,
                    transform=jitter,
                )
                arrays: dict[str, Any] = dict(maps)
                n_token = len(maps["token_chain_id"])
                arrays["token_plddt"] = np.linspace(0.95, 0.6, n_token)
                unavailable = {}
                if with_pae:
                    arrays["pae"] = synthetic_pae(
                        maps["token_chain_id"], inter=6.0 + index
                    )
                else:
                    unavailable["pae"] = "not returned by this replay"
                confidence_arrays.write(
                    confidence_arrays.staged_path(structure),
                    model=model,
                    arrays=arrays,
                    scales={"token_plddt": "0-1"},
                    unavailable=unavailable,
                    sample={"sample": index},
                )
                metadata = {
                    "job": job,
                    "sample": index,
                    **confidence_arrays.sample_metadata(structure),
                }
                built.append(
                    PredictionSample(
                        seed=request.seed,
                        structure_path=structure,
                        scores={
                            "confidence_score": 0.8 - 0.1 * index,
                            "complex_plddt": 0.85,
                            "ptm": 0.8,
                            "iptm": 0.7 - 0.1 * index,
                            "affinity_pred_value": 1.5 + index,
                            "affinity_probability_binary": 0.6 - 0.2 * index,
                        },
                        metadata=metadata,
                    )
                )
            return PredictionResult(
                model=model, samples=tuple(built), output_dir=request.output_dir
            )

    return Replay


def run_replay(
    out: Path,
    *,
    model: str = "boltz2",
    name: str = "pair",
    ligand: bool = False,
    samples: int = 2,
    seed: int = 7,
    with_pae: bool = True,
    scratch: Path | None = None,
) -> Path:
    """Run the replay backend into ``out`` and return it."""
    scratch = scratch or out.parent / f".scratch-{out.name}"
    job = write_job(scratch / "jobs", name, ligand=ligand)
    backend = replay_backend(
        model, name=name, ligand=ligand, samples=samples, with_pae=with_pae
    )
    weights = scratch / "weights.jax"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"replayed")
    request = PredictionRequest(
        model=model,
        input=job,
        weights=weights,
        output_dir=out,
        seed=seed,
        use_compile_cache=False,
    )
    with backend_override(model, backend):
        foldjax.predict(request)
    return out
