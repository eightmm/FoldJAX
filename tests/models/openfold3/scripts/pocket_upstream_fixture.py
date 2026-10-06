"""Regenerate ``fixtures/pocket_upstream.npz`` from upstream OpenFold3 v0.5.0.

Run with the upstream checkout's own interpreter (it needs torch, which the
FoldJAX environment does not have)::

    CUDA_VISIBLE_DEVICES= ../openfold3-v050/.venv/bin/python \\
        tests/models/openfold3/scripts/pocket_upstream_fixture.py

Two records, both on CPU:

* ``seed_<case>_*``: ``_build_pocket_sampling_seeds`` on synthetic parents,
  with every random draw it made recorded in FoldJAX's
  ``PocketProposalDraws`` layout, so the JAX builder can replay them.
* ``feature_<query>_*``: ``create_pocket_sampling_features`` without RDKit
  conformers, whose ensemble depends on the RDKit build rather than the code.
  ``feature_queries`` is the JSON of those queries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from openfold3.core.config.pocket_sampling_config import PocketSamplingSettings
from openfold3.core.data.pipelines.featurization.pocket_constraints import (
    create_pocket_sampling_features,
)
from openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mols_from_query,
)
from openfold3.core.model.structure import pocket_constraints
from openfold3.projects.of3_all_atom.config.inference_query_format import Query

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "pocket_upstream.npz"

#: name -> (conformers, num_parents, candidates, diversity_rmsd, samples)
SEED_CASES = {
    "conformers": (4, 3, 64, 0.5, 5),
    "parent_conformation": (0, 16, 128, 0.5, 5),
    "fill_duplicates": (3, 2, 32, 1e4, 5),
}

FEATURE_QUERIES = {
    "smiles": {
        "chains": [
            {"molecule_type": "protein", "chain_ids": "A", "sequence": "PVLSCGEWQCL"},
            {
                "molecule_type": "ligand",
                "chain_ids": "L",
                "smiles": "CC(=O)OC1C[NH+]2CCC1CC2",
            },
        ],
        "pocket_constraint": {
            "ligand_chain_id": "L",
            "pocket_residues": [["A", 2], ["A", 5], ["A", 9]],
        },
    },
    "ccd_order": {
        "chains": [
            {"molecule_type": "ligand", "chain_ids": "X", "smiles": "N#N"},
            {
                "molecule_type": "protein",
                "chain_ids": ["A", "B"],
                "sequence": "ACDEFGHIKLMN",
            },
            {"molecule_type": "ligand", "chain_ids": "L", "ccd_codes": ["ATP"]},
        ],
        "pocket_constraint": {
            "ligand_chain_id": "L",
            "pocket_residues": [["B", 3], ["A", 7]],
            "max_distance": 6.0,
        },
    },
}

_recorded: list[tuple[str, np.ndarray]] = []


def _recording(name, function):
    def wrapped(*args, **kwargs):
        value = function(*args, **kwargs)
        _recorded.append((name, value.detach().cpu().numpy().copy()))
        return value

    return wrapped


def _seed_case(name, n_conf, num_parents, candidates, diversity, samples, payload):
    rng = np.random.default_rng(len(name))
    n_protein, n_lig = 60, 9
    n_atoms = n_protein + n_lig
    protein = rng.normal(scale=6.0, size=(n_protein, 3))
    lig_mask = np.zeros(n_atoms, bool)
    lig_mask[n_protein:] = True
    pocket_mask = np.zeros(n_atoms, bool)
    pocket_mask[[3, 4, 5, 17, 18, 40, 41, 42]] = True
    xl = np.zeros((samples, n_atoms, 3), np.float32)
    for s in range(samples):
        xl[s, :n_protein] = protein + rng.normal(scale=0.3, size=protein.shape)
        centre = rng.normal(scale=8.0, size=3)
        xl[s, n_protein:] = rng.normal(loc=centre, size=(n_lig, 3))
    vdw = rng.choice([1.52, 1.55, 1.7, 1.8], size=n_atoms).astype(np.float32)
    batch = {
        "pocket_sampling_ligand_atom_mask": torch.tensor(
            lig_mask[None], dtype=torch.float32
        ),
        "pocket_sampling_pocket_atom_mask": torch.tensor(
            pocket_mask[None], dtype=torch.float32
        ),
        "pocket_sampling_vdw_radii": torch.tensor(vdw[None]),
        "pocket_sampling_contact_distance": torch.tensor([4.0]),
        "pocket_sampling_num_parents": torch.tensor([num_parents]),
        "pocket_sampling_candidates": torch.tensor([candidates]),
        "pocket_sampling_center_jitter": torch.tensor([4.0]),
        "pocket_sampling_surface_jitter": torch.tensor([1.5]),
        "pocket_sampling_vdw_buffer": torch.tensor([0.225]),
        "pocket_sampling_diversity_rmsd": torch.tensor([diversity]),
    }
    conformers = np.zeros((0, n_lig, 3), np.float32)
    if n_conf:
        conformers = rng.normal(scale=2.0, size=(n_conf, n_lig, 3)).astype(np.float32)
        conformers -= conformers.mean(axis=1, keepdims=True)
        batch["pocket_sampling_conformer_rels"] = torch.tensor(conformers[None])

    torch.manual_seed(1234)
    _recorded.clear()
    seeds = pocket_constraints._build_pocket_sampling_seeds(
        batch=batch,
        xl_base=torch.tensor(xl[None]),
        atom_mask=torch.ones(1, n_atoms),
        no_rollout_samples=samples,
    )[0]

    n_parents = max(1, min(samples, num_parents))
    n_candidates = max(samples, candidates)
    draws = {
        "conformer_index": np.zeros(n_candidates, np.int32),
        "quaternion": np.zeros((n_candidates, 4), np.float32),
        "coin": np.zeros(n_candidates, np.float32),
        "center_noise": np.zeros((n_candidates, 3), np.float32),
        "surface_index": np.zeros(n_candidates, np.int32),
        "surface_noise": np.zeros((n_candidates, 3), np.float32),
    }
    stream = iter(_recorded)

    def take(kind, shape):
        got, value = next(stream)
        assert got == kind and value.shape == shape, (got, value.shape, kind, shape)
        return value

    # Upstream's draw order per proposal (pocket_constraints.py:184-209).
    for i in range(n_parents, n_candidates):
        if n_conf:
            draws["conformer_index"][i] = int(take("randint", ()))
        draws["quaternion"][i] = take("randn", (4,))
        coin = float(take("rand", ()))
        draws["coin"][i] = coin
        if coin < 0.5:
            draws["center_noise"][i] = take("randn", (1, 3))[0]
        else:
            draws["surface_index"][i] = int(take("randint", ()))
            draws["surface_noise"][i] = take("randn", (1, 3))[0]
    assert next(stream, None) is None

    prefix = f"seed_{name}_"
    payload.update(
        {
            f"{prefix}xl": xl,
            f"{prefix}ligand_atom_mask": lig_mask.astype(np.float32),
            f"{prefix}pocket_atom_mask": pocket_mask.astype(np.float32),
            f"{prefix}vdw_radii": vdw,
            f"{prefix}conformer_rels": conformers,
            f"{prefix}settings": np.asarray(
                [n_conf, num_parents, candidates, diversity, samples], np.float64
            ),
            f"{prefix}seeds": seeds.numpy(),
            **{f"{prefix}draw_{key}": value for key, value in draws.items()},
        }
    )


def main() -> None:
    torch.randn = _recording("randn", torch.randn)
    torch.rand = _recording("rand", torch.rand)
    torch.randint = _recording("randint", torch.randint)
    payload: dict[str, np.ndarray] = {}
    for name, case in SEED_CASES.items():
        _seed_case(name, *case, payload)
    for name, spec in FEATURE_QUERIES.items():
        query = Query.model_validate(spec)
        structure = structure_with_ref_mols_from_query(query)
        features = create_pocket_sampling_features(
            query=query,
            atom_array=structure.atom_array,
            processed_reference_molecules=structure.processed_reference_mols,
            settings=PocketSamplingSettings(rdkit_num_conformers=0),
        )
        for key, value in features.items():
            payload[f"feature_{name}_{key}"] = value.numpy()
    payload["feature_queries"] = np.asarray(json.dumps(FEATURE_QUERIES))
    np.savez_compressed(OUT, **payload)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(payload)} arrays)")


if __name__ == "__main__":
    main()
