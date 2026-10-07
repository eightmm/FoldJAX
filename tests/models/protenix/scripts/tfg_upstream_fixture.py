"""Regenerate ``fixtures/tfg_upstream.npz`` from upstream Protenix 2.0.0.

Run with the upstream checkout's own interpreter (it needs torch, which the
FoldJAX environment does not have), from the FoldJAX checkout root::

    CUDA_VISIBLE_DEVICES= PROTENIX_ROOT_DIR=../protenix \\
        ../protenix/.venv/bin/python \\
        tests/models/protenix/scripts/tfg_upstream_fixture.py

Two records, both on CPU:

* ``engine_*``: one ``TFGEngine.step`` of upstream's default guidance config
  (``configs_base.py`` ``sample_diffusion.guidance`` with ``enable`` set) on a
  small stereo ligand, its geometry features from upstream's own extractors,
  and a fixed linear stand-in for the denoiser.
* ``geometry_<case>_*``: what ``SampleDictToFeatures(...,
  extract_features_for_tfg=True)`` -- the ``GeometryFeaturizer(...,
  exclude_std_residue=True)`` call of a guided run -- returns for each job in
  ``geometry_jobs``, plus the atom names and ``hetero`` flags it indexed.

Upstream ran with RDKit 2025.09.3; a later RDKit that changes only a
torsion-pattern count shows up here as drift, not as a port bug.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROTENIX = Path(os.environ.get("PROTENIX_ROOT_DIR", "../protenix")).resolve()
os.environ["PROTENIX_ROOT_DIR"] = str(PROTENIX)
sys.path.insert(0, str(PROTENIX))

from configs.configs_base import model_configs  # noqa: E402
from protenix.data.core import geometry_featurizer as gf  # noqa: E402
from protenix.data.inference.json_parser import add_entity_atom_array  # noqa: E402
from protenix.data.inference.json_to_feature import SampleDictToFeatures  # noqa: E402
from protenix.tfg.config import parse_tfg_config  # noqa: E402
from protenix.tfg.engine import TFGEngine  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "tfg_upstream.npz"

ENGINE_SMILES = "C[C@H](N)C(=O)O/C=C/C#N"

GEOMETRY_JOBS = {
    # Pickle-free: standard residues and two SMILES ligands, one with the
    # secondary amide whose torsion pattern needs explicit hydrogens.
    "smiles": {
        "name": "smiles",
        "modelSeeds": [101],
        "sequences": [
            {"proteinChain": {"sequence": "ACDEFGHIK", "count": 1}},
            {"ligand": {"ligand": ENGINE_SMILES, "count": 1}},
            {"ligand": {"ligand": "CC(=O)NC1CCCCC1", "count": 1}},
        ],
    },
    # Needs the CCD RDKit cache: a CCD ligand, a metal ion, and a modified
    # residue inside a protein chain.
    "ccd": {
        "name": "ccd",
        "modelSeeds": [101],
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "ACDSFGHIK",
                    "count": 1,
                    "modifications": [{"ptmType": "CCD_SEP", "ptmPosition": 4}],
                }
            },
            {"ligand": {"ligand": "CCD_ATP", "count": 1}},
            {"ion": {"ion": "MG", "count": 1}},
        ],
    },
    "protein": {
        "name": "protein",
        "modelSeeds": [101],
        "sequences": [{"proteinChain": {"sequence": "ACDEFGHIK", "count": 2}}],
    },
}


def _plain(value):
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    return getattr(value, "v", value)


def _engine_record() -> dict[str, np.ndarray]:
    mol = Chem.AddHs(Chem.MolFromSmiles(ENGINE_SMILES))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    mol = Chem.RemoveHs(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    n_atom = mol.GetNumAtoms()
    work = copy.deepcopy(mol)
    work.UpdatePropertyCache(strict=False)
    Chem.AssignStereochemistry(work, force=True, cleanIt=True)
    Chem.GetSymmSSSR(work)
    features = {
        **gf.extract_pairwise_distance_bounds_from_mol(work),
        **gf.extract_experimental_torsion_from_mol(work),
        **gf.extract_chiral_dihedral_from_mol(work),
        **gf.extract_linear_triple_bond_from_mol(work),
        **gf.extract_stereo_bond_from_mol(work),
        **gf.extract_planar_improper_from_mol(work),
        "interchain_bond_index": [],
    }
    features = gf.GeometryFeaturizer.dict_to_tensor(features)
    elements = torch.zeros((n_atom, 128))
    for atom in mol.GetAtoms():
        elements[atom.GetIdx(), atom.GetAtomicNum() - 1] = 1.0
    features["ref_element"] = elements
    features["atom_to_token_idx"] = torch.arange(n_atom)
    # Two chains, so the inter-chain steric term is live.
    features["asym_id"] = (torch.arange(n_atom) >= n_atom // 2).long()

    ref = mol.GetConformer().GetPositions().astype(np.float32)
    rng = np.random.default_rng(0)
    x = (ref[None] + 0.4 * rng.standard_normal((2, n_atom, 3))).astype(np.float32)

    cfg = _plain(copy.deepcopy(model_configs["sample_diffusion"]["guidance"]))
    cfg["enable"] = True
    engine = TFGEngine(
        parse_tfg_config(cfg), device=torch.device("cpu"), dtype=torch.float32
    )

    def denoise(x_noisy, t_hat_noise_level, **_):
        del t_hat_noise_level
        return 0.8 * x_noisy + 0.1

    torch.manual_seed(0)
    step = engine.step(
        denoise,
        x=torch.as_tensor(x),
        t_hat=torch.full((2,), 2.5),
        c_tau=torch.full((2,), 2.0),
        step_scale_eta=1.5,
        step_i=150,
        num_diffusion_steps=200,
        input_feature_dict=features,
        s_inputs=None,
        s_trunk=None,
        z_trunk=None,
        pair_z=None,
        p_lm=None,
        c_l=None,
        chunk_size=None,
        inplace_safe=False,
        enable_efficient_fusion=False,
    )
    record = {f"engine_feat_{key}": value.numpy() for key, value in features.items()}
    record["engine_ref"] = ref
    record["engine_x"] = x
    record["engine_step"] = step.numpy()
    return record


def _geometry_record(case: str, job: dict) -> dict[str, np.ndarray]:
    sample = add_entity_atom_array(job)
    features, atom_array, _ = SampleDictToFeatures(
        sample, extract_features_for_tfg=True
    ).get_feature_dict()
    record = {
        f"geometry_{case}_{key}": features[key].numpy()
        for key in gf.RDKIT_GEOMETRY_FEATURES + ["interchain_bond_index"]
    }
    record[f"geometry_{case}_atom_name"] = np.asarray(atom_array.atom_name, dtype=str)
    record[f"geometry_{case}_res_name"] = np.asarray(atom_array.res_name, dtype=str)
    record[f"geometry_{case}_hetero"] = np.asarray(atom_array.hetero, dtype=bool)
    return record


def main() -> None:
    record = _engine_record()
    for case, job in GEOMETRY_JOBS.items():
        record.update(_geometry_record(case, job))
    record["geometry_jobs"] = np.asarray(json.dumps(GEOMETRY_JOBS, sort_keys=True))
    np.savez_compressed(OUT, **record)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
