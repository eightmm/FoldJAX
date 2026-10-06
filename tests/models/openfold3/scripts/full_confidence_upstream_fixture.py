"""Regenerate ``fixtures/full_confidence_upstream.npz`` from upstream OpenFold3 v0.5.0.

Run with the upstream checkout's own interpreter (it needs torch and biotite,
which the FoldJAX environment does not have, or not at upstream's version)::

    CUDA_VISIBLE_DEVICES= ../openfold3-v050/.venv/bin/python \\
        tests/models/openfold3/scripts/full_confidence_upstream_fixture.py

Two records, both on CPU:

* ``conf_*``: upstream's aggregated and full confidence arithmetic
  (``core/metrics/aggregate_confidence_ranking.py`` and
  ``core/metrics/sample_ranking.py``) on random PAE/PDE/distogram logits for a
  three-chain complex whose third chain is a ligand: expected PAE and PDE,
  gPDE, chain pTM, chain-pair ipTM and bespoke ipTM.
* ``rasa_*``: ``core/metrics/rasa.py``'s ``process_disorder`` -- the disorder
  term of the sample ranking score -- on three coordinate sets of a real
  129-residue predicted chain, plus an 8-residue copy (shorter than the
  smoothing half-window) and a non-protein copy that must be filtered out.
"""

from __future__ import annotations

import gzip
import inspect
import io
from pathlib import Path

import biotite.structure as struc
import numpy as np
import torch
from biotite.structure.io import pdbx
from openfold3.core.data.resources.residues import RESIDUE_SASA_SCALES, MoleculeType
from openfold3.core.metrics import rasa
from openfold3.core.metrics.confidence import (
    compute_global_predicted_distance_error,
    compute_ptm,
    probs_to_expected_error,
)
from openfold3.core.metrics.sample_ranking import (
    compute_chain_pair_iptm,
    compute_chain_ptm,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "fixtures" / "full_confidence_upstream.npz"
STRUCTURE = ROOT / "fixtures" / "outputs" / "openfold3_e9_8reh" / "structure.cif.gz"

PAE = {"bin_min": 0, "bin_max": 32, "no_bins": 64}
DISTOGRAM = {"bin_min": 2, "bin_max": 22, "no_bins": 64}


def _confidence(record: dict[str, np.ndarray]) -> None:
    torch.manual_seed(20261006)
    samples, bins = 3, 64
    asym_id = torch.tensor([1] * 5 + [2] * 4 + [3] * 3)
    n_token = asym_id.numel()
    is_ligand = (asym_id == 3).long()
    token_mask = torch.ones(n_token)
    pae_logits = 2.0 * torch.randn(samples, n_token, n_token, bins)
    pde_logits = 2.0 * torch.randn(samples, n_token, n_token, bins)
    distogram_logits = 2.0 * torch.randn(n_token, n_token, bins)
    distogram_logits = 0.5 * (distogram_logits + distogram_logits.transpose(0, 1))
    has_frame = torch.rand(samples, n_token) > 0.25
    # Sample 1's second chain has no frame, so bespoke averaging differs per sample.
    has_frame[1, 5:9] = False

    pae = probs_to_expected_error(torch.softmax(pae_logits, dim=-1), **PAE)
    pde = probs_to_expected_error(torch.softmax(pde_logits, dim=-1), **PAE)
    gpde, contact = compute_global_predicted_distance_error(
        pde=pde, logits=distogram_logits, **DISTOGRAM
    )
    batch = {"token_mask": token_mask, "asym_id": asym_id, "is_ligand": is_ligand}
    chain_ptm = compute_chain_ptm(
        batch=batch, outputs={"pae_logits": pae_logits}, has_frame=has_frame, **PAE
    )["chain_ptm"]
    pairs = compute_chain_pair_iptm(
        batch=batch, logits=pae_logits, has_frame=has_frame, **PAE
    )
    chains = sorted(chain_ptm)
    dense = {
        name: np.zeros((samples, len(chains), len(chains)), dtype=np.float32)
        for name in ("chain_pair_iptm", "bespoke_iptm")
    }
    for name, matrix in dense.items():
        for key, value in pairs[name].items():
            i, j = (chains.index(int(part)) for part in key.strip("()").split(","))
            matrix[:, i, j] = matrix[:, j, i] = value.numpy()
    ptm = compute_ptm(pae_logits, has_frame, mask_i=token_mask, **PAE)
    iptm = compute_ptm(
        pae_logits, has_frame, mask_i=token_mask, asym_id=asym_id, interface=True, **PAE
    )
    record.update(
        conf_asym_id=asym_id.numpy() - 1,
        conf_is_ligand=is_ligand.numpy(),
        conf_has_frame=has_frame.numpy(),
        conf_pae_logits=pae_logits.numpy(),
        conf_pde_logits=pde_logits.numpy(),
        conf_distogram_logits=distogram_logits.numpy(),
        conf_pae=pae.numpy(),
        conf_pde=pde.numpy(),
        conf_gpde=gpde.numpy(),
        conf_contact_probs=contact.numpy(),
        conf_ptm=ptm.numpy(),
        conf_iptm=iptm.numpy(),
        conf_chain_ptm=np.stack([chain_ptm[c].numpy() for c in chains], axis=-1),
        conf_chain_pair_iptm=dense["chain_pair_iptm"],
        conf_bespoke_iptm=dense["bespoke_iptm"],
    )


def _structure():
    with gzip.open(STRUCTURE, "rt") as handle:
        cif = pdbx.CIFFile.read(io.StringIO(handle.read()))
    return pdbx.get_structure(cif, model=1)


def _rasa(record: dict[str, np.ndarray]) -> None:
    chain = _structure()
    short = chain[chain.res_id <= chain.res_id[0] + 7].copy()
    short.chain_id[:] = "B"
    short.coord = short.coord + np.array([150.0, 0.0, 0.0], dtype=np.float32)
    ligand = chain[chain.res_id == chain.res_id[0] + 10].copy()
    ligand.chain_id[:] = "C"
    ligand.res_name[:] = "LIG"
    ligand.coord = ligand.coord + np.array([0.0, 150.0, 0.0], dtype=np.float32)
    array = chain + short + ligand
    molecule_type = np.full(array.array_length(), MoleculeType.PROTEIN, dtype=int)
    molecule_type[array.chain_id == "C"] = MoleculeType.LIGAND
    array.set_annotation("molecule_type_id", molecule_type)

    centre = array.coord.mean(axis=0)
    coords = np.stack(
        [
            array.coord,
            centre + 1.6 * (array.coord - centre),
            centre + 0.85 * (array.coord - centre),
        ]
    ).astype(np.float32)
    # Upstream's compute_disorder, sample by sample, on a CPU tensor.
    disorder = rasa.compute_disorder(
        batch={"atom_array": array},
        outputs={"atom_positions_predicted": torch.from_numpy(coords)},
    ).numpy()
    # Per-residue smoothed RASA of the first sample, chains pooled, which is
    # what the disorder fraction thresholds. Called with the arguments
    # process_disorder passes: calculate_res_rasa's own max_acc_dict default
    # is the dict of scales, not one scale.
    # compute_disorder leaves the last sample's coordinates on the array.
    array.coord = coords[0]
    protein = array[array.molecule_type_id == MoleculeType.PROTEIN]
    residue_rasa = np.concatenate(
        [
            rasa.calculate_res_rasa(
                chain,
                window=25,
                max_acc_dict=RESIDUE_SASA_SCALES["Sander"],
                default_max_acc=113.0,
            )[0]
            for chain in struc.chain_iter(protein)
        ]
    )
    defaults = inspect.signature(rasa.compute_disorder).parameters
    threshold_default = (
        inspect.signature(rasa.process_disorder)
        .parameters["disorder_threshold"]
        .default
    )
    record.update(
        rasa_coords=coords,
        rasa_atom_name=array.atom_name.astype(str),
        rasa_element=array.element.astype(str),
        rasa_res_name=array.res_name.astype(str),
        rasa_res_id=array.res_id.astype(np.int64),
        rasa_chain_id=array.chain_id.astype(str),
        rasa_is_protein=molecule_type == MoleculeType.PROTEIN,
        rasa_disorder=disorder,
        rasa_residue_rasa_0=residue_rasa,
        rasa_window=np.asarray(defaults["window"].default),
        rasa_default_max_acc=np.asarray(defaults["default_max_acc"].default),
        rasa_threshold=np.asarray(defaults["disorder_threshold"].default),
        rasa_process_threshold=np.asarray(threshold_default),
        rasa_vdw_radii=np.asarray(defaults["vdw_radii"].default),
        rasa_sander_names=np.asarray(sorted(RESIDUE_SASA_SCALES["Sander"])),
        rasa_sander_values=np.asarray(
            [
                RESIDUE_SASA_SCALES["Sander"][name]
                for name in sorted(RESIDUE_SASA_SCALES["Sander"])
            ]
        ),
        rasa_smoothed_probe=rasa._smooth_rasa(np.linspace(0.0, 1.0, 8), 25),
    )


def main() -> None:
    torch.set_grad_enabled(False)
    record: dict[str, np.ndarray] = {}
    _confidence(record)
    _rasa(record)
    np.savez_compressed(OUT, **record)
    print(f"wrote {OUT}: rasa_disorder={record['rasa_disorder']}")


if __name__ == "__main__":
    main()
