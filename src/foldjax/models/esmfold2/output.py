"""Writing ESMFold2's samples and confidences to disk.

One file per diffusion sample plus one confidence JSON, which is the shape the
other FoldJAX backends produce and what `foldjax.output.normalize` expects to
tidy afterwards.

The confidence names are upstream's, with one clarification carried into the
file: `plddt` is on the model's own 0-1 scale here, while the b-factor column
of the structures is on the 0-100 scale viewers assume. Reporting one number
under one name on two scales in two places is how a confidence gets misread,
so the JSON says which it is.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from foldjax.models.esmfold2.data import pdb

#: Scalars the confidence head returns once per sample.
SAMPLE_SCORES = ("complex_plddt", "complex_iplddt", "ptm", "iptm")


def _numpy(value: object) -> np.ndarray:
    return np.asarray(value)


def sample_scores(
    output: Mapping[str, object], *, token_mask: object | None = None
) -> list[dict[str, float]]:
    """Per-sample confidences, in the order the sampler produced them."""
    lengths = [
        _numpy(output[name]).shape[0] for name in SAMPLE_SCORES if name in output
    ]
    n_samples = lengths[0] if lengths else 1
    scores: list[dict[str, float]] = []
    for index in range(n_samples):
        entry: dict[str, float] = {"sample": index}
        for name in SAMPLE_SCORES:
            if name in output:
                entry[name] = float(_numpy(output[name])[index])
        if "plddt" in output:
            # The per-token mean, masked to real tokens, which is the number
            # people quote as "the pLDDT".
            plddt = _numpy(output["plddt"])[index]
            if token_mask is not None:
                mask = _numpy(token_mask).astype(bool).reshape(-1)
                if plddt.shape[-1] != mask.size:
                    raise ValueError(
                        "ESMFold2 token mask does not match pLDDT width: "
                        f"{mask.size} versus {plddt.shape[-1]}"
                    )
                plddt = plddt[mask]
            entry["plddt"] = float(plddt.mean()) if plddt.size else 0.0
        scores.append(entry)
    return scores


def crop_prediction(
    output: Mapping[str, object], features: Mapping[str, np.ndarray]
) -> dict[str, object]:
    """Remove masked token/atom suffixes from every public prediction array."""

    token_mask = np.asarray(features["token_attention_mask"]).reshape(-1).astype(bool)
    atom_mask = np.asarray(features["atom_attention_mask"]).reshape(-1).astype(bool)
    n_token = int(token_mask.sum())
    n_atom = int(atom_mask.sum())
    if not np.array_equal(token_mask, np.arange(token_mask.size) < n_token):
        raise ValueError("ESMFold2 token padding must be a suffix")
    if not np.array_equal(atom_mask, np.arange(atom_mask.size) < n_atom):
        raise ValueError("ESMFold2 atom padding must be a suffix")

    token_arrays = {"plddt", "plddt_ca", "residue_index", "entity_id"}
    pair_logit_arrays = {
        "distogram_logits",
        "pae_logits",
        "pde_logits",
    }
    pair_scalar_arrays = {"pae", "pde"}
    atom_arrays = {"atom_pad_mask", "plddt_per_atom"}
    atom_channel_arrays = {
        "sample_atom_coords",
        "plddt_logits",
        "resolved_logits",
    }

    cropped: dict[str, object] = {}
    for name, value in output.items():
        array = value
        if name in token_arrays:
            array = value[..., :n_token]
        elif name in pair_logit_arrays:
            array = value[..., :n_token, :n_token, :]
        elif name in pair_scalar_arrays:
            array = value[..., :n_token, :n_token]
        elif name in atom_arrays:
            array = value[..., :n_atom]
        elif name in atom_channel_arrays:
            array = value[..., :n_atom, :]
        cropped[name] = array
    return cropped


def write_prediction_outputs(
    output: Mapping[str, object],
    features: Mapping[str, np.ndarray],
    output_dir: str | Path,
    *,
    name: str,
    plddt_scale: float = 100.0,
) -> dict[str, object]:
    """Write one PDB per sample and one confidence JSON beside them."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    cropped = crop_prediction(output, features)
    coords = _numpy(cropped["sample_atom_coords"])
    if coords.ndim == 2:
        coords = coords[None]
    per_atom = (
        _numpy(cropped["plddt_per_atom"])
        if "plddt_per_atom" in cropped
        else None
    )

    structures: list[Path] = []
    for index in range(coords.shape[0]):
        # mmCIF, because `foldjax.output.normalize` names every structure
        # `.cif` and edits its data block. A PDB file under that name would be
        # a file whose extension lies about its contents.
        path = directory / f"{name}_sample_{index}.cif"
        path.write_text(
            pdb.to_mmcif(
                coords[index],
                features,
                None if per_atom is None else per_atom[index],
                plddt_scale=plddt_scale,
                name=f"{name}_sample_{index}",
            ),
            encoding="utf-8",
        )
        structures.append(path)

    scores = sample_scores(cropped)
    summary = {
        "model": "esmfold2",
        "plddt_scale": "0-1 here; the structures' b-factor column is 0-100",
        "samples": scores,
    }
    scores_path = directory / f"{name}_confidence.json"
    scores_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"structures": structures, "scores": scores_path, "summary": scores}


__all__ = [
    "SAMPLE_SCORES",
    "crop_prediction",
    "sample_scores",
    "write_prediction_outputs",
]
