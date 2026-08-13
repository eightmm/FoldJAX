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


def sample_scores(output: Mapping[str, object]) -> list[dict[str, float]]:
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
            entry["plddt"] = float(plddt.mean())
        scores.append(entry)
    return scores


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

    coords = _numpy(output["sample_atom_coords"])
    if coords.ndim == 2:
        coords = coords[None]
    per_atom = (
        _numpy(output["plddt_per_atom"]) if "plddt_per_atom" in output else None
    )

    structures: list[Path] = []
    for index in range(coords.shape[0]):
        path = directory / f"{name}_sample_{index}.pdb"
        path.write_text(
            pdb.to_pdb(
                coords[index],
                features,
                None if per_atom is None else per_atom[index],
                plddt_scale=plddt_scale,
            ),
            encoding="utf-8",
        )
        structures.append(path)

    scores = sample_scores(output)
    summary = {
        "model": "esmfold2",
        "plddt_scale": "0-1 here; the structures' b-factor column is 0-100",
        "samples": scores,
    }
    scores_path = directory / f"{name}_confidence.json"
    scores_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"structures": structures, "scores": scores_path, "summary": scores}


__all__ = ["SAMPLE_SCORES", "sample_scores", "write_prediction_outputs"]
