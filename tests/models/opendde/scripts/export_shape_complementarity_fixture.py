"""Record upstream OpenDDE's shape-complementarity outputs on a two-chain case.

The port is 300 lines of geometry with no learned weights, which makes it both
easy to write and easy to write *plausibly wrong*: every intermediate has a
defensible shape and a defensible magnitude, so nothing fails loudly. The only
check worth having is upstream's own numbers on the same input.

Two chains, deliberately. Every pair in this computation is gated by
`asym_id[i] != asym_id[j]`, so a single-chain fixture scores identically zero
and would pass against almost any implementation -- including one that returns
zeros. The chains here are placed at a separation inside `interface_cutoff`
(16.0 A) so the interface is actually populated; the assertion that it is
non-trivial lives in the test, not here.

Run from the OpenDDE checkout, which owns the torch this needs:

    cd ../OpenDDE && PYTHONPATH=$PWD uv run --no-sync python \
      ../foldjax/tests/models/opendde/scripts/export_shape_complementarity_fixture.py \
      --out ../foldjax/tests/models/opendde/data/shape_complementarity.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def build_case(seed: int = 11) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Two small protein chains facing each other across a ~6 A gap.

    Four atoms per residue, the second of which is the representative -- enough
    that `atom_count > 1` holds and the per-token density gradient has more than
    one contributor, which is what makes the normal a direction rather than
    noise.
    """
    rng = np.random.default_rng(seed)
    residues_per_chain = 9
    atoms_per_residue = 4
    n_token = 2 * residues_per_chain
    n_atom = n_token * atoms_per_residue

    centers = []
    for chain in range(2):
        for index in range(residues_per_chain):
            # A slab per chain, the two separated along x by `gap_mean`.
            centers.append(
                [
                    0.0 if chain == 0 else 6.0,
                    3.4 * (index % 3),
                    3.4 * (index // 3),
                ]
            )
    centers = np.asarray(centers, dtype=np.float32)
    coordinate = np.repeat(centers, atoms_per_residue, axis=0)
    coordinate = coordinate + rng.normal(scale=0.45, size=coordinate.shape)
    coordinate = coordinate.astype(np.float32)

    atom_to_token_idx = np.repeat(np.arange(n_token), atoms_per_residue)
    rep_atom_mask = np.zeros(n_atom, dtype=bool)
    rep_atom_mask[np.arange(n_token) * atoms_per_residue + 1] = True

    features = {
        "token_index": np.arange(n_token, dtype=np.int64),
        "atom_to_token_idx": atom_to_token_idx.astype(np.int64),
        "asym_id": np.repeat([0, 1], residues_per_chain).astype(np.int64),
        "distogram_rep_atom_mask": rep_atom_mask,
        "is_protein": np.ones(n_atom, dtype=np.float32),
        "is_ligand": np.zeros(n_atom, dtype=np.float32),
        "is_dna": np.zeros(n_atom, dtype=np.float32),
        "is_rna": np.zeros(n_atom, dtype=np.float32),
    }
    atom_mask = np.ones(n_atom, dtype=bool)
    return coordinate, features, atom_mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    from opendde.config.model_base import configs as model_configs
    from opendde.model.shape_complementarity import (
        build_shape_comp_pred_outputs,
        compute_shape_complementarity_fields,
    )

    coordinate, features, atom_mask = build_case(args.seed)
    settings = dict(model_configs["confidence"]["shape_comp"])
    # `ValueMaybeNone(128)` and friends are config wrappers, not numbers.
    settings = {
        key: (value.value if hasattr(value, "value") else value)
        for key, value in settings.items()
    }
    settings.pop("debug_pair_map", None)
    settings.pop("checkpoint_chunks", None)

    shape_comp = compute_shape_complementarity_fields(
        coordinate=torch.from_numpy(coordinate),
        feat_dict={key: torch.from_numpy(value) for key, value in features.items()},
        atom_mask=torch.from_numpy(atom_mask),
        checkpoint_chunks=False,
        return_pair_map=True,
        **settings,
    )
    outputs = build_shape_comp_pred_outputs(shape_comp=shape_comp, keep_pair_map=False)

    payload: dict[str, np.ndarray] = {
        "coordinate": coordinate,
        "atom_mask": atom_mask,
    }
    payload.update({f"input_{k}": v for k, v in features.items()})
    payload.update(
        {k: np.asarray(v.detach().cpu().numpy()) for k, v in outputs.items()}
    )
    payload["settings_keys"] = np.asarray(sorted(settings), dtype=object)
    payload["settings_values"] = np.asarray(
        [float(settings[k]) for k in sorted(settings)], dtype=np.float64
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)

    interface = int(np.asarray(outputs["shape_comp_token_mask"]).sum())
    print(f"wrote {args.out}")
    print(f"  tokens with a cross-chain partner: {interface}")
    print(f"  global score: {float(outputs['shape_comp_global_pred']):.6f}")
    print(f"  pair mean:    {float(outputs['shape_comp_pair_mean_pred']):.6f}")
    print(f"  pair topk:    {float(outputs['shape_comp_pair_topk_mean_pred']):.6f}")
    if interface == 0:
        raise SystemExit(
            "the two chains never met: this fixture would pass against an "
            "implementation that returns zeros, so it is not worth storing"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
