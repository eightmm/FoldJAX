"""Entity RMSD between two Protenix runs of one input (native or FoldJAX).

Each run directory holds ``prediction.npz`` with ``coordinate`` of shape
(samples, atoms, 3). Atom identity comes from the native capture's
``native-identity.npz`` (chain id, residue id, residue name, atom name,
element); a FoldJAX replay directory without that file borrows the identity
of the native reference it replayed (``--identity``). Samples are paired
positionally; one whole-system Kabsch per sample; entity maxima by chain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.boltz_historical_replay import save_new
from bench.entity_parity import compare_entity_parity

IDENTITY = (
    "output_atom_chain_id", "output_atom_res_id", "output_atom_res_name",
    "output_atom_atom_name", "output_atom_element",
)


def identity_keys(root):
    with np.load(root / "native-identity.npz", allow_pickle=False) as data:
        columns = [
            data[name].tolist()
            for name in IDENTITY
            if name in data
        ]
        if len(columns) != len(IDENTITY):
            columns = [
                data["output_atom_chain_id"].tolist(),
                data["output_atom_res_id"].tolist(),
                data["output_atom_res_name"].tolist(),
                data["output_atom_name"].tolist(),
                data["output_atom_element"].tolist(),
            ]
    return list(zip(*columns, strict=True)), columns[0]


def coordinates(root):
    with np.load(root / "prediction.npz", allow_pickle=False) as data:
        return np.asarray(data["coordinate"], dtype=np.float64)


def compare(left, right, identity):
    keys, chains = identity_keys(identity)
    a, b = coordinates(left), coordinates(right)
    if a.shape != b.shape or a.shape[1] != len(keys):
        raise ValueError(f"coordinate shapes differ: {a.shape} vs {b.shape}")
    mask = np.ones(a.shape[:2], dtype=bool)
    report = compare_entity_parity(a, b, keys, keys, chains, chains, mask, mask)
    return {
        "scope": (
            "two Protenix runs of one input; samples paired positionally; one "
            "whole-system Kabsch per sample; entity maxima by chain"
        ),
        "left": str(left), "right": str(right),
        "coordinates_bitwise_equal": bool(np.array_equal(a, b)),
        "coordinates": report,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    identity = args.identity or args.left
    result = compare(args.left, args.right, identity)
    save_new(args.out, result)
    print(json.dumps({
        "entity_max_rmsd": result["coordinates"]["entity_max_rmsd"],
        "entity_rmsd": result["coordinates"]["entity_rmsd"],
        "bitwise": result["coordinates_bitwise_equal"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
