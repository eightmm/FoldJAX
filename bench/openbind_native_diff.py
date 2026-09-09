"""Compare two native OpenBind captures of the same input (e.g. Triton vs cuEq).

Both captures must share the forward batch (same ``input.npz`` digest) so the
atom order is trusted; the RNG tapes may differ and are reported. This
measures how far upstream moves against itself when only its kernel choice or
its process changes, which is the context any port-versus-native residual has
to be read in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.boltz_historical_replay import digest, save_new
from bench.entity_parity import compare_entity_parity


def compare(left, right):
    left, right = Path(left), Path(right)
    identity = {}
    for name, root in (("left", left), ("right", right)):
        trace = json.loads((root / "trace.json").read_text())
        calls = json.loads((root / "kernel-calls.json").read_text())["calls"]
        identity[name] = {
            "input_sha256": digest(root / "input.npz"),
            "tape_sha256": digest(root / "tape.npz"),
            "requested_triangle_backend": trace.get("requested_triangle_backend"),
            "kernel_calls": calls,
        }
    # Archive bytes differ between captures (zip metadata); the batch itself
    # must be equal array by array.
    with (
        np.load(left / "input.npz", allow_pickle=False) as a,
        np.load(right / "input.npz", allow_pickle=False) as b,
    ):
        if set(a) != set(b) or any(not np.array_equal(a[k], b[k]) for k in a):
            raise ValueError("captures do not share the forward batch")
    with np.load(left / "input.npz", allow_pickle=False) as features:
        columns = [
            features["atom_array.0.annotation." + name].tolist()
            for name in ("chain_id", "res_id", "res_name", "atom_name", "element")
        ]
        atom_mask = features["atom_mask"]
    keys = list(zip(*columns, strict=True))
    with (
        np.load(left / "coordinate.npz", allow_pickle=False) as a,
        np.load(right / "coordinate.npz", allow_pickle=False) as b,
    ):
        ca, cb = a["coordinate"], b["coordinate"]
    mask = np.broadcast_to(atom_mask.astype(bool), (ca.shape[0], len(keys)))
    geometry = compare_entity_parity(
        ca, cb, keys, keys, columns[0], columns[0], mask, mask
    )
    return {
        "scope": (
            "two native captures of one forward batch; one system Kabsch per "
            "sample; entity maxima; tapes compared by digest only"
        ),
        "identities": identity,
        "same_tape": (
            identity["left"]["tape_sha256"] == identity["right"]["tape_sha256"]
        ),
        "coordinates_bitwise_equal": bool(np.array_equal(ca, cb)),
        "coordinates": geometry,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    save_new(args.out, compare(args.left, args.right))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
