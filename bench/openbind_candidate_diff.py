"""Compare two FoldJAX OpenBind candidate runs of the same native capture.

Answers one question the native-versus-candidate report cannot: how far two
separately compiled processes of the same source move on the same fixed tape.
That is the candidate's own cross-process floor, to be read before any
native-relative residual is called a port defect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.boltz_historical_replay import digest, save_new
from bench.entity_parity import compare_entity_parity


def _atom_keys(capture):
    with np.load(capture / "input.npz", allow_pickle=False) as features:
        columns = [
            features["atom_array.0.annotation." + name].tolist()
            for name in ("chain_id", "res_id", "res_name", "atom_name", "element")
        ]
        atom_mask = features["atom_mask"]
    keys = list(zip(*columns, strict=True))
    return keys, columns[0], atom_mask


def compare(capture, left, right):
    capture, left, right = Path(capture), Path(left), Path(right)
    identities = {}
    for name, root in (("left", left), ("right", right)):
        preflight = json.loads((root / "preflight.json").read_text())
        finished = json.loads((root / "finished.json").read_text())
        if digest(root / "prediction.npz") != finished["prediction_sha256"]:
            raise ValueError(f"{name} prediction identity mismatch")
        identities[name] = {
            "input_sha256": preflight["input_sha256"],
            "tape_sha256": preflight["tape_sha256"],
            "checkpoint_sha256": preflight["checkpoint_sha256"],
            "candidate_backend": preflight["candidate_backend"],
            "source_sha256": digest_source(preflight["source"]),
            "prediction_sha256": finished["prediction_sha256"],
        }
    for key in ("input_sha256", "tape_sha256", "checkpoint_sha256"):
        if identities["left"][key] != identities["right"][key]:
            raise ValueError(f"candidates replay different captures: {key}")
    keys, labels, atom_mask = _atom_keys(capture)
    with (
        np.load(left / "prediction.npz", allow_pickle=False) as a,
        np.load(right / "prediction.npz", allow_pickle=False) as b,
    ):
        if set(a) != set(b):
            raise ValueError("candidate archives carry different fields")
        arrays = {}
        for name in sorted(a):
            x, y = a[name], b[name]
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError(f"field schema differs: {name}")
            delta = x.astype(np.float64) - y.astype(np.float64)
            arrays[name] = {
                "bitwise_equal": bool(np.array_equal(x, y)),
                "max_absolute_difference": float(np.abs(delta).max()),
            }
        samples = a["coordinates"].shape[0]
        mask = np.broadcast_to(atom_mask.astype(bool), (samples, len(keys)))
        geometry = compare_entity_parity(
            a["coordinates"], b["coordinates"], keys, keys, labels, labels, mask, mask
        )
    return {
        "scope": (
            "same native capture replayed by two candidate processes; "
            "one system Kabsch fit per sample; entity maxima; no native reference"
        ),
        "identities": identities,
        "same_source": (
            identities["left"]["source_sha256"] == identities["right"]["source_sha256"]
        ),
        "same_backend": (
            identities["left"]["candidate_backend"]
            == identities["right"]["candidate_backend"]
        ),
        "fields": arrays,
        "coordinates": geometry,
    }


def digest_source(source):
    import hashlib

    return hashlib.sha256(
        json.dumps(source, sort_keys=True).encode()
    ).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    save_new(args.out, compare(args.capture, args.left, args.right))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
