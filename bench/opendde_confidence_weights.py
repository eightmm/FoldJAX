"""CPU-only exact confidence-checkpoint mapping evidence; no operator parity claim."""

import argparse
import collections
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import save, sha


class TrackedState(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = collections.Counter()

    def __getitem__(self, key):
        self.reads[key] += 1
        return super().__getitem__(key)


def compare_trees(mapped, managed):
    import jax

    left, ld = jax.tree_util.tree_flatten_with_path(mapped, is_leaf=lambda x: x is None)
    right, rd = jax.tree_util.tree_flatten_with_path(
        managed, is_leaf=lambda x: x is None
    )
    if ld != rd:
        raise ValueError("different checkpoint tree definitions")
    counts = collections.Counter()
    failures, groups = [], collections.defaultdict(collections.Counter)
    digests = [hashlib.sha256(), hashlib.sha256()]
    for (lp, a), (rp, b) in zip(left, right, strict=True):
        name = jax.tree_util.keystr(lp)
        if name != jax.tree_util.keystr(rp):
            raise ValueError("different checkpoint leaf paths")
        if a is None or b is None:
            counts["none_leaves"] += 1
            if a is not None or b is not None:
                failures.append(name)
            continue
        if not hasattr(a, "shape") or not hasattr(b, "shape"):
            counts["static_leaves"] += 1
            if a != b:
                failures.append(name)
            continue
        a, b = np.asarray(a), np.asarray(b)
        exact = a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        finite = np.isfinite(a).all() and np.isfinite(b).all()
        if not exact or not finite:
            failures.append(name)
        counts.update(array_leaves=1, elements=int(a.size), array_bytes=int(a.nbytes))
        group = name.split(".", 2)[1]
        groups[group].update(array_leaves=1, exact=int(exact), bytes=int(a.nbytes))
        for value, digest in zip((a, b), digests, strict=True):
            header = json.dumps(
                {"path": name, "dtype": value.dtype.str, "shape": value.shape},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            digest.update(value.tobytes())
    return {
        "passed": not failures and counts["array_leaves"] > 0,
        "tree_definition_exact": True,
        "counts": counts,
        "groups": groups,
        "failures": failures,
        "canonical_contents_sha256": [d.hexdigest() for d in digests],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--managed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import jax

    from foldjax.models.protenix.bridge.torch_mapping import (
        map_confidence_head_state_dict,
    )
    from foldjax.models.protenix.bridge.weights_io import _load_native_numpy_tree
    from foldjax.torch_archive import load

    if any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("run this read-only weight audit with JAX_PLATFORMS=cpu")
    paths = {"native": args.native, "managed": args.managed}
    before = {name: path.stat() for name, path in paths.items()}
    managed = _load_native_numpy_tree(args.managed, prestack=False).confidence
    checkpoint = load(args.native)
    prefix = "module.confidence_head"
    tracked = TrackedState(
        {
            key: value
            for key, value in checkpoint["model"].items()
            if key.startswith(prefix + ".")
        }
    )
    del checkpoint
    gc.collect()
    mapped = map_confidence_head_state_dict(tracked, prefix)
    result = compare_trees(mapped, managed)
    result.update(
        scope=__doc__,
        native_leaves=len(tracked),
        consumed_leaves=len(tracked.reads),
        unconsumed=sorted(tracked.keys() - tracked.reads.keys()),
        multiply_consumed={
            key: value for key, value in tracked.reads.items() if value != 1
        },
        checkpoint_sha256={name: sha(path) for name, path in paths.items()},
        checkpoints_unchanged=all(
            path.stat() == before[name] for name, path in paths.items()
        ),
        jax_version=jax.__version__,
    )
    root = Path(__file__).resolve().parents[1]
    result["source_sha256"] = {
        name: sha(root / name)
        for name in (
            "bench/opendde_confidence_weights.py",
            "src/foldjax/torch_archive.py",
            "src/foldjax/models/opendde/bridge/torch_mapping.py",
            "src/foldjax/models/protenix/bridge/torch_mapping.py",
            "src/foldjax/models/protenix/bridge/weights_io.py",
            "src/foldjax/models/protenix/models/heads/confidence.py",
        )
    }
    result["passed"] &= (
        not result["unconsumed"]
        and not result["multiply_consumed"]
        and result["checkpoints_unchanged"]
    )
    save(args.out, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "passed",
                    "counts",
                    "native_leaves",
                    "consumed_leaves",
                    "canonical_contents_sha256",
                )
            }
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
