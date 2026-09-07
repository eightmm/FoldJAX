"""Two-runtime, CPU-only proof of native-to-managed Protenix weight mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stable_file_identity(path):
    """Reading a checkpoint can update atime without changing its contents."""
    stat = Path(path).stat()
    return tuple(
        getattr(stat, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


def normalize_state_keys(state):
    normalized, origins = {}, {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise TypeError("native parameter keys must be strings")
        target = key.removeprefix("module.")
        if target in normalized:
            raise ValueError(f"module-prefix normalization collision: {target}")
        normalized[target] = value
        origins[target] = key
    return normalized, origins


def export_native(checkpoint: Path, out: Path):
    """Executed only in the trusted native Torch environment, without JAX."""
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("native export requires CUDA_VISIBLE_DEVICES='' explicitly")
    out.mkdir(parents=True, exist_ok=False)
    before = stable_file_identity(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("native checkpoint must contain a state dictionary")
    root_key = next((key for key in ("model", "state_dict") if key in payload), None)
    state = payload if root_key is None else payload[root_key]
    if not isinstance(state, dict):
        raise TypeError("selected model state is not a dictionary")
    normalized, origins = normalize_state_keys(state)
    arrays, signatures = {}, {}
    for key, value in normalized.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"non-tensor model state entry: {key}: {type(value).__name__}"
            )
        array = value.detach().cpu().numpy()
        if array.dtype.hasobject:
            raise TypeError(f"object parameter dtype is not supported: {key}")
        arrays[key] = array
        signatures[key] = {
            "native_key": origins[key],
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    np.savez(out / "state.npz", **arrays)
    metadata = {
        "format_version": 1,
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": sha(checkpoint),
        "checkpoint_unchanged": before == stable_file_identity(checkpoint),
        "selected_state_key": root_key,
        "nonmodel_metadata": {
            key: type(value).__name__
            for key, value in payload.items()
            if root_key is not None and key != root_key
        },
        "state_metadata_attribute": getattr(state, "_metadata", None),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "export_script_sha256": sha(__file__),
        "state_archive_sha256": sha(out / "state.npz"),
        "state_signatures": signatures,
        "native_array_count": len(arrays),
        "native_elements": sum(array.size for array in arrays.values()),
        "native_bytes": sum(array.nbytes for array in arrays.values()),
    }
    if not metadata["checkpoint_unchanged"]:
        raise ValueError("native checkpoint changed during export")
    save(out / "export.json", metadata)
    return {
        key: metadata[key]
        for key in (
            "checkpoint_sha256",
            "native_array_count",
            "native_elements",
            "native_bytes",
        )
    }


class TrackedState(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = Counter()

    def __getitem__(self, key):
        self.reads[key] += 1
        return super().__getitem__(key)


def map_with_source_keys(state, mapper=None):
    """Track the actual mapper read and unchanged asarray result, not hashes."""
    import jax.numpy as jnp

    from foldjax.models.protenix.bridge import torch_mapping

    original = torch_mapping.require_key
    tracked = TrackedState(state)
    arrays_by_id = {}

    def read(values, key):
        original_array = original(values, key)
        array = jnp.asarray(original_array)
        converted = np.asarray(array)
        if (
            original_array.dtype != converted.dtype
            or original_array.shape != converted.shape
            or original_array.tobytes() != converted.tobytes()
        ):
            raise ValueError(f"mapper conversion changed native parameter bytes: {key}")
        # Hold the array to prevent id reuse and trace it through the mapper's
        # jnp.asarray. Any transformed/new leaf fails closed below.
        arrays_by_id[id(array)] = (key, array)
        return array

    with patch.object(torch_mapping, "require_key", read):
        mapped = (mapper or torch_mapping.map_protenix_inference_state_dict)(tracked)
    return mapped, tracked.reads, arrays_by_id


def compare_trees(mapped, managed, source_arrays):
    import jax

    left, left_def = jax.tree_util.tree_flatten_with_path(
        mapped, is_leaf=lambda x: x is None
    )
    right, right_def = jax.tree_util.tree_flatten_with_path(
        managed, is_leaf=lambda x: x is None
    )
    if left_def != right_def:
        raise ValueError("checkpoint tree definitions differ")
    counts, groups = Counter(), {}
    failures, mappings = [], []
    digests = [hashlib.sha256(), hashlib.sha256()]
    mapped_keys = Counter()
    for (left_path, a), (right_path, b) in zip(left, right, strict=True):
        path = jax.tree_util.keystr(left_path)
        if path != jax.tree_util.keystr(right_path):
            raise ValueError("checkpoint leaf paths differ")
        if hasattr(a, "shape") and hasattr(b, "shape"):
            source = source_arrays.get(id(a))
            source_key = None if source is None else source[0]
            if source_key is not None:
                mapped_keys[source_key] += 1
            a, b = np.asarray(a), np.asarray(b)
            exact = (
                a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
            )
            record = {
                "path": path,
                "source_key": source_key,
                "shape": list(a.shape),
                "dtype": a.dtype.str,
                "byte_exact": exact,
            }
            counts.update(
                array_leaves=1, elements=int(a.size), array_bytes=int(a.nbytes)
            )
            counts["nonfinite_elements"] += int(np.count_nonzero(~np.isfinite(a)))
            group = (
                getattr(left_path[0], "name", str(left_path[0]))
                if left_path
                else "root"
            )
            groups.setdefault(group, Counter()).update(
                array_leaves=1, exact=int(exact), bytes=int(a.nbytes)
            )
            if not exact or source_key is None:
                failures.append(path)
            for value, digest in zip((a, b), digests, strict=True):
                header = json.dumps(
                    {"path": path, "dtype": value.dtype.str, "shape": value.shape},
                    sort_keys=True,
                ).encode()
                digest.update(len(header).to_bytes(8, "big"))
                digest.update(header)
                digest.update(value.tobytes())
        else:
            counts["none_leaves" if a is None else "static_leaves"] += 1
            exact = type(a) is type(b) and a == b
            record = {
                "path": path,
                "static_type": type(a).__name__,
                "value": a,
                "exact": bool(exact),
            }
            if not exact:
                failures.append(path)
            for value, digest in zip((a, b), digests, strict=True):
                header = json.dumps(
                    {"path": path, "type": type(value).__name__, "value": value},
                    sort_keys=True,
                ).encode()
                digest.update(len(header).to_bytes(8, "big"))
                digest.update(header)
        mappings.append(record)
    return {
        "passed": not failures and counts["array_leaves"] > 0,
        "tree_definition_exact": True,
        "counts": counts,
        "groups": groups,
        "failures": failures,
        "leaf_mapping": mappings,
        "source_leaf_uses": mapped_keys,
        "canonical_contents_sha256": [digest.hexdigest() for digest in digests],
    }


def audit(export: Path, managed_path: Path, out: Path):
    import jax

    from foldjax.models.protenix.bridge.weights_io import load_native_weights

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
        device.platform != "cpu" for device in jax.devices()
    ):
        raise ValueError("audit requires CUDA_VISIBLE_DEVICES='' and JAX_PLATFORMS=cpu")
    out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((export / "export.json").read_text())
    if sha(export / "state.npz") != metadata["state_archive_sha256"]:
        raise ValueError("native state archive hash changed")
    before = stable_file_identity(managed_path)
    with np.load(export / "state.npz", allow_pickle=False) as archive:
        expected = metadata["state_signatures"]
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(
            expected
        ):
            raise ValueError("exported native key set differs from manifest")
        state = {key: archive[key] for key in archive.files}
    for key, array in state.items():
        signature = expected[key]
        if (
            list(array.shape) != signature["shape"]
            or array.dtype.str != signature["dtype"]
            or hashlib.sha256(array.tobytes()).hexdigest() != signature["sha256"]
        ):
            raise ValueError(f"exported native signature mismatch: {key}")
    mapped, reads, sources = map_with_source_keys(state)
    managed = load_native_weights(managed_path, prestack=False)
    result = compare_trees(mapped, managed, sources)
    for leaf in result["leaf_mapping"]:
        if leaf.get("source_key") is not None:
            leaf["native_key"] = expected[leaf["source_key"]]["native_key"]
    result.update(
        scope=__doc__,
        native_checkpoint_name=metadata["checkpoint_name"],
        managed_checkpoint_name=managed_path.name,
        native_checkpoint_sha256=metadata["checkpoint_sha256"],
        native_export_script_sha256=metadata["export_script_sha256"],
        native_export_sha256=sha(export / "export.json"),
        native_state_archive_sha256=metadata["state_archive_sha256"],
        managed_checkpoint_sha256=sha(managed_path),
        managed_checkpoint_unchanged=before == stable_file_identity(managed_path),
        native_state_key=metadata["selected_state_key"],
        key_normalization="strip at most one leading module.; reject collisions",
        native_nonmodel_metadata=metadata["nonmodel_metadata"],
        native_array_count=len(state),
        consumed_native_keys=len(reads),
        unconsumed_native_keys=sorted(state.keys() - reads.keys()),
        multiply_consumed_native_keys={
            key: count for key, count in reads.items() if count != 1
        },
        native_keys_without_output=sorted(
            state.keys() - result["source_leaf_uses"].keys()
        ),
        native_keys_with_multiple_outputs={
            key: count
            for key, count in result["source_leaf_uses"].items()
            if count != 1
        },
        runtimes={
            "jax": jax.__version__,
            "numpy": np.__version__,
            "native_torch": metadata["torch_version"],
            "native_numpy": metadata["numpy_version"],
        },
    )
    root = Path(__file__).resolve().parents[1]
    result["source_sha256"] = {
        name: sha(root / name)
        for name in (
            "bench/protenix_weight_audit.py",
            "src/foldjax/models/protenix/bridge/torch_mapping.py",
            "src/foldjax/models/protenix/bridge/weights_io.py",
        )
    }
    result["passed"] &= (
        metadata["checkpoint_unchanged"]
        and result["managed_checkpoint_unchanged"]
        and all(
            not result[key]
            for key in (
                "unconsumed_native_keys",
                "multiply_consumed_native_keys",
                "native_keys_without_output",
                "native_keys_with_multiple_outputs",
            )
        )
    )
    save(out / "audit.json", result)
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"leaf_mapping", "source_leaf_uses"}
    }
    summary["leaf_mapping_sha256"] = sha(out / "audit.json")
    save(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exporter = sub.add_parser("export")
    exporter.add_argument("--checkpoint", type=Path, required=True)
    exporter.add_argument("--out", type=Path, required=True)
    comparer = sub.add_parser("compare")
    comparer.add_argument("--export", type=Path, required=True)
    comparer.add_argument("--managed", type=Path, required=True)
    comparer.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = (
        export_native(args.checkpoint, args.out)
        if args.command == "export"
        else audit(args.export, args.managed, args.out)
    )
    print(json.dumps(result, indent=2))
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
