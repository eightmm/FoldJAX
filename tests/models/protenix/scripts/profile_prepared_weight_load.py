"""Measure and bit-audit a native prepared-weight load in a fresh process.

Run ``baseline`` and ``prepared`` as separate processes so JAX's allocator
high-water belongs to only one route.  ``--digest`` copies the final tree back
leaf by leaf and is intended for correctness gates, not timing measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _block_tree(tree: Any) -> None:
    for leaf in jax.tree.leaves(tree, is_leaf=lambda value: value is None):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _tree_digest(tree: Any) -> str:
    digest = hashlib.sha256()
    digest.update(repr(jax.tree.structure(tree)).encode())
    for path, leaf in jax.tree_util.tree_leaves_with_path(
        tree, is_leaf=lambda value: value is None
    ):
        digest.update(repr(path).encode())
        if hasattr(leaf, "dtype"):
            array = np.asarray(leaf)
            digest.update(array.dtype.str.encode())
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes(order="C"))
        else:
            digest.update(repr((type(leaf).__qualname__, leaf)).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("protenix", "opendde"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "prepared"), required=True)
    parser.add_argument("--digest", action="store_true")
    args = parser.parse_args()

    from foldjax.models.protenix.bridge.weights_io import (
        _load_native_weights_with_field_dtype,
        load_native_weights,
    )

    started = time.perf_counter()
    if args.model == "protenix":
        field_names = frozenset({"input_embedder", "pairformer_output"})
        from foldjax.models.protenix.models.model import cast_trunk_params
    else:
        field_names = frozenset(
            {
                "input_embedder",
                "pairformer_output",
                "structural_expander",
                "structural_refiner",
            }
        )
        from foldjax.models.opendde.models.model import cast_trunk_params

    if args.mode == "baseline":
        params = cast_trunk_params(load_native_weights(args.weights), jnp.bfloat16)
    else:
        params = _load_native_weights_with_field_dtype(
            args.weights,
            jnp.bfloat16,
            field_names,
        )
    _block_tree(params)
    elapsed = time.perf_counter() - started

    leaves = [
        leaf
        for leaf in jax.tree.leaves(params, is_leaf=lambda value: value is None)
        if hasattr(leaf, "dtype")
    ]
    stats = jax.devices()[0].memory_stats() or {}
    result = {
        "mode": args.mode,
        "model": args.model,
        "device": jax.devices()[0].platform,
        "elapsed_s": elapsed,
        "array_leaves": len(leaves),
        "logical_bytes": sum(int(leaf.size * leaf.dtype.itemsize) for leaf in leaves),
        "bytes_in_use": stats.get("bytes_in_use"),
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
    }
    if args.digest:
        result["digest"] = _tree_digest(params)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
