"""Torch-free native weight format for foldjax.models.boltz2.

Serializes an arbitrary nested pytree of JAX arrays (and scalar config leaves)
into a flat ``{dotted.key: array}`` representation that round-trips the exact
nested structure. Array leaves are stored via safetensors when available, else
``numpy.savez``; non-array leaves (int/float/bool/None) are stored in a sidecar
JSON metadata file.

This module MUST NOT import torch or boltz. It is the runtime weight loader.

Note on conditioning functions: some pytrees produced by the model at *runtime*
(e.g. the ``to_keys`` closure created inside diffusion conditioning) contain
Python functions. Those closures are NOT part of the saved parameter pytrees --
they are reconstructed at runtime by the model code (``diffusion_conditioning_
forward``). Only array and scalar leaves are serialized here; functions are
never written to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# Path component encoding ----------------------------------------------------
# Each pytree path component is encoded as ``<tag>:<value>`` where tag marks the
# container kind so unflatten can rebuild dict vs list/tuple:
#   d:<key>   dict key (string)
#   i:<idx>   list/tuple index (int)
# Components are joined with "/" to avoid collisions with "." inside dict keys.
_SEP = "/"


def _encode_path(path: tuple[Any, ...]) -> str:
    import jax.tree_util as jtu

    parts: list[str] = []
    for entry in path:
        if isinstance(entry, jtu.DictKey):
            parts.append(f"d:{entry.key}")
        elif isinstance(entry, jtu.SequenceKey):
            parts.append(f"i:{entry.idx}")
        elif isinstance(entry, jtu.GetAttrKey):
            parts.append(f"a:{entry.name}")
        else:  # FlattenedIndexKey or other
            parts.append(f"i:{getattr(entry, 'key', getattr(entry, 'idx', entry))}")
    return _SEP.join(parts)


def _decode_path(key: str) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for comp in key.split(_SEP):
        tag, _, val = comp.partition(":")
        if tag == "d":
            out.append(("dict", val))
        elif tag == "a":
            out.append(("dict", val))
        elif tag == "i":
            out.append(("list", int(val)))
        else:
            raise ValueError(f"bad path component: {comp!r}")
    return out


def _is_array_leaf(leaf: Any) -> bool:
    return isinstance(leaf, (jnp.ndarray, np.ndarray)) or hasattr(leaf, "shape")


def flatten_pytree(pytree: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Flatten ``pytree`` into (arrays, scalars) keyed by encoded path strings."""
    import jax.tree_util as jtu

    leaves_with_path, _ = jtu.tree_flatten_with_path(
        pytree, is_leaf=lambda x: x is None
    )
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, Any] = {}
    for path, leaf in leaves_with_path:
        key = _encode_path(path)
        if leaf is None:
            scalars[key] = {"__none__": True}
        elif _is_array_leaf(leaf):
            arrays[key] = np.asarray(leaf)
        elif isinstance(leaf, (bool, int, float)):
            scalars[key] = leaf
        else:
            raise TypeError(f"unserializable leaf at {key!r}: {type(leaf).__name__}")
    return arrays, scalars


def _set_path(root: Any, decoded: list[tuple[str, Any]], value: Any) -> Any:
    """Insert ``value`` at ``decoded`` path, creating dicts/lists as needed.

    Lists are built as dicts keyed by int then materialized later; we use a
    plain dict-with-int-keys placeholder and convert at the end.
    """
    if root is None:
        root = {}
    node = root
    for i, (_kind, key) in enumerate(decoded):
        if i == len(decoded) - 1:
            node[key] = value
        else:
            if key not in node:
                node[key] = {}
            node = node[key]
    return root


def _materialize(node: Any) -> Any:
    """Convert dict-with-contiguous-int-keys into lists, recursively."""
    if not isinstance(node, dict):
        return node
    node = {k: _materialize(v) for k, v in node.items()}
    keys = list(node.keys())
    if keys and all(isinstance(k, int) for k in keys):
        if sorted(keys) == list(range(len(keys))):
            return [node[i] for i in sorted(keys)]
    return node


def unflatten_pytree(
    arrays: dict[str, Any], scalars: dict[str, Any], to_device: bool = True
) -> Any:
    """Inverse of :func:`flatten_pytree`.

    ``to_device=False`` leaves the array leaves as NumPy, which lets a caller
    reshape the tree on the host before paying for the transfer.
    """
    root: Any = {}
    for key, arr in arrays.items():
        leaf = jnp.asarray(arr) if to_device else arr
        root = _set_path(root, _decode_path(key), leaf)
    for key, val in scalars.items():
        if isinstance(val, dict) and val.get("__none__"):
            val = None
        root = _set_path(root, _decode_path(key), val)
    return _materialize(root)


# Public save / load ---------------------------------------------------------
def _have_safetensors() -> bool:
    try:
        import safetensors  # noqa: F401

        return True
    except ImportError:
        return False


def _cast_float_array(arr: np.ndarray, dtype: Any) -> np.ndarray:
    """Cast a FLOAT numpy array to ``dtype``; leave int/bool arrays untouched.

    ``dtype`` may be a numpy dtype or a jnp dtype (e.g. ``jnp.bfloat16``);
    both are accepted via ``np.dtype``/ml_dtypes handling in ``np.asarray``.
    """
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(dtype)
    return arr


def save_params(pytree: Any, path: str | Path, dtype: Any = None) -> dict[str, Any]:
    """Save ``pytree`` of jnp arrays + scalar leaves to ``path``.

    Uses safetensors when available (``path`` -> .safetensors), else numpy
    ``.npz``. Scalar/None leaves and a backend marker go to a ``<path>.json``
    sidecar. Returns a small info dict (backend, byte sizes).

    If ``dtype`` is given, FLOAT array leaves are cast to it before saving
    (int/bool arrays and scalar/None leaves are left untouched). Default
    ``dtype=None`` preserves the original dtypes (fp32 round-trip unchanged).
    """
    path = Path(path)
    arrays, scalars = flatten_pytree(pytree)
    if dtype is not None:
        arrays = {k: _cast_float_array(v, dtype) for k, v in arrays.items()}
    meta = {"scalars": scalars, "keys": sorted(arrays.keys())}

    if _have_safetensors():
        from safetensors.numpy import save_file

        weights_path = path.with_suffix(".safetensors")
        # safetensors requires contiguous arrays; np.ascontiguousarray ensures.
        tensors = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
        save_file(tensors, str(weights_path))
        meta["backend"] = "safetensors"
    else:
        weights_path = path.with_suffix(".npz")
        np.savez(weights_path, **arrays)
        meta["backend"] = "npz"

    sidecar = weights_path.with_suffix(weights_path.suffix + ".json")
    sidecar.write_text(json.dumps(meta))
    return {
        "backend": meta["backend"],
        "weights_path": str(weights_path),
        "sidecar": str(sidecar),
        "num_arrays": len(arrays),
        "num_scalars": len(scalars),
        "bytes": weights_path.stat().st_size,
    }


def load_params(path: str | Path, dtype: Any = None, prestack: bool = True) -> Any:
    """Reconstruct the nested pytree from ``path``. No torch/boltz imports.

    If ``dtype`` is given, FLOAT array leaves are cast to it on load (int/bool
    arrays and scalar/None sidecar leaves are left untouched). Default
    ``dtype=None`` leaves dtypes exactly as stored (fp32 round-trip unchanged).

    ``prestack`` collapses each homogeneous layer list onto a leading layer
    axis here, on the host, so ``lax.scan`` does not have to stack it inside
    the traced graph -- which would copy the whole weight set into XLA's temp
    arena. The result still reads like the layer list it replaced. Pass
    ``prestack=False`` to get the plain per-layer structure back.
    """
    from foldjax.models.boltz2.weights import resolve_native_weight_bundle

    bundle = resolve_native_weight_bundle(path)
    if bundle is None:
        raise FileNotFoundError(f"no native weights found for {path}")
    weights_path, sidecar = bundle
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {"scalars": {}}

    if weights_path.suffix == ".safetensors":
        from safetensors.numpy import load_file

        arrays = load_file(str(weights_path))
    else:
        with np.load(weights_path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}

    if dtype is not None:
        arrays = {
            k: (
                v.astype(dtype)
                if np.issubdtype(np.asarray(v).dtype, np.floating)
                else v
            )
            for k, v in arrays.items()
        }

    if not prestack:
        return unflatten_pytree(arrays, meta.get("scalars", {}))

    from foldjax.models._stacking import prestack_layer_lists

    # Stack on the host and transfer once. Stacking after the transfer would
    # briefly hold both the per-layer arrays and the stacked copy on the
    # device -- twice the checkpoint, for nothing.
    params = unflatten_pytree(arrays, meta.get("scalars", {}), to_device=False)
    # The nested tree now owns every array leaf. Drop the flat loader mapping
    # before stacking so the unstacked tree can be reclaimed as soon as its
    # stacked replacement is ready, rather than retaining both forms through
    # the device transfer below.
    del arrays
    params = prestack_layer_lists(params)
    # ``jnp.asarray`` stages one tiny JIT program for every distinct leaf shape.
    # These are already concrete host arrays, so transfer them without tracing.
    return jax.tree.map(
        lambda leaf: jax.device_put(leaf) if isinstance(leaf, np.ndarray) else leaf,
        params,
        is_leaf=lambda leaf: leaf is None,
    )


# Torch-free feature loading -------------------------------------------------
def load_features_npz(path: str | Path) -> dict[str, jnp.ndarray]:
    """Load model features from a plain numpy ``.npz`` (no torch). -> jnp dict."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return {k: jnp.asarray(data[k]) for k in data.files}
