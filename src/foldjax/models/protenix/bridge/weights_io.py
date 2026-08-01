"""Native Protenix JAX weight serialization."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# Native weight files pickle their parameter NamedTuple classes, so each one
# records the module that defined them. Files exported before these ports were
# vendored into FoldJAX still carry the standalone top-level package names, and
# must keep loading.
_VENDORED_PACKAGE_ALIASES = {
    "opendde_jax.": "foldjax.models.opendde.",
    "protenix_jax.": "foldjax.models.protenix.",
}
_MODELS = "foldjax.models.protenix.models."
# Weight files predating the flat -> grouped module split inside this port.
_LEGACY_MODULE_ALIASES = {
    f"{_MODELS}{flat}": f"{_MODELS}{grouped}"
    for flat, grouped in {
        "atom": "diffusion.atom",
        "attention": "primitives.attention",
        "confidence": "heads.confidence",
        "diffusion": "diffusion.diffusion",
        "embedders": "trunk_blocks.embedders",
        "head": "heads.head",
        "msa": "trunk_blocks.msa",
        "pairformer": "trunk_blocks.pairformer",
        "primitives": "primitives.primitives",
        "template": "trunk_blocks.template",
        "transformer": "diffusion.transformer",
        "triangle": "triangle.triangle",
        "trunk": "trunk_blocks.trunk",
    }.items()
}
_SAFE_NUMPY_GLOBALS = frozenset(
    {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "_frombuffer"),
    }
)
_PARAMETER_MODULE_PREFIXES = (
    "foldjax.models.opendde.models.",
    "foldjax.models.protenix.models.",
)


class _NativeWeightsUnpickler(pickle.Unpickler):
    """Load only NumPy arrays and parameter NamedTuples from native weights."""

    def find_class(self, module: str, name: str) -> Any:
        for old, new in _VENDORED_PACKAGE_ALIASES.items():
            if module.startswith(old):
                module = new + module[len(old) :]
                break
        module = _LEGACY_MODULE_ALIASES.get(module, module)
        if (module, name) in _SAFE_NUMPY_GLOBALS:
            return super().find_class(module, name)
        if module.startswith(_PARAMETER_MODULE_PREFIXES):
            value = super().find_class(module, name)
            fields = getattr(value, "_fields", None)
            if (
                isinstance(value, type)
                and issubclass(value, tuple)
                and isinstance(fields, tuple)
                and all(isinstance(field, str) for field in fields)
            ):
                return value
        raise pickle.UnpicklingError(
            f"forbidden global in native weights: {module}.{name}"
        )


def save_native_weights(
    path: str | Path,
    params: Any,
    *,
    compress: bool = True,
) -> None:
    """Save a JAX parameter pytree without requiring torch at load time."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_tree = jax.tree.map(_leaf_to_numpy, params)
    opener = gzip.open if compress else open
    with opener(path, "wb") as fh:
        pickle.dump(numpy_tree, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_native_weights(path: str | Path, prestack: bool = True) -> Any:
    """Load native JAX weights produced by ``save_native_weights``.

    ``prestack`` collapses each homogeneous block list onto a leading layer
    axis before the arrays reach the device, so ``lax.scan`` does not have to
    stack them inside the traced graph -- which would copy the whole weight
    set into XLA's temp arena. See :mod:`foldjax.models._stacking`.
    """

    path = Path(path)
    opener = gzip.open if _is_gzip_file(path) else open
    with opener(path, "rb") as fh:
        numpy_tree = _NativeWeightsUnpickler(fh).load()
    if prestack:
        from foldjax.models._stacking import prestack_layer_lists

        numpy_tree = prestack_layer_lists(numpy_tree)
    return jax.tree.map(_leaf_to_jax, numpy_tree, is_leaf=lambda leaf: leaf is None)


def _is_gzip_file(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def _leaf_to_numpy(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.asarray(value)
    return value


def _leaf_to_jax(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    return value
