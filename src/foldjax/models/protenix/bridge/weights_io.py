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
# Bound the wide input bytes per conversion batch.  The device fallback can
# briefly hold both FP32 input and BF16 output, so its transient is at most
# 1.5x this value (192 MiB), except for one indivisible oversized leaf.
_PREPARED_CAST_INPUT_BATCH_BYTES = 128 * 1024 * 1024


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

    numpy_tree = _load_native_numpy_tree(path, prestack=prestack)
    return jax.tree.map(_leaf_to_jax, numpy_tree, is_leaf=lambda leaf: leaf is None)


def _load_native_weights_with_field_dtype(
    path: str | Path,
    dtype: jnp.dtype,
    field_names: frozenset[str],
    *,
    prestack: bool = True,
    cast_batch_bytes: int = _PREPARED_CAST_INPUT_BATCH_BYTES,
) -> Any:
    """Load selected root fields without first placing them in their wide dtype.

    The historical model CLIs loaded every checkpoint leaf onto the accelerator
    and then narrowed selected fields.  Their final trees were mixed precision,
    but while the cast was pending the complete wide field set remained resident
    next to its replacement.

    Native checkpoints are NumPy trees until this boundary.  When ``dtype`` is
    BF16, native-endian, C-contiguous, finite FP32 leaves can be rounded on the
    host before transfer: ml_dtypes and XLA use the same round-to-nearest-even
    conversion for finite values. NaN payload conversion is backend-specific,
    so any other target, non-finite, non-native, non-contiguous, or differently
    typed leaf takes the exact historical device
    ``asarray(...).astype(...)`` route instead. Both routes are bounded and
    completed before their source wrappers are released.

    This is private to prepared CLI loaders.  ``load_native_weights`` keeps its
    public behaviour, and direct callers that cast after loading keep behaving
    exactly as before.
    """

    if cast_batch_bytes <= 0:
        raise ValueError("cast_batch_bytes must be positive")
    field_names = frozenset(field_names)
    if not field_names:
        raise ValueError("field_names must not be empty")
    numpy_tree = _load_native_numpy_tree(path, prestack=prestack)
    missing = field_names.difference(getattr(numpy_tree, "_fields", ()))
    if missing:
        raise TypeError(
            "prepared weights are missing parameter fields: "
            + ", ".join(sorted(missing))
        )

    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(
        numpy_tree, is_leaf=lambda leaf: leaf is None
    )
    # The flat source list now owns every leaf.  Releasing the immutable root
    # lets each consumed host array die as soon as its bounded batch completes.
    del numpy_tree
    paths = [key_path for key_path, _ in path_leaves]
    sources = [leaf for _, leaf in path_leaves]
    del path_leaves
    outputs: list[Any] = [None] * len(sources)
    pending_indices: list[int] = []
    pending_bytes = 0
    pending_host_cast: bool | None = None
    target_dtype = jnp.dtype(dtype)

    def flush() -> None:
        nonlocal pending_bytes, pending_host_cast
        if not pending_indices:
            return
        values = [sources[index] for index in pending_indices]
        if pending_host_cast:
            narrowed = _host_narrow_batch(values, target_dtype)
        else:
            narrowed = _device_narrow_batch(values, target_dtype)
        for index, value in zip(pending_indices, narrowed, strict=True):
            outputs[index] = value
            sources[index] = None
        pending_indices.clear()
        pending_bytes = 0
        pending_host_cast = None

    for index, (key_path, value) in enumerate(zip(paths, sources, strict=True)):
        root_name = getattr(key_path[0], "name", None) if key_path else None
        in_narrowed_field = root_name in field_names
        is_float_array = isinstance(value, np.ndarray) and np.issubdtype(
            value.dtype, np.floating
        )
        if not (in_narrowed_field and is_float_array):
            # Finish the final narrowed batch before another root field starts
            # reaching the device. Otherwise its final residency would overlap
            # the last conversion batch unnecessarily.
            if not in_narrowed_field:
                flush()
            outputs[index] = _leaf_to_jax(value)
            sources[index] = None
            continue

        # Restrict the host fast path to the checkpoint dtype measured and used
        # by released native weights. Other source dtypes take the exact old
        # device conversion, avoiding e.g. float64 -> float32 -> BF16
        # double-rounding differences when JAX x64 is disabled.
        host_cast = (
            target_dtype == jnp.dtype(jnp.bfloat16)
            and value.dtype == np.dtype(np.float32)
            and value.dtype.isnative
            and value.flags.c_contiguous
            and bool(np.isfinite(value).all())
        )
        value_bytes = int(value.nbytes)
        if pending_indices and (
            pending_host_cast != host_cast
            or pending_bytes + value_bytes > cast_batch_bytes
        ):
            flush()
        pending_host_cast = host_cast
        pending_indices.append(index)
        pending_bytes += value_bytes
        # A single stacked leaf can exceed the target.  It remains one bounded
        # unit rather than being sliced and concatenated into a new device
        # allocation, which would trade the old peak for another full-size copy.
        if pending_bytes >= cast_batch_bytes:
            flush()
    flush()
    return jax.tree_util.tree_unflatten(treedef, outputs)


def _load_native_numpy_tree(path: str | Path, *, prestack: bool) -> Any:
    """Decode and optionally prestack one checkpoint while it is host-only."""

    path = Path(path)
    opener = gzip.open if _is_gzip_file(path) else open
    with opener(path, "rb") as fh:
        # A holder lets ``pop`` transfer the only root reference into the
        # consuming traversal; assigning ``numpy_tree = helper(numpy_tree)``
        # would retain an immutable root until the helper returned.
        loaded = [_NativeWeightsUnpickler(fh).load()]
    if prestack:
        from foldjax.models._stacking import prestack_loaded_tree_consuming

        numpy_tree = prestack_loaded_tree_consuming(loaded.pop())
    else:
        numpy_tree = loaded.pop()
    return numpy_tree


def _host_narrow_batch(values: list[np.ndarray], dtype: jnp.dtype) -> list[Any]:
    """Round finite FP32 leaves on the host, then complete their transfers."""

    host_values = [value.astype(dtype, copy=False) for value in values]
    device_values: list[Any] = []
    try:
        device_values = [jnp.asarray(value) for value in host_values]
        for value in device_values:
            value.block_until_ready()
    except Exception:
        for value in device_values:
            value.delete()
        raise
    return device_values


def _device_narrow_batch(values: list[np.ndarray], dtype: jnp.dtype) -> list[Any]:
    """Apply the historical JAX cast, releasing each completed wide buffer."""

    wide_values: list[Any] = []
    narrowed_values: list[Any] = []
    try:
        wide_values = [jnp.asarray(value) for value in values]
        narrowed_values = [
            value if value.dtype == dtype else value.astype(dtype)
            for value in wide_values
        ]
        for value in narrowed_values:
            value.block_until_ready()
    except Exception:
        for value in wide_values:
            value.delete()
        raise
    for wide, narrowed in zip(wide_values, narrowed_values, strict=True):
        if wide is not narrowed:
            wide.delete()
    return narrowed_values


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
