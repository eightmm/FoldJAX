"""Read a torch checkpoint archive without torch.

Every upstream ships its parameters as ``torch.save`` output: a zip archive
holding one pickle (`data.pkl`) whose tensors are persistent references into
raw little-endian buffers stored beside it (`data/<key>`). Nothing about that
container needs torch to read -- the pickle needs a handler for the persistent
IDs and for a handful of ``torch.*`` constructors, and the buffers need a dtype
table. This module supplies exactly those, returning NumPy arrays (bfloat16
via ``ml_dtypes``, which JAX already carries).

Two properties the torch loader does not have:

- **No arbitrary code.** ``find_class`` maps the tensor constructors it knows
  and returns an inert stub for everything else, so a pickle cannot reach any
  callable that was not written here. That is stricter than
  ``torch.load(weights_only=False)``, which the Lightning-style checkpoints
  otherwise force; stubbed entries (optimizer state, hyperparameter objects)
  deserialize as placeholders and the weight mappings never read them.
- **No torch in the environment.** Weight conversion -- the one step that read
  checkpoints -- previously needed the ``torch-bridge`` extra for this alone.

The compressed/legacy (non-zip) ``torch.save`` format is not handled; none of
the vendored models ship it. Callers fall back to real torch if it is both
needed and installed.
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

#: torch storage class name -> the dtype of its raw buffer.
_STORAGE_DTYPES: dict[str, np.dtype] = {
    "DoubleStorage": np.dtype(np.float64),
    "FloatStorage": np.dtype(np.float32),
    "HalfStorage": np.dtype(np.float16),
    "BFloat16Storage": np.dtype(ml_dtypes.bfloat16),
    "LongStorage": np.dtype(np.int64),
    "IntStorage": np.dtype(np.int32),
    "ShortStorage": np.dtype(np.int16),
    "CharStorage": np.dtype(np.int8),
    "ByteStorage": np.dtype(np.uint8),
    "BoolStorage": np.dtype(np.bool_),
}


class _Stub:
    """Placeholder for a pickled object the weight mappings never read."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> "_Stub":
        return _Stub(*args, **kwargs)

    def __setstate__(self, state: Any) -> None:
        self.state = state

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<torch-archive stub {self.args!r}>"


class _StorageType:
    """Marker for ``torch.FloatStorage`` and friends inside persistent IDs."""

    def __init__(self, name: str) -> None:
        self.name = name


class _Storage:
    """One raw buffer from the archive, typed but not yet shaped."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array


def _rebuild_tensor_v2(
    storage: _Storage,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool = False,
    backward_hooks: Any = None,
    metadata: Any = None,
) -> np.ndarray:
    """``torch._utils._rebuild_tensor_v2``, on a NumPy buffer.

    The stride walk is copied out immediately so the result owns its memory
    rather than aliasing the archive read.
    """
    base = storage.array[storage_offset:]
    itemsize = base.dtype.itemsize
    strided = np.lib.stride_tricks.as_strided(
        base,
        shape=tuple(size),
        strides=tuple(s * itemsize for s in stride),
    )
    return np.array(strided)


def _rebuild_parameter(data: np.ndarray, requires_grad: bool = False, *rest: Any):
    return data


_CONSTRUCTORS: dict[tuple[str, str], Any] = {
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
    ("torch", "Size"): tuple,
    # OrderedDict handles its own pickle BUILD protocol; a plain dict cannot.
    ("collections", "OrderedDict"): __import__("collections").OrderedDict,
}


class _Unpickler(pickle.Unpickler):
    def __init__(self, file: Any, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(file)
        self._archive = archive
        self._prefix = prefix

    def find_class(self, module: str, name: str) -> Any:
        constructor = _CONSTRUCTORS.get((module, name))
        if constructor is not None:
            return constructor
        if module.startswith("torch") and name.endswith("Storage"):
            return _StorageType(name)
        # Everything else -- optimizer classes, Lightning containers, dtypes
        # in metadata positions -- becomes an inert stub CLASS: NEWOBJ demands
        # a real type, and a fresh subclass per name keeps the repr honest.
        # The state dicts the mappings read are plain dict/str/ndarray by this
        # point; only the entries nothing reads are stubbed.
        return type(name, (_Stub,), {"__module__": module})

    def persistent_load(self, saved_id: Any) -> _Storage:
        kind, storage_type, key, _location, numel = saved_id
        if kind != "storage":
            raise pickle.UnpicklingError(f"unknown persistent record: {kind!r}")
        name = (
            storage_type.name
            if isinstance(storage_type, _StorageType)
            else getattr(storage_type, "__name__", str(storage_type))
        )
        dtype = _STORAGE_DTYPES.get(name)
        if dtype is None:
            raise pickle.UnpicklingError(f"unknown storage dtype: {name}")
        raw = self._archive.read(f"{self._prefix}/data/{key}")
        array = np.frombuffer(raw, dtype=dtype, count=numel)
        return _Storage(array)


def load(path: str | Path) -> Any:
    """Deserialize a ``torch.save`` zip archive into NumPy-backed objects."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        pickles = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if not pickles:
            raise pickle.UnpicklingError(
                f"{path} is not a torch zip checkpoint (no data.pkl)"
            )
        prefix = pickles[0].rsplit("/", 1)[0]
        with archive.open(pickles[0]) as handle:
            return _Unpickler(handle, archive, prefix).load()
