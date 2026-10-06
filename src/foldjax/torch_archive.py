"""Read a torch checkpoint archive without torch.

Several upstreams ship parameters as ``torch.save`` output: a zip archive
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
- **No torch in the environment.** An earlier FoldJAX release exposed a
  ``torch-bridge`` extra just to read these checkpoints; the restricted reader
  makes that extra unnecessary and it is no longer published.

The compressed/legacy (non-zip) ``torch.save`` format is not handled; none of
the supported published checkpoints uses it. Unsupported archives fail
explicitly instead of silently crossing into a PyTorch runtime.
"""

from __future__ import annotations

import os
import pickle
import struct
import zipfile
import zlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from operator import index
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

    def __call__(self, *args: Any, **kwargs: Any) -> _Stub:
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
        # Whether a tensor already owns ``array`` itself (see
        # `_rebuild_tensor_v2`); every later view of it is copied.
        self.handed_out = False


def _checkpoint_int(value: Any, *, field: str, minimum: int = 0) -> int:
    """Return one pickle integer without accepting booleans or floats."""
    if isinstance(value, bool):
        raise pickle.UnpicklingError(f"tensor {field} must be an integer")
    try:
        result = index(value)
    except TypeError as error:
        raise pickle.UnpicklingError(
            f"tensor {field} must be an integer; got {value!r}"
        ) from error
    if result < minimum:
        raise pickle.UnpicklingError(
            f"tensor {field} must be at least {minimum}; got {result}"
        )
    return result


def _checkpoint_shape(
    value: Any, *, field: str, minimum: int = 0
) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise pickle.UnpicklingError(f"tensor {field} must be an integer sequence")
    return tuple(
        _checkpoint_int(item, field=f"{field}[{position}]", minimum=minimum)
        for position, item in enumerate(value)
    )


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
    rather than aliasing the archive read -- except for the first tensor that
    is exactly its whole, writable storage, which takes the buffer itself.
    """
    if not isinstance(storage, _Storage):
        raise pickle.UnpicklingError("tensor references an invalid storage object")
    offset = _checkpoint_int(storage_offset, field="storage_offset")
    shape = _checkpoint_shape(size, field="size")
    strides = _checkpoint_shape(stride, field="stride")
    if len(shape) != len(strides):
        raise pickle.UnpicklingError(
            "tensor size and stride must have the same number of dimensions"
        )

    storage_size = storage.array.size
    if any(dimension == 0 for dimension in shape):
        if offset > storage_size:
            raise pickle.UnpicklingError(
                f"empty tensor offset {offset} exceeds storage size {storage_size}"
            )
    else:
        # NumPy deliberately performs no bounds checks in ``as_strided``. Check
        # the final reachable element first so a forged archive cannot make the
        # owning copy below read outside the checkpoint's backing buffer.
        maximum_index = offset + sum(
            (dimension - 1) * step
            for dimension, step in zip(shape, strides, strict=True)
        )
        if offset >= storage_size or maximum_index >= storage_size:
            raise pickle.UnpicklingError(
                "tensor view exceeds its storage: "
                f"offset={offset}, size={shape}, stride={strides}, "
                f"storage_size={storage_size}"
            )

    if (
        not storage.handed_out
        and storage.array.flags.writeable
        and offset == 0
        and storage_size == int(np.prod(shape, dtype=np.int64))
        and strides == _contiguous_strides(shape)
    ):
        # The one tensor that is its whole storage, in order: the prefetched
        # buffer already is the owning copy the stride walk would make, and no
        # other tensor will be handed the same memory.
        storage.handed_out = True
        return storage.array.reshape(shape)
    base = storage.array[offset:]
    itemsize = base.dtype.itemsize
    strided = np.lib.stride_tricks.as_strided(
        base,
        shape=shape,
        strides=tuple(step * itemsize for step in strides),
        writeable=False,
    )
    return np.array(strided)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides, step = [], 1
    for dimension in reversed(shape):
        strides.append(step)
        step *= dimension
    return tuple(reversed(strides))


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
    def __init__(
        self,
        file: Any,
        archive: zipfile.ZipFile,
        prefix: str,
        prefetched: dict[str, np.ndarray] | None = None,
    ) -> None:
        super().__init__(file)
        self._archive = archive
        self._prefix = prefix
        self._prefetched = {} if prefetched is None else prefetched
        # torch.save writes multiple tensor views against one storage record.
        # Re-reading a large storage for every view causes extreme allocator
        # churn (ESM-2 references each ~150 MiB layer storage 16 times).  Keep
        # one immutable buffer per archive member for the duration of unpickle,
        # just as torch's own loader preserves storage identity.
        self._storage_cache: dict[
            str, tuple[np.dtype[Any], int, _Storage]
        ] = {}

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
        if (
            not isinstance(saved_id, Sequence)
            or isinstance(saved_id, (str, bytes, bytearray))
            or len(saved_id) != 5
        ):
            raise pickle.UnpicklingError(
                f"malformed persistent storage record: {saved_id!r}"
            )
        kind, storage_type, key, _location, raw_numel = saved_id
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
        numel = _checkpoint_int(raw_numel, field="storage numel")
        member = f"{self._prefix}/data/{key}"
        try:
            stored_bytes = self._archive.getinfo(member).file_size
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"checkpoint storage {key!r} is missing"
            ) from error
        expected_bytes = numel * dtype.itemsize
        if stored_bytes != expected_bytes:
            raise pickle.UnpicklingError(
                f"checkpoint storage {key!r} has {stored_bytes} bytes; "
                f"expected {expected_bytes} for {numel} values of {dtype}"
            )
        cached = self._storage_cache.get(member)
        if cached is not None:
            cached_dtype, cached_numel, storage = cached
            if cached_dtype != dtype or cached_numel != numel:
                raise pickle.UnpicklingError(
                    f"checkpoint storage {key!r} was referenced with "
                    "inconsistent dtype or size"
                )
            return storage
        raw = self._prefetched.pop(member, None)
        if raw is None:
            raw = self._archive.read(member)
        array = np.frombuffer(raw, dtype=dtype, count=numel)
        storage = _Storage(array)
        self._storage_cache[member] = (dtype, numel, storage)
        return storage


def load(path: str | Path) -> Any:
    """Deserialize a ``torch.save`` zip archive into NumPy-backed objects."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        pickles = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if not pickles:
            raise pickle.UnpicklingError(
                f"{path} is not a torch zip checkpoint (no data.pkl)"
            )
        if len(pickles) != 1:
            raise pickle.UnpicklingError(
                f"{path} has multiple data.pkl records: {pickles}"
            )
        prefix = pickles[0].rsplit("/", 1)[0]
        prefetched = _prefetch_stored_members(path, archive, f"{prefix}/data/")
        with archive.open(pickles[0]) as handle:
            return _Unpickler(handle, archive, prefix, prefetched).load()


#: Members read by one thread before the pickle needs them.
_PREFETCH_MAX_WORKERS = 16


def pread_into(fd: int, view: memoryview, offset: int) -> int:
    """Read from ``fd`` at ``offset`` straight into ``view``; bytes read.

    `os.preadv` where the platform has it. CPython drops it from `os` on
    macOS before 11 (`posixmodule.c`, keyed on the `pwritev` runtime check),
    which its 3.13 builds still support; there this takes an `os.pread` copy.
    """
    if hasattr(os, "preadv"):
        return os.preadv(fd, [view], offset)
    data = os.pread(fd, len(view), offset)
    view[: len(data)] = data
    return len(data)
_LOCAL_HEADER = struct.Struct("<4s2B4HL2L2H")


def _prefetch_stored_members(
    path: Path, archive: zipfile.ZipFile, directory: str
) -> dict[str, np.ndarray]:
    """Read every uncompressed tensor buffer at once, CRC-checked, in parallel.

    ``ZipFile.read`` copies each member through its buffered reader and then
    checks its CRC, one member at a time: 2.6 s of OpenFold3's 4.2 s archive
    load. The same bytes are read here with positioned reads into owned,
    writable arrays and checked against the same central-directory CRC, the
    way ``zipfile`` would -- a mismatch raises ``BadZipFile``. A member this
    cannot locate exactly (compressed, encrypted, or a local header that does
    not name it) is left to ``ZipFile.read``.
    """
    size = path.stat().st_size
    wanted = []
    for info in archive.infolist():
        if (
            info.filename.startswith(directory)
            and info.compress_type == zipfile.ZIP_STORED
            and not info.flag_bits & 0x1
            and info.compress_size == info.file_size
        ):
            wanted.append(info)
    if not wanted:
        return {}
    fd = os.open(path, os.O_RDONLY)
    try:
        located = []
        for info in wanted:
            header = os.pread(fd, _LOCAL_HEADER.size, info.header_offset)
            if len(header) != _LOCAL_HEADER.size:
                continue
            fields = _LOCAL_HEADER.unpack(header)
            name_length, extra_length = fields[10], fields[11]
            if fields[0] != zipfile.stringFileHeader:
                continue
            encoding = "utf-8" if info.flag_bits & 0x800 else "cp437"
            name = info.orig_filename.encode(encoding)
            name_at = info.header_offset + _LOCAL_HEADER.size
            if os.pread(fd, name_length, name_at) != name:
                continue
            start = name_at + name_length + extra_length
            if start + info.file_size > size:
                continue
            located.append((info, start))

        def read(item: tuple[zipfile.ZipInfo, int]) -> tuple[str, np.ndarray]:
            info, start = item
            buffer = np.empty(info.file_size, np.uint8)
            view, done = memoryview(buffer), 0
            while done < info.file_size:
                count = pread_into(fd, view[done:], start + done)
                if count <= 0:
                    raise zipfile.BadZipFile(f"truncated member {info.filename!r}")
                done += count
            if zlib.crc32(buffer) != info.CRC:
                raise zipfile.BadZipFile(f"Bad CRC-32 for file {info.filename!r}")
            return info.filename, buffer

        located.sort(key=lambda item: -item[0].file_size)
        workers = max(1, min(os.process_cpu_count() or 1, _PREFETCH_MAX_WORKERS))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(pool.map(read, located))
    finally:
        os.close(fd)
