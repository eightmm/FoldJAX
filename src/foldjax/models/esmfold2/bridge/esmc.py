"""Reading the ESMC-6B checkpoint, which is sharded, prefixed and TE-flavoured.

Three differences from the structure weights, all of them in the file rather
than the model:

* it ships as six safetensors behind a `model.safetensors.index.json`;
* it is published as `ESMCForMaskedLM`, so every key carries an `esmc.`
  prefix that the encoder itself does not use. Stripped on load, because the
  alternative is spelling the prefix at 1,048 call sites;
* because upstream builds it out of Transformer Engine modules when they are
  available, it carries `_extra_state` entries that are pickled Python objects
  rather than tensors. Upstream drops them with a `_load_state_dict_pre_hook`;
  so does this.

The published file is float32 and 25.4 GB. `dtype="bfloat16"` is the default
here and halves what it occupies: the output is fed through a softmax over 81
layers and then a layer norm, which is not a place where the last eight bits
of mantissa decide anything.
"""

from __future__ import annotations

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open

from foldjax.models.esmfold2.models.esmc import ESMCSettings, settings_from_config
from foldjax.torch_archive import pread_into

INDEX_NAME = "model.safetensors.index.json"
WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
#: The masked-LM wrapper's submodule name, which the encoder does not use.
CHECKPOINT_PREFIX = "esmc."


def shard_paths(directory: str | Path) -> list[Path]:
    """Every safetensors file of the checkpoint, index or single file."""
    directory = Path(directory)
    index = directory / INDEX_NAME
    if index.exists():
        with index.open() as handle:
            mapping = json.load(handle)["weight_map"]
        return [directory / name for name in sorted(set(mapping.values()))]
    single = directory / WEIGHTS_NAME
    if single.exists():
        return [single]
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors checkpoint under {directory}")
    return shards


def load_settings(directory: str | Path) -> ESMCSettings:
    """The checkpoint's own width, head count and depth."""
    with Path(directory, CONFIG_NAME).open() as handle:
        return settings_from_config(json.load(handle))


def load_parameters(
    directory: str | Path, *, dtype: str | None = "bfloat16", to_device: bool = True
) -> dict[str, jnp.ndarray]:
    """Every tensor in the checkpoint, keyed the way `esmc.encode` asks.

    The `esmc.` prefix of the masked-LM wrapper is stripped, and the masked-LM
    head itself -- which the structure model never reads -- is dropped with it.
    `_extra_state` entries go too: Transformer Engine writes its quantisation
    bookkeeping there, it is not a tensor, and nothing here reads it.
    """
    parameters: dict[str, jnp.ndarray] = {}
    target = None if dtype is None else jnp.dtype(dtype)
    for shard in shard_paths(directory):
        with safe_open(shard, framework="numpy") as handle:
            names = [
                name
                for name in handle.keys()  # noqa: SIM118 -- safetensors has no __iter__
                if not name.endswith("_extra_state")
                and _is_encoder_key(name.removeprefix(CHECKPOINT_PREFIX))
            ]
            direct = {} if target is None else _float32_extents(shard, handle, names)
            with _CastBatches(shard, target, direct) as batches:
                for name in names:
                    key = name.removeprefix(CHECKPOINT_PREFIX)
                    if name in direct:
                        batches.add(key, name)
                    else:
                        array = handle.get_tensor(name)
                        if target is not None:
                            # Cast while this is still a host array. The
                            # published ESMC checkpoint is 25.4 GB of float32
                            # and normally loads as bfloat16; transferring
                            # first briefly stages the full-width leaf on
                            # device and builds a conversion executable for
                            # each distinct shape.
                            array = array.astype(target)
                        batches.flush(parameters, to_device)
                        parameters[key] = jax.device_put(array) if to_device else array
                    if batches.full:
                        batches.flush(parameters, to_device)
                batches.flush(parameters, to_device)
    return parameters


def _is_encoder_key(key: str) -> bool:
    return key.startswith("transformer.") or key == "embed.weight"


#: A safetensors header larger than this is refused rather than parsed.
_MAX_HEADER_BYTES = 100_000_000
#: Source bytes cast before the batch is handed to the device, bounding the
#: narrowed host copies held at once.
_CAST_BATCH_BYTES = 1024 * 1024 * 1024
#: Elements per cast task: one tensor still spreads over the threads, and each
#: thread reads into one reused buffer of this size.
_CAST_CHUNK_ELEMENTS = 1 << 22


def _float32_extents(
    shard: Path, handle: Any, names: list[str]
) -> dict[str, tuple[int, tuple[int, ...]]]:
    """Byte offset and shape of each float32 tensor, read from the file itself.

    ``safe_open`` has already validated the header; this re-reads it only for
    the data offsets it does not expose, and admits a tensor only when the
    header agrees with ``safe_open`` on dtype and shape and its extent is the
    exact float32 size inside the file. Anything else keeps the per-tensor
    ``get_tensor`` route.
    """
    size = shard.stat().st_size
    with shard.open("rb") as file:
        length = int.from_bytes(file.read(8), "little")
        if not 0 < length <= min(_MAX_HEADER_BYTES, size - 8):
            return {}
        header = json.loads(file.read(length))
    start = 8 + length
    extents = {}
    for name in names:
        entry = header.get(name)
        if not isinstance(entry, dict) or entry.get("dtype") != "F32":
            continue
        shape = tuple(int(n) for n in entry.get("shape", ()))
        begin, end = (int(n) for n in entry["data_offsets"])
        view = handle.get_slice(name)
        if view.get_dtype() != "F32" or tuple(view.get_shape()) != shape:
            continue
        if end - begin != 4 * math.prod(shape) or not 0 <= begin <= end <= size - start:
            continue
        extents[name] = (start + begin, shape)
    return extents


class _CastBatches:
    """Narrow float32 tensors read straight from the shard, in parallel.

    Each element goes through the same ``float32 -> dtype`` cast that
    ``ndarray.astype`` applies (ml_dtypes' round-to-nearest-even for
    bfloat16); the work is only split into slices of one tensor, so the
    narrowed array is bitwise the one ``get_tensor(name).astype(dtype)``
    returned. What goes is the single-threaded cast and the full-width
    per-tensor copy ``get_tensor`` made first: 29.6 s of the 25.4 GB
    checkpoint's load on 8 cores, against 2.1 s here.
    """

    def __init__(self, shard: Path, target: Any, extents: dict) -> None:
        self._shard, self._target, self._extents = shard, target, extents
        self._pending: list[tuple[str, np.ndarray]] = []
        self._jobs: list[tuple[int, int, np.ndarray]] = []
        self._bytes = 0
        self._fd: int | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._buffers = threading.local()

    def __enter__(self) -> _CastBatches:
        if self._extents:
            self._fd = os.open(self._shard, os.O_RDONLY)
            workers = min(os.process_cpu_count() or 1, 16)
            self._pool = ThreadPoolExecutor(max_workers=max(workers, 1))
        return self

    def __exit__(self, *exc: object) -> None:
        self._pending.clear()
        self._jobs.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        if self._fd is not None:
            os.close(self._fd)

    @property
    def full(self) -> bool:
        return self._bytes >= _CAST_BATCH_BYTES

    def add(self, key: str, name: str) -> None:
        offset, shape = self._extents[name]
        count = math.prod(shape)
        narrowed = np.empty(shape, self._target)
        self._pending.append((key, narrowed))
        flat = narrowed.reshape(-1)
        for first in range(0, count, _CAST_CHUNK_ELEMENTS):
            size = min(_CAST_CHUNK_ELEMENTS, count - first)
            self._jobs.append((offset + 4 * first, size, flat[first : first + size]))
        self._bytes += 4 * count

    def _cast(self, job: tuple[int, int, np.ndarray]) -> None:
        offset, count, out = job
        buffer = getattr(self._buffers, "array", None)
        if buffer is None:
            buffer = self._buffers.array = np.empty(_CAST_CHUNK_ELEMENTS, "<f4")
        source = buffer[:count]
        view, done = memoryview(source).cast("B"), 0
        while done < view.nbytes:
            read = pread_into(self._fd, view[done:], offset + done)
            if read <= 0:
                raise OSError(f"short read from {self._shard} at byte {offset + done}")
            done += read
        np.copyto(out, source, casting="unsafe")

    def flush(self, parameters: dict, to_device: bool) -> None:
        if self._jobs:
            for future in [self._pool.submit(self._cast, job) for job in self._jobs]:
                future.result()
            self._jobs.clear()
        for key, array in self._pending:
            parameters[key] = jax.device_put(array) if to_device else array
        self._pending.clear()
        self._bytes = 0


__all__ = [
    "CONFIG_NAME",
    "INDEX_NAME",
    "WEIGHTS_NAME",
    "load_parameters",
    "load_settings",
    "shard_paths",
]
