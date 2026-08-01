"""Read Chai's released conformer archive without antipickle, torch, or chai_lab.

The archive is an ``antipickle`` file: a msgpack stream whose first object is a
header and whose second is the payload. Anything that is not a msgpack native
type is encoded as a mapping carrying a magic key naming its adapter.

Only four adapters appear in this archive, and none of them needs torch:

``np.ndarray`` / ``np.scalar``
    a ``.npy`` buffer, loadable with ``allow_pickle=False``
``tuple`` / ``set``
    a list to rebuild
``torch``
    ``{"data": <ndarray>}`` — Chai's own adapter serialises tensors *through*
    numpy, so the file already holds plain arrays
``dataclass``
    ``{"classname": ..., "data": {field: value}}``, where the only class in this
    archive is Chai's ``ConformerData``

Upstream reads this by importing ``chai_lab`` for its private adapters, which
drags in torch and pins an rdkit FoldJAX cannot hold. Decoding the format
directly is both smaller and torch-free.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

#: antipickle marks adapted values with this key. Chosen upstream for being a
#: byte that never appears in ordinary dictionary keys.
MAGIC_KEY = "\x7f"
INTRODUCTION_PHRASE = "unpack me with antipickle"
FORMAT_VERSION = 1


class AntipickleFormatError(ValueError):
    """The file is not the antipickle archive we expect."""


def _load_npy(payload: bytes) -> np.ndarray:
    return np.load(BytesIO(payload), allow_pickle=False)


def _adapters(dataclasses: Mapping[str, Callable[..., Any]]) -> dict[str, Any]:
    return {
        "np.ndarray": lambda d: _load_npy(d["data"]),
        # A 0-d npy buffer; unwrap it to the scalar it holds.
        "np.scalar": lambda d: _load_npy(d["data"])[()],
        "tuple": lambda d: tuple(d["data"]),
        "set": lambda d: set(d["data"]),
        # Chai's torch adapter stores `tensor.numpy()`, so the payload is
        # already an ndarray by the time the inner adapter has run.
        "torch": lambda d: np.asarray(d["data"]),
        "dataclass": lambda d: _build(dataclasses, d),
    }


def _build(dataclasses: Mapping[str, Callable[..., Any]], entry: Mapping) -> Any:
    name = entry["classname"]
    factory = dataclasses.get(name)
    if factory is None:
        raise AntipickleFormatError(
            f"archive holds an unknown dataclass {name!r}; "
            f"expected one of {', '.join(sorted(dataclasses)) or '(none)'}"
        )
    return factory(**entry["data"])


def load(
    path: str | Path,
    *,
    dataclasses: Mapping[str, Callable[..., Any]] | None = None,
) -> Any:
    """Load the single object stored in an antipickle archive.

    ``dataclasses`` maps the archive's class names onto the callables that
    should rebuild them, which is how ``ConformerData`` is reconstructed without
    importing Chai.
    """
    import msgpack

    table = _adapters(dataclasses or {})

    def hook(entry: dict) -> Any:
        typestring = entry.get(MAGIC_KEY)
        if typestring is None:
            return entry
        adapter = table.get(typestring)
        if adapter is None:
            raise AntipickleFormatError(
                f"archive uses the {typestring!r} adapter, which this reader "
                "does not implement"
            )
        return adapter(entry)

    path = Path(path)
    with path.open("rb") as handle:
        unpacker = msgpack.Unpacker(
            handle,
            object_hook=hook,
            strict_map_key=False,
            # The released conformer archive is ~124 MB and a single object.
            max_buffer_size=4 * 1024**3,
        )
        try:
            header = next(unpacker)
        except StopIteration:
            raise AntipickleFormatError(f"{path} is empty") from None
        _check_header(path, header)

        try:
            payload = next(unpacker)
        except StopIteration:
            raise AntipickleFormatError(f"{path} holds no object") from None
        if next(unpacker, _SENTINEL) is not _SENTINEL:
            raise AntipickleFormatError(f"{path} holds more than one object")
    return payload


_SENTINEL = object()


def _check_header(path: Path, header: Any) -> None:
    if (
        not isinstance(header, list)
        or len(header) != 3
        or header[0] != INTRODUCTION_PHRASE
        or not isinstance(header[1], int)
    ):
        raise AntipickleFormatError(f"{path} is not an antipickle archive")
    if header[1] > FORMAT_VERSION:
        raise AntipickleFormatError(
            f"{path} uses antipickle format {header[1]}; this reader "
            f"implements {FORMAT_VERSION}"
        )
