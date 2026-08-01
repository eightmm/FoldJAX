"""FoldJAX's own reader for Chai's released conformer archive.

Upstream decodes this by importing chai_lab for its private antipickle
adapters, which needs torch and pins an rdkit this environment cannot hold.
These tests build archives in the real on-disk format and read them back, so
the decoder is exercised without either dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from foldjax.models.chai.bridge.antipickle_io import (
    INTRODUCTION_PHRASE,
    MAGIC_KEY,
    AntipickleFormatError,
    load,
)


@dataclass(frozen=True)
class _Conf:
    position: Any
    element: Any
    charge: Any
    atom_names: Any
    bonds: Any
    symmetries: Any


def _npy(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _adapted(typestring: str, **fields: Any) -> dict:
    return {MAGIC_KEY: typestring, **fields}


def _array(array: np.ndarray) -> dict:
    return _adapted("np.ndarray", data=_npy(array))


def _write(tmp_path: Path, payload: Any, *, version: int = 1) -> Path:
    import msgpack

    path = tmp_path / "archive.apkl"
    with path.open("wb") as handle:
        handle.write(msgpack.packb([INTRODUCTION_PHRASE, version, None]))
        handle.write(msgpack.packb(payload))
    return path


def test_reads_the_adapters_the_released_archive_actually_uses(
    tmp_path: Path,
) -> None:
    positions = np.arange(6, dtype=np.float32).reshape(2, 3)
    symmetries = np.array([[0, 1]], dtype=np.int64)
    payload = {
        "ALA": _adapted(
            "dataclass",
            classname="conf",
            data={
                # Chai's torch adapter wraps a numpy array, so a tensor field
                # arrives already decoded.
                "position": _adapted("torch", data=_array(positions)),
                "element": _array(np.array([6, 7], dtype=np.int32)),
                "charge": _array(np.array([0, 0], dtype=np.int32)),
                "atom_names": _adapted("tuple", data=["N", "CA"]),
                "bonds": _adapted(
                    "tuple", data=[_adapted("tuple", data=[0, 1])]
                ),
                "symmetries": _array(symmetries),
            },
        )
    }
    loaded = load(_write(tmp_path, payload), dataclasses={"conf": _Conf})

    conf = loaded["ALA"]
    assert isinstance(conf, _Conf)
    np.testing.assert_array_equal(conf.position, positions)
    assert conf.position.dtype == positions.dtype
    assert conf.atom_names == ("N", "CA")
    assert conf.bonds == ((0, 1),)
    np.testing.assert_array_equal(conf.symmetries, symmetries)


def test_numpy_scalars_and_sets_round_trip(tmp_path: Path) -> None:
    payload = {
        "scalar": _adapted("np.scalar", data=_npy(np.float32(2.5))),
        "members": _adapted("set", data=[1, 2, 2]),
    }
    loaded = load(_write(tmp_path, payload))
    assert loaded["scalar"] == np.float32(2.5)
    assert loaded["members"] == {1, 2}


def test_plain_mappings_are_left_alone(tmp_path: Path) -> None:
    """Only mappings carrying the magic key are adapted."""
    loaded = load(_write(tmp_path, {"a": 1, "nested": {"b": [1, 2]}}))
    assert loaded == {"a": 1, "nested": {"b": [1, 2]}}


def test_an_unknown_adapter_is_reported_not_guessed(tmp_path: Path) -> None:
    path = _write(tmp_path, {"x": _adapted("pd.DataFrame", data=b"")})
    with pytest.raises(AntipickleFormatError, match="pd.DataFrame"):
        load(path)


def test_an_unknown_dataclass_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path, _adapted("dataclass", classname="other", data={}))
    with pytest.raises(AntipickleFormatError, match="unknown dataclass 'other'"):
        load(path, dataclasses={"conf": _Conf})


def test_a_foreign_file_is_rejected(tmp_path: Path) -> None:
    import msgpack

    path = tmp_path / "not-antipickle.bin"
    path.write_bytes(msgpack.packb(["something else", 1, None]))
    with pytest.raises(AntipickleFormatError, match="not an antipickle archive"):
        load(path)


def test_a_newer_format_version_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, {"a": 1}, version=99)
    with pytest.raises(AntipickleFormatError, match="format 99"):
        load(path)


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.apkl"
    path.write_bytes(b"")
    with pytest.raises(AntipickleFormatError, match="is empty"):
        load(path)


def test_the_decoder_never_unpickles(tmp_path: Path) -> None:
    """npy buffers are loaded with allow_pickle=False; object arrays must fail."""
    buffer = BytesIO()
    np.save(buffer, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    path = _write(tmp_path, {"x": _adapted("np.ndarray", data=buffer.getvalue())})
    with pytest.raises(ValueError, match="allow_pickle=False"):
        load(path)
