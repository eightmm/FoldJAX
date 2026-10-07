"""The archive reader in the environment it exists for: no torch at all.

`test_torch_archive.py` needs torch to *write* its fixture, so it skips
without it -- which is every environment the reader was written for, CI
included. This module is the other half: a fixture written once and committed
beside its expected bytes, read back with nothing installed. Kept apart
because a module-level `importorskip` skips the whole file, and a torch-free
test that only runs where torch exists is not a test of anything.
"""

from __future__ import annotations

import io
import json
import pickle
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from foldjax import torch_archive

_FIXTURE = Path(__file__).parent / "data" / "torch_archive_fixture.pt"


def test_reads_the_checked_in_archive_without_torch() -> None:
    """The reader's home environment has no torch, so neither does this test.

    The fixture was written once by `torch.save` (seed 101) and committed with
    its expected bytes beside it; the assertion is bitwise, dtype for dtype --
    bfloat16 included, which NumPy alone cannot even spell without ml_dtypes.
    """
    expected = json.loads(_FIXTURE.with_suffix(".json").read_text())
    loaded = torch_archive.load(_FIXTURE)
    assert loaded["epoch"] == 3
    state = loaded["state_dict"]
    assert set(state) == set(expected)
    for key, spec in expected.items():
        got = state[key]
        assert str(got.dtype) == spec["dtype"], key
        assert list(got.shape) == spec["shape"], key
        assert got.tobytes() == bytes.fromhex(spec["bytes"]), key


@pytest.mark.parametrize(
    ("offset", "size", "stride", "message"),
    [
        (-1, (1,), (1,), "storage_offset"),
        (0, (-1,), (1,), r"size\[0\]"),
        (0, (2,), (-1,), r"stride\[0\]"),
        (0, (2, 2), (1,), "same number"),
        (8, (1,), (1,), "exceeds its storage"),
        (7, (2,), (1,), "exceeds its storage"),
        (0, (2,), (8,), "exceeds its storage"),
    ],
)
def test_forged_tensor_views_cannot_read_outside_storage(
    offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    message: str,
) -> None:
    storage = torch_archive._Storage(np.arange(8, dtype=np.float32))
    with pytest.raises(Exception, match=message):
        torch_archive._rebuild_tensor_v2(storage, offset, size, stride)


def test_valid_strided_and_empty_views_remain_supported() -> None:
    storage = torch_archive._Storage(np.arange(12, dtype=np.float32))
    view = torch_archive._rebuild_tensor_v2(storage, 1, (3, 2), (4, 2))
    np.testing.assert_array_equal(view, np.array([[1, 3], [5, 7], [9, 11]]))

    empty = torch_archive._rebuild_tensor_v2(storage, 12, (0, 4), (4, 1))
    assert empty.shape == (0, 4)


def test_shared_storage_is_read_once() -> None:
    class _Info:
        file_size = 16

    class _Archive:
        def __init__(self) -> None:
            self.reads = 0

        def getinfo(self, member: str) -> _Info:
            assert member == "checkpoint/data/7"
            return _Info()

        def read(self, member: str) -> bytes:
            assert member == "checkpoint/data/7"
            self.reads += 1
            return np.arange(4, dtype=np.float32).tobytes()

    archive = _Archive()
    unpickler = torch_archive._Unpickler(io.BytesIO(), archive, "checkpoint")
    saved_id = (
        "storage",
        torch_archive._StorageType("FloatStorage"),
        "7",
        "cpu",
        4,
    )

    first = unpickler.persistent_load(saved_id)
    second = unpickler.persistent_load(saved_id)

    assert first is second
    assert archive.reads == 1


def _rewritten(tmp_path: Path, compression: int) -> Path:
    import zipfile

    target = tmp_path / "rewritten.pt"
    with (
        zipfile.ZipFile(_FIXTURE) as source,
        zipfile.ZipFile(target, "w", compression=compression) as sink,
    ):
        for info in source.infolist():
            sink.writestr(info.filename, source.read(info.filename))
    return target


def _state_bytes(loaded) -> dict:
    return {
        key: (str(value.dtype), value.shape, value.tobytes())
        for key, value in loaded["state_dict"].items()
    }


def test_compressed_members_take_the_zipfile_route(tmp_path) -> None:
    """Only stored members are prefetched; anything else reads as before."""
    import zipfile

    stored = torch_archive.load(_FIXTURE)
    deflated = torch_archive.load(_rewritten(tmp_path, zipfile.ZIP_DEFLATED))
    assert _state_bytes(deflated) == _state_bytes(stored)


def test_a_corrupted_tensor_buffer_fails_its_crc(tmp_path) -> None:
    import zipfile

    path = _rewritten(tmp_path, zipfile.ZIP_STORED)
    with zipfile.ZipFile(path) as archive:
        info = max(
            (i for i in archive.infolist() if "/data/" in i.filename),
            key=lambda i: i.file_size,
        )
    payload = bytearray(path.read_bytes())
    # The member's data follows its local header, name and extra field.
    header = info.header_offset
    extra = int.from_bytes(payload[header + 28 : header + 30], "little")
    start = header + 30 + len(info.filename.encode()) + extra
    payload[start] ^= 0xFF
    path.write_bytes(bytes(payload))
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        torch_archive.load(path)


def test_only_the_first_whole_view_takes_its_storage_buffer() -> None:
    buffer = np.arange(6, dtype=np.float32)
    storage = torch_archive._Storage(buffer)
    first = torch_archive._rebuild_tensor_v2(storage, 0, (2, 3), (3, 1))
    second = torch_archive._rebuild_tensor_v2(storage, 0, (2, 3), (3, 1))
    assert np.shares_memory(first, buffer)
    assert not np.shares_memory(second, buffer)
    np.testing.assert_array_equal(second, first)

    transposed = torch_archive._Storage(np.arange(6, dtype=np.float32))
    view = torch_archive._rebuild_tensor_v2(transposed, 0, (3, 2), (1, 3))
    assert not np.shares_memory(view, transposed.array)

    read_only = torch_archive._Storage(
        np.frombuffer(np.arange(6, dtype=np.float32).tobytes(), np.float32)
    )
    copied = torch_archive._rebuild_tensor_v2(read_only, 0, (6,), (1,))
    assert copied.flags.writeable and copied.flags.owndata


# -- hostile archives ---------------------------------------------------------
#
# A checkpoint is downloaded input, and the reader's promise is that a forged
# one is refused at the field it lies about rather than read past its buffer
# or into anything but the constructors `torch_archive` names. Each archive
# below is built by hand, with no torch: the pickle is written against stand-in
# `torch` modules so it spells the same globals `torch.save` does.


class _Persistent:
    """Pickled as the persistent record it carries, as torch writes storages."""

    def __init__(self, *record) -> None:
        self.record = record


class _Tensor:
    """Pickled as `torch._utils._rebuild_tensor_v2(*args)`."""

    def __init__(self, *args) -> None:
        self.args = args

    def __reduce__(self):
        return (sys.modules["torch._utils"]._rebuild_tensor_v2, self.args)


def _write_archive(
    path: Path,
    payload,
    *,
    members: dict[str, bytes] | None = None,
    pickles: tuple[str, ...] = ("archive/data.pkl",),
) -> Path:
    """A zip shaped like `torch.save` output, holding ``payload``'s pickle."""
    torch = ModuleType("torch")
    utils = ModuleType("torch._utils")

    def _rebuild_tensor_v2(*args):  # pragma: no cover - only pickled by name
        raise AssertionError("only pickled by reference")

    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    torch._utils = utils
    for name in ("FloatStorage", "LongStorage", "UnheardOfStorage"):
        setattr(torch, name, type(name, (), {"__module__": "torch"}))

    class _Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, _Persistent):
                return tuple(
                    getattr(torch, item)
                    if isinstance(item, str) and item.endswith("Storage")
                    else item
                    for item in obj.record
                )
            return None

    stream = io.BytesIO()
    saved = {name: sys.modules.get(name) for name in ("torch", "torch._utils")}
    sys.modules.update({"torch": torch, "torch._utils": utils})
    try:
        _Pickler(stream, protocol=2).dump(payload)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    with zipfile.ZipFile(path, "w") as archive:
        for name in pickles:
            archive.writestr(name, stream.getvalue())
        for key, data in (members or {}).items():
            archive.writestr(f"archive/data/{key}", data)
    return path


_FLOATS = np.arange(4, dtype=np.float32).tobytes()


def _storage(*, kind="storage", type_="FloatStorage", key="0", numel=4):
    return _Persistent(kind, type_, key, "cpu", numel)


@pytest.mark.parametrize(
    ("payload", "members", "message"),
    [
        pytest.param(
            _Tensor(_storage(numel=True), 0, (4,), (1,)),
            {"0": _FLOATS},
            "storage numel must be an integer$",
            id="boolean-numel",
        ),
        pytest.param(
            _Tensor(_storage(numel="4"), 0, (4,), (1,)),
            {"0": _FLOATS},
            "storage numel must be an integer; got '4'",
            id="string-numel",
        ),
        pytest.param(
            _Tensor(_storage(), 0, 4, (1,)),
            {"0": _FLOATS},
            "size must be an integer sequence",
            id="scalar-size",
        ),
        pytest.param(
            _Tensor("not a storage", 0, (4,), (1,)),
            {},
            "invalid storage object",
            id="storage-is-not-a-record",
        ),
        pytest.param(
            _Tensor(_storage(), 5, (0,), (1,)),
            {"0": _FLOATS},
            "empty tensor offset 5 exceeds storage size 4",
            id="empty-view-past-its-storage",
        ),
        pytest.param(
            _Tensor(_Persistent("storage", "FloatStorage", "0"), 0, (4,), (1,)),
            {"0": _FLOATS},
            "malformed persistent storage record",
            id="short-record",
        ),
        pytest.param(
            _Tensor(_storage(kind="module"), 0, (4,), (1,)),
            {"0": _FLOATS},
            "unknown persistent record: 'module'",
            id="not-a-storage-record",
        ),
        pytest.param(
            _Tensor(_storage(type_="UnheardOfStorage"), 0, (4,), (1,)),
            {"0": _FLOATS},
            "unknown storage dtype: UnheardOfStorage",
            id="unknown-dtype",
        ),
        pytest.param(
            _Tensor(_storage(key="9"), 0, (4,), (1,)),
            {"0": _FLOATS},
            "storage '9' is missing",
            id="missing-member",
        ),
        pytest.param(
            _Tensor(_storage(numel=8), 0, (4,), (1,)),
            {"0": _FLOATS},
            "has 16 bytes; expected 32",
            id="numel-larger-than-its-bytes",
        ),
        pytest.param(
            [
                _Tensor(_storage(), 0, (4,), (1,)),
                _Tensor(_storage(type_="LongStorage", numel=2), 0, (2,), (1,)),
            ],
            {"0": _FLOATS},
            "referenced with inconsistent dtype or size",
            id="one-storage-two-dtypes",
        ),
    ],
)
def test_a_forged_archive_is_refused_at_the_field_it_forges(
    tmp_path: Path, payload, members: dict[str, bytes], message: str
) -> None:
    path = _write_archive(tmp_path / "forged.pt", payload, members=members)
    with pytest.raises(pickle.UnpicklingError, match=message):
        torch_archive.load(path)


def test_the_hand_built_archive_reads_when_nothing_is_forged(tmp_path: Path) -> None:
    """The control for the refusals above: the same builder, honest fields."""
    path = _write_archive(
        tmp_path / "honest.pt",
        {"weight": _Tensor(_storage(), 0, (2, 2), (2, 1))},
        members={"0": _FLOATS},
    )
    np.testing.assert_array_equal(
        torch_archive.load(path)["weight"],
        np.arange(4, dtype=np.float32).reshape(2, 2),
    )


@pytest.mark.parametrize(
    ("pickles", "message"),
    [((), "no data.pkl"), (("a/data.pkl", "b/data.pkl"), "multiple data.pkl")],
    ids=["none", "two"],
)
def test_an_archive_needs_exactly_one_pickle(
    tmp_path: Path, pickles: tuple[str, ...], message: str
) -> None:
    path = _write_archive(tmp_path / "archive.pt", {}, pickles=pickles)
    with pytest.raises(pickle.UnpicklingError, match=message):
        torch_archive.load(path)


def test_a_member_that_comes_back_short_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """Its size was checked up front; a read that then returns nothing stops."""
    path = _write_archive(
        tmp_path / "archive.pt",
        {"weight": _Tensor(_storage(), 0, (4,), (1,))},
        members={"0": _FLOATS},
    )
    monkeypatch.setattr(torch_archive.os, "preadv", lambda *args: 0)
    with pytest.raises(zipfile.BadZipFile, match="truncated member"):
        torch_archive.load(path)
