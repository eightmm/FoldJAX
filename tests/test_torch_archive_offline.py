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
from pathlib import Path

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
