"""The archive reader in the environment it exists for: no torch at all.

`test_torch_archive.py` needs torch to *write* its fixture, so it skips
without it -- which is every environment the reader was written for, CI
included. This module is the other half: a fixture written once and committed
beside its expected bytes, read back with nothing installed. Kept apart
because a module-level `importorskip` skips the whole file, and a torch-free
test that only runs where torch exists is not a test of anything.
"""

from __future__ import annotations

import json
from pathlib import Path

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


