import json

import numpy as np
import pytest

from bench.openbind_native_capture import DrawRecorder, KernelCensus, control, driver


class _Tensor:
    def __init__(self, array, dtype):
        self._array = np.asarray(array)
        self.dtype = dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def test_tape_keys_follow_the_adapter_spelling_in_call_order():
    recorder = DrawRecorder()
    draw = recorder.wrap("randint", lambda: _Tensor([1024], "torch.int64"))
    noise = recorder.wrap(
        "randn", lambda: _Tensor(np.zeros((1, 5, 3)), "torch.float32")
    )
    draw()
    noise()
    archive = recorder.archive()
    assert list(archive) == ["000000_randint_torch_int64", "000001_randn_torch_float32"]
    assert archive["000001_randn_torch_float32"].shape == (1, 5, 3)


def test_kernel_census_counts_and_forwards():
    census = KernelCensus()
    counted = census.wrap("attention.cueq", lambda x: x + 1)
    assert counted(1) == 2 and counted(2) == 3
    assert census.calls == {"attention.cueq": 2}


def test_digest_and_save_are_fail_closed(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    identity = driver.digest_file(path)
    assert identity["bytes"] == 3 and len(identity["sha256"]) == 64
    out = tmp_path / "x.json"
    control.save(out, {"b": np.int64(2), "a": (1, 2)})
    assert json.loads(out.read_text()) == {"a": [1, 2], "b": 2}
    with pytest.raises(FileExistsError):
        control.save(out, {})
