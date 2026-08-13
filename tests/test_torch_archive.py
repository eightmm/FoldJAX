"""The torch-free checkpoint reader against torch's own writer.

The contract is bitwise: whatever `torch.save` wrote, `torch_archive.load`
returns the same bytes as `torch.load`, tensor for tensor -- including the
cases that break naive readers: bfloat16 (no NumPy dtype), non-contiguous
views (stride walks), zero-dim scalars, and Lightning-style payloads whose
non-tensor objects must deserialize as inert stubs rather than executing
anything. torch is required to *write* the fixture, so the module skips
without it -- which is exactly the environment the reader exists for.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import ml_dtypes  # noqa: E402

from foldjax import torch_archive  # noqa: E402


class _Hyper:
    """Stands in for the arbitrary objects Lightning pickles beside weights."""

    def __init__(self) -> None:
        self.learning_rate = 1e-3


def _reference(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return tensor.numpy()


def test_reads_back_what_torch_wrote_bit_for_bit(tmp_path) -> None:
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    payload = {
        "state_dict": {
            "dense.weight": torch.randn(8, 3, dtype=torch.float32),
            "dense.half": torch.randn(5, 5).to(torch.float16),
            "dense.bf16": torch.randn(7, 2).to(torch.bfloat16),
            "index": torch.arange(11, dtype=torch.int64),
            "flag": torch.tensor(True),
            "scalar": torch.tensor(0.0415771715),
            # A strided view: the reader must honour offset and stride, not
            # assume the buffer is the tensor.
            "view": base[1:, ::2],
        },
        "epoch": 3,
        "hyper_parameters": _Hyper(),
    }
    path = tmp_path / "fixture.ckpt"
    torch.save(payload, path)

    loaded = torch_archive.load(path)

    assert loaded["epoch"] == 3
    # The stub is inert data, not the class it stood for.
    assert type(loaded["hyper_parameters"]).__name__ == "_Hyper"
    assert not isinstance(loaded["hyper_parameters"], _Hyper)

    for key, tensor in payload["state_dict"].items():
        expected = _reference(tensor)
        got = loaded["state_dict"][key]
        assert got.shape == expected.shape, key
        assert got.dtype == expected.dtype, key
        assert np.asarray(got).tobytes() == expected.tobytes(), key


def test_refuses_a_non_checkpoint_zip(tmp_path) -> None:
    import zipfile

    path = tmp_path / "not_a_checkpoint.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("readme.txt", "hello")
    with pytest.raises(Exception, match="data.pkl"):
        torch_archive.load(path)
