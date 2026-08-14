import builtins
import json
import subprocess
import sys

import numpy as np
import pytest

from foldjax import torch_archive
from foldjax.models.boltz2 import __version__
from foldjax.models.boltz2.bridge.checkpoint import load_checkpoint_state_dict
from tests.models.boltz2.torch_checkpoint import load_torch_checkpoint_state_dict


def test_import() -> None:
    assert __version__


def test_import_is_lazy_and_torch_free() -> None:
    code = """
import json
import sys
import foldjax.models.boltz2
print(json.dumps({name: name in sys.modules for name in (
    'torch', 'boltz', 'jax', 'foldjax.models.boltz2.api'
)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "torch": False,
        "boltz": False,
        "jax": False,
        "foldjax.models.boltz2.api": False,
    }


def test_checkpoint_conversion_never_imports_torch(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "boltz.ckpt"
    checkpoint.write_bytes(b"trusted fixture")
    state = {"weight": object()}
    seen = []

    def fake_load(path):
        seen.append(path)
        return {"state_dict": state}

    monkeypatch.setattr(torch_archive, "load", fake_load)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("checkpoint conversion imported torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert load_checkpoint_state_dict(checkpoint) is state
    assert seen == [checkpoint]


def test_parity_helper_converts_checkpoint_arrays_to_torch(
    tmp_path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "boltz.ckpt"
    checkpoint.write_bytes(b"trusted fixture")
    state = {
        "weight": np.asarray([1.5, -2.0], dtype=np.float32),
        "step": 7,
    }
    monkeypatch.setattr(torch_archive, "load", lambda path: {"state_dict": state})

    converted = load_torch_checkpoint_state_dict(checkpoint)

    assert converted is not state
    assert isinstance(converted["weight"], torch.Tensor)
    assert converted["weight"].dtype == torch.float32
    torch.testing.assert_close(converted["weight"], torch.tensor([1.5, -2.0]))
    assert converted["step"] == 7
    assert isinstance(state["weight"], np.ndarray)


def test_numpy_featurizer_is_the_default_even_when_torch_is_installed() -> None:
    code = """
from foldjax.models.boltz2.data._torch import BACKEND
print(BACKEND)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "numpy"


def test_numpy_featurizer_cannot_be_switched_to_torch_by_environment() -> None:
    code = """
import builtins
import os
import sys

os.environ['FOLDJAX_BOLTZ_FEATURIZER'] = 'torch'
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.') or name == 'pytorch_lightning':
        raise AssertionError(f'production featurizer imported {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from foldjax.models.boltz2.data._torch import BACKEND
from foldjax.models.boltz2.data.module.inferencev2 import PredictionDataset

assert BACKEND == 'numpy'
assert PredictionDataset is not None
assert 'torch' not in sys.modules
assert 'pytorch_lightning' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
