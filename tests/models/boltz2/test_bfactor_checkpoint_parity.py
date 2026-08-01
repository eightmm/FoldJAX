import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.bridge.torch_checkpoint import load_checkpoint_state_dict
from foldjax.models.boltz2.bridge.torch_mapping import map_bfactor_state_dict
from foldjax.models.boltz2.models.heads.bfactor import bfactor_forward

CHECKPOINT = Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
PREFIX = "bfactor_module"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


def test_bfactor_matches_boltz_torch(checkpoint_state) -> None:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.trunkv2 import BFactorModule

    token_s = int(checkpoint_state[f"{PREFIX}.bfactor.weight"].shape[1])
    num_bins = int(checkpoint_state[f"{PREFIX}.bfactor.weight"].shape[0])
    module = BFactorModule(token_s, num_bins).eval()
    local = {
        k.removeprefix(f"{PREFIX}."): v
        for k, v in checkpoint_state.items()
        if k.startswith(f"{PREFIX}.")
    }
    module.load_state_dict(local)

    rng = np.random.default_rng(0)
    s = rng.standard_normal((1, 16, token_s)).astype(np.float32)

    with torch.no_grad():
        expected = module(torch.from_numpy(s)).numpy()

    params = map_bfactor_state_dict(checkpoint_state, PREFIX)
    actual = np.asarray(bfactor_forward(params, jnp.asarray(s)))

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
