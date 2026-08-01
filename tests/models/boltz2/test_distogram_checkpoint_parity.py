import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.bridge.torch_checkpoint import load_checkpoint_state_dict
from foldjax.models.boltz2.bridge.torch_mapping import map_distogram_state_dict
from foldjax.models.boltz2.models.heads.distogram import distogram_forward

CHECKPOINT = Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
PREFIX = "distogram_module"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


def test_distogram_matches_boltz_torch(checkpoint_state) -> None:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.trunkv2 import DistogramModule

    token_z, num_bins = 128, 64
    module = DistogramModule(token_z, num_bins).eval()
    local = {
        k.removeprefix(f"{PREFIX}."): v
        for k, v in checkpoint_state.items()
        if k.startswith(f"{PREFIX}.")
    }
    module.load_state_dict(local)

    rng = np.random.default_rng(0)
    z = rng.standard_normal((1, 12, 12, token_z)).astype(np.float32)

    with torch.no_grad():
        expected = module(torch.from_numpy(z)).numpy()

    params = map_distogram_state_dict(checkpoint_state, PREFIX)
    actual = np.asarray(
        distogram_forward(params, jnp.asarray(z), num_bins=num_bins)
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
