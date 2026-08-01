import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.models.trunk_blocks.trunk import _weighted_rigid_align

BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"


def _torch_align():
    if not BOLTZ_SRC.is_dir():
        pytest.skip("upstream Boltz source is not available")
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.loss.diffusionv2 import weighted_rigid_align

    return weighted_rigid_align


def test_weighted_rigid_align_matches_boltz_torch() -> None:
    rng = np.random.default_rng(0)
    b, n = 2, 40
    true = rng.standard_normal((b, n, 3)).astype(np.float32)
    pred = rng.standard_normal((b, n, 3)).astype(np.float32)
    mask = np.ones((b, n), dtype=np.float32)
    mask[:, -5:] = 0.0  # padded atoms
    weights = mask.copy()

    torch_align = _torch_align()
    with torch.no_grad():
        expected = torch_align(
            torch.from_numpy(true),
            torch.from_numpy(pred),
            torch.from_numpy(weights),
            torch.from_numpy(mask),
        ).numpy()

    actual = np.asarray(
        _weighted_rigid_align(
            jnp.asarray(true),
            jnp.asarray(pred),
            jnp.asarray(weights),
            jnp.asarray(mask),
        )
    )

    # Compare only over real (unmasked) atoms; padded rows are arbitrary.
    sel = mask.astype(bool)
    np.testing.assert_allclose(actual[sel], expected[sel], rtol=1e-4, atol=1e-4)
