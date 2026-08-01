"""Torch-vs-JAX parity gate for the first ported Chai leaf.

This is the parity pattern every future module port MUST follow (the lesson
from foldjax.models.protenix, which shipped a real bug for lack of such a gate):

  1. extract weights from the configured real Chai component via the torch bridge,
  2. run the torch reference forward,
  3. run the JAX implementation with the SAME extracted weights,
  4. assert max abs diff < atol (1e-4 for fp32 leaves).

``bond_loss_input_proj.pt`` is a no-bias ``Linear(1, 512)`` (state_dict key
'weight', shape (512, 1)) -- the smallest real component (5.5 KB), used here as
the proof leaf. Requires the torch-bridge extra and the official asset directory.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.official_parity

torch = pytest.importorskip("torch")

from foldjax.models.chai.bridge.component_io import (  # noqa: E402
    load_component_state_dict,
)
from foldjax.models.chai.bridge.torch_mapping import (  # noqa: E402
    apply_linear,
    map_bond_loss_input_proj,
)


def test_bond_loss_input_proj_torch_vs_jax(official_asset_path) -> None:
    path = official_asset_path("bond_loss_input_proj.pt")
    state = load_component_state_dict(path)
    assert set(state) == {"weight"}
    assert state["weight"].shape == (512, 1)

    # torch reference (reconstructed Linear from extracted weight)
    w = torch.from_numpy(state["weight"])
    lin = torch.nn.Linear(1, 512, bias=False)
    with torch.no_grad():
        lin.weight.copy_(w)
        x = torch.randn(3, 5, 1)
        y_torch = lin(x).numpy()

    # jax with the SAME extracted weights
    params = map_bond_loss_input_proj(state)
    y_jax = np.asarray(apply_linear(params, x.numpy()))

    assert y_jax.shape == y_torch.shape == (3, 5, 512)
    np.testing.assert_allclose(y_jax, y_torch, rtol=1e-4, atol=1e-4)
