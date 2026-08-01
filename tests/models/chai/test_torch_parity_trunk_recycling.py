"""Direct official-component parity for Chai trunk recycling projections."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.official_parity

torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from foldjax.models.chai.models.trunk import (  # noqa: E402
    map_recycling,
    recycling_projection,
)


@pytest.mark.parametrize(
    ("attribute", "shape"),
    [
        ("token_single_recycle_proj", (1, 7, 384)),
        ("token_pair_recycle_proj", (1, 7, 7, 256)),
    ],
)
def test_recycling_projection_matches_callable_torch_child(
    chai_trunk_module, attribute, shape
) -> None:
    rng = np.random.default_rng(31)
    x = rng.normal(size=shape).astype(np.float32)
    child = getattr(chai_trunk_module, attribute)
    state = {
        name: value.detach().cpu().numpy()
        for name, value in child.state_dict().items()
    }
    params = map_recycling(state)

    with torch.no_grad():
        expected = child(torch.from_numpy(x)).float().numpy()
    actual = np.asarray(
        recycling_projection(jnp.asarray(x), params), dtype=np.float32
    )

    # CPU Torch and GPU XLA can select adjacent BF16 values after an otherwise
    # FP32-accumulated dot product; one BF16 ULP is the strict portable bound.
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=7e-5)
