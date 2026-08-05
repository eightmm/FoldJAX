"""Guards that the parity gate is not passing vacuously.

OpenFold3 zero-initializes output and gate projections (``init="final"`` and
``init="gating"``). A default-constructed ``Attention``,``SwiGLUTransition`` or
``TriangleMultiplicativeUpdate`` therefore returns exactly zero, so a parity test
that forgets to randomize compares 0 against 0 and passes for any port,
correct or not. That happened during this port and these tests exist so it
cannot happen silently again.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.torch_parity


def _modules(torch):
    from openfold3.core.model.layers.transition import SwiGLUTransition
    from openfold3.core.model.layers.triangular_multiplicative_update import (
        TriangleMultiplicationOutgoing,
    )
    from openfold3.core.model.primitives import AdaLN, Attention

    return {
        "Attention": (
            Attention(c_q=12, c_k=10, c_v=10, c_hidden=4, no_heads=3),
            lambda m: m(torch.randn(2, 7, 12), torch.randn(2, 5, 10)),
        ),
        "SwiGLUTransition": (
            SwiGLUTransition(c_in=8, n=4),
            lambda m: m(torch.randn(2, 5, 8)),
        ),
        "AdaLN": (
            AdaLN(c_a=8, c_s=6),
            lambda m: m(torch.randn(2, 5, 8), torch.randn(2, 5, 6)),
        ),
        "TriangleMultiplicationOutgoing": (
            TriangleMultiplicationOutgoing(c_z=8, c_hidden=6),
            lambda m: m(torch.randn(2, 5, 5, 8)),
        ),
    }


def test_default_initialization_really_is_degenerate(openfold3_source: Path) -> None:
    """Documents the trap: three of these return exactly zero out of the box."""
    import torch

    torch.manual_seed(0)
    all_zero = {}
    for name, (module, call) in _modules(torch).items():
        with torch.no_grad():
            out = call(module.eval())
        all_zero[name] = bool((out == 0).all())
    assert all_zero["Attention"] is True
    assert all_zero["SwiGLUTransition"] is True
    assert all_zero["TriangleMultiplicationOutgoing"] is True
    # AdaLN is not all-zero, but its linear_g/linear_s weights are, so its
    # conditioning path is still untested without randomization.
    assert all_zero["AdaLN"] is False


def test_randomized_fixture_makes_every_module_non_degenerate(
    openfold3_source: Path, randomized
) -> None:
    import torch

    torch.manual_seed(0)
    for name, (module, call) in _modules(torch).items():
        with torch.no_grad():
            out = call(randomized(module))
        array = np.asarray(out.detach().numpy())
        assert np.isfinite(array).all(), name
        assert np.abs(array).max() > 1e-3, f"{name} is still degenerate"


def test_randomized_fixture_leaves_no_zero_parameters(
    openfold3_source: Path, randomized
) -> None:
    import torch

    torch.manual_seed(0)
    for name, (module, _call) in _modules(torch).items():
        for key, value in randomized(module).state_dict().items():
            assert float(value.abs().sum()) > 0.0, f"{name}.{key} is still zero"
