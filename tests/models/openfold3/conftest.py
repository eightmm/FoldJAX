"""Test configuration, including access to the upstream OpenFold3 checkout.

The parity gate imports real OpenFold3 modules rather than reimplementing them
in the test. It adds the sibling checkout to ``sys.path`` instead of installing
the package, because installing ``openfold3`` would pull in its full data stack
(lmdb, biotite, pdbeccdutils, awscli) that the model layers do not need.

Where the checkout is comes from :mod:`foldjax.models.openfold3.upstream`, so
the tests and the featurization CLI look in the same places.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from foldjax.models.openfold3.upstream import find_source

# ``.../<workspace>/foldjax/tests/models/openfold3/conftest.py``: the workspace
# holding both checkouts is four levels up. This was ``parents[2]`` while the
# port was a top-level package with its own ``tests/`` directory.
OPENFOLD3_SOURCE = find_source() or Path(__file__).resolve().parents[4] / "openfold3"


def _torch_bridge_available() -> tuple[bool, str]:
    if not (OPENFOLD3_SOURCE / "openfold3" / "core" / "model").is_dir():
        return False, f"no OpenFold3 checkout at {OPENFOLD3_SOURCE}"
    for module in ("torch", "ml_collections"):
        try:
            __import__(module)
        except ImportError:
            return False, f"{module} is not installed (need the torch-bridge extra)"
    return True, ""


@pytest.fixture(scope="session")
def openfold3_source() -> Path:
    """Make the upstream OpenFold3 package importable for parity tests."""
    available, reason = _torch_bridge_available()
    if not available:
        pytest.skip(reason)
    if str(OPENFOLD3_SOURCE) not in sys.path:
        sys.path.insert(0, str(OPENFOLD3_SOURCE))
    return OPENFOLD3_SOURCE


@pytest.fixture(scope="session")
def randomized():
    """Return a helper that replaces every parameter of a module with noise.

    OpenFold3 zero-initializes output and gate projections on purpose
    (``init="final"`` and ``init="gating"`` set the weight to 0). A freshly
    constructed ``Attention``, ``SwiGLUTransition`` or
    ``TriangleMultiplicativeUpdate`` therefore returns exactly zero, so a parity
    test on a default-initialized module compares 0 against 0 and passes no
    matter how wrong the port is.

    Every parity test must randomize before comparing. The gate checks forward
    math and key mapping, so the distribution is irrelevant as long as it is
    non-degenerate.
    """

    def _randomize(module, seed: int = 0, scale: float = 0.5):
        """``scale`` keeps activations in range.

        Whole-model tests need a smaller scale than single-layer ones: with 0.5,
        a full trunk drives activations to ~1e4, where a float32 ulp is ~1e-3 and
        differences of large numbers lose the precision a 1e-4 gate needs. That
        is arithmetic, not a porting error, but it makes the gate unusable, so
        composite tests pass a smaller scale rather than a looser tolerance.
        """
        import torch

        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator) * scale
                )
        return module.eval()

    return _randomize
