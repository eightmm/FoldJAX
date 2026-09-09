"""How the port reaches cuEquivariance when autocast boundaries are in play.

Upstream makes one fused `triangle_multiplicative_update` call and lets
`torch.autocast` place the boundaries. This port decomposes the operator into
`norm`, `gemm` and `gemm_dual` and places them by hand. Both sides use the same
library; only the call granularity differs, and a fused kernel does not
accumulate the way three primitive calls do.

The switch exists because that difference is measurable -- 0.67% of the
first-cycle `delta_z` residual against a native 5SAK capture -- and it defaults
off because a stage measurement is not a coordinate one.
"""

from __future__ import annotations

from foldjax.models.boltz2.models.triangle.triangle_cueq import (
    triangle_amp_granularity,
)

_VARIABLE = "BOLTZ_JAX_TRIANGLE_AMP_GRANULARITY"


def test_the_shipped_granularity_is_the_hand_placed_one(monkeypatch) -> None:
    """An unasked run keeps the decomposition it had before this switch."""
    monkeypatch.delenv(_VARIABLE, raising=False)
    assert triangle_amp_granularity() == "decomposed"


def test_upstreams_granularity_can_be_asked_for(monkeypatch) -> None:
    monkeypatch.setenv(_VARIABLE, "fused")
    assert triangle_amp_granularity() == "fused"


def test_the_value_is_case_insensitive(monkeypatch) -> None:
    """Environment values arrive however a shell script wrote them."""
    monkeypatch.setenv(_VARIABLE, "FUSED")
    assert triangle_amp_granularity() == "fused"


def test_an_unrecognised_value_is_refused(monkeypatch) -> None:
    """A misspelling must not quietly become upstream's path.

    The dispatch reads `== "decomposed"`, so anything else falls through to the
    fused call. Refusing here keeps a typo from making two machines run two
    different programs under one setting -- the same reason the triangle
    backend has no automatic fallback.
    """
    import pytest

    monkeypatch.setenv(_VARIABLE, "decomposd")
    with pytest.raises(ValueError, match="decomposed"):
        triangle_amp_granularity()
