"""An optional kernel that failed to import says why when it is selected.

The guarded imports used to drop the exception, so explicitly selecting the
kernel reported only "unavailable" and hid the actual cause (a missing CUDA
library, a version skew).
"""

from __future__ import annotations

import importlib
import sys

import jax.numpy as jnp
import pytest

from foldjax.models import _tokamax_attention
from foldjax.models.boltz2.models.triangle import triangle_attention_pallas
from foldjax.models.protenix.models.primitives import attention_tokamax


@pytest.fixture
def tokamax_import_fails(monkeypatch: pytest.MonkeyPatch):
    """Re-import both tokamax wrappers with ``import tokamax`` failing."""

    monkeypatch.setitem(sys.modules, "tokamax", None)
    try:
        yield (
            importlib.reload(_tokamax_attention),
            importlib.reload(attention_tokamax),
        )
    finally:
        monkeypatch.undo()
        importlib.reload(_tokamax_attention)
        importlib.reload(attention_tokamax)


def _operands():
    q = jnp.zeros((1, 2, 1, 2, 4))
    return q, q, q, jnp.zeros((1, 1, 1, 2, 2)), jnp.zeros((1, 2, 1, 1, 2))


def test_the_import_error_is_kept_and_chained(tokamax_import_fails) -> None:
    shared, protenix = tokamax_import_fails
    assert shared.tokamax_available() is False
    cause = shared.tokamax_import_error()
    assert isinstance(cause, ImportError)

    with pytest.raises(
        RuntimeError, match="importing it failed with ModuleNotFoundError"
    ) as info:
        shared.tokamax_attention_core(*_operands())
    assert info.value.__cause__ is cause

    q = jnp.zeros((1, 2, 1, 4))
    with pytest.raises(RuntimeError, match="ModuleNotFoundError") as info:
        protenix.tokamax_attention(q, q, q)
    assert isinstance(info.value.__cause__, ImportError)


def test_a_ring_kernel_selection_names_the_cause(tokamax_import_fails) -> None:
    from foldjax.models import _cp_attention

    with pytest.raises(RuntimeError, match="tokamax.*ModuleNotFoundError") as info:
        _cp_attention.resolve_ring_tile_kernel("tokamax")
    assert isinstance(info.value.__cause__, ImportError)


def test_the_pallas_error_carries_its_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    cause = ImportError("libtriton missing")
    monkeypatch.setattr(triangle_attention_pallas, "_PALLAS_AVAILABLE", False)
    monkeypatch.setattr(triangle_attention_pallas, "_IMPORT_ERROR", cause)
    with pytest.raises(RuntimeError, match="libtriton missing") as info:
        triangle_attention_pallas.pallas_attention_core(*_operands())
    assert info.value.__cause__ is cause
