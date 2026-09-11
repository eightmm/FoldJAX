"""Tokamax fused triangle attention -- re-exported from the shared module.

Boltz-2 carried a byte-identical copy of this wrapper, so it moved to
`models/_tokamax_attention.py`, beside `_cueq.py`. Kept importable from here so
existing call sites do not move.
"""

from __future__ import annotations

from foldjax.models._tokamax_attention import (
    tokamax_attention_core,
    tokamax_available,
)

__all__ = ["tokamax_attention_core", "tokamax_available"]
