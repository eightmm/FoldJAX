"""Native Protenix JAX weight serialization.

The implementation is shared with OpenDDE, whose checkpoints carry Protenix
parameter classes and are read by the same restricted unpickler; it lives in
:mod:`foldjax.models._weights_io`. This module keeps the import path every
Protenix caller already uses -- including the two private names its prepared
CLI loader reaches for, and the unpickler that fixes which modules a weight
file may name. OpenDDE now imports the shared module directly instead of
reaching through here.
"""

from __future__ import annotations

from foldjax.models._weights_io import (
    _PREPARED_CAST_INPUT_BATCH_BYTES,
    _load_native_weights_with_field_dtype,
    _NativeWeightsUnpickler,
    load_native_weights,
    save_native_weights,
)

__all__ = [
    "_NativeWeightsUnpickler",
    "_PREPARED_CAST_INPUT_BATCH_BYTES",
    "_load_native_weights_with_field_dtype",
    "load_native_weights",
    "save_native_weights",
]
