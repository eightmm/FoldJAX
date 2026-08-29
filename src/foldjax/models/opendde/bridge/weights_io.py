"""Native OpenDDE-JAX weights and isolated trusted-checkpoint conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foldjax.models.opendde.bridge.checkpoint import unwrap_state_dict
from foldjax.models.opendde.bridge.torch_mapping import map_opendde_inference_state_dict
from foldjax.models.protenix.bridge.weights_io import (
    _PREPARED_CAST_INPUT_BATCH_BYTES,
    _load_native_weights_with_field_dtype,
    load_native_weights,
    save_native_weights,
)

_TRUNK_FIELDS = frozenset(
    {
        "input_embedder",
        "pairformer_output",
        "structural_expander",
        "structural_refiner",
    }
)


def _load_prepared_native_weights(
    path: str | Path,
    trunk_dtype: Any,
    *,
    _cast_batch_bytes: int = _PREPARED_CAST_INPUT_BATCH_BYTES,
) -> Any:
    """Load OpenDDE's mixed-precision tree without a full FP32 device copy.

    The historical path first transferred every FP32 checkpoint leaf and then
    rebuilt the four trunk subtrees in bfloat16.  Both complete device trees
    were live during that cast.  Here the checkpoint is decoded and pre-stacked
    on the host.  Finite eligible leaves are narrowed there before their first
    device transfer.  A leaf containing non-finite values takes the *same* JAX
    ``asarray(...).astype(...)`` route in bounded batches, preserving NaN
    payload behavior as well as ordinary values.  Once such a batch's narrowed
    outputs are ready, its temporary FP32 device inputs are deleted; only the
    final mixed tree accumulates.

    The common loader keeps each fallback cast batch bounded and releases its
    wide inputs once the narrowed outputs are ready. Diffusion, distogram, and
    confidence remain on the ordinary FP32 path.
    """
    return _load_native_weights_with_field_dtype(
        path,
        trunk_dtype,
        _TRUNK_FIELDS,
        cast_batch_bytes=_cast_batch_bytes,
    )


def load_torch_checkpoint(path: str | Path) -> Any:
    """Convert one trusted official ``.pt`` checkpoint to JAX parameters.

    FoldJAX's restricted archive reader reconstructs tensor storage directly as
    NumPy arrays. Production prediction loads the resulting native archive;
    neither operation imports PyTorch.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    from foldjax import torch_archive

    checkpoint = torch_archive.load(checkpoint_path)
    return map_opendde_inference_state_dict(unwrap_state_dict(checkpoint))


__all__ = [
    "load_native_weights",
    "load_torch_checkpoint",
    "save_native_weights",
]
