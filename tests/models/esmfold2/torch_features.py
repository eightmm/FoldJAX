"""Publisher-parity adapter for handing JAX-built features to torch."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from foldjax.models.esmfold2.data.features import ATOM_BLOCK
from foldjax.models.esmfold2.data.features import build_features as _build

__all__ = ["ATOM_BLOCK", "build_features"]


def build_features(
    chains: Sequence[tuple[str, str, int, int]],
    alignments: dict[int, Path] | None = None,
    *,
    msa_depth: int | None = None,
) -> dict[str, Any]:
    """Featurize a parity job as publisher-runtime tensors."""

    import torch

    arrays = _build(chains, alignments, msa_depth=msa_depth)
    return {name: torch.from_numpy(value) for name, value in arrays.items()}
