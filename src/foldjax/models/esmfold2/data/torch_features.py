"""The same features as `features`, handed to torch.

`foldjax.backends.esmfold2` drives upstream's torch model and wants torch
tensors; the JAX port wants arrays. Rather than two builders that agree until
one of them is edited, there is one builder in `features` and this converts its
output. The dtypes carry across unchanged -- `torch.from_numpy` preserves them,
including `ref_charge`'s int8 and the bool masks -- so the tensor-for-tensor
check against upstream's `prepare_protein_features` still gates both.
"""

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
    """Featurize a job as torch tensors.

    `chains` is one entry per chain copy: `(sequence, chain id, entity index,
    symmetry index)`. `alignments` maps an entity index to its a3m.
    """
    import torch

    arrays = _build(chains, alignments, msa_depth=msa_depth)
    return {name: torch.from_numpy(value) for name, value in arrays.items()}
