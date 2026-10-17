"""Host-side chain extent and fixed-capacity confidence outputs."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np


def chain_capacity(n_chain: int) -> int:
    """Share a four-slot graph for monomers and small complexes."""
    if isinstance(n_chain, bool) or not isinstance(n_chain, Integral) or n_chain < 0:
        raise ValueError("n_chain must be a non-negative integer")
    return max(4, 1 << (max(1, int(n_chain)) - 1).bit_length())


def resolve_chain_count(features: Mapping[str, Any], n_chain: int | None = None) -> int:
    """Preserve the native max-id-plus-one convention, including empty ids."""
    if n_chain is not None:
        chain_capacity(n_chain)
        return int(n_chain)
    asym_id = np.asarray(features["asym_id"])
    valid = features.get("token_padding_mask")
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != asym_id.shape:
            raise ValueError("token_padding_mask must match asym_id")
        asym_id = asym_id[valid]
    return int(np.max(asym_id)) + 1 if asym_id.size else 0


def chain_mask(n_chain: int) -> np.ndarray:
    return np.arange(chain_capacity(n_chain)) < n_chain


_CHAIN_VECTORS = frozenset({"chain_ptm", "chain_iptm", "chain_gpde", "chain_plddt"})
_CHAIN_MATRICES = frozenset(
    {
        "chain_pair_iptm",
        "chain_pair_iptm_global",
        "chain_pair_gpde",
        "chain_pair_plddt",
        "chain_pair_pae_mean",
        "chain_pair_pae_min",
    }
)


def crop_chain_scores(output: Mapping[str, Any], n_chain: int) -> dict[str, Any]:
    """Crop only confidence chain axes; retain token/structural output ownership."""
    result = dict(output)
    for key in result.keys() & _CHAIN_VECTORS:
        result[key] = result[key][..., :n_chain]
    for key in result.keys() & _CHAIN_MATRICES:
        result[key] = result[key][..., :n_chain, :n_chain]
    return result
