"""OpenDDE-compatible MSA resampling for recycling inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def sample_opendde_msa_cycle_features(
    input_feature_dict: Mapping[str, Any],
    *,
    n_cycle: int,
    seed: int,
    msa_depth: int = 1280,
    gap_token: int = 31,
) -> tuple[dict[str, np.ndarray], ...]:
    """Build fixed-depth, valid-first MSA samples for every recycle.

    Pinned OpenDDE shuffles rows with at least one valid token ahead of fully
    padded/all-gap rows, then takes at most ``msa_depth`` rows.  In particular,
    it does not choose a random depth and it does not remove duplicate rows.
    """

    if n_cycle <= 0:
        raise ValueError("n_cycle must be positive")
    if msa_depth <= 0:
        raise ValueError("msa_depth must be positive")

    msa_fields = ("msa", "has_deletion", "deletion_value")
    if any(field not in input_feature_dict for field in msa_fields):
        return tuple()

    arrays = {field: np.asarray(input_feature_dict[field]) for field in msa_fields}
    msa = arrays["msa"]
    if msa.ndim != 2 or any(values.shape != msa.shape for values in arrays.values()):
        raise ValueError("MSA and deletion features must share shape [N_msa, N_token]")

    n_msa = int(msa.shape[0])
    if n_msa == 0:
        return tuple()

    source_mask = input_feature_dict.get("msa_mask")
    if source_mask is None:
        msa_mask = np.ones(msa.shape, dtype=np.float32)
        row_valid = np.ones(n_msa, dtype=bool)
    else:
        msa_mask = np.asarray(source_mask)
        if msa_mask.shape != msa.shape:
            raise ValueError("msa_mask must share shape [N_msa, N_token] with msa")
        row_valid = np.any(msa_mask.astype(bool), axis=-1)

    # OpenDDE's current featurizer stores an all-one mask, which carries no
    # padded-row signal.  Its sampler therefore falls back to the gap token.
    if np.all(row_valid):
        row_valid = np.any(msa != gap_token, axis=-1)

    valid_idx = np.flatnonzero(row_valid)
    invalid_idx = np.flatnonzero(~row_valid)
    sample_size = min(msa_depth, n_msa)
    rng = np.random.default_rng(seed)
    cycles: list[dict[str, np.ndarray]] = []
    for _ in range(n_cycle):
        valid_perm = rng.permutation(valid_idx)
        invalid_perm = rng.permutation(invalid_idx)
        indices = np.concatenate((valid_perm, invalid_perm))[:sample_size]
        cycle = {
            field: np.take(values, indices, axis=0) for field, values in arrays.items()
        }
        # Upstream uses the source mask only to prioritize rows.  Once selected,
        # every row/token enters the MSA module without a separate mask.
        cycle["msa_mask"] = np.ones(cycle["msa"].shape, dtype=np.float32)
        cycles.append(cycle)
    return tuple(cycles)
