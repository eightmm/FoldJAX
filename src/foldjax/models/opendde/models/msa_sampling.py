"""OpenDDE-compatible MSA resampling for recycling inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from foldjax.padding import PaddingPlan, resolve_axis
from foldjax.schema import PaddingConfig

_MSA_VALUE_FIELDS = ("msa", "has_deletion", "deletion_value")
_RAW_MSA_SOURCE_FIELDS = (*_MSA_VALUE_FIELDS, "msa_mask")


def drop_sampled_msa_source_features(
    input_feature_dict: Mapping[str, Any],
    cycle_msa_features: Sequence[Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    """Drop raw MSA inputs only after complete sampled cycles replace them.

    The sampled-cycle tuple is the model's sole MSA input once every cycle has
    a non-empty, shape-consistent value for all four consumed fields.  Keeping
    the much deeper source alignment in the feature mapping would still make
    it part of the JIT input signature and context-parallel placement even
    though XLA dead-code elimination removes it from the computation.

    This is deliberately a conservative boundary helper: absent, empty,
    incomplete, or malformed cycle data leaves the original mapping object
    untouched so direct/custom callers retain their historical fallback.
    """

    if cycle_msa_features is None or len(cycle_msa_features) == 0:
        return input_feature_dict

    sampled_shape: tuple[int, ...] | None = None
    for cycle in cycle_msa_features:
        if not isinstance(cycle, Mapping):
            return input_feature_dict
        if any(name not in cycle for name in _RAW_MSA_SOURCE_FIELDS):
            return input_feature_dict
        shapes: list[tuple[int, ...]] = []
        for name in _RAW_MSA_SOURCE_FIELDS:
            value = cycle[name]
            try:
                shape = getattr(value, "shape", None)
                if shape is None:
                    shape = np.shape(value)
                concrete_shape = tuple(int(dimension) for dimension in shape)
            except (TypeError, ValueError):
                return input_feature_dict
            if len(concrete_shape) != 2 or any(size <= 0 for size in concrete_shape):
                return input_feature_dict
            shapes.append(concrete_shape)
        if any(shape != shapes[0] for shape in shapes[1:]):
            return input_feature_dict
        if sampled_shape is None:
            sampled_shape = shapes[0]
        elif shapes[0] != sampled_shape:
            return input_feature_dict

    if not any(name in input_feature_dict for name in _RAW_MSA_SOURCE_FIELDS):
        return input_feature_dict
    return {
        name: value
        for name, value in input_feature_dict.items()
        if name not in _RAW_MSA_SOURCE_FIELDS
    }


def sample_opendde_msa_cycle_features(
    input_feature_dict: Mapping[str, Any],
    *,
    num_recycles: int,
    seed: int,
    msa_depth: int = 1280,
    gap_token: int = 31,
) -> tuple[dict[str, np.ndarray], ...]:
    """Build fixed-depth, valid-first MSA samples for every recycle.

    Pinned OpenDDE shuffles rows with at least one valid token ahead of fully
    padded/all-gap rows, then takes at most ``msa_depth`` rows.  In particular,
    it does not choose a random depth and it does not remove duplicate rows.
    """

    if num_recycles <= 0:
        raise ValueError("num_recycles must be positive")
    if msa_depth <= 0:
        raise ValueError("msa_depth must be positive")

    if any(field not in input_feature_dict for field in _MSA_VALUE_FIELDS):
        return tuple()

    arrays = {
        field: np.asarray(input_feature_dict[field]) for field in _MSA_VALUE_FIELDS
    }
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
    for _ in range(num_recycles):
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


def pad_opendde_msa_cycle_features(
    cycles: tuple[dict[str, np.ndarray], ...],
    config: PaddingConfig,
    *,
    gap_token: int = 31,
    token_target: int | None = None,
) -> tuple[tuple[dict[str, np.ndarray], ...], PaddingPlan]:
    """Right-pad sampled OpenDDE MSA cycles with rigorously masked axes.

    OpenDDE samples the same number of MSA rows for every recycle and its MSA
    stack consumes ``msa_mask`` in every operation that communicates a row
    back into the pair representation.  That makes both the row and token axes
    safe to pad as long as real values form prefixes and every added position
    has a zero mask.  ``token_target`` is supplied by the full OpenDDE feature
    padder after it has resolved the residue-token bucket.

    This helper intentionally operates *after* native MSA sampling.  Padding
    the raw alignment first would let synthetic rows enter the sampler and
    change its seeded permutation, violating the default stochastic contract.
    """

    if not cycles:
        raise ValueError(
            "OpenDDE MSA padding requires sampled MSA features; this input "
            "does not contain msa, has_deletion, and deletion_value"
        )

    storage_depth: int | None = None
    storage_tokens: int | None = None
    actual_depth: int | None = None
    validated: list[tuple[dict[str, np.ndarray], int]] = []
    for cycle_index, cycle in enumerate(cycles):
        missing = [name for name in _RAW_MSA_SOURCE_FIELDS if name not in cycle]
        if missing:
            raise ValueError(
                "OpenDDE MSA padding requires complete sampled cycle fields; "
                f"cycle {cycle_index} is missing: {', '.join(missing)}"
            )
        arrays = {name: np.asarray(cycle[name]) for name in _MSA_VALUE_FIELDS}
        msa = arrays["msa"]
        if msa.ndim != 2 or any(value.shape != msa.shape for value in arrays.values()):
            raise ValueError(
                "OpenDDE sampled MSA and deletion features must share "
                "shape [N_msa, N_token]"
            )
        mask = np.asarray(cycle["msa_mask"])
        if mask.shape != msa.shape:
            raise ValueError(
                "OpenDDE sampled msa_mask must share shape [N_msa, N_token]"
            )
        row_valid = np.any(mask.astype(bool), axis=-1)
        real_depth = int(np.count_nonzero(row_valid))
        if real_depth < 1:
            raise ValueError("OpenDDE MSA padding requires at least one real MSA row")
        if not np.all(row_valid[:real_depth]) or np.any(row_valid[real_depth:]):
            raise ValueError(
                "OpenDDE MSA padding requires real rows to form a contiguous prefix"
            )
        # A partially masked source row would alter normalization differently
        # from OpenDDE's native all-token rows.  Token padding is appended only
        # after this validation, so its synthetic zero columns remain distinct
        # from an unknown pre-existing layout.
        if not np.all(mask[:real_depth].astype(bool)):
            raise ValueError(
                "OpenDDE MSA padding requires every real row to be valid across "
                "the full unpadded token axis"
            )
        if storage_depth is None:
            storage_depth = int(msa.shape[0])
            storage_tokens = int(msa.shape[1])
            actual_depth = real_depth
        elif (
            int(msa.shape[0]) != storage_depth
            or int(msa.shape[1]) != storage_tokens
            or real_depth != actual_depth
        ):
            raise ValueError(
                "OpenDDE MSA padding requires every recycle to share one "
                "storage shape and real-row prefix"
            )
        validated.append(({**cycle, **arrays, "msa_mask": mask}, real_depth))

    assert (
        storage_depth is not None
        and storage_tokens is not None
        and actual_depth is not None
    )
    if token_target is None:
        token_target = storage_tokens
    if token_target < storage_tokens:
        raise ValueError(
            "OpenDDE sampled MSA token target is smaller than its storage width "
            f"{storage_tokens}: {token_target}"
        )
    target_depth = resolve_axis(
        actual_depth,
        config,
        "msa",
        minimum=storage_depth,
    )
    padding_rows = target_depth - storage_depth
    padding_tokens = token_target - storage_tokens
    padded_cycles: list[dict[str, np.ndarray]] = []
    for cycle, _real_depth in validated:
        padded = dict(cycle)
        for name in _MSA_VALUE_FIELDS:
            constant = gap_token if name == "msa" else 0
            padded[name] = np.pad(
                np.asarray(cycle[name]),
                ((0, padding_rows), (0, padding_tokens)),
                mode="constant",
                constant_values=constant,
            )
        padded["msa_mask"] = np.pad(
            np.asarray(cycle["msa_mask"]),
            ((0, padding_rows), (0, padding_tokens)),
            mode="constant",
            constant_values=0,
        )
        padded_cycles.append(padded)

    return tuple(padded_cycles), PaddingPlan(
        actual={"msa": actual_depth},
        storage={"msa": storage_depth},
        target={"msa": target_depth},
    )
