"""Portable feature archives fail before entering checkpoint/JAX code."""

from __future__ import annotations

import numpy as np
import pytest

from foldjax.models.openfold3.data.featurize import load_features, save_features
from foldjax.models.openfold3.data.validation import validate_features
from foldjax.models.openfold3.models.representative_atoms import (
    RepresentativeAtomTable,
)

from .feature_fixture import minimal_features


def _features() -> dict[str, np.ndarray]:
    return minimal_features(tokens=3, atoms=5)


def test_valid_released_feature_abi_is_accepted() -> None:
    validate_features(_features())


@pytest.mark.parametrize(
    "mask",
    [
        np.ones((1, 3), np.int32),
        np.ones((1, 2), bool),
    ],
)
def test_cyclic_mask_has_explicit_boolean_token_contract(mask) -> None:
    features = _features()
    features["cyclic_mask"] = mask
    with pytest.raises(ValueError, match="cyclic_mask"):
        validate_features(features)


def test_cyclic_mask_cannot_include_padding() -> None:
    from foldjax.models.openfold3.data.featurize import pad_features

    features = pad_features(_features(), n_token=4, n_atom=5)
    features["cyclic_mask"] = np.ones((1, 4), bool)
    with pytest.raises(ValueError, match="cyclic_mask.*padded"):
        validate_features(features)


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    [
        ("restype", np.zeros((1, 3, 31), dtype=np.int32), "expected"),
        ("ref_element", np.zeros((1, 5, 120), dtype=np.int32), "expected"),
        ("msa", np.zeros((1, 2, 3, 32), dtype=np.float32), "dtype"),
        ("token_mask", np.array([[1.0, 0.0, 1.0]], dtype=np.float32), "padding"),
        ("atom_mask", np.array([[1, 1, 0, 1, 0]], dtype=np.float32), "padding"),
        ("ref_pos", np.full((1, 5, 3), np.nan, dtype=np.float32), "NaN"),
        (
            "atom_to_token_index",
            np.array([[0, 0, 2, 2, 2]], dtype=np.int32),
            "inconsistent",
        ),
    ],
)
def test_malformed_feature_abi_is_refused(
    name: str, replacement: np.ndarray, message: str
) -> None:
    features = _features()
    features[name] = replacement
    with pytest.raises(ValueError, match=message):
        validate_features(features)


def test_atom_counts_and_slots_must_agree() -> None:
    features = _features()
    features["num_atoms_per_token"] = np.array([[1, 2, 2]], dtype=np.int32)
    with pytest.raises(ValueError, match="start_atom_index|atom_to_token_index|mask"):
        validate_features(features)


def test_non_numeric_extra_payload_cannot_reach_jax() -> None:
    features = _features()
    features["surprise"] = np.array(["not a tensor argument"])
    with pytest.raises(ValueError, match="must be numeric"):
        validate_features(features)


def test_validation_checks_only_unique_values_in_zero_stride_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from foldjax.models.openfold3.data import validation

    features = _features()
    features["broadcast_extra"] = np.broadcast_to(
        np.zeros((), dtype=np.float32), (128, 128, 39)
    )
    features["empty_extra"] = np.broadcast_to(
        np.zeros((0, 1), dtype=np.float32), (0, 128)
    )
    original = np.isfinite
    observed_sizes: list[int] = []

    def tracked(value):
        observed_sizes.append(np.asarray(value).size)
        return original(value)

    monkeypatch.setattr(validation.np, "isfinite", tracked)
    validate_features(features)

    assert 128 * 128 * 39 not in observed_sizes
    assert 1 in observed_sizes


def test_validation_rejects_a_repeated_nonfinite_value() -> None:
    features = _features()
    features["broadcast_extra"] = np.broadcast_to(
        np.asarray(np.nan, dtype=np.float32), (128, 128, 39)
    )

    with pytest.raises(ValueError, match="NaN or infinity"):
        validate_features(features)


def test_archive_loader_enforces_the_feature_abi(tmp_path) -> None:
    features = _features()
    features["token_bonds"] = np.zeros((1, 2, 3), dtype=np.int32)
    table = RepresentativeAtomTable(
        *(np.zeros(32, dtype=np.float32) for _ in RepresentativeAtomTable._fields)
    )
    path = save_features(features, tmp_path / "invalid.npz", representative_atoms=table)
    with pytest.raises(ValueError, match="token_bonds.*expected"):
        load_features(path)
