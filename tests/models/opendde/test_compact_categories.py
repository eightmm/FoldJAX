"""OpenDDE's managed atom-category storage is private and lossless."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.opendde.data.compact_categories import (
    COMPACT_REF_ATOM_CATEGORIES_MARKER,
    COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES,
    COMPACT_REF_ATOM_NAME_CHAR_IDS,
    COMPACT_REF_ELEMENT_IDS,
    compact_ref_atom_category_storage,
    validate_compact_ref_atom_categories,
)
from foldjax.models.opendde.models.model import _restore_ref_atom_category_one_hot


def _dense_categories() -> dict[str, np.ndarray]:
    element = np.zeros((3, 128), dtype=np.int64)
    chars = np.zeros((3, 4, 64), dtype=np.int64)
    element[0, 6] = 1
    element[2, 127] = 1
    chars[0, 0, 32] = 1
    chars[0, 2, 47] = 1
    chars[2, :, 63] = 1
    return {"ref_element": element, "ref_atom_name_chars": chars}


def test_exact_int64_categories_compact_and_restore_inside_jit() -> None:
    dense = _dense_categories()
    compact = compact_ref_atom_category_storage(dense)

    assert "ref_element" not in compact
    assert "ref_atom_name_chars" not in compact
    assert compact[COMPACT_REF_ATOM_CATEGORIES_MARKER].dtype == np.uint8
    assert int(compact[COMPACT_REF_ATOM_CATEGORIES_MARKER]) == 1
    np.testing.assert_array_equal(
        compact[COMPACT_REF_ELEMENT_IDS], np.asarray([6, 128, 127], np.uint8)
    )
    np.testing.assert_array_equal(
        compact[COMPACT_REF_ATOM_NAME_CHAR_IDS][1], np.full(4, 64, np.uint8)
    )

    @jax.jit
    def restore(features):
        restored = _restore_ref_atom_category_one_hot(features)
        return restored["ref_element"], restored["ref_atom_name_chars"]

    element, chars = restore(compact)
    np.testing.assert_array_equal(element, dense["ref_element"])
    np.testing.assert_array_equal(chars, dense["ref_atom_name_chars"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda features: features["ref_element"].astype(np.float32),
        lambda features: features["ref_atom_name_chars"].astype(np.float16),
        lambda features: np.full((3, 128), np.nan, dtype=np.float32),
        lambda features: np.full((3, 128), -0.0, dtype=np.float32),
        lambda features: np.full((3, 128), -1, dtype=np.int64),
        lambda features: np.full((3, 128), 2, dtype=np.int64),
    ],
)
def test_nonexact_or_nonint64_categories_fall_back_to_dense(mutate) -> None:
    dense = _dense_categories()
    value = mutate(dense)
    if value.shape == dense["ref_element"].shape:
        dense["ref_element"] = value
    else:
        dense["ref_atom_name_chars"] = value
    compact = compact_ref_atom_category_storage(dense)

    assert compact["ref_element"] is dense["ref_element"]
    assert compact["ref_atom_name_chars"] is dense["ref_atom_name_chars"]
    assert not (set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) & compact.keys())


@pytest.mark.parametrize(
    "field,value",
    [
        ("ref_element", np.zeros((3, 127), dtype=np.int64)),
        ("ref_atom_name_chars", np.zeros((3, 3, 64), dtype=np.int64)),
        ("ref_atom_name_chars", np.zeros((3, 4, 63), dtype=np.int64)),
        ("ref_element", np.zeros((128,), dtype=np.int64)),
    ],
)
def test_wrong_dense_category_shape_falls_back(field: str, value: np.ndarray) -> None:
    dense = _dense_categories()
    dense[field] = value
    compact = compact_ref_atom_category_storage(dense)
    assert compact[field] is value
    assert "ref_element" in compact
    assert "ref_atom_name_chars" in compact


def test_bfloat16_custom_categories_fall_back_to_dense() -> None:
    dense = _dense_categories()
    dense["ref_element"] = jnp.asarray(dense["ref_element"], dtype=jnp.bfloat16)
    compact = compact_ref_atom_category_storage(dense)
    assert compact["ref_element"] is dense["ref_element"]
    assert compact["ref_atom_name_chars"] is dense["ref_atom_name_chars"]


def test_dense_categories_win_and_strip_stale_private_provenance() -> None:
    dense = _dense_categories()
    dense.update(
        {
            COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(99, dtype=np.int16),
            COMPACT_REF_ELEMENT_IDS: np.asarray([255], dtype=np.uint8),
        }
    )
    compact = compact_ref_atom_category_storage(dense)

    assert set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) <= compact.keys()
    assert int(compact[COMPACT_REF_ATOM_CATEGORIES_MARKER]) == 1
    assert compact[COMPACT_REF_ELEMENT_IDS].shape == (3,)


@pytest.mark.parametrize(
    "private,match",
    [
        ({COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8)}, "incomplete"),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(2, np.uint8),
                COMPACT_REF_ELEMENT_IDS: np.zeros(3, np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.zeros((3, 4), np.uint8),
            },
            "v1",
        ),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8),
                COMPACT_REF_ELEMENT_IDS: np.asarray([129], np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.zeros((1, 4), np.uint8),
            },
            "sentinel 128",
        ),
    ],
)
def test_marker_bearing_compact_provenance_is_strict(private, match: str) -> None:
    with pytest.raises((KeyError, ValueError), match=match):
        validate_compact_ref_atom_categories(private)
    with pytest.raises((KeyError, ValueError), match=match):
        compact_ref_atom_category_storage(private)


def test_compact_inputs_have_a_distinct_small_jit_argument_signature() -> None:
    compact = compact_ref_atom_category_storage(_dense_categories())

    @jax.jit
    def category_projection(features):
        restored = _restore_ref_atom_category_one_hot(features)
        return restored["ref_element"].sum() + restored["ref_atom_name_chars"].sum()

    lowered = category_projection.lower(compact)
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    assert "tensor<3xui8>" in hlo
    assert "tensor<3x4xui8>" in hlo
    assert "tensor<3x128xi64>" not in hlo
    assert "tensor<3x4x64xi64>" not in hlo
