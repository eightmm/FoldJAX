from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models._feature_storage import compact_msa_storage
from foldjax.models.boltz2.models.trunk_blocks.msa import _msa_input_embedding


def _model_msa_features() -> dict[str, object]:
    shape = (2, 4, 5)
    return {
        "msa": (np.arange(np.prod(shape), dtype=np.int64) % 32).reshape(shape),
        "has_deletion": (np.arange(np.prod(shape)) % 2).reshape(shape).astype(
            np.float32
        ),
        "deletion_value": np.linspace(0.0, 1.0, np.prod(shape), dtype=np.float32)
        .reshape(shape),
        "msa_mask": np.ones(shape, dtype=np.int64),
        "msa_paired": np.zeros(shape, dtype=np.float32),
        "msa_attention_mask": np.ones(shape, dtype=np.float32),
        "msa_loop_tape": (
            np.arange(2 * np.prod(shape), dtype=np.int32) % 33
        ).reshape((2, *shape)),
        "has_deletion_loop_tape": np.zeros((2, *shape), dtype=np.float32),
        "msa_attention_mask_loop_tape": np.ones((2, *shape), dtype=np.float32),
        "metadata": object(),
    }


def test_compact_msa_storage_narrows_only_exact_categorical_fields() -> None:
    features = _model_msa_features()

    compact = compact_msa_storage(features)

    assert compact is not features
    assert compact["msa"].dtype == np.uint8
    for name in (
        "has_deletion",
        "msa_mask",
        "msa_paired",
        "msa_attention_mask",
        "has_deletion_loop_tape",
        "msa_attention_mask_loop_tape",
    ):
        assert compact[name].dtype == np.bool_
        np.testing.assert_array_equal(compact[name], features[name])
    np.testing.assert_array_equal(compact["msa"], features["msa"])
    assert compact["msa_loop_tape"].dtype == np.uint8
    np.testing.assert_array_equal(compact["msa_loop_tape"], features["msa_loop_tape"])
    assert compact["deletion_value"] is features["deletion_value"]
    assert compact["metadata"] is features["metadata"]

    original_bytes = sum(
        features[name].nbytes
        for name in (
            "msa",
            "msa_loop_tape",
            "has_deletion",
            "deletion_value",
            "msa_mask",
            "msa_paired",
            "msa_attention_mask",
            "has_deletion_loop_tape",
            "msa_attention_mask_loop_tape",
        )
    )
    compact_bytes = sum(
        compact[name].nbytes
        for name in (
            "msa",
            "msa_loop_tape",
            "has_deletion",
            "deletion_value",
            "msa_mask",
            "msa_paired",
            "msa_attention_mask",
            "has_deletion_loop_tape",
            "msa_attention_mask_loop_tape",
        )
    )
    assert compact_bytes * 2 < original_bytes


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("msa", np.asarray([[-1, 0]], dtype=np.int64)),
        ("msa", np.asarray([[0, 256]], dtype=np.int64)),
        ("msa", np.asarray([[0.0, 1.0]], dtype=np.float32)),
        ("msa", np.empty((0, 3), dtype=np.int64)),
        ("msa_loop_tape", np.asarray([[[-1, 0]]], dtype=np.int64)),
        ("has_deletion", np.asarray([[0.0, 0.5]], dtype=np.float32)),
        ("has_deletion", np.asarray([[0.0, -0.0]], dtype=np.float32)),
        ("msa_mask", np.asarray([[1.0, np.nan]], dtype=np.float32)),
        ("msa_paired", np.asarray([[0, 2]], dtype=np.int64)),
        ("msa_paired", np.asarray([[0.0 + 0.0j, 1.0 + 0.0j]])),
        ("msa_paired", np.asarray([[object(), object()]], dtype=object)),
    ],
)
def test_compact_msa_storage_falls_back_by_identity_for_custom_values(
    name: str,
    value: np.ndarray,
) -> None:
    features = _model_msa_features()
    features[name] = value

    assert compact_msa_storage(features) is features


def test_compact_msa_storage_does_not_copy_already_compact_arrays() -> None:
    features = _model_msa_features()
    features["msa"] = features["msa"].astype(np.uint8)
    features["msa_loop_tape"] = features["msa_loop_tape"].astype(np.uint8)
    for name in (
        "has_deletion",
        "msa_mask",
        "msa_paired",
        "msa_attention_mask",
        "has_deletion_loop_tape",
        "msa_attention_mask_loop_tape",
    ):
        features[name] = features[name].astype(bool)

    compact = compact_msa_storage(features)

    for name in (
        "msa",
        "msa_loop_tape",
        "has_deletion",
        "msa_mask",
        "msa_paired",
        "msa_attention_mask",
        "has_deletion_loop_tape",
        "msa_attention_mask_loop_tape",
    ):
        assert compact[name] is features[name]


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_boltz_msa_projection_preserves_compact_storage_arithmetic(dtype) -> None:
    rng = np.random.default_rng(19)
    batch, rows, tokens, c_s, c_m = 2, 3, 5, 7, 11
    alphabet = 33
    params = {
        "msa_proj": {
            "kernel": jnp.asarray(
                rng.normal(size=(alphabet + 3, c_m)), dtype=dtype
            )
        },
        "s_proj": {
            "kernel": jnp.asarray(rng.normal(size=(c_s, c_m)), dtype=dtype)
        },
    }
    emb = jnp.asarray(rng.normal(size=(batch, tokens, c_s)), dtype=dtype)
    msa = rng.integers(0, alphabet, size=(batch, rows, tokens), dtype=np.int32)
    has_deletion = rng.integers(0, 2, size=msa.shape).astype(np.float32)
    deletion_value = rng.normal(size=msa.shape).astype(np.float32)
    msa_paired = rng.integers(0, 2, size=msa.shape).astype(np.float32)

    with jax.numpy_dtype_promotion("strict"):
        historical = _msa_input_embedding(
            params,
            emb,
            jnp.asarray(msa),
            jnp.asarray(has_deletion),
            jnp.asarray(deletion_value),
            jnp.asarray(msa_paired),
            num_tokens=alphabet,
        )
        compact = _msa_input_embedding(
            params,
            emb,
            jnp.asarray(msa.astype(np.uint8)),
            jnp.asarray(has_deletion.astype(bool)),
            jnp.asarray(deletion_value),
            jnp.asarray(msa_paired.astype(bool)),
            num_tokens=alphabet,
        )

    np.testing.assert_array_equal(np.asarray(compact), np.asarray(historical))
