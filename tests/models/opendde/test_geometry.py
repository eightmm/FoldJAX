from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.opendde.models.geometry import (
    centre_random_augmentation,
    uniform_random_rotations,
)


def test_centre_random_augmentation_matches_shared_rigid_tape() -> None:
    coords = jnp.asarray(
        [[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [100.0, 0.0, 0.0]]],
        dtype=jnp.float32,
    )
    mask = jnp.asarray([[1.0, 1.0, 0.0]], dtype=jnp.float32)
    rotations = jnp.asarray(
        [
            [
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    translations = jnp.asarray([[[1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]]], dtype=jnp.float32)

    actual = centre_random_augmentation(
        coords,
        num_samples=2,
        mask=mask,
        rotations=rotations,
        translations=translations,
    )

    centered = np.asarray(coords) - np.asarray([[[2.0, 0.0, 0.0]]])
    expanded = np.broadcast_to(centered[:, None], (1, 2, 3, 3))
    expected = np.einsum("...sij,...saj->...sai", rotations, expanded)
    expected += np.asarray(translations)[..., None, :]
    expected *= np.asarray(mask)[:, None, :, None]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.shape == (1, 2, 3, 3)


def test_centre_only_repeats_masked_centered_coordinates() -> None:
    coords = jnp.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    mask = jnp.asarray([1.0, 1.0, 0.0], dtype=jnp.float32)

    actual = centre_random_augmentation(
        coords,
        num_samples=3,
        centre_only=True,
        mask=mask,
    )

    expected = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, np.broadcast_to(expected, (3, 3, 3)))


def test_uniform_random_rotations_are_proper_and_deterministic() -> None:
    key = jax.random.key(29)

    first = np.asarray(uniform_random_rotations(key, (2, 4)))
    second = np.asarray(uniform_random_rotations(key, (2, 4)))

    np.testing.assert_array_equal(first, second)
    identity = np.einsum("...ji,...jk->...ik", first, first)
    expected_identity = np.broadcast_to(np.eye(3), identity.shape)
    np.testing.assert_allclose(identity, expected_identity, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(first), 1.0, rtol=1e-5, atol=1e-5)


def test_random_augmentation_requires_key_or_shared_tape() -> None:
    with pytest.raises(ValueError, match="key"):
        centre_random_augmentation(jnp.zeros((2, 3), dtype=jnp.float32))
    with pytest.raises(ValueError, match="provided together"):
        centre_random_augmentation(
            jnp.zeros((2, 3), dtype=jnp.float32),
            rotations=jnp.eye(3)[None],
        )


def test_random_augmentation_from_key_is_deterministic() -> None:
    coords = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    key = jax.random.key(31)

    first = centre_random_augmentation(coords, num_samples=2, key=key)
    second = centre_random_augmentation(coords, num_samples=2, key=key)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 4, 3)


def test_rigid_augmentation_keeps_native_fp32_scalar_rotation() -> None:
    """Native rot_vec_mul explicitly avoids AMP/TF32 matrix multiplication."""
    coords = jnp.arange(30, dtype=jnp.float32).reshape(2, 5, 3)
    rotations = uniform_random_rotations(jax.random.key(7), (2, 1))
    translations = jnp.zeros((2, 1, 3), dtype=jnp.float32)

    def augment(x, r, t):
        return centre_random_augmentation(x, rotations=r, translations=t)

    assert "dot_general" not in str(
        jax.make_jaxpr(augment)(coords, rotations, translations)
    )
    results = []
    for precision in ("high", "highest"):
        with jax.default_matmul_precision(precision):
            results.append(jax.jit(augment)(coords, rotations, translations))
    np.testing.assert_array_equal(*results)
