"""Torch-vs-JAX parity for coordinate augmentation (AF3 Algorithm 19).

The rotation and translation are random, so parity is checked on the
deterministic parts: the quaternion-to-matrix conversion against upstream, and
the centring/masking invariants that must hold for any sampled transform.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.augmentation import (
    centre_random_augmentation,
    quat_to_rot,
    sample_rotations,
)

pytestmark = pytest.mark.torch_parity

N_ATOM = 7


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def test_quat_to_rot_matches_torch(openfold3_source: Path) -> None:
    torch = _torch()
    from openfold3.core.utils.rigid_utils import quat_to_rot as torch_quat_to_rot

    quat = torch.randn(4, 4)
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
    expected = torch_quat_to_rot(quat)
    actual = quat_to_rot(jnp.asarray(quat.numpy()))
    assert actual.shape == tuple(expected.shape)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_sampled_rotations_are_orthonormal(openfold3_source: Path) -> None:
    rots = np.asarray(sample_rotations(jax.random.key(0), (16,)))
    identity = np.einsum("...ij,...kj->...ik", rots, rots)
    np.testing.assert_allclose(
        identity, np.broadcast_to(np.eye(3), identity.shape), rtol=1e-5, atol=1e-5
    )
    dets = np.linalg.det(rots)
    np.testing.assert_allclose(dets, np.ones_like(dets), rtol=1e-5, atol=1e-5)


def test_augmentation_preserves_pairwise_distances(openfold3_source: Path) -> None:
    """A rigid transform cannot change internal geometry."""
    rng = np.random.default_rng(0)
    xl = jnp.asarray(rng.normal(size=(2, N_ATOM, 3)), dtype=jnp.float32)
    mask = jnp.ones((2, N_ATOM))
    out = centre_random_augmentation(jax.random.key(1), xl, mask)

    def dists(x):
        x = np.asarray(x)
        return np.linalg.norm(x[:, :, None, :] - x[:, None, :, :], axis=-1)

    np.testing.assert_allclose(dists(out), dists(xl), rtol=1e-4, atol=1e-4)


def test_centroid_uses_only_unmasked_atoms(openfold3_source: Path) -> None:
    """A wild masked coordinate must not drag the centroid."""
    xl = jnp.asarray(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1e6, 1e6, 1e6]]], dtype=jnp.float32
    )
    mask = jnp.asarray([[1.0, 1.0, 0.0]])
    out = centre_random_augmentation(
        jax.random.key(2), xl, mask, scale_trans=0.0
    )
    # With no translation, the two real atoms straddle the origin.
    real = np.asarray(out)[0, :2]
    np.testing.assert_allclose(real.mean(axis=0), np.zeros(3), rtol=1e-4, atol=1e-4)


def test_masked_atoms_are_zeroed(openfold3_source: Path) -> None:
    rng = np.random.default_rng(3)
    xl = jnp.asarray(rng.normal(size=(1, N_ATOM, 3)), dtype=jnp.float32)
    mask = jnp.asarray(
        np.concatenate([np.ones(4), np.zeros(3)])[None, :], dtype=jnp.float32
    )
    out = np.asarray(centre_random_augmentation(jax.random.key(4), xl, mask))
    assert np.allclose(out[:, 4:], 0.0)


def test_is_deterministic_for_a_fixed_key(openfold3_source: Path) -> None:
    rng = np.random.default_rng(5)
    xl = jnp.asarray(rng.normal(size=(1, N_ATOM, 3)), dtype=jnp.float32)
    mask = jnp.ones((1, N_ATOM))
    first = centre_random_augmentation(jax.random.key(7), xl, mask)
    again = centre_random_augmentation(jax.random.key(7), xl, mask)
    other = centre_random_augmentation(jax.random.key(8), xl, mask)
    np.testing.assert_allclose(np.asarray(first), np.asarray(again))
    assert not np.allclose(np.asarray(first), np.asarray(other))
