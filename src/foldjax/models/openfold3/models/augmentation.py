"""Centre-and-randomly-augment atom coordinates (AF3 Algorithm 19).

Centres on the masked centroid, applies a uniformly random rotation, adds a
random translation, then re-masks. Used by the sampler between steps so the
denoiser never sees a preferred global frame.

The rotation is sampled the same way as upstream: a Gaussian quaternion,
normalized, converted to a matrix. Sampling Euler angles instead would bias the
distribution, so the quaternion route is kept.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def quat_to_rot(quat: jnp.ndarray) -> jnp.ndarray:
    """Convert ``[..., 4]`` quaternions ``(a, b, c, d)`` to ``[..., 3, 3]`` matrices."""
    a, b, c, d = (quat[..., i] for i in range(4))
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    row0 = [aa + bb - cc - dd, 2 * (b * c - a * d), 2 * (b * d + a * c)]
    row1 = [2 * (b * c + a * d), aa - bb + cc - dd, 2 * (c * d - a * b)]
    row2 = [2 * (b * d - a * c), 2 * (c * d + a * b), aa - bb - cc + dd]
    return jnp.stack(
        [jnp.stack(row, axis=-1) for row in (row0, row1, row2)], axis=-2
    )


def sample_rotations(key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
    """Sample ``[*shape, 3, 3]`` rotations from normalized Gaussian quaternions."""
    quat = jax.random.normal(key, (*shape, 4))
    quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
    return quat_to_rot(quat)


def centre_random_augmentation(
    key: jax.Array,
    xl: jnp.ndarray,
    atom_mask: jnp.ndarray,
    *,
    scale_trans: float = 1.0,
) -> jnp.ndarray:
    """Centre, randomly rotate and translate atom coordinates.

    Args:
        key: PRNG key.
        xl: ``[..., N_atom, 3]`` coordinates.
        atom_mask: ``[..., N_atom]`` atom mask; the centroid uses only real atoms.
        scale_trans: translation standard deviation.

    Returns:
        ``[..., N_atom, 3]`` augmented coordinates, re-masked.
    """
    batch_shape = xl.shape[:-2]
    rot_key, trans_key = jax.random.split(key)

    rots = sample_rotations(rot_key, batch_shape)
    trans = scale_trans * jax.random.normal(trans_key, (*batch_shape, 3))

    mask = atom_mask[..., None]
    centroid = jnp.sum(xl * mask, axis=-2, keepdims=True) / jnp.sum(
        mask, axis=-2, keepdims=True
    )

    centred = xl - centroid
    out = jnp.matmul(centred, jnp.swapaxes(rots, -1, -2)) + trans[..., None, :]
    return out * mask
