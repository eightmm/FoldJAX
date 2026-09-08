"""Centre-and-randomly-augment atom coordinates (AF3 Algorithm 19).

Centres on the masked centroid, applies a uniformly random rotation, adds a
random translation, then re-masks. Used by the sampler between steps so the
denoiser never sees a preferred global frame.

The rotation is sampled the same way as upstream: a Gaussian quaternion,
normalized, converted to a matrix. Sampling Euler angles instead would bias the
distribution, so the quaternion route is kept.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class AugmentationTape(NamedTuple):
    """Unnormalised quaternion and unscaled translation draws, in step order.

    Arrays use ``[steps, samples, 4]`` and ``[steps, samples, 3]``. Native
    OpenBind's batch-one ``[steps, 1, samples, ...]`` layout is also accepted;
    no sample sorting, quaternion sign change or dtype conversion is applied.
    Draws must be finite and quaternion norms nonzero. Public prediction entry
    points validate concrete values; direct traced sampler callers must run
    ``validate_augmentation_tape(..., check_values=True)`` before tracing.
    Value-validated replay supports FP32/FP64 draws only, not an unvalidated
    low-precision native policy. Gaussian quaternion magnitudes must also keep
    their sum of squares away from subnormal/overflow arithmetic.
    """

    quaternions: jnp.ndarray
    translations: jnp.ndarray


def validate_augmentation_tape(
    tape: AugmentationTape,
    *,
    steps: int,
    samples: int,
    dtype=None,
    check_values: bool = False,
) -> AugmentationTape:
    """Validate replay dimensions and optionally concrete finite draw values.

    Value checks require host-readable arrays and never insert device callbacks.
    The default static checks also work on tracers. Forward normalisation keeps
    the native norm order without an epsilon or clamp.
    """
    if not isinstance(tape, AugmentationTape):
        raise TypeError("augmentation_tape must be an AugmentationTape")
    result = []
    for name, value, width in zip(tape._fields, tape, (4, 3), strict=True):
        original_dtype = getattr(value, "dtype", None)
        value = jnp.asarray(value)
        if original_dtype is not None and value.dtype != original_dtype:
            raise TypeError(
                f"augmentation_tape.{name} dtype {original_dtype} cannot be "
                f"preserved by the active JAX dtype policy"
            )
        expected = (steps, samples, width)
        native = (steps, 1, samples, width)
        if value.shape == native:
            value = value[:, 0]
        if value.shape != expected:
            raise ValueError(
                f"augmentation_tape.{name} expected shape {expected} or {native}, "
                f"got {value.shape}"
            )
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError(f"augmentation_tape.{name} must have floating dtype")
        if dtype is not None and value.dtype != dtype:
            raise TypeError(
                f"augmentation_tape.{name} dtype {value.dtype} does not match "
                f"coordinate dtype {dtype}"
            )
        result.append(value)
    if result[0].dtype != result[1].dtype:
        raise TypeError("augmentation_tape draw dtypes must match")
    if check_values:
        if result[0].dtype not in (jnp.float32, jnp.float64):
            raise TypeError(
                "value-validated native augmentation tape requires FP32/FP64 draws"
            )
        for name, value in zip(tape._fields, result, strict=True):
            if isinstance(value, jax.core.Tracer):
                raise ValueError(
                    "augmentation tape value checks require concrete arrays"
                )
            host = np.asarray(value)
            if not np.isfinite(host).all():
                raise ValueError(f"augmentation_tape.{name} must contain finite draws")
            if name == "quaternions":
                # This is a rejection gate, not replacement arithmetic. The
                # forward path still performs its native, unclamped norm.
                norm_dtype = np.float64 if value.dtype == jnp.float64 else np.float32
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    norms = np.linalg.norm(host.astype(norm_dtype), axis=-1)
                if not np.isfinite(norms).all() or np.any(norms == 0):
                    raise ValueError(
                        "augmentation_tape.quaternions require finite nonzero norms"
                    )
                # XLA may flush subnormal products that NumPy retains. One
                # normal squared component prevents a zero device norm; the
                # upper margin prevents a four-term sum from overflowing.
                # Ordinary native Gaussian draws are far inside these bounds.
                limits = np.finfo(norm_dtype)
                magnitude = np.max(np.abs(host), axis=-1)
                if np.any(magnitude < np.sqrt(limits.tiny)) or np.any(
                    magnitude > np.sqrt(limits.max) / 4
                ):
                    raise ValueError(
                        "augmentation_tape.quaternions require finite nonzero "
                        "normal-range norm arithmetic"
                    )
    return AugmentationTape(*result)


def quat_to_rot(quat: jnp.ndarray) -> jnp.ndarray:
    """Convert ``[..., 4]`` quaternions ``(a, b, c, d)`` to ``[..., 3, 3]`` matrices."""
    a, b, c, d = (quat[..., i] for i in range(4))
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    row0 = [aa + bb - cc - dd, 2 * (b * c - a * d), 2 * (b * d + a * c)]
    row1 = [2 * (b * c + a * d), aa - bb + cc - dd, 2 * (c * d - a * b)]
    row2 = [2 * (b * d - a * c), 2 * (c * d + a * b), aa - bb - cc + dd]
    return jnp.stack([jnp.stack(row, axis=-1) for row in (row0, row1, row2)], axis=-2)


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
    return _centre_and_apply(xl, atom_mask, rots, trans)


def centre_augmentation_from_draws(
    xl: jnp.ndarray,
    atom_mask: jnp.ndarray,
    quaternions: jnp.ndarray,
    translations: jnp.ndarray,
) -> jnp.ndarray:
    """Apply native raw draws: normalise quaternion, centre, rotate, translate.

    Native sampler augmentation uses translation scale one. These are Gaussian
    draws, not already-normalised quaternions or pre-scaled translations.
    Direct traced calls require prevalidated draws: zero quaternions retain
    native undefined/nonfinite behaviour, rather than being clamped to identity.
    """
    batch_shape = xl.shape[:-2]
    if quaternions.shape != (*batch_shape, 4):
        raise ValueError("quaternion draw shape must match coordinate sample axes")
    if translations.shape != (*batch_shape, 3):
        raise ValueError("translation draw shape must match coordinate sample axes")
    quat = quaternions / jnp.linalg.norm(quaternions, axis=-1, keepdims=True)
    return _centre_and_apply(xl, atom_mask, quat_to_rot(quat), translations)


def _centre_and_apply(xl, atom_mask, rots, trans):
    mask = atom_mask[..., None]
    # Native c4771653 core/model/structure/augmentation.py:69 clamps mask count
    # to one; this is separate from its unclamped quaternion normalization.
    centroid = jnp.sum(xl * mask, axis=-2, keepdims=True) / jnp.maximum(
        jnp.sum(mask, axis=-2, keepdims=True), 1
    )

    centred = xl - centroid
    out = jnp.matmul(centred, jnp.swapaxes(rots, -1, -2)) + trans[..., None, :]
    return out * mask
