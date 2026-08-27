"""OpenDDE diffusion sampler with per-step rigid augmentation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

from foldjax.models._random import masked_prefix_draw, supports_masked_prefix_draw
from foldjax.models.opendde.models.geometry import (
    centre_random_augmentation,
    uniform_random_rotations,
)


def make_padded_random_tapes(
    *,
    key: jax.Array,
    num_samples: int,
    n_steps: int,
    actual_atom: int,
    target_atom: int,
    batch_shape: Sequence[int] = (),
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate OpenDDE's real-size random stream, then pad its atom axis.

    Drawing directly at ``target_atom`` changes the flat RNG offsets for later
    samples.  This helper follows :func:`sample_diffusion`'s exact key split and
    real shapes first, so every real atom retains the default unpadded stream.
    Step noise is returned as one ``[steps, ..., samples, atoms, 3]`` array so
    the scanned sampler can consume it directly. Rotations and translations
    have no atom axis and are reused unchanged.
    """

    if num_samples < 1 or n_steps < 1 or actual_atom < 1:
        raise ValueError(
            "padded random tapes require positive sample/step/atom sizes"
        )
    if target_atom < actual_atom:
        raise ValueError(
            f"target_atom={target_atom} is smaller than actual_atom={actual_atom}"
        )
    from foldjax.models.protenix.data.padding import pad_atom_noise

    init_key, step_key, rotation_key, translation_key = jax.random.split(key, 4)
    real_shape = (*batch_shape, num_samples, actual_atom, 3)
    init_noise = jax.random.normal(init_key, real_shape, dtype=dtype)
    step_noises = jnp.stack(
        tuple(
            jax.random.normal(step_key_i, real_shape, dtype=dtype)
            for step_key_i in jax.random.split(step_key, n_steps)
        ),
        axis=0,
    )
    leading_shape = real_shape[:-2]
    rotations = uniform_random_rotations(
        rotation_key,
        (n_steps, *leading_shape),
    )
    translations = jax.random.normal(
        translation_key,
        (n_steps, *leading_shape, 3),
        dtype=jnp.float32,
    )
    return (
        pad_atom_noise(init_noise, actual=actual_atom, target=target_atom),
        pad_atom_noise(step_noises, actual=actual_atom, target=target_atom),
        rotations,
        translations,
    )


def _prefix_atom_normal(
    key: jax.Array,
    atom_mask: jnp.ndarray,
    *,
    leading_shape: Sequence[int],
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Draw one padded coordinate field from the compact atom prefix."""

    prefix_mask = jnp.broadcast_to(
        jnp.asarray(atom_mask, dtype=bool),
        (*tuple(leading_shape), atom_mask.shape[0]),
    )
    return masked_prefix_draw(
        lambda normal_key, shape: jax.random.normal(
            normal_key, shape, dtype=dtype
        ),
        key,
        prefix_mask,
        trailing_shape=(3,),
    )


def sample_diffusion(
    denoise_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    noise_schedule: jnp.ndarray,
    *,
    num_samples: int,
    n_atom: int,
    key: jax.Array | None,
    init_noise: jnp.ndarray | None = None,
    step_noises: jnp.ndarray | Sequence[jnp.ndarray] | None = None,
    rotations: jnp.ndarray | None = None,
    translations: jnp.ndarray | None = None,
    batch_shape: Sequence[int] = (),
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    dtype: jnp.dtype = jnp.float32,
    use_scan: bool = False,
    atom_mask: jnp.ndarray | None = None,
    preserve_prefix_rng: bool = False,
) -> jnp.ndarray:
    """Run OpenDDE's loop sampler with optional shared random tapes.

    ``use_scan`` rolls the step loop into ``lax.scan`` instead of writing one
    copy of the denoiser per step into the graph. Every other repeated stack in
    this port is already scanned; the sampler was the one that was not, and it
    is the one whose count is a user-facing knob. At the released 200-step
    schedule the unrolled form puts 200 copies of the structural refiner in a
    single executable, which is why OpenDDE was the slowest model here by far.
    """

    noise_schedule = jnp.asarray(noise_schedule, dtype=dtype)
    n_steps = int(noise_schedule.shape[0]) - 1
    if n_steps < 1:
        raise ValueError("noise schedule must contain at least two levels")
    if atom_mask is not None:
        atom_mask = jnp.asarray(atom_mask, dtype=dtype)
        if tuple(atom_mask.shape) != (n_atom,):
            raise ValueError(
                f"atom_mask expected shape {(n_atom,)}, got {atom_mask.shape}"
            )
    if preserve_prefix_rng:
        if not supports_masked_prefix_draw():
            raise ValueError(
                "prefix-preserving padding noise requires "
                "jax_default_prng_impl='threefry2x32' and "
                "jax_threefry_partitionable=True"
            )
        if atom_mask is None:
            raise ValueError("prefix-preserving padding noise requires atom_mask")
        if any(
            tape is not None
            for tape in (init_noise, step_noises, rotations, translations)
        ):
            raise ValueError(
                "prefix-preserving padding noise and explicit random tapes are "
                "mutually exclusive"
            )
    tapes_complete = all(
        tape is not None for tape in (init_noise, step_noises, rotations, translations)
    )
    if key is None and not tapes_complete:
        raise ValueError("sample_diffusion requires key or complete random tapes")

    if key is not None:
        init_key, step_key, rotation_key, translation_key = jax.random.split(key, 4)
    leading_shape = (*batch_shape, num_samples)

    def draw_normal(draw_key):
        if preserve_prefix_rng:
            return _prefix_atom_normal(
                draw_key,
                atom_mask,
                leading_shape=leading_shape,
                dtype=dtype,
            )
        return jax.random.normal(
            draw_key,
            (*leading_shape, n_atom, 3),
            dtype=dtype,
        )

    if init_noise is None:
        init_noise = draw_normal(init_key)
    else:
        init_noise = jnp.asarray(init_noise, dtype=dtype)
    expected_shape = (*batch_shape, num_samples, n_atom, 3)
    if tuple(init_noise.shape) != expected_shape:
        raise ValueError(
            f"init_noise expected shape {expected_shape}, got {init_noise.shape}"
        )
    if atom_mask is not None:
        init_noise = init_noise * atom_mask[..., None]

    leading_shape = init_noise.shape[:-2]
    step_noise_keys = preserve_prefix_rng and step_noises is None
    packed_step_noises = step_noises is not None and hasattr(step_noises, "shape")
    if step_noises is None:
        step_keys = jax.random.split(step_key, n_steps)
        step_noises = (
            step_keys
            if step_noise_keys
            else tuple(draw_normal(step_key_i) for step_key_i in step_keys)
        )
    elif packed_step_noises:
        step_noises = jnp.asarray(step_noises, dtype=dtype)
    else:
        step_noises = tuple(jnp.asarray(noise, dtype=dtype) for noise in step_noises)
    if packed_step_noises:
        expected_step_shape = (n_steps, *init_noise.shape)
        if tuple(step_noises.shape) != expected_step_shape:
            raise ValueError(
                f"packed step_noises expected shape {expected_step_shape}, "
                f"got {step_noises.shape}"
            )
    elif step_noise_keys:
        if len(step_noises) != n_steps:
            raise ValueError("step noise keys must match the number of steps")
    elif len(step_noises) != n_steps or any(
        noise.shape != init_noise.shape for noise in step_noises
    ):
        raise ValueError("step_noises must match the number of steps and init shape")

    if rotations is None:
        rotations = uniform_random_rotations(
            rotation_key,
            (n_steps, *leading_shape),
        )
    else:
        rotations = jnp.asarray(rotations, dtype=jnp.float32)
    expected_rotations = (n_steps, *leading_shape, 3, 3)
    if tuple(rotations.shape) != expected_rotations:
        raise ValueError(
            f"rotations expected shape {expected_rotations}, got {rotations.shape}"
        )

    if translations is None:
        translations = jax.random.normal(
            translation_key,
            (n_steps, *leading_shape, 3),
            dtype=jnp.float32,
        )
    else:
        translations = jnp.asarray(translations, dtype=jnp.float32)
    expected_translations = (n_steps, *leading_shape, 3)
    if tuple(translations.shape) != expected_translations:
        raise ValueError(
            "translations expected shape "
            f"{expected_translations}, got {translations.shape}"
        )

    x_l = noise_schedule[0] * init_noise

    def one_step(x_current, c_tau_last, c_tau, step_noise, rotation, translation):
        """One Algorithm 18 step. Shared by the rolled and unrolled paths."""
        augmented = centre_random_augmentation(
            x_current,
            num_samples=1,
            mask=atom_mask,
            rotations=rotation[..., None, :, :],
            translations=translation[..., None, :],
        )
        x_current = jnp.squeeze(augmented, axis=-3).astype(dtype)
        gamma = jnp.where(c_tau > gamma_min, gamma0, 0.0).astype(dtype)
        t_hat_scalar = c_tau_last * (gamma + 1.0)
        # `t_hat**2 - c**2` is a difference of nearly equal squares and exactly
        # zero when gamma is zero, so XLA's contraction can leave it slightly
        # negative and `sqrt` returns NaN. The identity
        # `t_hat**2 - c**2 == c**2 * gamma * (gamma + 2)` avoids the
        # cancellation. Same defect and same fix as the Protenix sampler.
        delta_noise_level = c_tau_last * jnp.sqrt(gamma * (gamma + 2.0))
        x_noisy = x_current + noise_scale_lambda * delta_noise_level * step_noise
        if atom_mask is not None:
            x_noisy = x_noisy * atom_mask[..., None]
        t_hat = jnp.full(x_noisy.shape[:-2], t_hat_scalar, dtype=dtype)
        x_denoised = denoise_fn(x_noisy, t_hat)
        delta = (x_noisy - x_denoised) / t_hat[..., None, None]
        dt = c_tau - t_hat
        x_next = x_noisy + step_scale_eta * dt[..., None, None] * delta
        if atom_mask is not None:
            x_next = x_next * atom_mask[..., None]
        return x_next

    if use_scan:
        stacked_noises = (
            step_noises
            if packed_step_noises or step_noise_keys
            else jnp.stack(tuple(step_noises), axis=0)
        )

        def body(x_carry, xs):
            c_tau_last, c_tau, step_noise, rotation, translation = xs
            if step_noise_keys:
                step_noise = draw_normal(step_noise)
            return one_step(
                x_carry, c_tau_last, c_tau, step_noise, rotation, translation
            ), None

        x_l, _ = jax.lax.scan(
            body,
            x_l,
            xs=(
                noise_schedule[:-1],
                noise_schedule[1:],
                stacked_noises,
                rotations,
                translations,
            ),
        )
        return x_l

    for step_index in range(n_steps):
        step_noise = step_noises[step_index]
        if step_noise_keys:
            step_noise = draw_normal(step_noise)
        x_l = one_step(
            x_l,
            noise_schedule[step_index],
            noise_schedule[step_index + 1],
            step_noise,
            rotations[step_index],
            translations[step_index],
        )
    return x_l
