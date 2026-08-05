"""EDM sampler rollout (AF3 Algorithm 18).

A first-order Euler rollout over the noise schedule with optional noise inflation
(``gamma``) per step. Three details are transcribed deliberately:

* ``gamma`` is applied only while the *next* noise level exceeds ``gamma_min``,
  so the last steps run without extra churn.
* The step direction uses ``xl_noisy``, not the pre-noise ``xl``. Upstream flags
  this as an intentional deviation from the AF3 SI, following the EDM paper.
* ``dt`` is ``c_tau - t``, i.e. relative to the *inflated* ``t``, not to
  ``noise_schedule[tau]``.

Randomness is explicit: JAX has no global RNG, so the caller passes a key. The
centre/random-augmentation step is injected as a callable because it is a
structural operation on coordinates, independent of the sampler's schedule logic.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def sample_diffusion(
    key: jax.Array,
    noise_schedule: jnp.ndarray,
    shape: tuple[int, ...],
    denoise_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    *,
    gamma_0: float,
    gamma_min: float,
    noise_scale: float,
    step_scale: float,
    augment_fn: Callable[[jax.Array, jnp.ndarray], jnp.ndarray] | None = None,
    noise_fn: Callable[[int, tuple[int, ...]], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Roll out the EDM sampler.

    Args:
        key: PRNG key.
        noise_schedule: ``[no_steps + 1]`` descending noise levels.
        shape: shape of the coordinate tensor to sample, e.g.
            ``(n_sample, N_atom, 3)``.
        denoise_fn: ``(xl_noisy, t) -> xl_denoised``; close over the batch,
            conditioning and parameters before calling.
        gamma_0: noise inflation factor.
        gamma_min: inflate only while the next noise level exceeds this.
        noise_scale: multiplier on the injected noise.
        step_scale: multiplier on the Euler step.
        augment_fn: ``(key, xl) -> xl``, applied at the top of every step.
            ``None`` skips augmentation.
        noise_fn: ``(step, shape) -> noise``, where step ``0`` is the initial draw
            and step ``tau + 1`` is the injection before rollout step ``tau``.
            Supplied to compare the rollout against another implementation's
            random stream, which cannot otherwise be matched; ``None`` uses
            ``key``.

    Returns:
        Coordinates of shape ``shape``.
    """
    n_steps = noise_schedule.shape[0] - 1
    # Split once up front: typed PRNG keys are opaque, so they are split into
    # per-step arrays rather than reshaped.
    init_key, noise_root, augment_root = jax.random.split(key, 3)
    noise_keys = jax.random.split(noise_root, n_steps)
    augment_keys = jax.random.split(augment_root, n_steps)

    if noise_fn is None:
        injected = None
        xl = noise_schedule[0] * jax.random.normal(init_key, shape)
    else:
        # Materialize the injected draws so the rollout can be scanned. The
        # callback is a pure function of the step index, so this changes only when
        # it is called, not what it returns.
        injected = jnp.stack([noise_fn(step, shape) for step in range(n_steps + 1)])
        xl = noise_schedule[0] * injected[0]

    def step(xl: jnp.ndarray, carry) -> tuple[jnp.ndarray, None]:
        previous, c_tau, step_noise, noise_key, augment_key = carry

        if augment_fn is not None:
            xl = augment_fn(augment_key, xl)

        # Inflate the noise level, but only while there is schedule left to run.
        gamma = jnp.where(c_tau > gamma_min, gamma_0, 0.0)
        t = previous * (gamma + 1.0)

        drawn = (
            jax.random.normal(noise_key, shape) if injected is None else step_noise
        )
        xl_noisy = xl + (
            noise_scale * jnp.sqrt(jnp.maximum(t**2 - previous**2, 0.0)) * drawn
        )

        xl_denoised = denoise_fn(xl_noisy, jnp.atleast_1d(t))

        # Deliberately from xl_noisy, and dt relative to the inflated t.
        delta = (xl_noisy - xl_denoised) / t
        dt = c_tau - t
        return xl_noisy + step_scale * dt * delta, None

    # A scan rather than a Python loop: the released rollout is 200 steps over a
    # 24-block transformer, and unrolling that under jit produces a graph whose
    # compile time dominates everything else.
    xl, _ = jax.lax.scan(
        step,
        xl,
        (
            noise_schedule[:-1],
            noise_schedule[1:],
            injected[1:] if injected is not None else jnp.zeros((n_steps,)),
            noise_keys,
            augment_keys,
        ),
    )
    return xl
