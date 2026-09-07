"""Native Algorithm 19 augmentation and sampler tape boundaries."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.models.diffusion.diffusion import (
    _uniform_random_rotations,
    centre_random_augmentation,
    sample_diffusion,
)


def _reference_augmentation(x, rotations, translations, mask=None):
    if mask is None:
        center = np.mean(x, axis=-2, keepdims=True)
    else:
        center = np.sum(x * mask[..., None], axis=-2, keepdims=True) / (
            np.sum(mask, axis=-1, keepdims=True)[..., None] + 1e-12
        )
    centered = (x - center).astype(np.float32)
    result = (
        np.stack(
            [
                rotations[..., axis, 0, None] * centered[..., 0]
                + rotations[..., axis, 1, None] * centered[..., 1]
                + rotations[..., axis, 2, None] * centered[..., 2]
                for axis in range(3)
            ],
            axis=-1,
        )
        + translations[..., None, :]
    )
    return result if mask is None else result * mask[..., None]


@pytest.mark.parametrize("masked", [False, True])
def test_explicit_augmentation_matches_native_scalar_formula(masked):
    coords = np.arange(2 * 5 * 4 * 3, dtype=np.float32).reshape(2, 5, 4, 3) / 7
    rotations = np.broadcast_to(
        np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32),
        (2, 5, 3, 3),
    )
    translations = np.arange(30, dtype=np.float32).reshape(2, 5, 3) / 11
    mask = np.asarray([1, 1, 1, 0], np.float32) if masked else None
    actual = centre_random_augmentation(
        jnp.asarray(coords), mask, rotations=rotations, translations=translations
    )
    expected = _reference_augmentation(coords, rotations, translations, mask)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_rotation_stays_scalar_fp32_under_matmul_policy():
    coords = jnp.arange(30, dtype=jnp.float32).reshape(2, 5, 3) / 7
    rotations = jnp.broadcast_to(jnp.eye(3), (2, 3, 3))
    translations = jnp.ones((2, 3), jnp.float32)

    def run(x, r, t):
        return centre_random_augmentation(x, rotations=r, translations=t)

    graph = str(jax.make_jaxpr(run)(coords, rotations, translations))
    assert "dot_general" not in graph
    with jax.default_matmul_precision("high"):
        high = jax.jit(run)(coords, rotations, translations)
    with jax.default_matmul_precision("highest"):
        highest = jax.jit(run)(coords, rotations, translations)
    np.testing.assert_array_equal(high, highest)
    assert (
        run(coords.astype(jnp.bfloat16), rotations, translations).dtype == jnp.float32
    )


def test_random_augmentation_is_reproducible_and_rigid_not_center_only():
    coords = jnp.broadcast_to(
        jnp.asarray([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], jnp.float32),
        (5, 4, 3),
    )
    first = centre_random_augmentation(coords, key=jax.random.key(19))
    repeated = centre_random_augmentation(coords, key=jax.random.key(19))
    other = centre_random_augmentation(coords, key=jax.random.key(20))
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)
    assert not np.allclose(np.mean(first, axis=-2), 0)
    expected_distances = np.sum(
        (np.asarray(coords)[..., :, None, :] - coords[..., None, :, :]) ** 2, axis=-1
    )
    distances = np.sum((first[..., :, None, :] - first[..., None, :, :]) ** 2, axis=-1)
    np.testing.assert_allclose(distances, expected_distances, atol=2e-6)


def test_ordinary_rotations_are_proper():
    rotations = _uniform_random_rotations(jax.random.key(91), (5, 7))
    gram = rotations @ jnp.swapaxes(rotations, -1, -2)
    np.testing.assert_allclose(
        gram, jnp.broadcast_to(jnp.eye(3), gram.shape), atol=1e-6
    )
    np.testing.assert_allclose(jnp.linalg.det(rotations), 1, atol=1e-6)


def _tape(num_samples=5, n_steps=7, batch=False):
    shape = ((2,) if batch else ()) + (num_samples, 4, 3)
    init = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100
    noises = np.stack(
        [np.full(shape, value / 9, np.float32) for value in range(n_steps)]
    )
    rotations = np.broadcast_to(
        np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32),
        (n_steps, *shape[:-2], 3, 3),
    ).copy()
    rotations[::2] = np.eye(3, dtype=np.float32)
    translation_shape = (n_steps, *shape[:-2], 3)
    translations = (
        np.arange(np.prod(translation_shape), dtype=np.float32).reshape(
            translation_shape
        )
        / 37
    )
    schedule = np.linspace(4, 0, n_steps + 1, dtype=np.float32)
    return init, noises, rotations, translations, schedule


@pytest.mark.parametrize("num_samples,batch", [(1, False), (5, False), (5, True)])
@pytest.mark.parametrize("chunk_size", [None, 2, 5])
@pytest.mark.parametrize("use_scan", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_sampler_consumes_each_sample_rigid_tape(
    num_samples, batch, chunk_size, use_scan, packed
):
    init, noises, rotations, translations, schedule = _tape(num_samples, batch=batch)
    mask = np.asarray([1, 1, 1, 0], np.float32)
    expected = schedule[0] * init * mask[:, None]
    for index in range(len(schedule) - 1):
        expected = _reference_augmentation(
            expected, rotations[index], translations[index], mask
        )
        # gamma0=0 and identity denoiser isolate actual augmentation consumers.
    actual = sample_diffusion(
        lambda x, _: x,
        jnp.asarray(schedule),
        num_samples=num_samples,
        n_atom=4,
        key=None,
        init_noise=jnp.asarray(init),
        step_noises=jnp.asarray(noises)
        if packed
        else tuple(jnp.asarray(x) for x in noises),
        rotations=jnp.asarray(rotations)
        if packed
        else tuple(jnp.asarray(x) for x in rotations),
        translations=jnp.asarray(translations)
        if packed
        else tuple(jnp.asarray(x) for x in translations),
        gamma0=0,
        diffusion_chunk_size=chunk_size,
        use_scan=use_scan,
        atom_mask=mask,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-6)
    np.testing.assert_array_equal(np.asarray(actual)[..., -1, :], 0)


@pytest.mark.parametrize("use_scan", [False, True])
def test_explicit_noise_still_gets_ordinary_augmentation(use_scan):
    init, noises, _, _, schedule = _tape(n_steps=2)
    kwargs = dict(
        denoise_fn=lambda x, _: x,
        noise_schedule=jnp.asarray(schedule),
        num_samples=5,
        n_atom=4,
        init_noise=jnp.asarray(init),
        step_noises=jnp.asarray(noises),
        gamma0=0,
        use_scan=use_scan,
    )
    actual = sample_diffusion(key=jax.random.key(9), **kwargs)
    repeated = sample_diffusion(key=jax.random.key(9), **kwargs)
    other = sample_diffusion(key=jax.random.key(10), **kwargs)
    np.testing.assert_array_equal(actual, repeated)
    assert not np.array_equal(actual, other)
    assert not np.allclose(np.mean(actual, axis=-2), 0)


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_ordinary_sampler_scan_matches_loop(chunk_size):
    kwargs = dict(
        denoise_fn=lambda x, t: x / (1 + t[..., None, None]),
        noise_schedule=jnp.asarray([4, 2, 0.5, 0], jnp.float32),
        num_samples=5,
        n_atom=4,
        key=jax.random.key(81),
        diffusion_chunk_size=chunk_size,
    )
    loop = sample_diffusion(use_scan=False, **kwargs)
    scan = sample_diffusion(use_scan=True, **kwargs)
    np.testing.assert_allclose(scan, loop, rtol=1e-5, atol=1e-5)


def test_missing_augmentation_rng_is_not_silently_center_only():
    init, noises, _, _, schedule = _tape(n_steps=2)
    with pytest.raises(ValueError, match="augmentation.*key|key.*augmentation"):
        sample_diffusion(
            lambda x, _: x,
            jnp.asarray(schedule),
            num_samples=5,
            n_atom=4,
            key=None,
            init_noise=jnp.asarray(init),
            step_noises=jnp.asarray(noises),
        )


@pytest.mark.parametrize(
    "bad", ["rotation_length", "translation_samples", "missing_pair", "disabled"]
)
def test_invalid_augmentation_tape_is_rejected(bad):
    init, noises, rotations, translations, schedule = _tape(n_steps=2)
    if bad == "rotation_length":
        rotations = rotations[:1]
    elif bad == "translation_samples":
        translations = translations[:, :2]
    elif bad == "missing_pair":
        translations = None
    with pytest.raises(ValueError, match="rotations|translations|centre_each_step"):
        sample_diffusion(
            lambda x, _: x,
            jnp.asarray(schedule),
            num_samples=5,
            n_atom=4,
            key=None,
            init_noise=jnp.asarray(init),
            step_noises=jnp.asarray(noises),
            rotations=rotations,
            translations=translations,
            centre_each_step=bad != "disabled",
        )
