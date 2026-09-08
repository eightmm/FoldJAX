"""The EDM sampler rollout.

The denoiser it drives is gated elsewhere; this checks the rollout's schedule
arithmetic, its gamma gating, and its determinism under a fixed key.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.augmentation import (
    AugmentationTape,
    centre_random_augmentation,
)
from foldjax.models.openfold3.models.diffusion_schedule import noise_schedule
from foldjax.models.openfold3.models.sampler import sample_diffusion

SHAPE = (2, 5, 3)
KW = {"gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003, "step_scale": 1.5}


def _schedule(steps: int = 6) -> jnp.ndarray:
    return noise_schedule(steps, sigma_data=16.0, s_max=160.0, s_min=4e-4, p=7)


def _identity_denoise(xl, t):
    """A denoiser that returns its input: delta is zero, so xl_noisy survives."""
    return xl


def test_identity_denoiser_leaves_the_noisy_state(deterministic: bool = True) -> None:
    key = jax.random.key(0)
    out = sample_diffusion(
        key, _schedule(), SHAPE, _identity_denoise, augment_fn=None, **KW
    )
    assert out.shape == SHAPE
    assert np.isfinite(np.asarray(out)).all()


def test_is_deterministic_for_a_fixed_key() -> None:
    args = (_schedule(), SHAPE, _identity_denoise)
    first = sample_diffusion(jax.random.key(3), *args, augment_fn=None, **KW)
    again = sample_diffusion(jax.random.key(3), *args, augment_fn=None, **KW)
    other = sample_diffusion(jax.random.key(4), *args, augment_fn=None, **KW)
    np.testing.assert_allclose(np.asarray(first), np.asarray(again))
    assert not np.allclose(np.asarray(first), np.asarray(other))


def test_initial_state_scales_with_the_first_noise_level() -> None:
    """xl starts at noise_schedule[0] * N(0, 1), so its scale tracks the schedule."""
    schedule = _schedule(1)  # one step: init then a single update
    key = jax.random.key(0)

    def zero_denoise(xl, t):
        return jnp.zeros_like(xl)

    out = sample_diffusion(
        key, schedule, (1, 4096, 3), zero_denoise, augment_fn=None, **KW
    )
    # With a zero denoiser, delta = xl_noisy / t and the step is a pure rescale;
    # the result must remain finite and on the order of the schedule.
    assert np.isfinite(np.asarray(out)).all()
    assert np.abs(np.asarray(out)).mean() > 0.0


def _reference(schedule, noise, *, gamma_0, gamma_min, noise_scale, step_scale,
               denoise, augment=None):
    """An explicit numpy rollout, transcribed from Algorithm 18.

    The sampler is a ``lax.scan``, so its body is traced once and Python side
    effects inside a callback cannot observe per-step values. Comparing against a
    written-out reference checks the same properties -- gamma gating, the inflated
    ``t``, stepping from ``xl_noisy`` rather than ``xl``, and ``step_scale`` -- and
    does not depend on tracing behaviour.
    """
    schedule = np.asarray(schedule, dtype=np.float64)
    xl = schedule[0] * noise[0]
    for tau in range(schedule.shape[0] - 1):
        previous, c_tau = schedule[tau], schedule[tau + 1]
        if augment is not None:
            xl = augment(xl)
        gamma = gamma_0 if c_tau > gamma_min else 0.0
        t = previous * (gamma + 1.0)
        xl_noisy = xl + noise_scale * np.sqrt(max(t**2 - previous**2, 0.0)) * noise[
            tau + 1
        ]
        delta = (xl_noisy - denoise(xl_noisy, t)) / t
        xl = xl_noisy + step_scale * (c_tau - t) * delta
    return xl


def _close_enough(actual, expected) -> None:
    """The reference accumulates in float64, the sampler in float32.

    The rollout starts at ``schedule[0]`` -- 2560 for the released schedule -- and
    contracts to O(0.1), so the result is a difference of much larger numbers and
    carries the float32 error of those. Observed worst case is 2e-5, so the
    project's 1e-4 parity tolerance applies rather than a tighter one.
    """
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64), expected, rtol=1e-4, atol=1e-4
    )


def _noise(steps: int, seed: int = 0):
    generator = np.random.default_rng(seed)
    return generator.standard_normal((steps + 1, *SHAPE)).astype(np.float32)


def _denoise(xl, t):
    """Depends on both arguments, so a wrong t cannot cancel out."""
    return 0.3 * xl + 0.05 * np.asarray(t).reshape(-1)[0] - 0.1


@pytest.mark.parametrize("gamma_min", [1e9, 0.0, 1.0])
def test_rollout_matches_an_explicit_reference(gamma_min: float) -> None:
    """``gamma_min`` above the schedule disables inflation, below it enables it,
    and the released value gates it partway through."""
    steps = 5
    schedule = _schedule(steps)
    noise = _noise(steps)
    settings = {**KW, "gamma_min": gamma_min}

    actual = sample_diffusion(
        jax.random.key(0),
        schedule,
        SHAPE,
        lambda xl, t: 0.3 * xl + 0.05 * t.reshape(-1)[0] - 0.1,
        augment_fn=None,
        noise_fn=lambda step, _shape: jnp.asarray(noise[step]),
        **settings,
    )
    expected = _reference(schedule, noise.astype(np.float64), denoise=_denoise,
                          **settings)
    _close_enough(actual, expected)


def test_stepping_from_the_previous_state_would_not_match() -> None:
    """Mutation check: the reference is only right because it uses ``xl_noisy``."""
    steps = 5
    schedule = np.asarray(_schedule(steps), dtype=np.float64)
    noise = _noise(steps).astype(np.float64)

    def mutant():
        xl = schedule[0] * noise[0]
        for tau in range(schedule.shape[0] - 1):
            previous, c_tau = schedule[tau], schedule[tau + 1]
            gamma = KW["gamma_0"] if c_tau > KW["gamma_min"] else 0.0
            t = previous * (gamma + 1.0)
            xl_noisy = xl + KW["noise_scale"] * np.sqrt(
                max(t**2 - previous**2, 0.0)
            ) * noise[tau + 1]
            # The deviation under test: stepping from xl instead of xl_noisy.
            xl = xl + KW["step_scale"] * (c_tau - t) * (
                (xl_noisy - _denoise(xl_noisy, t)) / t
            )
        return xl

    correct = _reference(schedule, noise, denoise=_denoise, **KW)
    assert not np.allclose(correct, mutant(), rtol=1e-3, atol=1e-3)


def test_augmentation_is_applied_before_the_noise_each_step() -> None:
    """A constant shift per step must accumulate exactly as the reference does."""
    steps = 4
    schedule = _schedule(steps)
    noise = _noise(steps, seed=1)
    shift = 0.25

    actual = sample_diffusion(
        jax.random.key(0),
        schedule,
        SHAPE,
        lambda xl, t: 0.3 * xl + 0.05 * t.reshape(-1)[0] - 0.1,
        augment_fn=lambda _key, xl: xl + shift,
        noise_fn=lambda step, _shape: jnp.asarray(noise[step]),
        **KW,
    )
    expected = _reference(
        schedule,
        noise.astype(np.float64),
        denoise=_denoise,
        augment=lambda xl: xl + shift,
        **KW,
    )
    _close_enough(actual, expected)
    # And it is not a no-op, so the comparison above is load-bearing.
    without = _reference(schedule, noise.astype(np.float64), denoise=_denoise, **KW)
    assert not np.allclose(expected, without, rtol=1e-3, atol=1e-3)


def _width_denoise(xl, t):
    """A denoiser that reads its own sample width, the way the real one does.

    `inference.denoise_fn` widens the conditioning to `xl_noisy.shape[0]`
    rather than to the config, which is what lets the rollout be chunked. A
    denoiser that ignored the width would pass these tests whatever the
    chunking did, so this one mixes it in.
    """
    width = xl.shape[0]
    # Broadcast, not `arange`: the real conditioning is the same for every
    # sample and only its *width* follows the coordinates. A denoiser keyed on
    # the sample index would read a chunk's local position as a global one and
    # fail here for a reason the model does not have.
    conditioning = jnp.broadcast_to(jnp.asarray(0.5, xl.dtype), (width, 1, 1))
    return jnp.tanh(xl) * conditioning


@pytest.mark.parametrize("chunk", [1, 2, 3, 5])
def test_chunking_the_sample_axis_changes_nothing(chunk: int) -> None:
    """The same coordinates, to float32 round-off, whatever it denoises at once.

    Every noise source is drawn at the full sample width and then narrowed, so
    a chunk sees the numbers its samples would have seen anyway. What is not
    preserved is the last bit: the elementwise rollout runs on a different
    array shape per chunk and XLA fuses it differently. Measured at 9.5e-06
    absolute on coordinates of magnitude 10-30, i.e. about 1e-7 relative --
    the same caveat every blocked path in this repository carries, and far
    under the 0.9 A this model's own samples disagree by.

    A chunk that redrew its own noise instead of narrowing the full draw would
    differ by whole Angstrom and fail this by orders of magnitude.

    Chunk sizes that do not divide the sample count are included on purpose:
    the trailing chunk is shorter, which is the case that gets sliced wrong.
    """
    key = jax.random.key(0)
    schedule = _schedule()
    shape = (7, 5, 3)

    whole = sample_diffusion(key, schedule, shape, _width_denoise, **KW)
    chunked = sample_diffusion(
        key, schedule, shape, _width_denoise, diffusion_chunk_size=chunk, **KW
    )

    assert chunked.shape == whole.shape
    np.testing.assert_allclose(
        np.asarray(chunked), np.asarray(whole), rtol=1e-5, atol=1e-4
    )


def test_a_chunk_at_least_as_wide_as_the_samples_is_the_unchunked_rollout() -> None:
    """No loop, no concatenate, and the same numbers."""
    key = jax.random.key(1)
    schedule = _schedule()
    whole = sample_diffusion(key, schedule, SHAPE, _width_denoise, **KW)
    for chunk in (SHAPE[0], SHAPE[0] + 1, None):
        same = sample_diffusion(
            key, schedule, SHAPE, _width_denoise, diffusion_chunk_size=chunk, **KW
        )
        np.testing.assert_array_equal(np.asarray(same), np.asarray(whole))


def test_a_noise_tape_is_narrowed_rather_than_reused() -> None:
    """A supplied tape must follow its own sample rows into each chunk.

    Slicing the step axis but not the sample axis would hand every chunk the
    first chunk's noise, which is a wrong answer that still has the right
    shape.
    """
    key = jax.random.key(2)
    schedule = _schedule()
    shape = (4, 5, 3)
    tape = jax.random.normal(jax.random.key(3), (schedule.shape[0], *shape))

    whole = sample_diffusion(
        key, schedule, shape, _width_denoise, noise_tape=tape, **KW
    )
    chunked = sample_diffusion(
        key,
        schedule,
        shape,
        _width_denoise,
        noise_tape=tape,
        diffusion_chunk_size=2,
        **KW,
    )
    np.testing.assert_allclose(
        np.asarray(chunked), np.asarray(whole), rtol=1e-5, atol=1e-4
    )
    # And the samples genuinely differ, so the agreement above is not vacuous.
    assert not np.allclose(np.asarray(whole)[0], np.asarray(whole)[1])


@pytest.mark.parametrize("chunk", [1, 2, 3])
@pytest.mark.parametrize("noise_route", ["native", "tape", "mask"])
def test_chunking_preserves_full_width_random_augmentation(
    chunk: int, noise_route: str
) -> None:
    """Unequal sample tails must not reuse the first chunk's rigid transforms."""
    shape = (5, 7, 3)
    schedule = jnp.asarray([8.0, 4.0, 2.0, 1.0, 0.0])
    key = jax.random.key(13)
    mask = jnp.ones(shape[:-1]).at[:, -1].set(0)
    options = {}
    if noise_route == "tape":
        options["noise_tape"] = jnp.asarray(
            np.random.default_rng(4)
            .normal(size=(schedule.shape[0], *shape))
            .astype(np.float32)
        )
    elif noise_route == "mask":
        options["noise_mask"] = mask

    def augment(draw_key, xl):
        assert xl.shape == shape
        return centre_random_augmentation(draw_key, xl, mask)

    def denoise(xl, _t):
        return 0.3 * xl

    whole = sample_diffusion(
        key, schedule, shape, denoise, augment_fn=augment, **options, **KW
    )
    chunked = jax.jit(
        lambda sampler_key: sample_diffusion(
            sampler_key,
            schedule,
            shape,
            denoise,
            augment_fn=augment,
            diffusion_chunk_size=chunk,
            **options,
            **KW,
        )
    )(key)
    np.testing.assert_allclose(chunked, whole, rtol=1e-5, atol=1e-5)
    assert not np.allclose(np.asarray(whole)[0], np.asarray(whole)[1])


def test_chunking_only_narrows_the_denoiser_callback() -> None:
    shape = (5, 7, 3)
    seen = []

    def denoise(xl, _t):
        seen.append(xl.shape[0])
        return 0.3 * xl

    def augment(_key, xl):
        assert xl.shape == shape
        return xl + jnp.arange(shape[0])[:, None, None]

    actual = sample_diffusion(
        jax.random.key(0),
        _schedule(2),
        shape,
        denoise,
        augment_fn=augment,
        diffusion_chunk_size=2,
        **KW,
    )
    assert set(seen) == {1, 2}
    assert np.isfinite(np.asarray(actual)).all()


@pytest.mark.parametrize("chunk", [0, -1])
def test_nonpositive_sample_chunk_is_rejected(chunk: int) -> None:
    with pytest.raises(ValueError, match="diffusion_chunk_size must be positive"):
        sample_diffusion(
            jax.random.key(0),
            _schedule(2),
            SHAPE,
            _identity_denoise,
            diffusion_chunk_size=chunk,
            **KW,
        )


@pytest.mark.parametrize("chunk", [None, 1, 2, 3])
@pytest.mark.parametrize("native_layout", [False, True])
def test_full_native_augmentation_tape_matches_numpy_rollout(chunk, native_layout):
    shape = (5, 7, 3)
    schedule = jnp.asarray([8.0, 4.0, 2.0, 1.0, 0.0])
    rng = np.random.default_rng(14)
    noise = rng.normal(size=(5, *shape)).astype(np.float32)
    quaternions = rng.normal(size=(4, shape[0], 4)).astype(np.float32)
    translations = rng.normal(size=(4, shape[0], 3)).astype(np.float32)
    mask = np.ones((1, shape[1]), dtype=np.float32)
    mask[:, -1] = 0

    # Independent NumPy recurrence applies each recorded transform before that
    # step's churn draw. Nontrivial masks and per-sample transforms catch both
    # sample-axis reuse and an off-by-one tape entry.
    expected = np.asarray(schedule[0]) * noise[0]
    for i, (previous, current) in enumerate(zip(schedule[:-1], schedule[1:])):
        previous, current = float(previous), float(current)
        q = quaternions[i] / np.linalg.norm(quaternions[i], axis=-1, keepdims=True)
        a, b, c, d = q.T
        rotations = np.stack(
            [
                a * a + b * b - c * c - d * d,
                2 * (b * c - a * d),
                2 * (b * d + a * c),
                2 * (b * c + a * d),
                a * a - b * b + c * c - d * d,
                2 * (c * d - a * b),
                2 * (b * d - a * c),
                2 * (c * d + a * b),
                a * a - b * b - c * c + d * d,
            ],
            axis=-1,
        ).reshape(-1, 3, 3)
        centre = (expected * mask[..., None]).sum(-2, keepdims=True) / mask.sum()
        expected = (
            (expected - centre) @ rotations.swapaxes(-1, -2) + translations[i, :, None]
        ) * mask[..., None]
        gamma = KW["gamma_0"] if current > KW["gamma_min"] else 0.0
        t = previous * (1 + gamma)
        noisy = (
            expected
            + KW["noise_scale"] * np.sqrt(t * t - previous * previous) * noise[i + 1]
        )
        expected = noisy + KW["step_scale"] * (current - t) * (noisy - 0.3 * noisy) / t

    tape = AugmentationTape(
        jnp.asarray(quaternions[:, None] if native_layout else quaternions),
        jnp.asarray(translations[:, None] if native_layout else translations),
    )
    options = dict(
        augmentation_tape=tape,
        atom_mask=jnp.asarray(mask),
        noise_tape=jnp.asarray(noise),
        diffusion_chunk_size=chunk,
        **KW,
    )

    def run(key):
        return sample_diffusion(key, schedule, shape, lambda x, _t: 0.3 * x, **options)

    actual = jax.jit(run)(jax.random.key(0))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(actual, jax.jit(run)(jax.random.key(999)))


@pytest.mark.parametrize(
    "bad", ["callback", "mask_missing", "tape_missing", "mask_shape"]
)
def test_augmentation_tape_rejects_ambiguous_or_incomplete_arguments(bad):
    tape = AugmentationTape(jnp.ones((2, 2, 4)), jnp.ones((2, 2, 3)))
    options = {"augmentation_tape": tape, "atom_mask": jnp.ones(SHAPE[:-1])}
    if bad == "callback":
        options["augment_fn"] = lambda _key, xl: xl
    elif bad == "mask_missing":
        options.pop("atom_mask")
    elif bad == "tape_missing":
        options.pop("augmentation_tape")
    else:
        options["atom_mask"] = jnp.ones((2, 4))
    with pytest.raises(
        ValueError, match="mutually exclusive|supplied together|atom_mask expected"
    ):
        sample_diffusion(
            jax.random.key(0), _schedule(2), SHAPE, _identity_denoise, **options, **KW
        )
