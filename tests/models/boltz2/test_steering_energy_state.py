"""FK steering retains only the one energy vector its next step consumes."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.trunk_blocks import trunk


def test_incremental_state_matches_the_historical_growing_trajectory() -> None:
    energies = [
        jnp.asarray([1.0, -0.0, 3.0, -4.0], dtype=jnp.float32),
        jnp.asarray([0.5, 2.0, -1.0, 7.0], dtype=jnp.float32),
        jnp.asarray([-2.0, 4.0, 5.0, 1.0], dtype=jnp.float32),
    ]
    resamples = [
        jnp.asarray([3, 0, 0, 2], dtype=jnp.int32),
        jnp.asarray([1, 1, 3, 0], dtype=jnp.int32),
        jnp.asarray([2], dtype=jnp.int32),
    ]
    history = jnp.empty((4, 0), dtype=jnp.float32)
    previous = None

    for step, (energy, indices) in enumerate(zip(energies, resamples, strict=True)):
        history = jnp.concatenate((history, energy[:, None]), axis=1)
        historical = -energy if step == 0 else history[:, -2] - history[:, -1]
        compact = trunk._fk_energy_increment(previous, energy, step)

        assert np.asarray(compact).tobytes() == np.asarray(historical).tobytes()
        history = history[indices]
        previous = energy[indices]
        if step + 1 < len(energies):
            energies[step + 1] = energies[step + 1][: indices.shape[0]]

    assert history.shape == (1, 3)
    assert previous.shape == (1,)


def test_step_zero_resample_has_the_initial_fk_increment() -> None:
    energy = jnp.asarray([1.0, -0.0, jnp.inf, jnp.nan], dtype=jnp.float32)
    actual = trunk._fk_energy_increment(None, energy, 0)

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(-energy))
    finite = ~np.isnan(np.asarray(actual))
    assert np.asarray(actual)[finite].tobytes() == np.asarray(-energy)[
        finite
    ].tobytes()


def test_late_first_resample_preserves_historical_clamped_subtraction() -> None:
    energy = jnp.asarray([1.0, -0.0, jnp.inf, -jnp.inf, jnp.nan], dtype=jnp.float32)
    trajectory = energy[:, None]
    historical = trajectory[:, -2] - trajectory[:, -1]

    actual = trunk._fk_energy_increment(None, energy, 1)

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(historical))
    finite = ~np.isnan(np.asarray(historical))
    assert np.asarray(actual)[finite].tobytes() == np.asarray(historical)[
        finite
    ].tobytes()


def test_sampling_loop_no_longer_concatenates_an_energy_trajectory() -> None:
    source = inspect.getsource(trunk.boltz2_sample_forward)

    assert "energy_traj" not in source
    assert "_fk_energy_increment(previous_energy, energy, step_idx)" in source


def test_full_steering_loop_matches_the_historical_increment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Potential:
        @staticmethod
        def compute_parameters(_time):
            return {"resampling_weight": 1.0}

        @staticmethod
        def compute(coordinate, _features, _parameters):
            return jnp.sum(coordinate * coordinate, axis=(-1, -2))

    from foldjax.models.boltz2.models.heads import potentials

    monkeypatch.setattr(
        potentials,
        "get_potentials",
        lambda *_args, **_kwargs: [Potential()],
    )
    monkeypatch.setattr(
        trunk,
        "diffusion_conditioning_forward",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        trunk,
        "_preconditioned_score_forward",
        lambda *_args, r_noisy, **_kwargs: r_noisy * 0.75,
    )

    captured_logits: list[list[np.ndarray]] = [[], []]
    capture_index = 0
    historical_trajectory: list[jax.Array] | None = None

    def deterministic_resample(_key, logits, *, axis, shape):
        del axis
        captured_logits[capture_index].append(np.asarray(logits))
        groups, particles = logits.shape
        draws = shape[-1]
        order = jnp.flip(jnp.argsort(logits, axis=1), axis=1)
        selected = order[:, :draws].astype(jnp.int32)
        if historical_trajectory is not None:
            flat = (
                selected + particles * jnp.arange(groups)[:, None]
            ).reshape(-1)
            historical_trajectory[:] = [value[flat] for value in historical_trajectory]
        return selected

    monkeypatch.setattr(jax.random, "categorical", deterministic_resample)

    params = {
        "trunk": {},
        "conditioned_diffusion": {
            "diffusion_conditioning": {},
            "score_model": {},
        },
    }
    features = {
        "token_pad_mask": jnp.ones((1, 2), dtype=bool),
        "atom_pad_mask": jnp.ones((1, 4), dtype=bool),
    }
    supplied_trunk = {
        "s": jnp.zeros((1, 2, 4), dtype=jnp.float32),
        "z": jnp.zeros((1, 2, 2, 4), dtype=jnp.float32),
        "s_inputs": jnp.zeros((1, 2, 4), dtype=jnp.float32),
        "relative_position_encoding": jnp.zeros(
            (1, 2, 2, 4), dtype=jnp.float32
        ),
    }
    rng = np.random.default_rng(3)
    init_noise = jnp.asarray(rng.normal(size=(2, 4, 3)), dtype=jnp.float32)
    step_noises = jnp.asarray(rng.normal(size=(4, 2, 4, 3)), dtype=jnp.float32)
    steering = {
        "fk_steering": True,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
        "num_particles": 2,
        "fk_resampling_interval": 2,
        "fk_lambda": 1.0,
    }

    def run():
        return trunk.boltz2_sample_forward(
            params,
            features,
            jax.random.key(5),
            num_sampling_steps=4,
            multiplicity=1,
            augmentation=False,
            alignment_reverse_diff=False,
            steering_args=steering,
            init_noise=init_noise,
            step_noises=step_noises,
            use_scan=True,
            trunk=supplied_trunk,
        )["sample_atom_coords"]

    compact = run()
    history = []
    capture_index = 1
    historical_trajectory = history

    def historical_increment(_previous, energy, step_idx):
        history.append(energy)
        trajectory = jnp.stack(history, axis=-1)
        if step_idx == 0:
            return -energy
        return trajectory[:, -2] - trajectory[:, -1]

    monkeypatch.setattr(trunk, "_fk_energy_increment", historical_increment)
    historical = run()

    assert len(captured_logits[0]) == len(captured_logits[1]) == 2
    for compact_logits, historical_logits in zip(
        captured_logits[0], captured_logits[1], strict=True
    ):
        assert compact_logits.tobytes() == historical_logits.tobytes()
    assert np.asarray(compact).tobytes() == np.asarray(historical).tobytes()


def _one_hot_weights(samples: int, particles: int) -> tuple[jax.Array, np.ndarray]:
    """Weights that leave one particle per sample, a different one per sample."""

    winners = (2 * np.arange(samples) + 1) % particles
    weights = np.full((samples, particles), 1e-30, dtype=np.float32)
    weights[np.arange(samples), winners] = 1.0
    return jnp.asarray(weights), winners


@pytest.mark.parametrize("samples", [1, 2, 3, 5])
@pytest.mark.parametrize("particles", [2, 3])
def test_resampling_draws_every_sample_from_its_own_weights(
    samples: int, particles: int
) -> None:
    """Real categorical draws: a sample only ever receives its own particles.

    ``samples == particles`` is the case one batched call got wrong without
    raising, by giving every sample one draw from each sample's weights.
    """

    weights, winners = _one_hot_weights(samples, particles)
    offsets = particles * np.arange(samples)

    resampled = trunk._fk_resample_indices(jax.random.key(11), weights, particles)
    np.testing.assert_array_equal(
        np.asarray(resampled).reshape(samples, particles),
        np.broadcast_to((winners + offsets)[:, None], (samples, particles)),
    )

    # The last step keeps one particle per sample. A legacy key this time,
    # which is the kind the released API hands the sampler.
    final = trunk._fk_resample_indices(jax.random.PRNGKey(12), weights, 1)
    np.testing.assert_array_equal(np.asarray(final), winners + offsets)


@pytest.mark.parametrize("samples", [1, 2, 3, 5])
def test_resampling_follows_each_samples_distribution(samples: int) -> None:
    """Undecided weights still never select a particle of another sample."""

    particles = 4
    rng = np.random.default_rng(samples)
    weights = rng.uniform(0.1, 1.0, size=(samples, particles)).astype(np.float32)
    # One particle per sample is impossible; the draws must never return it.
    banned = rng.integers(0, particles, size=samples)
    weights[np.arange(samples), banned] = 0.0
    weights /= weights.sum(axis=1, keepdims=True)

    for n_draw in (particles, 1):
        seen = np.zeros((samples, particles), dtype=np.int64)
        for seed in range(40):
            flat = np.asarray(
                trunk._fk_resample_indices(
                    jax.random.key(seed), jnp.asarray(weights), n_draw
                )
            ).reshape(samples, n_draw)
            local = flat - particles * np.arange(samples)[:, None]
            assert local.min() >= 0 and local.max() < particles
            for sample in range(samples):
                np.add.at(seen[sample], local[sample], 1)
        assert not seen[np.arange(samples), banned].any()
        assert (seen.sum(axis=1) == 40 * n_draw).all()
        # Every particle with weight is reachable within its own sample.
        if n_draw == particles:
            assert ((seen > 0) == (weights > 0)).all()


def test_one_sample_keeps_its_historical_draws() -> None:
    weights = jnp.asarray([[0.2, 0.5, 0.3]], dtype=jnp.float32)
    key = jax.random.key(7)

    for n_draw in (3, 1):
        expected = jax.random.categorical(
            key, jnp.log(weights), axis=1, shape=(1, n_draw)
        ).reshape(-1)
        actual = trunk._fk_resample_indices(key, weights, n_draw)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("samples", [1, 2, 3, 5])
def test_full_steering_loop_resamples_within_each_sample(
    samples: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sampler end to end, with the real ``jax.random.categorical``.

    Every particle carries its sample and particle number as a ratio of two
    coordinates, which survives the loop's rescaling. The energy leaves one
    particle per sample, a different one for each, so the returned structures
    say which particle each sample kept -- through the intermediate resamples
    and the last step's one-particle selection alike.
    """

    particles = 3
    winners = (2 * np.arange(samples) + 1) % particles
    seen: list[np.ndarray] = []

    def identity(coordinate):
        scale = coordinate[:, 0, 0]
        sample = jnp.rint(coordinate[:, 1, 0] / scale) - 1
        particle = jnp.rint(coordinate[:, 1, 1] / scale) - 1
        return sample, particle

    class Potential:
        @staticmethod
        def compute_parameters(_time):
            return dict(resampling_weight=1.0)

        @staticmethod
        def compute(coordinate, _features, _parameters):
            sample, particle = identity(coordinate)
            seen.append(np.asarray(sample))
            winner = jnp.mod(2 * sample + 1, particles)
            return jnp.where(particle == winner, 0.0, 60.0)

    from foldjax.models.boltz2.models.heads import potentials

    monkeypatch.setattr(
        potentials, "get_potentials", lambda *_args, **_kwargs: [Potential()]
    )
    monkeypatch.setattr(
        trunk, "diffusion_conditioning_forward", lambda *_args, **_kwargs: dict()
    )
    monkeypatch.setattr(
        trunk,
        "_preconditioned_score_forward",
        lambda *_args, r_noisy, **_kwargs: r_noisy * 0.75,
    )

    init_noise = np.zeros((samples * particles, 4, 3), dtype=np.float32)
    init_noise[:, 0, 0] = 1.0
    init_noise[:, 1, 0] = np.repeat(np.arange(samples), particles) + 1
    init_noise[:, 1, 1] = np.tile(np.arange(particles), samples) + 1
    steps = 4
    coords = trunk.boltz2_sample_forward(
        dict(
            trunk=dict(),
            conditioned_diffusion=dict(
                diffusion_conditioning=dict(), score_model=dict()
            ),
        ),
        dict(
            token_pad_mask=jnp.ones((1, 2), dtype=bool),
            atom_pad_mask=jnp.ones((1, 4), dtype=bool),
        ),
        jax.random.key(5),
        num_sampling_steps=steps,
        multiplicity=samples,
        augmentation=False,
        alignment_reverse_diff=False,
        steering_args=dict(
            fk_steering=True,
            physical_guidance_update=False,
            contact_guidance_update=False,
            num_particles=particles,
            fk_resampling_interval=2,
            fk_lambda=1.0,
        ),
        init_noise=jnp.asarray(init_noise),
        step_noises=jnp.zeros((steps, samples * particles, 4, 3), dtype=jnp.float32),
        use_scan=True,
        trunk=dict(
            s=jnp.zeros((1, 2, 4), dtype=jnp.float32),
            z=jnp.zeros((1, 2, 2, 4), dtype=jnp.float32),
            s_inputs=jnp.zeros((1, 2, 4), dtype=jnp.float32),
            relative_position_encoding=jnp.zeros((1, 2, 2, 4), dtype=jnp.float32),
        ),
    )["sample_atom_coords"]

    assert coords.shape == (samples, 4, 3)
    sample, particle = identity(coords)
    np.testing.assert_array_equal(np.asarray(sample), np.arange(samples))
    np.testing.assert_array_equal(np.asarray(particle), winners)
    # More than one resample ran, and none of them moved a particle across.
    assert len(seen) > 1
    for stage in seen:
        np.testing.assert_array_equal(
            stage, np.repeat(np.arange(samples), particles)
        )
