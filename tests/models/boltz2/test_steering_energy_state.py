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
