from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.opendde.models.sampling import (
    _prefix_atom_normal,
    make_padded_random_tapes,
    sample_diffusion,
)


def _bytes(value) -> np.ndarray:
    return np.asarray(value).reshape(-1).view(np.uint8)


@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_masked_draw_matches_the_materialized_opendde_tape_bitwise(
    batch_shape,
) -> None:
    key = jax.random.PRNGKey(13)
    num_samples, num_steps, actual_atom, target_atom = 2, 3, 3, 5
    atom_mask = jnp.arange(target_atom) < actual_atom

    with jax.threefry_partitionable(True):
        expected = make_padded_random_tapes(
            key=key,
            num_samples=num_samples,
            num_steps=num_steps,
            actual_atom=actual_atom,
            target_atom=target_atom,
            batch_shape=batch_shape,
        )
        init_key, step_key, _rotation_key, _translation_key = jax.random.split(
            key, 4
        )
        actual_init = _prefix_atom_normal(
            init_key,
            atom_mask,
            leading_shape=(*batch_shape, num_samples),
            dtype=jnp.float32,
        )
        actual_steps = jnp.stack(
            tuple(
                _prefix_atom_normal(
                    step_key_i,
                    atom_mask,
                    leading_shape=(*batch_shape, num_samples),
                    dtype=jnp.float32,
                )
                for step_key_i in jax.random.split(step_key, num_steps)
            )
        )

    np.testing.assert_array_equal(_bytes(actual_init), _bytes(expected[0]))
    np.testing.assert_array_equal(_bytes(actual_steps), _bytes(expected[1]))


@pytest.mark.parametrize("batch_shape", [(), (2,)])
@pytest.mark.parametrize("use_scan", [False, True])
def test_lazy_padding_rng_matches_the_complete_tape_sampler_bitwise(
    batch_shape,
    use_scan,
) -> None:
    key = jax.random.PRNGKey(19)
    num_samples, num_steps, actual_atom, target_atom = 2, 3, 3, 5
    atom_mask = jnp.arange(target_atom) < actual_atom
    schedule = jnp.asarray([4.0, 2.0, 1.1, 0.0], dtype=jnp.float32)

    def denoise(x, t):
        return x * jnp.asarray(0.83, x.dtype) + t[..., None, None] * 0.01

    with jax.threefry_partitionable(True):
        tapes = make_padded_random_tapes(
            key=key,
            num_samples=num_samples,
            num_steps=num_steps,
            actual_atom=actual_atom,
            target_atom=target_atom,
            batch_shape=batch_shape,
        )
        expected = sample_diffusion(
            denoise,
            schedule,
            num_samples=num_samples,
            n_atom=target_atom,
            key=None,
            init_noise=tapes[0],
            step_noises=tapes[1],
            rotations=tapes[2],
            translations=tapes[3],
            batch_shape=batch_shape,
            atom_mask=atom_mask,
            use_scan=use_scan,
        )
        actual = sample_diffusion(
            denoise,
            schedule,
            num_samples=num_samples,
            n_atom=target_atom,
            key=key,
            batch_shape=batch_shape,
            atom_mask=atom_mask,
            use_scan=use_scan,
            preserve_prefix_rng=True,
        )

    np.testing.assert_array_equal(_bytes(actual), _bytes(expected))


def test_lazy_padding_rng_removes_the_full_step_tape_from_arguments() -> None:
    key = jax.random.PRNGKey(11)
    num_samples, num_steps, actual_atom, target_atom = 2, 8, 7, 32
    atom_mask = jnp.arange(target_atom) < actual_atom
    schedule = jnp.linspace(2.0, 0.0, num_steps + 1, dtype=jnp.float32)

    def denoise(x, _t):
        return x * jnp.asarray(0.9, x.dtype)

    with jax.threefry_partitionable(True):
        tapes = make_padded_random_tapes(
            key=key,
            num_samples=num_samples,
            num_steps=num_steps,
            actual_atom=actual_atom,
            target_atom=target_atom,
        )
        old = jax.jit(
            lambda init, steps, rotations, translations, mask: sample_diffusion(
                denoise,
                schedule,
                num_samples=num_samples,
                n_atom=target_atom,
                key=None,
                init_noise=init,
                step_noises=steps,
                rotations=rotations,
                translations=translations,
                atom_mask=mask,
                use_scan=True,
            )
        )
        compact = jax.jit(
            lambda random_key, mask: sample_diffusion(
                denoise,
                schedule,
                num_samples=num_samples,
                n_atom=target_atom,
                key=random_key,
                atom_mask=mask,
                use_scan=True,
                preserve_prefix_rng=True,
            )
        )
        old_executable = old.lower(*tapes, atom_mask).compile()
        compact_executable = compact.lower(key, atom_mask).compile()

    old_memory = old_executable.memory_analysis()
    compact_memory = compact_executable.memory_analysis()
    assert (
        old_memory.argument_size_in_bytes
        > 25 * compact_memory.argument_size_in_bytes
    )
    compact_hlo = compact.lower(key, atom_mask).compiler_ir(dialect="stablehlo")
    assert "stablehlo.while" in str(compact_hlo)


def test_lazy_padding_rng_rejects_unsupported_or_ambiguous_routes() -> None:
    schedule = jnp.asarray([2.0, 0.0], dtype=jnp.float32)
    key = jax.random.PRNGKey(7)
    atom_mask = jnp.asarray([1, 0], dtype=bool)
    common = {
        "denoise_fn": lambda x, _t: x,
        "noise_schedule": schedule,
        "num_samples": 1,
        "n_atom": 2,
        "key": key,
        "atom_mask": atom_mask,
        "preserve_prefix_rng": True,
    }

    with jax.threefry_partitionable(False):
        with pytest.raises(ValueError, match="jax_threefry_partitionable"):
            sample_diffusion(**common)
    with jax.default_prng_impl("rbg"):
        with pytest.raises(ValueError, match="jax_default_prng_impl"):
            sample_diffusion(**common)
    with jax.threefry_partitionable(True):
        with pytest.raises(ValueError, match="mutually exclusive"):
            sample_diffusion(
                **common,
                init_noise=jnp.zeros((1, 2, 3), dtype=jnp.float32),
            )
