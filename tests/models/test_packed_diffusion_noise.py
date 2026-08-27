from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.opendde.models.geometry import uniform_random_rotations
from foldjax.models.opendde.models.sampling import sample_diffusion as sample_opendde
from foldjax.models.protenix.models.diffusion.diffusion import (
    sample_diffusion as sample_protenix,
)


def _denoise(x_noisy: jnp.ndarray, t_hat: jnp.ndarray) -> jnp.ndarray:
    return x_noisy / (1.0 + t_hat[..., None, None])


def _opendde_runner(
    schedule: jnp.ndarray,
    init_noise: jnp.ndarray,
    *,
    use_scan: bool = True,
) -> Callable[[jnp.ndarray | Sequence[jnp.ndarray]], jnp.ndarray]:
    n_steps = schedule.shape[0] - 1
    num_samples, n_atom = init_noise.shape[:2]
    rotations = jnp.broadcast_to(
        jnp.eye(3, dtype=jnp.float32), (n_steps, num_samples, 3, 3)
    )
    translations = jnp.zeros((n_steps, num_samples, 3), dtype=jnp.float32)

    def run(step_noises):
        return sample_opendde(
            _denoise,
            schedule,
            num_samples=num_samples,
            n_atom=n_atom,
            key=None,
            init_noise=init_noise,
            step_noises=step_noises,
            rotations=rotations,
            translations=translations,
            use_scan=use_scan,
        )

    return run


def _protenix_runner(
    schedule: jnp.ndarray,
    init_noise: jnp.ndarray,
    *,
    diffusion_chunk_size: int | None = None,
    use_scan: bool = True,
) -> Callable[[jnp.ndarray | Sequence[jnp.ndarray]], jnp.ndarray]:
    num_samples, n_atom = init_noise.shape[:2]

    def run(step_noises):
        return sample_protenix(
            _denoise,
            schedule,
            num_samples=num_samples,
            n_atom=n_atom,
            key=None,
            init_noise=init_noise,
            step_noises=step_noises,
            diffusion_chunk_size=diffusion_chunk_size,
            centre_each_step=False,
            use_scan=use_scan,
        )

    return run


@pytest.mark.parametrize("model", ["opendde", "protenix"])
def test_packed_step_noise_matches_direct_tuple_and_list_bitwise(model: str) -> None:
    schedule = jnp.asarray([8.0, 4.0, 0.5, 0.0], dtype=jnp.float32)
    init_noise = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3) / 10.0
    packed = jnp.stack(
        tuple(jnp.full_like(init_noise, value) for value in (0.25, -0.5, 0.75))
    )
    tuple_tape = tuple(packed[index] for index in range(3))
    list_tape = list(tuple_tape)
    run = (
        _opendde_runner(schedule, init_noise)
        if model == "opendde"
        else _protenix_runner(schedule, init_noise)
    )

    expected = run(tuple_tape)
    np.testing.assert_array_equal(run(list_tape), expected)
    np.testing.assert_array_equal(run(packed), expected)

    loop_run = (
        _opendde_runner(schedule, init_noise, use_scan=False)
        if model == "opendde"
        else _protenix_runner(schedule, init_noise, use_scan=False)
    )
    loop_expected = loop_run(tuple_tape)
    np.testing.assert_array_equal(loop_run(list_tape), loop_expected)
    np.testing.assert_array_equal(loop_run(packed), loop_expected)


def test_protenix_packed_step_noise_keeps_sample_chunking_bitwise() -> None:
    schedule = jnp.asarray([8.0, 4.0, 0.5, 0.0], dtype=jnp.float32)
    init_noise = jnp.arange(60, dtype=jnp.float32).reshape(5, 4, 3) / 10.0
    packed = jnp.stack(
        tuple(jnp.full_like(init_noise, value) for value in (0.25, -0.5, 0.75))
    )
    tuple_tape = tuple(packed[index] for index in range(3))
    run = _protenix_runner(schedule, init_noise, diffusion_chunk_size=2)

    expected = run(tuple_tape)
    np.testing.assert_array_equal(run(list(tuple_tape)), expected)
    np.testing.assert_array_equal(run(packed), expected)


def test_protenix_batched_packed_noise_chunks_the_minus_three_sample_axis() -> None:
    schedule = jnp.asarray([8.0, 4.0, 0.5, 0.0], dtype=jnp.float32)
    n_batch, num_samples, n_atom, n_steps = 2, 5, 4, 3
    init_noise = (
        jnp.arange(n_batch * num_samples * n_atom * 3, dtype=jnp.float32).reshape(
            n_batch, num_samples, n_atom, 3
        )
        / 10.0
    )
    packed = jnp.stack(
        tuple(jnp.full_like(init_noise, value) for value in (0.25, -0.5, 0.75))
    )
    tuple_tape = tuple(packed[index] for index in range(n_steps))

    def run(step_noises):
        return sample_protenix(
            _denoise,
            schedule,
            num_samples=num_samples,
            n_atom=n_atom,
            key=None,
            init_noise=init_noise,
            step_noises=step_noises,
            diffusion_chunk_size=2,
            centre_each_step=False,
            use_scan=True,
        )

    expected = run(tuple_tape)
    actual = run(packed)
    assert actual.shape == (n_batch, num_samples, n_atom, 3)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("model", ["opendde", "protenix"])
def test_malformed_packed_step_noise_has_an_explicit_shape_error(model: str) -> None:
    schedule = jnp.asarray([4.0, 2.0, 0.0], dtype=jnp.float32)
    init_noise = jnp.zeros((2, 3, 3), dtype=jnp.float32)
    malformed = jnp.zeros((1, 2, 3, 3), dtype=jnp.float32)
    run = (
        _opendde_runner(schedule, init_noise)
        if model == "opendde"
        else _protenix_runner(
            schedule,
            init_noise,
            diffusion_chunk_size=1,
        )
    )

    with pytest.raises(ValueError, match="packed step_noises expected shape"):
        run(malformed)


@pytest.mark.parametrize("model", ["opendde", "protenix"])
def test_no_tape_rng_route_matches_the_historical_split_order(model: str) -> None:
    schedule = jnp.asarray([8.0, 4.0, 0.5, 0.0], dtype=jnp.float32)
    n_steps, num_samples, n_atom = 3, 2, 4
    key = jax.random.PRNGKey(19)
    if model == "opendde":
        init_key, step_key, rotation_key, translation_key = jax.random.split(key, 4)
        init_noise = jax.random.normal(
            init_key, (num_samples, n_atom, 3), dtype=jnp.float32
        )
        tuple_tape = tuple(
            jax.random.normal(
                step_key_i, (num_samples, n_atom, 3), dtype=jnp.float32
            )
            for step_key_i in jax.random.split(step_key, n_steps)
        )
        rotations = uniform_random_rotations(rotation_key, (n_steps, num_samples))
        translations = jax.random.normal(
            translation_key, (n_steps, num_samples, 3), dtype=jnp.float32
        )
        random_result = sample_opendde(
            _denoise,
            schedule,
            num_samples=num_samples,
            n_atom=n_atom,
            key=key,
            use_scan=True,
        )
        explicit_result = sample_opendde(
            _denoise,
            schedule,
            num_samples=num_samples,
            n_atom=n_atom,
            key=None,
            init_noise=init_noise,
            step_noises=tuple_tape,
            rotations=rotations,
            translations=translations,
            use_scan=True,
        )
    else:
        step_key, init_key = jax.random.split(key)
        init_noise = jax.random.normal(
            init_key, (num_samples, n_atom, 3), dtype=jnp.float32
        )
        tuple_tape = tuple(
            jax.random.normal(
                step_key_i, (num_samples, n_atom, 3), dtype=jnp.float32
            )
            for step_key_i in jax.random.split(step_key, n_steps)
        )
        random_result = sample_protenix(
            _denoise,
            schedule,
            num_samples=num_samples,
            n_atom=n_atom,
            key=key,
            centre_each_step=False,
            use_scan=True,
        )
        explicit_result = sample_protenix(
            _denoise,
            schedule,
            num_samples=num_samples,
            n_atom=n_atom,
            key=None,
            init_noise=init_noise,
            step_noises=tuple_tape,
            centre_each_step=False,
            use_scan=True,
        )

    np.testing.assert_array_equal(random_result, explicit_result)


@pytest.mark.parametrize("model", ["opendde", "protenix"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_packed_step_noise_preserves_nonfinite_classification(
    model: str,
    nonfinite: float,
) -> None:
    schedule = jnp.asarray([8.0, 4.0, 0.0], dtype=jnp.float32)
    init_noise = jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3) / 10.0
    packed = jnp.zeros((2, 2, 3, 3), dtype=jnp.float32).at[0, 0, 0, 0].set(
        nonfinite
    )
    tuple_tape = tuple(packed[index] for index in range(2))
    run = (
        _opendde_runner(schedule, init_noise)
        if model == "opendde"
        else _protenix_runner(schedule, init_noise)
    )

    expected = np.asarray(run(tuple_tape))
    actual = np.asarray(run(packed))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite], expected[finite])


@pytest.mark.parametrize("model", ["opendde", "protenix"])
def test_packed_and_sequence_tapes_have_distinct_reused_jit_identities(
    model: str,
) -> None:
    schedule = jnp.asarray([4.0, 2.0, 0.0], dtype=jnp.float32)
    init_noise = jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3) / 10.0
    packed = jnp.zeros((2, 2, 3, 3), dtype=jnp.float32)
    tuple_tape = tuple(packed[index] for index in range(2))
    sampler = (
        _opendde_runner(schedule, init_noise)
        if model == "opendde"
        else _protenix_runner(schedule, init_noise)
    )
    traced_structures = []

    def run(step_noises):
        traced_structures.append(jax.tree_util.tree_structure(step_noises))
        return sampler(step_noises)

    compiled = jax.jit(run)
    tuple_result = compiled(tuple_tape)
    packed_result = compiled(packed)
    np.testing.assert_array_equal(compiled(tuple_tape), tuple_result)
    np.testing.assert_array_equal(compiled(packed), packed_result)

    assert len(traced_structures) == 2
    assert traced_structures[0] != traced_structures[1]
    np.testing.assert_array_equal(packed_result, tuple_result)


@pytest.mark.parametrize("model", ["opendde", "protenix"])
def test_packed_200_step_scan_removes_stack_hlo_and_temporary(model: str) -> None:
    cpu = jax.devices("cpu")[0]
    n_steps, num_samples, n_atom = 200, 5, 32
    schedule = jax.device_put(
        jnp.linspace(8.0, 0.05, n_steps + 1, dtype=jnp.float32), cpu
    )
    init_noise = jax.device_put(
        jnp.zeros((num_samples, n_atom, 3), dtype=jnp.float32), cpu
    )
    packed = jax.device_put(
        jnp.zeros((n_steps, num_samples, n_atom, 3), dtype=jnp.float32), cpu
    )
    tuple_tape = tuple(packed[index] for index in range(n_steps))
    run = (
        _opendde_runner(schedule, init_noise)
        if model == "opendde"
        else _protenix_runner(schedule, init_noise)
    )
    compiled_run = jax.jit(run)

    tuple_lowered = compiled_run.lower(tuple_tape)
    packed_lowered = compiled_run.lower(packed)
    tuple_hlo = str(tuple_lowered.compiler_ir(dialect="stablehlo"))
    packed_hlo = str(packed_lowered.compiler_ir(dialect="stablehlo"))
    tuple_executable = tuple_lowered.compile()
    packed_executable = packed_lowered.compile()
    tuple_memory = tuple_executable.memory_analysis()
    packed_memory = packed_executable.memory_analysis()

    assert tuple_hlo.count("stablehlo.concatenate") > 0
    assert packed_hlo.count("stablehlo.concatenate") == 0
    assert len(packed_hlo) < len(tuple_hlo)
    assert packed_memory.temp_size_in_bytes < tuple_memory.temp_size_in_bytes
    np.testing.assert_array_equal(
        packed_executable(packed), tuple_executable(tuple_tape)
    )
