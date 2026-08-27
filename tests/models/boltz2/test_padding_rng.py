from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.boltz2.models.trunk_blocks.trunk as trunk_impl
from foldjax.models.boltz2.api import (
    _prefix_stable_noise_tape,
    _runner_identity,
    _storage_prefix_size,
)
from foldjax.models.boltz2.models.trunk_blocks.trunk import (
    _prefix_atom_normal,
    boltz2_sample_forward,
)


def _bytes(value) -> np.ndarray:
    return np.asarray(value).reshape(-1).view(np.uint8)


@pytest.mark.parametrize("multiplicity", [1, 3])
@pytest.mark.parametrize(("storage_atoms", "target_atoms"), [(0, 8), (5, 8), (8, 8)])
def test_storage_prefix_draw_matches_the_materialized_tape_bitwise(
    multiplicity: int,
    storage_atoms: int,
    target_atoms: int,
) -> None:
    key = jax.random.PRNGKey(17)
    steps = 4

    with jax.threefry_partitionable(True):
        expected_init, expected_steps = _prefix_stable_noise_tape(
            key,
            multiplicity=multiplicity,
            storage_atoms=storage_atoms,
            target_atoms=target_atoms,
            steps=steps,
        )
        run_key, init_key = jax.random.split(key)
        actual_init = _prefix_atom_normal(
            init_key,
            jnp.asarray(storage_atoms, dtype=jnp.int32),
            multiplicity=multiplicity,
            target_atoms=target_atoms,
            dtype=jnp.float32,
        )
        actual_steps = []
        for _ in range(steps):
            run_key, noise_key = jax.random.split(run_key)
            actual_steps.append(
                _prefix_atom_normal(
                    noise_key,
                    jnp.asarray(storage_atoms, dtype=jnp.int32),
                    multiplicity=multiplicity,
                    target_atoms=target_atoms,
                    dtype=jnp.float32,
                )
            )

    np.testing.assert_array_equal(_bytes(actual_init), _bytes(expected_init))
    np.testing.assert_array_equal(
        _bytes(jnp.stack(actual_steps)), _bytes(expected_steps)
    )


def _sampler_inputs(*, target_atoms: int):
    params = {
        "trunk": {},
        "conditioned_diffusion": {
            "diffusion_conditioning": {},
            "score_model": {},
        },
    }
    feats = {
        # Deliberately only two semantically valid atoms while the storage
        # stride below is three: masked atoms already present in native storage
        # still consume RNG offsets and must not be confused with padding.
        "atom_pad_mask": jnp.asarray(
            [[1, 1, *([0] * (target_atoms - 2))]], dtype=jnp.float32
        ),
        "token_pad_mask": jnp.ones((1, 2), dtype=jnp.float32),
    }
    trunk = {
        "s_inputs": jnp.zeros((1, 2, 1), dtype=jnp.float32),
        "s": jnp.zeros((1, 2, 1), dtype=jnp.float32),
        "z": jnp.zeros((1, 2, 2, 1), dtype=jnp.float32),
        "relative_position_encoding": jnp.zeros(
            (1, 2, 2, 1), dtype=jnp.float32
        ),
    }
    return params, feats, trunk


@pytest.mark.parametrize("use_scan", [False, True])
def test_storage_prefix_sampler_matches_the_materialized_tape_bitwise(
    monkeypatch: pytest.MonkeyPatch,
    use_scan: bool,
) -> None:
    monkeypatch.setattr(
        trunk_impl,
        "diffusion_conditioning_forward",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        trunk_impl,
        "_preconditioned_score_forward",
        lambda *_args, r_noisy, **_kwargs: r_noisy
        * jnp.asarray(0.83, r_noisy.dtype),
    )
    key = jax.random.PRNGKey(23)
    multiplicity, storage_atoms, target_atoms, steps = 2, 3, 5, 3
    params, feats, trunk = _sampler_inputs(target_atoms=target_atoms)

    with jax.threefry_partitionable(True):
        tape = _prefix_stable_noise_tape(
            key,
            multiplicity=multiplicity,
            storage_atoms=storage_atoms,
            target_atoms=target_atoms,
            steps=steps,
        )
        common = {
            "params": params,
            "feats": feats,
            "key": key,
            "recycling_steps": 0,
            "num_sampling_steps": steps,
            "multiplicity": multiplicity,
            "augmentation": False,
            "alignment_reverse_diff": False,
            "use_scan": use_scan,
            "trunk": trunk,
        }
        expected = boltz2_sample_forward(
            init_noise=tape[0],
            step_noises=tape[1],
            **common,
        )["sample_atom_coords"]
        actual = boltz2_sample_forward(
            noise_storage_atoms=jnp.asarray(storage_atoms, dtype=jnp.int32),
            **common,
        )["sample_atom_coords"]

    np.testing.assert_array_equal(_bytes(actual), _bytes(expected))


def test_storage_prefix_rng_rejects_unsupported_or_ambiguous_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trunk_impl,
        "diffusion_conditioning_forward",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        trunk_impl,
        "_preconditioned_score_forward",
        lambda *_args, r_noisy, **_kwargs: r_noisy,
    )
    params, feats, trunk = _sampler_inputs(target_atoms=5)
    common = {
        "params": params,
        "feats": feats,
        "key": jax.random.PRNGKey(7),
        "num_sampling_steps": 2,
        "augmentation": False,
        "alignment_reverse_diff": False,
        "trunk": trunk,
        "noise_storage_atoms": jnp.asarray(3, dtype=jnp.int32),
    }

    with jax.threefry_partitionable(False):
        with pytest.raises(ValueError, match="jax_threefry_partitionable"):
            boltz2_sample_forward(**common)
    with jax.default_prng_impl("rbg"):
        with pytest.raises(ValueError, match="jax_default_prng_impl"):
            boltz2_sample_forward(**common)
    with jax.threefry_partitionable(True):
        with pytest.raises(ValueError, match="mutually exclusive"):
            boltz2_sample_forward(
                **common,
                init_noise=jnp.zeros((1, 5, 3), dtype=jnp.float32),
            )
        with pytest.raises(ValueError, match="must be a scalar"):
            boltz2_sample_forward(
                **{
                    **common,
                    "noise_storage_atoms": jnp.asarray([3], dtype=jnp.int32),
                }
            )
        batched_feats = {
            **feats,
            "atom_pad_mask": jnp.repeat(feats["atom_pad_mask"], 2, axis=0),
        }
        with pytest.raises(ValueError, match="singleton feature batch"):
            boltz2_sample_forward(**{**common, "feats": batched_feats})


def test_storage_prefix_rng_removes_the_full_step_tape_from_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trunk_impl,
        "diffusion_conditioning_forward",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        trunk_impl,
        "_preconditioned_score_forward",
        lambda *_args, r_noisy, **_kwargs: r_noisy
        * jnp.asarray(0.9, r_noisy.dtype),
    )
    key = jax.random.PRNGKey(11)
    multiplicity, storage_atoms, target_atoms, steps = 2, 7, 32, 8
    params, feats, trunk = _sampler_inputs(target_atoms=target_atoms)
    common = {
        "params": params,
        "feats": feats,
        "recycling_steps": 0,
        "num_sampling_steps": steps,
        "multiplicity": multiplicity,
        "augmentation": False,
        "alignment_reverse_diff": False,
        "use_scan": True,
        "trunk": trunk,
    }

    with jax.threefry_partitionable(True):
        tape = _prefix_stable_noise_tape(
            key,
            multiplicity=multiplicity,
            storage_atoms=storage_atoms,
            target_atoms=target_atoms,
            steps=steps,
        )
        old = jax.jit(
            lambda random_key, init, step: boltz2_sample_forward(
                key=random_key,
                init_noise=init,
                step_noises=step,
                **common,
            )["sample_atom_coords"]
        )
        compact = jax.jit(
            lambda random_key, storage: boltz2_sample_forward(
                key=random_key,
                noise_storage_atoms=storage,
                **common,
            )["sample_atom_coords"]
        )
        old_executable = old.lower(key, *tape).compile()
        compact_executable = compact.lower(
            key, jnp.asarray(storage_atoms, dtype=jnp.int32)
        ).compile()

    old_memory = old_executable.memory_analysis()
    compact_memory = compact_executable.memory_analysis()
    assert (
        old_memory.argument_size_in_bytes
        > 25 * compact_memory.argument_size_in_bytes
    )
    compact_hlo = compact.lower(
        key, jnp.asarray(storage_atoms, dtype=jnp.int32)
    ).compiler_ir(dialect="stablehlo")
    assert "stablehlo.while" in str(compact_hlo)


def test_storage_prefix_size_validates_the_host_bucket() -> None:
    np.testing.assert_array_equal(
        _storage_prefix_size(storage_atoms=5, target_atoms=8),
        np.asarray(5, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="smaller than storage"):
        _storage_prefix_size(storage_atoms=9, target_atoms=8)


def test_runner_identity_separates_each_padding_noise_route() -> None:
    identities = {
        _runner_identity(
            predict_function=test_runner_identity_separates_each_padding_noise_route,
            predict_kwargs={},
            noise_mode=mode,
            runtime=(),
        )
        for mode in ("none", "tape", "storage_prefix")
    }
    assert len(identities) == 3
