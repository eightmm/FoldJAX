"""Native augmentation draws remain dynamic through every compile wrapper."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3 import inference
from foldjax.models.openfold3.models.augmentation import (
    AugmentationTape,
    centre_random_augmentation,
)
from foldjax.models.openfold3.models.sampler import sample_diffusion
from tests.models.openfold3.test_stable_compile import _config, _table


def _tape(offset=0.0, *, native=False):
    q = jnp.arange(40, dtype=jnp.float32).reshape(2, 5, 4) + 1
    t = jnp.arange(30, dtype=jnp.float32).reshape(2, 5, 3) / 10 + offset
    return (
        AugmentationTape(q[:, None], t[:, None]) if native else AugmentationTape(q, t)
    )


@pytest.fixture
def sampler_predict(monkeypatch):
    traces = []

    def predict(
        key,
        batch,
        params,
        config,
        representative_atoms,
        *,
        augmentation_tape=None,
        noise_tape=None,
        noise_mask=None,
        augment=True,
        **kwargs,
    ):
        del representative_atoms, kwargs
        traces.append(augmentation_tape is not None)
        callback = None
        if augment and augmentation_tape is None:

            def callback(k, x):
                return centre_random_augmentation(k, x, batch["atom_mask"])

        return sample_diffusion(
            key,
            jnp.asarray([4.0, 2.0, 0.0]),
            (config.num_samples, config.n_atom, 3),
            lambda x, _t: params * x,
            gamma_0=0.8,
            gamma_min=1.0,
            noise_scale=1.003,
            step_scale=1.5,
            augment_fn=callback,
            noise_tape=noise_tape,
            noise_mask=noise_mask,
            augmentation_tape=augmentation_tape,
            atom_mask=batch["atom_mask"] if augmentation_tape is not None else None,
            diffusion_chunk_size=config.diffusion_chunk_size,
        )

    monkeypatch.setattr(inference, "predict", predict)
    inference._compiled_predict.clear_cache()
    yield traces
    inference._compiled_predict.clear_cache()


def _args():
    return (
        jax.random.key(0),
        {"atom_mask": jnp.ones((1, 4), dtype=jnp.float32)},
        jnp.asarray(0.3, dtype=jnp.float32),
    )


def _factory(**kwargs):
    config = _config(num_samples=5, num_steps=2, diffusion_chunk_size=2)
    return inference.compile_predict(config, _table(), triangle_kernel="xla", **kwargs)


def test_tape_values_stay_dynamic_and_native_layout_reuses_the_graph(sampler_predict):
    run = _factory()
    noise = jnp.arange(180, dtype=jnp.float32).reshape(3, 5, 4, 3) / 100
    baseline = run(*_args(), noise_tape=noise)
    first = run(*_args(), noise_tape=noise, augmentation_tape=_tape())
    second = run(*_args(), noise_tape=noise, augmentation_tape=_tape(2.0))
    native = run(*_args(), noise_tape=noise, augmentation_tape=_tape(native=True))
    jax.block_until_ready((baseline, first, second, native))
    np.testing.assert_array_equal(first, native)
    assert not np.allclose(first, second)
    assert not np.allclose(first, baseline)
    assert sampler_predict == [False, True]
    assert inference._compiled_predict._cache_size() == 2


def test_lowered_executable_preserves_dynamic_tape_and_rejects_route_changes(
    sampler_predict,
):
    run = _factory()
    noise = jnp.ones((3, 5, 4, 3), dtype=jnp.float32)
    expected = run(*_args(), noise_tape=noise, augmentation_tape=_tape(2.0))
    compiled = run.lower(
        *_args(), noise_tape=noise, augmentation_tape=_tape(native=True)
    ).compile()
    actual = compiled(*_args(), noise_tape=noise, augmentation_tape=_tape(2.0))
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="augmentation tape route"):
        compiled(*_args(), noise_tape=noise)

    ordinary = run.lower(*_args(), noise_tape=noise).compile()
    with pytest.raises(ValueError, match="augmentation tape route"):
        ordinary(*_args(), noise_tape=noise, augmentation_tape=_tape())


def test_compile_rejects_invalid_tapes_before_tracing(sampler_predict):
    bad = AugmentationTape(jnp.ones((1, 5, 4)), jnp.ones((2, 5, 3)))
    for operation in (_factory(), _factory().lower):
        with pytest.raises(ValueError, match="expected shape"):
            operation(*_args(), augmentation_tape=bad)
    with pytest.raises(ValueError, match="requires augment=True"):
        _factory(augment=False)(*_args(), augmentation_tape=_tape())
    assert sampler_predict == []
    assert inference._compiled_predict._cache_size() == 0


@pytest.mark.parametrize("bad", ["zero", "nan", "inf"])
def test_public_compile_entry_rejects_invalid_values_before_tracing(
    sampler_predict, bad
):
    tape = _tape()
    value = {"zero": 0, "nan": jnp.nan, "inf": jnp.inf}[bad]
    tape = tape._replace(quaternions=tape.quaternions.at[0, 0].set(value))
    for operation in (_factory(), _factory().lower):
        with pytest.raises(ValueError, match="finite"):
            operation(*_args(), augmentation_tape=tape)
    assert sampler_predict == []
    assert inference._compiled_predict._cache_size() == 0


def test_compiled_executable_revalidates_dynamic_tape_values(sampler_predict):
    run = _factory()
    compiled = run.lower(*_args(), augmentation_tape=_tape()).compile()
    bad = _tape()._replace(translations=jnp.full((2, 5, 3), jnp.nan))
    with pytest.raises(ValueError, match="finite draws"):
        compiled(*_args(), augmentation_tape=bad)


def test_eager_predict_rejects_invalid_tape_before_model_execution():
    config = _config(num_samples=5, num_steps=2)
    bad = AugmentationTape(jnp.ones((1, 5, 4)), jnp.ones((2, 5, 3)))
    with pytest.raises(ValueError, match="expected shape"):
        inference.predict(
            jax.random.key(0), {}, None, config, _table(), augmentation_tape=bad
        )
    with pytest.raises(ValueError, match="requires augment=True"):
        inference.predict(
            jax.random.key(0),
            {},
            None,
            config,
            _table(),
            augment=False,
            augmentation_tape=_tape(),
        )
    zero = _tape()._replace(quaternions=jnp.zeros((2, 5, 4)))
    with pytest.raises(ValueError, match="finite nonzero norms"):
        inference.predict(
            jax.random.key(0), {}, None, config, _table(), augmentation_tape=zero
        )
