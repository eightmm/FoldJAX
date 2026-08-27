from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.models.diffusion.diffusion import inference_noise_schedule
from foldjax.models.protenix.models.model import protenix_infer_static
from foldjax.models.protenix.models.predict import protenix_predict_static
from foldjax.models.protenix.models.trunk_blocks.msa import (
    sample_msa_cycle_features,
    sample_msa_cycle_index_tape,
)

from .test_model import _toy_features, _toy_params


def test_predict_wrapper_matches_static_infer_direct_call() -> None:
    params = _toy_params()
    features = _toy_features()
    init_noise = jnp.ones((1, 3, 3), dtype=jnp.float32)
    step_noise = jnp.zeros_like(init_noise)

    def wrapper(*, graph_jit: bool):
        return protenix_predict_static(
            params,
            features,
            key=None,
            num_samples=1,
            num_sampling_steps=1,
            recycling_steps=1,
            input_atom_heads=1,
            atom_encoder_heads=1,
            token_heads=1,
            atom_decoder_heads=1,
            n_queries=2,
            n_keys=4,
            sigma_data=4.0,
            centre_each_step=False,
            init_noise=init_noise,
            step_noises=(step_noise,),
            graph_jit=graph_jit,
        )

    # What this pins is the wrapper's argument plumbing, so it compares against
    # the same op-by-op path the direct call takes. The consolidated graph is
    # checked below, where exact equality is the wrong bar: tracing the whole
    # model lets XLA reassociate the same arithmetic.
    actual = wrapper(graph_jit=False)
    expected = protenix_infer_static(
        features,
        params,
        inference_noise_schedule(num_steps=1, sigma_data=4.0),
        key=None,
        num_samples=1,
        init_noise=init_noise,
        step_noises=(step_noise,),
        num_recycles=1,
        input_atom_heads=1,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
        centre_each_step=False,
    )

    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_array_equal(np.asarray(actual[key]), np.asarray(value))

    compiled = wrapper(graph_jit=True)
    assert compiled.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_allclose(
            np.asarray(compiled[key]),
            np.asarray(value),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"consolidated graph diverged on {key}",
        )


def test_compiled_predict_accepts_the_compact_cycle_msa_tape() -> None:
    features = dict(_toy_features())
    features.update(
        {
            "msa": jnp.asarray([[1, 2], [3, 4], [5, 6]], dtype=jnp.int32),
            "has_deletion": jnp.asarray(
                [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=jnp.float32
            ),
            "deletion_value": jnp.asarray(
                [[0.0, 0.5], [0.25, 0.0], [1.0, -0.0]], dtype=jnp.float32
            ),
        }
    )
    cycles = sample_msa_cycle_features(features, num_recycles=2, seed=5)
    tape = sample_msa_cycle_index_tape(features, num_recycles=2, seed=5)
    assert tape is not None

    def run(**cycle_kwargs):
        return protenix_predict_static(
            _toy_params(),
            features,
            key=None,
            num_samples=1,
            num_sampling_steps=1,
            recycling_steps=2,
            input_atom_heads=1,
            atom_encoder_heads=1,
            token_heads=1,
            atom_decoder_heads=1,
            n_queries=2,
            n_keys=4,
            sigma_data=4.0,
            stop_after_trunk=True,
            capture_names=("single", "pair"),
            graph_jit=True,
            **cycle_kwargs,
        )

    expected = run(cycle_msa_features=cycles)
    actual = run(cycle_msa_index_tape=tape)

    assert actual.keys() == expected.keys() == {"single", "pair"}
    for name in expected:
        np.testing.assert_array_equal(
            np.asarray(actual[name]).reshape(-1).view(np.uint8),
            np.asarray(expected[name]).reshape(-1).view(np.uint8),
        )



def test_the_matmul_precision_pin_is_a_scope_not_a_latch() -> None:
    """The pin must not follow the caller out of `protenix_predict_static`.

    JAX's matmul precision is process-global. This was a `jax.config.update`,
    so a process that ran Protenix left everything after it in TF32 -- another
    port, a notebook cell, a test collected later. It went unnoticed while every
    port pinned the same value; Boltz-2 now pins `"highest"` because its own
    upstream does, so the difference is observable.

    The default is checked here too: upstream's released inference config sets
    `enable_tf32: True` (`configs_inference.py:32`), and a port that ran full
    float32 there was more precise than the model it ports, at ~17% of its
    runtime.

    The call has to be a real one. Asserting that `jax.default_matmul_precision`
    restores its own setting would pass whether or not this port ever used it --
    a test that cannot fail.
    """
    import inspect

    import jax

    default = (
        inspect.signature(protenix_predict_static)
        .parameters["matmul_precision"]
        .default
    )
    assert default == "high"

    before = jax.config.jax_default_matmul_precision
    assert before != default, "ambient precision already matches; nothing is proven"

    protenix_predict_static(
        _toy_params(),
        _toy_features(),
        key=None,
        num_samples=1,
        num_sampling_steps=1,
        recycling_steps=1,
        input_atom_heads=1,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
        centre_each_step=False,
        init_noise=jnp.ones((1, 3, 3), dtype=jnp.float32),
        step_noises=(jnp.zeros((1, 3, 3), dtype=jnp.float32),),
        graph_jit=False,
    )

    assert jax.config.jax_default_matmul_precision == before


def test_confidence_sample_sequential_matches_batched() -> None:
    """Scoring inside the per-sample loop must change memory, not numbers.

    The sequential path is the default; the batched path is the reference it
    replaced. Same toy model, three samples, every output key equal in shape
    and value -- including `contact_probs`, which has no sample axis and is
    the case the leading-axis collapse gets wrong first.
    """
    from foldjax.models.protenix.models.model import protenix_infer_static

    params = _toy_params()
    features = _toy_features()
    init_noise = jnp.ones((3, 3, 3), dtype=jnp.float32)
    step_noise = jnp.zeros_like(init_noise)

    def run(sequential: bool):
        return protenix_infer_static(
            features,
            params,
            inference_noise_schedule(num_steps=1, sigma_data=4.0),
            key=None,
            num_samples=3,
            init_noise=init_noise,
            step_noises=(step_noise,),
            num_recycles=1,
            input_atom_heads=1,
            atom_encoder_heads=1,
            token_heads=1,
            atom_decoder_heads=1,
            n_queries=2,
            n_keys=4,
            sigma_data=4.0,
            centre_each_step=False,
            confidence_sample_sequential=sequential,
        )

    sequential = run(True)
    batched = run(False)
    assert set(sequential) == set(batched)
    for name, value in batched.items():
        got = sequential[name]
        assert got.shape == value.shape, name
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(value), atol=1e-5, err_msg=name
        )
