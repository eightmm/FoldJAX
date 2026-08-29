"""Regression gates for compact, diffusion-step-invariant atom caches."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.diffusion.atom import (
    _broadcast_window_s_terms,
    diffusion_transformer_s_terms,
)
from foldjax.models.boltz2.models.diffusion.diffusion import (
    diffusion_score_model_forward,
)
from foldjax.models.boltz2.models.diffusion.diffusion_conditioning import (
    diffusion_conditioning_forward,
)
from foldjax.models.boltz2.models.trunk_blocks import trunk as trunk_impl
from tests.models.boltz2.test_atom_cp_model_integration import (
    _model_inputs,
    _native_params,
)


def _score_inputs(multiplicity: int):
    native = _native_params()
    inputs = _model_inputs()
    conditioning = diffusion_conditioning_forward(
        native["diffusion_conditioning"],
        s_trunk=inputs["s_trunk"],
        z_trunk=inputs["z_trunk"],
        relative_position_encoding=inputs["relative_position_encoding"],
        feats=inputs["feats"],
        token_layers=1,
        lazy_token_trans_bias=True,
    )
    return native, inputs, conditioning, {
        "r_noisy": jnp.repeat(inputs["r_noisy"], multiplicity, axis=0),
        "times": jnp.repeat(inputs["times"], multiplicity, axis=0),
    }


def _compact_terms(score_params, atom_c):
    atom_c = atom_c.reshape((-1, 32, atom_c.shape[-1]))
    return (
        diffusion_transformer_s_terms(
            score_params["atom_attention_encoder"]["atom_encoder"][
                "diffusion_transformer"
            ],
            atom_c,
        ),
        diffusion_transformer_s_terms(
            score_params["atom_attention_decoder"]["atom_decoder"][
                "diffusion_transformer"
            ],
            atom_c,
        ),
    )


@pytest.mark.parametrize("use_scan", [False, True])
@pytest.mark.parametrize("multiplicity", [1, 5])
def test_score_compact_atom_cache_matches_historical_projection(
    use_scan: bool,
    multiplicity: int,
) -> None:
    native, inputs, conditioning, repeated = _score_inputs(multiplicity)
    score_params = native["score_model"]
    encoder_terms, decoder_terms = _compact_terms(
        score_params,
        conditioning["c"],
    )

    def score(encoder_cache, decoder_cache):
        return diffusion_score_model_forward(
            score_params,
            s_inputs=inputs["s_inputs"],
            s_trunk=inputs["s_trunk"],
            r_noisy=repeated["r_noisy"],
            times=repeated["times"],
            feats=inputs["feats"],
            diffusion_conditioning=conditioning,
            multiplicity=multiplicity,
            use_scan=use_scan,
            token_layers=1,
            atom_encoder_s_terms=encoder_cache,
            atom_decoder_s_terms=decoder_cache,
        )

    reference = jax.jit(lambda: score(None, None))()
    cached = jax.jit(lambda: score(encoder_terms, decoder_terms))()
    np.testing.assert_allclose(reference, cached, rtol=0.0, atol=1e-7)

    zero_encoder_terms = jax.tree.map(jnp.zeros_like, encoder_terms)
    zero_decoder_terms = jax.tree.map(jnp.zeros_like, decoder_terms)
    zeroed = jax.jit(lambda: score(zero_encoder_terms, zero_decoder_terms))()
    assert float(jnp.max(jnp.abs(reference - zeroed))) > 1e-4


@pytest.mark.parametrize("use_scan", [False, True])
def test_score_compact_cache_preserves_mixed_bfloat16_cast(use_scan: bool) -> None:
    multiplicity = 5
    native, inputs, conditioning, repeated = _score_inputs(multiplicity)

    def to_bfloat16(value):
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
            return value.astype(jnp.bfloat16)
        return value

    score_params = jax.tree.map(to_bfloat16, native["score_model"])
    wrong_terms = _compact_terms(score_params, conditioning["c"])
    cast_terms = _compact_terms(
        score_params,
        conditioning["c"].astype(jnp.bfloat16),
    )

    def score(encoder_cache, decoder_cache):
        return diffusion_score_model_forward(
            score_params,
            s_inputs=inputs["s_inputs"].astype(jnp.bfloat16),
            s_trunk=inputs["s_trunk"].astype(jnp.bfloat16),
            r_noisy=repeated["r_noisy"].astype(jnp.bfloat16),
            times=repeated["times"].astype(jnp.bfloat16),
            feats=inputs["feats"],
            diffusion_conditioning=conditioning,
            multiplicity=multiplicity,
            use_scan=use_scan,
            token_layers=1,
            atom_encoder_s_terms=encoder_cache,
            atom_decoder_s_terms=decoder_cache,
        )

    # Keep the precision-contract variants in one eager schedule. Separate
    # compilations may select slightly different BF16 schedules even when the
    # supplied values are identical, obscuring the cast behavior under test.
    reference = score(None, None)
    wrong = score(*wrong_terms)
    cached = score(*cast_terms)
    np.testing.assert_array_equal(reference, cached)
    assert float(jnp.max(jnp.abs(reference - wrong))) > 1e-4


@pytest.mark.parametrize("multiplicity", [1, 5])
def test_window_cache_broadcast_matches_batched_repeat_bitwise(
    multiplicity: int,
) -> None:
    batch = 2
    windows = 3
    width = 2
    channels = 2
    values = np.arange(batch * windows * width * channels, dtype=np.float32)
    values[0] = -0.0
    values[1] = np.nan
    values[2] = np.inf
    values[3] = -np.inf
    conditioning = jnp.asarray(values.reshape(batch, windows * width, channels))
    compact = conditioning.reshape(batch * windows, width, channels)

    (actual,) = _broadcast_window_s_terms(
        (compact,),
        multiplicity=multiplicity,
        num_windows=windows,
    )
    expected = jnp.repeat(conditioning, multiplicity, axis=0).reshape(
        batch * multiplicity * windows,
        width,
        channels,
    )

    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)
    np.testing.assert_array_equal(
        actual_np.view(np.uint32),
        expected_np.view(np.uint32),
    )
    np.testing.assert_array_equal(np.isnan(actual_np), np.isnan(expected_np))
    np.testing.assert_array_equal(np.signbit(actual_np), np.signbit(expected_np))
    if multiplicity == 1:
        assert actual is compact


def test_released_1ubq_logical_cache_bytes_are_multiplicity_independent() -> None:
    # Released shapes: A=608 -> K=19, width=32, atom channels=128, three
    # encoder plus three decoder layers, six cached terms per layer, FP32.
    leaves = 2 * 6
    layers = 3
    windows = 608 // 32
    width = 32
    channels = 128
    itemsize = np.dtype(np.float32).itemsize
    compact_bytes = leaves * layers * windows * width * channels * itemsize
    old_m1_bytes = compact_bytes
    old_m5_bytes = compact_bytes * 5

    assert compact_bytes == 11_206_656
    assert compact_bytes / 2**20 == 10.6875
    assert old_m1_bytes == compact_bytes
    assert old_m5_bytes == 56_033_280
    assert old_m5_bytes / 2**20 == 53.4375


_FK_ARGS = {
    "fk_steering": True,
    "physical_guidance_update": False,
    "contact_guidance_update": False,
    "num_particles": 3,
    "fk_resampling_interval": 1,
    "fk_lambda": 1.0,
}


@pytest.mark.parametrize(
    (
        "use_scan",
        "steering_args",
        "requested_multiplicity",
        "effective_multiplicity",
        "score_calls",
    ),
    [
        (False, None, 1, 1, 3),
        (True, None, 1, 1, 1),
        (False, None, 5, 5, 3),
        (True, None, 5, 5, 1),
        (True, _FK_ARGS, 2, 6, 3),
    ],
)
def test_sampler_builds_one_base_cache_for_eager_scan_m5_and_fk(
    monkeypatch: pytest.MonkeyPatch,
    use_scan: bool,
    steering_args: dict[str, object] | None,
    requested_multiplicity: int,
    effective_multiplicity: int,
    score_calls: int,
) -> None:
    score_params = {
        "s_to_a_linear": {
            "linear": {"kernel": jnp.zeros((4, 4), dtype=jnp.bfloat16)}
        },
        "atom_attention_encoder": {
            "atom_encoder": {"diffusion_transformer": {"tag": "encoder"}}
        },
        "atom_attention_decoder": {
            "atom_decoder": {"diffusion_transformer": {"tag": "decoder"}}
        },
    }
    params = {
        "trunk": {},
        "conditioned_diffusion": {
            "diffusion_conditioning": {},
            "score_model": score_params,
        },
    }
    # The production sampler's semantic feature batch is singleton; B>1
    # ordering is exercised independently above without conflating FK's
    # particle-group contract with multi-job batching.
    batch = 1
    atoms = 64
    windows = atoms // 32
    feats = {
        "token_pad_mask": jnp.ones((batch, 2), dtype=bool),
        "atom_pad_mask": jnp.ones((batch, atoms), dtype=bool),
    }
    supplied_trunk = {
        "s": jnp.zeros((batch, 2, 4), dtype=jnp.float32),
        "z": jnp.zeros((batch, 2, 2, 4), dtype=jnp.float32),
        "s_inputs": jnp.zeros((batch, 2, 4), dtype=jnp.float32),
        "relative_position_encoding": jnp.zeros(
            (batch, 2, 2, 4), dtype=jnp.float32
        ),
    }
    base_conditioning = jnp.arange(
        batch * atoms * 4,
        dtype=jnp.float32,
    ).reshape(batch, atoms, 4)
    precompute_calls: list[tuple[str, tuple[int, ...], jnp.dtype, np.ndarray]] = []
    caches: dict[str, tuple[jnp.ndarray, ...]] = {}
    received: list[
        tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], int, int]
    ] = []

    def fake_s_terms(transformer_params, atom_c, **_kwargs):
        tag = transformer_params["tag"]
        precompute_calls.append(
            (tag, atom_c.shape, atom_c.dtype, np.asarray(atom_c[:, 0, 0]))
        )
        cache = (atom_c[:, :1, :1],)
        caches[tag] = cache
        return cache

    def fake_score(
        *_args,
        r_noisy,
        multiplicity,
        atom_encoder_s_terms,
        atom_decoder_s_terms,
        **_kwargs,
    ):
        received.append(
            (
                atom_encoder_s_terms,
                atom_decoder_s_terms,
                multiplicity,
                r_noisy.shape[0],
            )
        )
        return r_noisy * jnp.asarray(0.83, dtype=r_noisy.dtype)

    monkeypatch.setattr(
        trunk_impl,
        "diffusion_conditioning_forward",
        lambda *_args, **_kwargs: {"c": base_conditioning},
    )
    monkeypatch.setattr(trunk_impl, "diffusion_transformer_s_terms", fake_s_terms)
    monkeypatch.setattr(trunk_impl, "_preconditioned_score_forward", fake_score)
    if steering_args is not None:
        from foldjax.models.boltz2.models.heads import potentials

        monkeypatch.setattr(potentials, "get_potentials", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(
            jax.random,
            "categorical",
            lambda _key, _logits, *, axis, shape: jnp.zeros(
                shape, dtype=jnp.int32
            ),
        )

    steps = 3
    expanded_batch = batch * effective_multiplicity
    result = trunk_impl.boltz2_sample_forward(
        params,
        feats,
        jax.random.PRNGKey(5),
        num_sampling_steps=steps,
        multiplicity=requested_multiplicity,
        augmentation=False,
        alignment_reverse_diff=False,
        steering_args=steering_args,
        init_noise=jnp.zeros((expanded_batch, atoms, 3), dtype=jnp.float32),
        step_noises=jnp.zeros(
            (steps, expanded_batch, atoms, 3), dtype=jnp.float32
        ),
        use_scan=use_scan,
        trunk=supplied_trunk,
    )

    expected_window_starts = np.asarray(
        base_conditioning.reshape(batch * windows, 32, 4)[:, 0, 0],
        dtype=np.float32,
    ).astype(jnp.bfloat16)
    assert [call[:3] for call in precompute_calls] == [
        ("encoder", (batch * windows, 32, 4), jnp.bfloat16),
        ("decoder", (batch * windows, 32, 4), jnp.bfloat16),
    ]
    for call in precompute_calls:
        np.testing.assert_array_equal(call[3], expected_window_starts)
    assert len(received) == score_calls
    assert all(call[0] is caches["encoder"] for call in received)
    assert all(call[1] is caches["decoder"] for call in received)
    assert all(
        call[2:] == (effective_multiplicity, expanded_batch) for call in received
    )
    output_multiplicity = (
        requested_multiplicity if steering_args is not None else effective_multiplicity
    )
    assert result["sample_atom_coords"].shape == (
        batch * output_multiplicity,
        atoms,
        3,
    )


def test_sampler_leaves_projection_inside_context_parallel_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_params = {
        "atom_attention_encoder": {"atom_encoder": {}},
        "atom_attention_decoder": {"atom_decoder": {}},
    }
    params = {
        "trunk": {},
        "conditioned_diffusion": {
            "diffusion_conditioning": {},
            "score_model": score_params,
        },
    }
    feats = {
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
    monkeypatch.setattr(
        trunk_impl,
        "diffusion_conditioning_forward",
        lambda *_args, **_kwargs: {"c": jnp.zeros((1, 4, 4), dtype=jnp.float32)},
    )
    monkeypatch.setattr(trunk_impl, "_cp_mesh", lambda: object())
    monkeypatch.setattr(
        trunk_impl,
        "diffusion_transformer_s_terms",
        lambda *_args, **_kwargs: pytest.fail("CP must retain the historical path"),
    )
    monkeypatch.setattr(
        trunk_impl,
        "_preconditioned_score_forward",
        lambda *_args, r_noisy, **_kwargs: r_noisy,
    )

    result = trunk_impl.boltz2_sample_forward(
        params,
        feats,
        jax.random.PRNGKey(3),
        num_sampling_steps=2,
        augmentation=False,
        alignment_reverse_diff=False,
        use_scan=True,
        trunk=supplied_trunk,
        atom_context_parallel=True,
    )
    assert result["sample_atom_coords"].shape == (1, 4, 3)
