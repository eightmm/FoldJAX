"""Regression gates for compact, diffusion-step-invariant atom caches."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.diffusion.atom import (
    _broadcast_window_axis,
    _broadcast_window_s_terms,
    atom_transformer_forward,
    diffusion_transformer_forward,
    diffusion_transformer_s_terms,
    get_indexing_matrix,
    single_to_keys,
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


@pytest.mark.parametrize("multiplicity", [1, 5])
def test_window_bias_broadcast_matches_batched_repeat_bitwise(
    multiplicity: int,
) -> None:
    batch = 2
    windows = 3
    width = 2
    keys = 4
    channels = 2
    size = batch * windows * width * keys * channels
    values = np.arange(size, dtype=np.float32)
    values[0] = -0.0
    values[1] = np.nan
    values[2] = np.inf
    values[3] = -np.inf
    bias = jnp.asarray(values.reshape(batch, windows, width, keys, channels))
    compact = bias.reshape(batch * windows, width, keys, channels)

    actual = _broadcast_window_axis(
        compact,
        multiplicity=multiplicity,
        num_windows=windows,
    )
    expected = jnp.repeat(bias, multiplicity, axis=0).reshape(
        batch * multiplicity * windows,
        width,
        keys,
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


def _bias_scan_params(layers: int, dim: int, heads: int) -> dict[str, object]:
    """Distinct transformer layers on the released schema, at test scale."""

    template = _native_params()["score_model"]["atom_attention_encoder"]
    template = template["atom_encoder"]["diffusion_transformer"]["layers"][0]
    assert template["adaln"]["s_scale"]["kernel"].shape == (dim, dim)
    assert template["pair_bias_attn"]["num_heads"] == heads

    def scaled(value, factor):
        if isinstance(value, dict):
            return {k: scaled(v, factor) for k, v in value.items()}
        if isinstance(value, int):
            return value
        return value * factor

    return {"layers": [scaled(template, 1.0 + i) for i in range(layers)]}


@pytest.mark.parametrize("use_scan", [True, False])
def test_compact_pair_bias_matches_a_sample_repeated_bias_bitwise(
    use_scan: bool,
) -> None:
    # Broadcasting the sample axis per layer must reproduce, bit for bit, the
    # sample-repeated bias the layer loop used to be handed.
    batch, multiplicity, windows = 2, 3, 2
    width, keys, dim, heads, layers = 4, 4, 8, 2, 3
    params = _bias_scan_params(layers, dim, heads)
    rng = np.random.default_rng(20260911)

    def draw(*shape):
        return jnp.asarray(rng.normal(size=shape), jnp.float32)

    rows = batch * multiplicity * windows
    a = draw(rows, width, dim)
    s = draw(rows, width, dim)
    mask = jnp.ones((rows, width), jnp.float32)
    compact = draw(batch, windows, width, keys, layers * heads)
    repeated = jnp.reshape(
        jnp.repeat(compact, multiplicity, axis=0),
        (rows, width, keys, layers * heads),
    )
    flat = jnp.reshape(compact, (batch * windows, width, keys, layers * heads))

    def run(bias, bias_multiplicity, num_windows):
        return diffusion_transformer_forward(
            params,
            a=a,
            s=s,
            bias=bias,
            mask=mask,
            to_keys=None,
            multiplicity=1,
            use_scan=use_scan,
            bias_multiplicity=bias_multiplicity,
            bias_num_windows=num_windows,
        )

    reference = jax.jit(lambda: run(repeated, 1, None))()
    actual = jax.jit(lambda: run(flat, multiplicity, windows))()
    np.testing.assert_array_equal(np.asarray(reference), np.asarray(actual))

    # A sample-major/window-major ordering slip has to be visible, or the
    # comparison above proves nothing about the broadcast's index map.
    permuted = jnp.reshape(
        compact[::-1], (batch * windows, width, keys, layers * heads)
    )
    wrong = jax.jit(lambda: run(permuted, multiplicity, windows))()
    assert float(jnp.max(jnp.abs(reference - wrong))) > 1e-4


@pytest.mark.parametrize("use_scan", [False, True])
def test_singleton_conditioning_still_broadcasts_over_the_activation_batch(
    use_scan: bool,
) -> None:
    # A width-1 `s` against a wider activation batch, at the default
    # multiplicity, broadcast before the s-term alignment and still must.
    rows, width, dim, heads, layers = 2, 4, 8, 2, 3
    params = _bias_scan_params(layers, dim, heads)
    rng = np.random.default_rng(20261127)
    a = jnp.asarray(rng.normal(size=(rows, width, dim)), jnp.float32)
    s = jnp.asarray(rng.normal(size=(1, width, dim)), jnp.float32)
    bias = jnp.asarray(rng.normal(size=(rows, width, width, layers * heads)))
    bias = bias.astype(jnp.float32)
    mask = jnp.ones((rows, width), jnp.float32)

    def run(s_arg):
        return diffusion_transformer_forward(
            params,
            a=a,
            s=s_arg,
            bias=bias,
            mask=mask,
            to_keys=None,
            multiplicity=1,
            use_scan=use_scan,
        )

    expected = np.asarray(run(jnp.repeat(s, rows, axis=0)))
    for fn in (run, jax.jit(run)):
        np.testing.assert_allclose(np.asarray(fn(s)), expected, rtol=0.0, atol=1e-6)


def test_compact_pair_bias_shrinks_the_scan_operand() -> None:
    # The scan carries every layer's bias at once, so the sample axis must not
    # reach it: the repeat is what the merging reshape would make un-bitcastable.
    batch, multiplicity, windows = 2, 3, 2
    width, keys, dim, heads, layers = 4, 8, 8, 2, 3
    params = {
        "diffusion_transformer": _bias_scan_params(layers, dim, heads),
    }
    rng = np.random.default_rng(20260911)

    def draw(*shape):
        return jnp.asarray(rng.normal(size=shape), jnp.float32)

    rows = batch * multiplicity
    atoms = windows * width
    indexing = get_indexing_matrix(k=windows, w=width, h_keys=keys)
    bias = draw(batch, windows, width, keys, layers * heads)

    jaxpr = jax.make_jaxpr(
        lambda bias: atom_transformer_forward(
            params,
            q=draw(rows, atoms, dim),
            c=draw(rows, atoms, dim),
            bias=bias,
            to_keys=lambda x: single_to_keys(x, indexing, w=width, h_keys=keys),
            mask=jnp.ones((rows, atoms), jnp.float32),
            attn_window_queries=width,
            attn_window_keys=keys,
            multiplicity=multiplicity,
        )
    )(bias)

    scan = next(e for e in jaxpr.jaxpr.eqns if e.primitive.name == "scan")
    # The pair bias is the only rank-5 operand the layer scan takes.
    (operand,) = [v for v in scan.invars if v.aval.ndim == 5]
    assert operand.aval.shape == (layers, batch * windows, width, keys, heads)

    # Nothing between the caller's bias and that operand may grow it.
    producers = {v: eqn for eqn in jaxpr.jaxpr.eqns for v in eqn.outvars}
    chain, cursor = [], operand
    while cursor in producers:
        eqn = producers[cursor]
        chain.append(eqn.primitive.name)
        cursor = eqn.invars[0]
    assert set(chain) <= {"reshape", "transpose"}, chain
    assert cursor is jaxpr.jaxpr.invars[0]
    assert int(np.prod(cursor.aval.shape)) == int(np.prod(operand.aval.shape))

    body = scan.params["jaxpr"]
    body = getattr(body, "jaxpr", body)
    assert "broadcast_in_dim" in {eqn.primitive.name for eqn in body.eqns}


def test_eager_token_bias_casts_per_layer_inside_scan() -> None:
    """Eager token-transformer bias stays compact across the layer scan.

    The conditioning cache is stored in bfloat16 under the released mixed
    trunk, and `diffusion_score_model_forward` must pass it into
    `diffusion_transformer_forward` without widening the full `[1, N, N, L*H]`
    tensor beforehand. The cast happens per layer inside the scan body.
    """
    native = _native_params()
    inputs = _model_inputs()
    conditioning = diffusion_conditioning_forward(
        native["diffusion_conditioning"],
        s_trunk=inputs["s_trunk"],
        z_trunk=inputs["z_trunk"],
        relative_position_encoding=inputs["relative_position_encoding"],
        feats=inputs["feats"],
        token_layers=1,
        lazy_token_trans_bias=False,
        compute_dtype=jnp.bfloat16,
    )
    assert conditioning["token_trans_bias"].dtype == jnp.bfloat16

    jaxpr = jax.make_jaxpr(
        lambda bias: diffusion_score_model_forward(
            native["score_model"],
            s_inputs=inputs["s_inputs"],
            s_trunk=inputs["s_trunk"],
            r_noisy=inputs["r_noisy"],
            times=inputs["times"],
            feats=inputs["feats"],
            diffusion_conditioning={**conditioning, "token_trans_bias": bias},
            multiplicity=1,
            use_scan=True,
            token_layers=1,
        )
    )(conditioning["token_trans_bias"])

    # The token transformer is the scan whose rank-5 bias operand carries the
    # token pair axes `(layers, 1, N, N, heads)` (N = token count).
    n_token = inputs["feats"]["token_pad_mask"].shape[1]
    token_scans = [
        eqn
        for eqn in jaxpr.jaxpr.eqns
        if eqn.primitive.name == "scan"
        and any(
            v.aval.ndim == 5 and v.aval.shape[2:4] == (n_token, n_token)
            for v in eqn.invars
        )
    ]
    assert len(token_scans) == 1
    scan = token_scans[0]
    (operand,) = [
        v
        for v in scan.invars
        if v.aval.ndim == 5 and v.aval.shape[2:4] == (n_token, n_token)
    ]
    assert operand.aval.dtype == jnp.bfloat16

    producers = {v: eqn for eqn in jaxpr.jaxpr.eqns for v in eqn.outvars}
    chain, cursor = [], operand
    while cursor in producers:
        eqn = producers[cursor]
        chain.append(eqn.primitive.name)
        cursor = eqn.invars[0]
    assert set(chain) <= {"reshape", "transpose"}, chain
    assert cursor is jaxpr.jaxpr.invars[0]

    body = scan.params["jaxpr"]
    body = getattr(body, "jaxpr", body)
    converts = [
        eqn
        for eqn in body.eqns
        if eqn.primitive.name == "convert_element_type"
        and eqn.invars[0].aval.dtype == jnp.bfloat16
        and eqn.outvars[0].aval.dtype == jnp.float32
        and eqn.invars[0].aval.shape[-3:-1] == (n_token, n_token)
    ]
    assert converts

    with pytest.raises(ValueError, match="bias_layer_dtype requires an explicit bias"):
        diffusion_transformer_forward(
            native["score_model"]["token_transformer"],
            a=jnp.zeros((1, n_token, 8), dtype=jnp.float32),
            s=jnp.zeros((1, n_token, 8), dtype=jnp.float32),
            bias=None,
            mask=jnp.ones((1, n_token), dtype=jnp.float32),
            layer_limit=1,
            bias_layer_dtype=jnp.float32,
        )


@pytest.mark.parametrize("compute_dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("use_scan", [True, False])
def test_eager_per_layer_cast_matches_lazy_token_bias_bitwise_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
    compute_dtype: jnp.dtype,
    use_scan: bool,
) -> None:
    """Eager token-transformer bias with per-layer cast matches lazy on CPU."""
    from foldjax.models.boltz2.models.diffusion import diffusion as diffusion_impl

    native = _native_params()
    inputs = _model_inputs()
    params = {"trunk": {}, "conditioned_diffusion": native}
    trunk = {
        "s_inputs": inputs["s_inputs"],
        "s": inputs["s_trunk"],
        "z": inputs["z_trunk"],
        "relative_position_encoding": inputs["relative_position_encoding"],
    }
    feats = dict(inputs["feats"])

    seen_bias_calls: list[tuple[jnp.dtype | None, jnp.dtype | None]] = []
    real_dt = diffusion_impl.diffusion_transformer_forward

    def spy_dt(*args, **kwargs):
        bias = kwargs.get("bias")
        seen_bias_calls.append(
            (None if bias is None else bias.dtype, kwargs.get("bias_layer_dtype"))
        )
        return real_dt(*args, **kwargs)

    monkeypatch.setattr(diffusion_impl, "diffusion_transformer_forward", spy_dt)

    def run(lazy: bool):
        return trunk_impl.boltz2_sample_forward(
            params,
            feats,
            jax.random.PRNGKey(0),
            trunk=trunk,
            num_sampling_steps=3,
            multiplicity=2,
            token_layers=1,
            compute_dtype=compute_dtype,
            use_scan=use_scan,
            lazy_token_trans_bias=lazy,
            augmentation=False,
            alignment_reverse_diff=False,
        )["sample_atom_coords"]

    eager = np.asarray(jax.jit(lambda: run(False))())
    assert seen_bias_calls and all(
        call == (jnp.dtype(compute_dtype), jnp.dtype(jnp.float32))
        for call in seen_bias_calls
    )
    seen_bias_calls.clear()
    lazy = np.asarray(jax.jit(lambda: run(True))())
    assert seen_bias_calls and all(call == (None, None) for call in seen_bias_calls)
    assert eager.dtype == lazy.dtype == np.float32
    np.testing.assert_array_equal(eager.view(np.uint32), lazy.view(np.uint32))


def test_eager_and_lazy_token_bias_have_distinct_lowered_compile_keys() -> None:
    """Lazy and eager token-bias programs never share a lowered cache entry."""
    from foldjax import memory_policy
    from foldjax.executable_cache_verify import lowered_cache_key
    from foldjax.models.boltz2 import api as native_api

    native = _native_params()
    inputs = _model_inputs()
    params = {"trunk": {}, "conditioned_diffusion": native}
    trunk = {
        "s_inputs": inputs["s_inputs"],
        "s": inputs["s_trunk"],
        "z": inputs["z_trunk"],
        "relative_position_encoding": inputs["relative_position_encoding"],
    }
    feats = dict(inputs["feats"])
    key = jax.random.PRNGKey(0)

    lazy_choice = native_api._resolve_lazy_token_trans_bias(
        3000,
        budget=memory_policy.resolve_budget(
            pool_bytes=24 * 2**30, card_bytes=24 * 2**30, override_gib=None
        ),
    )
    eager_choice = native_api._resolve_lazy_token_trans_bias(
        3000,
        budget=memory_policy.resolve_budget(
            pool_bytes=48 * 2**30, card_bytes=48 * 2**30, override_gib=None
        ),
    )
    assert (lazy_choice, eager_choice) == (True, False)

    def lower(lazy: bool):
        return jax.jit(
            lambda p, f, k, t: trunk_impl.boltz2_sample_forward(
                p,
                f,
                k,
                trunk=t,
                num_sampling_steps=2,
                multiplicity=1,
                token_layers=1,
                compute_dtype=jnp.bfloat16,
                use_scan=True,
                lazy_token_trans_bias=lazy,
                augmentation=False,
                alignment_reverse_diff=False,
            )
        ).lower(params, feats, key, trunk)

    lazy_lowered = lower(lazy_choice)
    eager_lowered = lower(eager_choice)
    assert lazy_lowered.as_text() != eager_lowered.as_text()
    assert lowered_cache_key(lazy_lowered) != lowered_cache_key(eager_lowered)


def _width1_score_case(
    *,
    multiplicity: int,
    batch: int = 1,
    token_layers: int = 2,
    score_compute_dtype: jnp.dtype | None = None,
):
    """Build a multi-layer, non-zero-transition score case with distinct samples."""

    native = _native_params()
    inputs = _model_inputs()
    rng = np.random.default_rng(20261127)

    def draw(*shape: int, scale: float = 0.08) -> jnp.ndarray:
        return jnp.asarray(rng.normal(scale=scale, size=shape), dtype=jnp.float32)

    single_dim = native["score_model"]["a_norm"]["scale"].shape[0]
    hidden_dim = 2 * single_dim
    transitions = [
        {
            "norm": {
                "scale": jnp.ones((single_dim,), dtype=jnp.float32),
                "bias": draw(single_dim, scale=0.02),
            },
            "fc1": {"kernel": draw(single_dim, hidden_dim)},
            "fc2": {"kernel": draw(single_dim, hidden_dim)},
            "fc3": {"kernel": draw(hidden_dim, single_dim)},
        }
        for _ in range(2)
    ]

    def scale_tree(node, factor: float):
        if isinstance(node, dict):
            return {k: scale_tree(v, factor) for k, v in node.items()}
        if isinstance(node, list):
            return [scale_tree(v, factor) for v in node]
        if isinstance(node, int):
            return node
        return node * factor

    token_layer_template = native["score_model"]["token_transformer"]["layers"][0]
    token_proj_template = native["diffusion_conditioning"]["token_trans_proj_z"][0]
    score_params = {
        **native["score_model"],
        "single_conditioner": {
            **native["score_model"]["single_conditioner"],
            "transitions": transitions,
        },
        "token_transformer": {
            "layers": [
                scale_tree(token_layer_template, 1.0 + 0.25 * i)
                for i in range(token_layers)
            ]
        },
    }
    if score_compute_dtype is not None:
        score_params = trunk_impl._cast_score_params(score_params, score_compute_dtype)

    cond_params = {
        **native["diffusion_conditioning"],
        "token_trans_proj_z": [
            scale_tree(token_proj_template, 1.0 + 0.15 * i)
            for i in range(token_layers)
        ],
    }

    s_inputs = inputs["s_inputs"]
    s_trunk = inputs["s_trunk"]
    z_trunk = inputs["z_trunk"]
    relp = inputs["relative_position_encoding"]
    feats = dict(inputs["feats"])
    if batch > 1:
        s_inputs = jnp.concatenate(
            [s_inputs + draw(*s_inputs.shape, scale=0.05) for _ in range(batch)],
            axis=0,
        )
        s_trunk = jnp.concatenate(
            [s_trunk + draw(*s_trunk.shape, scale=0.05) for _ in range(batch)],
            axis=0,
        )
        z_trunk = jnp.concatenate(
            [z_trunk + draw(*z_trunk.shape, scale=0.02) for _ in range(batch)],
            axis=0,
        )
        relp = jnp.repeat(relp, batch, axis=0)
        feats = {k: jnp.repeat(v, batch, axis=0) for k, v in feats.items()}

    conditioning = diffusion_conditioning_forward(
        cond_params,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        relative_position_encoding=relp,
        feats=feats,
        token_layers=token_layers,
        lazy_token_trans_bias=False,
        bias_dtype=score_compute_dtype,
    )
    # Give every sample distinct coordinates so sample-broadcast errors are visible.
    r_noisy = jnp.repeat(inputs["r_noisy"], batch * multiplicity, axis=0) + draw(
        batch * multiplicity, *inputs["r_noisy"].shape[1:], scale=0.15
    )
    times_compact = jnp.asarray(
        [0.17 + 0.23 * b for b in range(batch)], dtype=jnp.float32
    )
    times_expanded = jnp.repeat(times_compact, multiplicity, axis=0)
    return {
        "score_params": score_params,
        "s_inputs": s_inputs,
        "s_trunk": s_trunk,
        "r_noisy": r_noisy,
        "times_compact": times_compact,
        "times_expanded": times_expanded,
        "feats": feats,
        "conditioning": conditioning,
        "token_layers": token_layers,
    }


def _historical_width_m_score_forward(
    params,
    *,
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    r_noisy: jnp.ndarray,
    times: jnp.ndarray,
    feats,
    diffusion_conditioning,
    multiplicity: int,
    use_scan: bool,
    token_layers: int,
    score_compute_dtype: jnp.dtype | None = None,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Explicit historical width-M single-conditioning reference path."""

    from foldjax.models.boltz2.models.diffusion.atom import (
        atom_attention_decoder_forward,
        atom_attention_encoder_forward,
    )
    from foldjax.models.boltz2.models.primitives._common import (
        layer_norm as _layer_norm,
    )
    from foldjax.models.boltz2.models.primitives._common import linear as _linear
    from foldjax.models.boltz2.models.trunk_blocks.conditioning import (
        single_conditioning_forward,
    )

    residual_dtype = r_noisy.dtype
    bias_dtype = (
        residual_dtype
        if score_compute_dtype is None
        else jnp.dtype(score_compute_dtype)
    )
    s, _ = single_conditioning_forward(
        params["single_conditioner"],
        times,
        jnp.repeat(s_trunk, multiplicity, axis=0),
        jnp.repeat(s_inputs, multiplicity, axis=0),
        eps=eps,
    )
    if score_compute_dtype is not None:
        s = s.astype(residual_dtype)

    atom_to_token_idx = diffusion_conditioning.get("atom_to_token_idx")
    a, q_skip, c_skip = atom_attention_encoder_forward(
        params["atom_attention_encoder"],
        feats=feats,
        q=diffusion_conditioning["q"].astype(residual_dtype),
        c=diffusion_conditioning["c"].astype(residual_dtype),
        atom_enc_bias=diffusion_conditioning["atom_enc_bias"].astype(bias_dtype),
        to_keys=diffusion_conditioning["to_keys"],
        r=r_noisy,
        multiplicity=multiplicity,
        eps=eps,
        atom_to_token_idx=atom_to_token_idx,
        atom_to_token_plan=diffusion_conditioning.get("atom_to_token_plan"),
    )

    s_to_a = params["s_to_a_linear"]
    a = a + _linear(
        _layer_norm(s, s_to_a["norm"]["scale"], s_to_a["norm"]["bias"], eps),
        s_to_a["linear"]["kernel"],
    )

    mask = jnp.repeat(feats["token_pad_mask"], multiplicity, axis=0)
    token_bias = diffusion_conditioning.get("token_trans_bias")
    bias_precision = diffusion_conditioning.get("token_trans_bias_precision")
    a = diffusion_transformer_forward(
        params["token_transformer"],
        a=a,
        s=s,
        bias=token_bias,
        mask=mask.astype(jnp.float32),
        multiplicity=multiplicity,
        eps=eps,
        use_scan=use_scan,
        layer_limit=token_layers,
        bias_params=diffusion_conditioning.get("token_trans_bias_params"),
        bias_input=diffusion_conditioning.get("token_trans_bias_input"),
        bias_normed_input=diffusion_conditioning.get("token_trans_bias_normed_input"),
        bias_compute_dtype=None if bias_precision is None else bias_precision.dtype,
        bias_out_dtype=None if score_compute_dtype is None else bias_dtype,
        bias_layer_dtype=None if token_bias is None else bias_dtype,
    )
    a_norm = params["a_norm"]
    a = _layer_norm(a, a_norm["scale"], a_norm["bias"], eps)
    return atom_attention_decoder_forward(
        params["atom_attention_decoder"],
        a=a,
        q=q_skip,
        c=c_skip,
        atom_dec_bias=diffusion_conditioning["atom_dec_bias"].astype(bias_dtype),
        feats=feats,
        to_keys=diffusion_conditioning["to_keys"],
        multiplicity=multiplicity,
        eps=eps,
        atom_to_token_idx=atom_to_token_idx,
    )


@pytest.mark.parametrize("jit_mode", [False, True])
@pytest.mark.parametrize("use_scan", [False, True])
@pytest.mark.parametrize("multiplicity", [1, 5])
def test_width1_single_conditioning_matches_historical_width_m(
    jit_mode: bool,
    use_scan: bool,
    multiplicity: int,
) -> None:
    """Width-1 single conditioning reproduces width-M within FP32 GEMM rounding."""

    case = _width1_score_case(multiplicity=multiplicity, batch=1, token_layers=2)

    def run_actual(times_arg, s_trunk_arg):
        return diffusion_score_model_forward(
            case["score_params"],
            s_inputs=case["s_inputs"],
            s_trunk=s_trunk_arg,
            r_noisy=case["r_noisy"],
            times=times_arg,
            feats=case["feats"],
            diffusion_conditioning=case["conditioning"],
            multiplicity=multiplicity,
            use_scan=use_scan,
            token_layers=case["token_layers"],
        )

    def run_reference(times_arg, s_trunk_arg):
        return _historical_width_m_score_forward(
            case["score_params"],
            s_inputs=case["s_inputs"],
            s_trunk=s_trunk_arg,
            r_noisy=case["r_noisy"],
            times=times_arg,
            feats=case["feats"],
            diffusion_conditioning=case["conditioning"],
            multiplicity=multiplicity,
            use_scan=use_scan,
            token_layers=case["token_layers"],
        )

    actual_fn = jax.jit(run_actual) if jit_mode else run_actual
    ref_fn = jax.jit(run_reference) if jit_mode else run_reference

    reference = np.asarray(ref_fn(case["times_expanded"], case["s_trunk"]))
    actual_from_expanded = np.asarray(
        actual_fn(case["times_expanded"], case["s_trunk"])
    )
    actual_from_compact = np.asarray(actual_fn(case["times_compact"], case["s_trunk"]))

    assert actual_from_compact.shape == reference.shape
    # Per-sample times keep the historical path exactly.
    np.testing.assert_array_equal(
        actual_from_expanded.view(np.uint32),
        reference.view(np.uint32),
    )
    if multiplicity == 1:
        np.testing.assert_array_equal(
            actual_from_compact.view(np.uint32),
            reference.view(np.uint32),
        )
    else:
        np.testing.assert_allclose(actual_from_compact, reference, rtol=0.0, atol=1e-6)

    # Confirm the single-conditioning path actively contributes to the output.
    perturbed = np.asarray(
        actual_fn(case["times_compact"], case["s_trunk"] + jnp.float32(1.0))
    )
    assert float(np.max(np.abs(actual_from_compact - perturbed))) > 1e-4


@pytest.mark.parametrize("jit_mode", [False, True])
@pytest.mark.parametrize("use_scan", [False, True])
def test_width1_single_conditioning_supports_multi_structure_batch_and_bf16(
    jit_mode: bool,
    use_scan: bool,
) -> None:
    """B>1 keeps the historical path; BF16 score kernels stay aligned with width-M."""

    batch_case = _width1_score_case(multiplicity=3, batch=2, token_layers=2)

    def run_batch_actual(times_arg):
        return diffusion_score_model_forward(
            batch_case["score_params"],
            s_inputs=batch_case["s_inputs"],
            s_trunk=batch_case["s_trunk"],
            r_noisy=batch_case["r_noisy"],
            times=times_arg,
            feats=batch_case["feats"],
            diffusion_conditioning=batch_case["conditioning"],
            multiplicity=3,
            use_scan=use_scan,
            token_layers=batch_case["token_layers"],
        )

    def run_batch_ref(times_arg):
        return _historical_width_m_score_forward(
            batch_case["score_params"],
            s_inputs=batch_case["s_inputs"],
            s_trunk=batch_case["s_trunk"],
            r_noisy=batch_case["r_noisy"],
            times=times_arg,
            feats=batch_case["feats"],
            diffusion_conditioning=batch_case["conditioning"],
            multiplicity=3,
            use_scan=use_scan,
            token_layers=batch_case["token_layers"],
        )

    batch_actual_fn = jax.jit(run_batch_actual) if jit_mode else run_batch_actual
    batch_ref_fn = jax.jit(run_batch_ref) if jit_mode else run_batch_ref
    batch_ref = np.asarray(batch_ref_fn(batch_case["times_expanded"]))
    batch_expanded = np.asarray(batch_actual_fn(batch_case["times_expanded"]))
    # More than one structure keeps the historical path exactly.
    np.testing.assert_array_equal(
        batch_expanded.view(np.uint32), batch_ref.view(np.uint32)
    )

    bf16_case = _width1_score_case(
        multiplicity=5,
        batch=1,
        token_layers=2,
        score_compute_dtype=jnp.bfloat16,
    )

    def run_bf16_actual():
        return diffusion_score_model_forward(
            bf16_case["score_params"],
            s_inputs=bf16_case["s_inputs"],
            s_trunk=bf16_case["s_trunk"],
            r_noisy=bf16_case["r_noisy"],
            times=bf16_case["times_compact"],
            feats=bf16_case["feats"],
            diffusion_conditioning=bf16_case["conditioning"],
            multiplicity=5,
            use_scan=use_scan,
            token_layers=bf16_case["token_layers"],
            score_compute_dtype=jnp.bfloat16,
        )

    def run_bf16_ref():
        return _historical_width_m_score_forward(
            bf16_case["score_params"],
            s_inputs=bf16_case["s_inputs"],
            s_trunk=bf16_case["s_trunk"],
            r_noisy=bf16_case["r_noisy"],
            times=bf16_case["times_expanded"],
            feats=bf16_case["feats"],
            diffusion_conditioning=bf16_case["conditioning"],
            multiplicity=5,
            use_scan=use_scan,
            token_layers=bf16_case["token_layers"],
            score_compute_dtype=jnp.bfloat16,
        )

    bf16_actual_fn = jax.jit(run_bf16_actual) if jit_mode else run_bf16_actual
    bf16_ref_fn = jax.jit(run_bf16_ref) if jit_mode else run_bf16_ref
    bf16_actual = np.asarray(bf16_actual_fn())
    bf16_ref = np.asarray(bf16_ref_fn())
    assert bf16_actual.dtype == np.float32
    np.testing.assert_allclose(bf16_actual, bf16_ref, rtol=0.0, atol=5e-3)


def test_width1_single_conditioning_keeps_single_path_and_scan_terms_at_width_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single conditioning and token-transformer AdaLN terms stay width-1 at M=5."""

    from foldjax.models.boltz2.models.diffusion import diffusion as diffusion_impl

    multiplicity = 5
    token_layers = 2
    case = _width1_score_case(
        multiplicity=multiplicity, batch=1, token_layers=token_layers
    )
    n_tokens = case["feats"]["token_pad_mask"].shape[1]
    single_dim = case["score_params"]["a_norm"]["scale"].shape[0]

    seen_single: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    real_single = diffusion_impl.single_conditioning_forward

    def spy_single(params, times, s_trunk, s_inputs, **kwargs):
        seen_single.append((times.shape, s_trunk.shape, s_inputs.shape))
        return real_single(params, times, s_trunk, s_inputs, **kwargs)

    seen_dt: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    real_dt = diffusion_impl.diffusion_transformer_forward

    def spy_dt(*args, **kwargs):
        seen_dt.append((kwargs["a"].shape, kwargs["s"].shape))
        return real_dt(*args, **kwargs)

    monkeypatch.setattr(diffusion_impl, "single_conditioning_forward", spy_single)
    monkeypatch.setattr(diffusion_impl, "diffusion_transformer_forward", spy_dt)

    jaxpr = jax.make_jaxpr(
        lambda r_noisy, times: diffusion_score_model_forward(
            case["score_params"],
            s_inputs=case["s_inputs"],
            s_trunk=case["s_trunk"],
            r_noisy=r_noisy,
            times=times,
            feats=case["feats"],
            diffusion_conditioning=case["conditioning"],
            multiplicity=multiplicity,
            use_scan=True,
            token_layers=token_layers,
        )
    )(case["r_noisy"], case["times_compact"])

    assert seen_single == [
        (
            (1,),
            (1, n_tokens, case["s_trunk"].shape[-1]),
            (1, n_tokens, case["s_inputs"].shape[-1]),
        )
    ]
    assert seen_dt == [
        ((multiplicity, n_tokens, single_dim), (1, n_tokens, single_dim))
    ]

    token_scans = [
        eqn
        for eqn in jaxpr.jaxpr.eqns
        if eqn.primitive.name == "scan"
        and any(
            v.aval.ndim == 5 and v.aval.shape[2:4] == (n_tokens, n_tokens)
            for v in eqn.invars
        )
    ]
    assert len(token_scans) == 1
    scan = token_scans[0]
    s_term_operands = [
        v
        for v in scan.invars
        if v.aval.ndim == 4
        and v.aval.shape[0] == token_layers
        and v.aval.shape[2:] == (n_tokens, single_dim)
    ]
    assert len(s_term_operands) == 6
    assert all(
        v.aval.shape == (token_layers, 1, n_tokens, single_dim)
        for v in s_term_operands
    )


def test_width1_single_conditioning_edge_and_error_contracts() -> None:
    """Only one structure with one-entry times changes path; the rest is historical."""

    def outcome(fn, **kwargs):
        try:
            return np.asarray(fn(**kwargs))
        except Exception as error:  # noqa: BLE001 - compared by type below
            return type(error)

    def assert_historical(case, *, multiplicity, **overrides):
        kwargs = {
            "s_inputs": case["s_inputs"],
            "s_trunk": case["s_trunk"],
            "r_noisy": case["r_noisy"],
            "times": case["times_compact"],
            "feats": case["feats"],
            "diffusion_conditioning": case["conditioning"],
            "multiplicity": multiplicity,
            "use_scan": True,
            "token_layers": case["token_layers"],
        }
        kwargs.update(overrides)
        expected = outcome(
            lambda **kw: _historical_width_m_score_forward(case["score_params"], **kw),
            **kwargs,
        )
        actual = outcome(
            lambda **kw: diffusion_score_model_forward(case["score_params"], **kw),
            **kwargs,
        )
        if isinstance(expected, type):
            assert actual is expected
        else:
            np.testing.assert_array_equal(
                actual.view(np.uint32), expected.view(np.uint32)
            )

    # Width-1 conditioning beside two activation rows with distinct times, at
    # the default multiplicity, takes the historical path.
    single = _width1_score_case(multiplicity=1, batch=1, token_layers=1)
    assert_historical(
        single,
        multiplicity=1,
        r_noisy=jnp.repeat(single["r_noisy"], 2, axis=0),
        times=jnp.asarray([0.2, 0.7], dtype=jnp.float32),
    )
    # A scalar times, a mismatched s_inputs batch, and coordinates that do not
    # match batch * multiplicity keep the historical result or error.
    assert_historical(single, multiplicity=1, times=jnp.float32(0.2))
    assert_historical(single, multiplicity=1, times=0.2)
    # Rank-2 and scalar conditioning keep the historical result or error.
    triple_rank2 = _width1_score_case(multiplicity=3, batch=1, token_layers=1)
    assert_historical(
        triple_rank2,
        multiplicity=3,
        s_trunk=triple_rank2["s_trunk"][:, 0, :],
        s_inputs=triple_rank2["s_inputs"][:, 0, :],
    )
    assert_historical(single, multiplicity=1, s_inputs=jnp.float32(0.0))
    # Unequal conditioning token counts, a non-int multiplicity, and a missing
    # conditioner with no s_inputs keep the historical result or error.
    assert_historical(
        triple_rank2, multiplicity=3, s_inputs=triple_rank2["s_inputs"][:, :-1, :]
    )
    assert_historical(triple_rank2, multiplicity=np.int64(3))
    missing = dict(triple_rank2, score_params={
        k: v for k, v in triple_rank2["score_params"].items()
        if k != "single_conditioner"
    })
    assert_historical(missing, multiplicity=3, s_inputs=None)
    assert_historical(
        single, multiplicity=1, s_inputs=jnp.repeat(single["s_inputs"], 2, axis=0)
    )
    triple = _width1_score_case(multiplicity=3, batch=1, token_layers=1)
    assert_historical(
        triple,
        multiplicity=3,
        r_noisy=jnp.repeat(triple["r_noisy"], 2, axis=0),
    )
