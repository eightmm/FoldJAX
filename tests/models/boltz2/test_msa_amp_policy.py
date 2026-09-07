"""CUDA AMP boundaries from pinned Boltz trunkv2.py and its MSA layers.

References spell out operation-level casts, not a blanket parameter cast.
They run on CPU; native CUDA dispatch was independently checked with FakeTensor.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.trunk_blocks import msa


def _norm(x, params):
    x = x.astype(jnp.float32)
    centered = x - x.mean(-1, keepdims=True)
    normalized = centered * jax.lax.rsqrt(
        jnp.mean(centered * centered, -1, keepdims=True) + 1e-5
    )
    return normalized * params["scale"] + params["bias"]


def _amp_linear(x, kernel, bias=None):
    dtype = kernel.dtype
    if bias is None:
        return jnp.matmul(x.astype(dtype), kernel)
    output = jnp.matmul(x.astype(dtype), kernel, preferred_element_type=jnp.float32)
    output = output + bias.astype(dtype).astype(jnp.float32)
    return output.astype(dtype)


def _array(rng, shape, dtype=jnp.float32):
    return jnp.asarray(rng.normal(0, 0.25, shape), dtype=dtype)


def _affine(rng, channels):
    return {"scale": 1 + _array(rng, (channels,)), "bias": _array(rng, (channels,))}


def _pwa_case(tokens, dtype=jnp.bfloat16):
    rng = np.random.default_rng(48)
    c_m, c_z, heads, hidden = 4, 3, 3, 2
    params = {
        "norm_m": _affine(rng, c_m),
        "norm_z": _affine(rng, c_z),
        "proj_m": {"kernel": _array(rng, (c_m, heads * hidden), dtype)},
        "proj_g": {"kernel": _array(rng, (c_m, heads * hidden), dtype)},
        "proj_z": {"kernel": _array(rng, (c_z, heads), dtype)},
        "proj_o": {"kernel": _array(rng, (heads * hidden, c_m), dtype)},
    }
    m = _array(rng, (1, 3, tokens, c_m), dtype)
    z = _array(rng, (1, tokens, tokens, c_z))
    mask = jnp.asarray(rng.integers(0, 2, (1, tokens, tokens)), jnp.float32)
    return params, m, z, mask


def _native_pwa(params, m, z, mask, *, chunk_heads):
    m, z = _norm(m, params["norm_m"]), _norm(z, params["norm_z"])
    n_heads = params["proj_z"]["kernel"].shape[-1]
    hidden = params["proj_m"]["kernel"].shape[-1] // n_heads
    groups = n_heads if chunk_heads else 1
    result = None
    for group in range(groups):
        first, last = (group, group + 1) if chunk_heads else (0, n_heads)
        start, stop = first * hidden, last * hidden
        v = _amp_linear(m, params["proj_m"]["kernel"][:, start:stop])
        v = v.reshape(*m.shape[:3], last - first, hidden).transpose(0, 3, 1, 2, 4)
        logits = _amp_linear(z, params["proj_z"]["kernel"][:, first:last])
        logits = logits.transpose(0, 3, 1, 2).astype(jnp.float32)
        weights = jax.nn.softmax(logits + (1 - mask[:, None]) * -1e6, -1)
        gate = _amp_linear(m, params["proj_g"]["kernel"][:, start:stop])
        gate = jax.nn.sigmoid(gate.astype(jnp.float32)).astype(gate.dtype)
        output = jnp.einsum("bhij,bhsjd->bhsid", weights.astype(v.dtype), v)
        output = output.transpose(0, 2, 3, 1, 4).reshape(*m.shape[:3], stop - start)
        output = _amp_linear(gate * output, params["proj_o"]["kernel"][start:stop])
        result = output if result is None else result + output
    return result


def _opm_case(tokens, depth=5, dtype=jnp.bfloat16):
    rng = np.random.default_rng(187)
    channels, hidden, output = 4, 8, 3
    params = {
        "norm": _affine(rng, channels),
        "proj_a": {"kernel": _array(rng, (channels, hidden), dtype)},
        "proj_b": {"kernel": _array(rng, (channels, hidden), dtype)},
        "proj_o": {
            "kernel": _array(rng, (hidden * hidden, output), dtype),
            "bias": _array(rng, (output,)),
        },
    }
    m = _array(rng, (1, depth, tokens, channels), dtype)
    mask = jnp.asarray(rng.integers(0, 2, (1, depth, tokens)), jnp.float32)
    return params, m, mask


def _native_opm(params, m, mask, *, chunk_hidden):
    dtype = params["proj_a"]["kernel"].dtype
    mask = mask.astype(m.dtype)
    m = _norm(m, params["norm"])
    a = _amp_linear(m, params["proj_a"]["kernel"]) * mask[..., None]
    b = _amp_linear(m, params["proj_b"]["kernel"]) * mask[..., None]
    count = (mask[:, :, :, None] * mask[:, :, None, :]).sum(1, dtype=jnp.float32)
    count = jnp.maximum(count, 1)[..., None]
    hidden = a.shape[-1]
    result = None
    for start in range(0, hidden, 4 if chunk_hidden else hidden):
        stop = min(start + (4 if chunk_hidden else hidden), hidden)
        product = jnp.einsum(
            "bsic,bsjd->bijcd", a[..., start:stop].astype(dtype), b.astype(dtype)
        )
        product = product.reshape(*product.shape[:3], -1).astype(jnp.float32) / count
        output = _amp_linear(
            product,
            params["proj_o"]["kernel"][start * hidden : stop * hidden],
            None if chunk_hidden else params["proj_o"]["bias"],
        )
        result = output if result is None else result + output
    return result + params["proj_o"]["bias"] if chunk_hidden else result


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_sparse_msa_embedding_rounds_full_projection_once(jit, dtype):
    rng = np.random.default_rng(6)
    codes = jnp.asarray(rng.integers(0, 5, (1, 7, 11)))
    params = {
        "msa_proj": {"kernel": _array(rng, (8, 4), dtype)},
        "s_proj": {"kernel": jnp.zeros((3, 4), dtype)},
    }
    extras = [_array(rng, codes.shape) for _ in range(3)]
    features = jnp.concatenate((jax.nn.one_hot(codes, 5), jnp.stack(extras, -1)), -1)
    expected = _amp_linear(features, params["msa_proj"]["kernel"])
    function = partial(msa._msa_input_embedding, num_tokens=5)
    actual = (jax.jit(function) if jit else function)(
        params, jnp.zeros((1, 11, 3)), codes, *extras
    )
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("budget,expected_chunk", [(1 << 30, 385), (4096, 7)])
def test_native_amp_opm_shape_respects_existing_product_budget(
    monkeypatch, budget, expected_chunk
):
    params, m, mask = _opm_case(385)
    monkeypatch.setattr(msa, "_OPM_BUDGET_BYTES", budget)
    original = msa._auto_outer_product_chunk
    seen = []

    def record(n_tokens, widened, requested):
        seen.append(requested)
        return original(n_tokens, widened, requested)

    monkeypatch.setattr(msa, "_auto_outer_product_chunk", record)
    # Trace only: the test checks the policy before the budget's usual clamp.
    jax.eval_shape(
        lambda p, m, mask: msa.outer_product_mean_forward(
            p, m, mask, chunk_size=7, preserve_native_amp_shape=True
        ),
        params,
        m,
        mask,
    )
    assert seen == [expected_chunk]


def test_native_shape_option_leaves_fp32_path_unchanged():
    params, m, mask = _opm_case(11, dtype=jnp.float32)
    original = jax.make_jaxpr(
        lambda p, m, mask: msa.outer_product_mean_forward(p, m, mask, chunk_size=7)
    )(params, m, mask)
    native_shape = jax.make_jaxpr(
        lambda p, m, mask: msa.outer_product_mean_forward(
            p, m, mask, chunk_size=7, preserve_native_amp_shape=True
        )
    )(params, m, mask)
    assert str(original) == str(native_shape)


@pytest.mark.parametrize("tokens", [384, 385])
@pytest.mark.parametrize("row_chunk", [0, 2])
@pytest.mark.parametrize("jit", [False, True])
def test_pwa_matches_native_head_rounding_at_threshold(tokens, row_chunk, jit):
    params, m, z, mask = _pwa_case(tokens)
    reference = partial(_native_pwa, chunk_heads=tokens > 384)
    expected = (jax.jit(reference) if jit else reference)(params, m, z, mask)
    function = partial(msa.pair_weighted_averaging_forward, row_chunk_size=row_chunk)
    actual = (jax.jit(function) if jit else function)(params, m, z, mask)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, expected)
    if tokens > 384:
        assert not np.array_equal(
            expected, _native_pwa(params, m, z, mask, chunk_heads=False)
        )


@pytest.mark.parametrize("tokens", [384, 385])
@pytest.mark.parametrize("token_chunk", [127, 512])
@pytest.mark.parametrize("jit", [False, True])
def test_opm_matches_native_contraction_and_external_bias(tokens, token_chunk, jit):
    params, m, mask = _opm_case(tokens)
    reference = partial(_native_opm, chunk_hidden=tokens > 384)
    expected = (jax.jit(reference) if jit else reference)(params, m, mask)
    function = partial(msa.outer_product_mean_forward, chunk_size=token_chunk)
    actual = (jax.jit(function) if jit else function)(params, m, mask)
    assert actual.dtype == (jnp.float32 if tokens > 384 else jnp.bfloat16)
    np.testing.assert_array_equal(actual, expected)


def test_opm_keeps_odd_large_msa_counts_in_float32():
    params, m, mask = _opm_case(3, depth=257)
    mask = jnp.ones_like(mask)
    expected = _native_opm(params, m, mask, chunk_hidden=False)
    actual = msa.outer_product_mean_forward(params, m, mask)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("input_dtype", [jnp.bfloat16, jnp.float32])
def test_amp_policy_follows_kernel_not_input_dtype(dtype, input_dtype):
    params, m, z, mask = _pwa_case(5, dtype)
    m = m.astype(input_dtype)
    reference = partial(_native_pwa, chunk_heads=False)
    actual = jax.jit(msa.pair_weighted_averaging_forward)(params, m, z, mask)
    expected = jax.jit(reference)(params, m, z, mask)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected)

    params, m, mask = _opm_case(5, dtype=dtype)
    m = m.astype(input_dtype)
    actual = jax.jit(msa.outer_product_mean_forward)(params, m, mask)
    expected = jax.jit(partial(_native_opm, chunk_hidden=False))(params, m, mask)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected)


def test_fp32_msa_projections_keep_unrounded_outputs():
    params, m, z, mask = _pwa_case(5, jnp.float32)
    expected = _native_pwa(params, m, z, mask, chunk_heads=False)
    actual = msa.pair_weighted_averaging_forward(params, m, z, mask)
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(actual, expected)
    params, m, mask = _opm_case(5, dtype=jnp.float32)
    expected = _native_opm(params, m, mask, chunk_hidden=False)
    actual = msa.outer_product_mean_forward(params, m, mask)
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("tokens, expected_chunk", [(384, None), (385, 32)])
def test_msa_transition_uses_native_hidden_chunk(monkeypatch, tokens, expected_chunk):
    seen = {}
    monkeypatch.setattr(
        msa, "pair_weighted_averaging_forward", lambda p, m, *a, **k: jnp.zeros_like(m)
    )
    monkeypatch.setattr(
        msa,
        "outer_product_mean_forward",
        lambda p, *a, **k: jnp.zeros((1, tokens, tokens, 1)),
    )
    monkeypatch.setattr(msa, "pairformer_no_seq_layer_forward", lambda p, z, *a, **k: z)

    def transition(params, m, **kwargs):
        seen.update(kwargs)
        return jnp.zeros_like(m)

    monkeypatch.setattr(msa, "transition_forward", transition)
    params = {
        "pair_weighted_averaging": {},
        "outer_product_mean": {},
        "pairformer_layer": {},
        "msa_transition": {"fc1": {"kernel": jnp.zeros((1, 1), jnp.bfloat16)}},
    }
    msa.msa_layer_forward(
        params,
        jnp.zeros((1, tokens, tokens, 1)),
        jnp.zeros((1, 1, tokens, 1), jnp.bfloat16),
        jnp.ones((1, tokens, tokens)),
        jnp.ones((1, 1, tokens)),
    )
    assert seen.get("chunk_size") == expected_chunk
    assert seen.get("compute_dtype") == jnp.bfloat16
