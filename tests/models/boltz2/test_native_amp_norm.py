import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.diffusion.atom import _projection_layer_forward
from foldjax.models.boltz2.models.diffusion.diffusion_conditioning import (
    _projection_input_norm,
    _projection_list_forward,
)
from foldjax.models.boltz2.models.primitives import native_amp_norm, transition
from foldjax.models.boltz2.models.primitives._common import layer_norm
from foldjax.models.boltz2.models.triangle import triangle_attention
from foldjax.models.boltz2.models.trunk_blocks import msa, pairformer


@pytest.mark.parametrize("affine_dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
def test_private_cuda_norm_promotes_affine_before_final_fma(affine_dtype):
    # ESM calls this private helper directly, unlike Boltz's FP32-casting wrapper.
    traced = jax.make_jaxpr(
        lambda x, s, b: native_amp_norm._cuda_layer_norm(x, s, b)
    )(
        jnp.zeros((1, 256), jnp.float32),
        jnp.ones(256, affine_dtype),
        jnp.zeros(256, affine_dtype),
    )
    call = next(e for e in traced.jaxpr.eqns if e.primitive.name == "pallas_call")
    kernel = call.params["jaxpr"]
    kernel = getattr(kernel, "jaxpr", kernel)
    affine = [
        e for e in kernel.eqns if e.params.get("asm") == "fma.rn.f32 $0, $1, $2, $3;"
    ][-1]
    assert all(v.aval.shape == (256,) for v in affine.invars)
    assert all(v.aval.dtype == jnp.float32 for v in affine.invars)
    assert affine.outvars[0].aval.dtype == jnp.float32
    if affine_dtype != jnp.float32:
        producers = {v: e for e in kernel.eqns for v in e.outvars}
        for operand in (affine.invars[0], affine.invars[2]):
            convert = producers[operand]
            assert convert.primitive.name == "convert_element_type"
            assert convert.invars[0].aval.dtype == affine_dtype
            assert convert.params["new_dtype"] == jnp.float32


@pytest.mark.parametrize("input_dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("kernel_dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("starting", [True, False])
def test_pair_attention_selects_native_norm_only_for_mixed_amp(
    monkeypatch, input_dtype, kernel_dtype, starting
):
    calls = []

    def norm(x, scale, bias, eps):
        calls.append(x)
        return x

    monkeypatch.setattr(triangle_attention, "amp_layer_norm", norm)
    monkeypatch.setattr(triangle_attention, "_layer_norm", lambda x, *args: x)
    monkeypatch.setattr(triangle_attention, "_attention", lambda p, **kw: kw["q_x"])
    params = {
        "layer_norm": {"scale": jnp.ones(128), "bias": jnp.zeros(128)},
        "linear": {"kernel": jnp.zeros((128, 4), kernel_dtype)},
        "mha": {},
    }
    x = jnp.arange(512, dtype=jnp.float32).reshape(1, 2, 2, 128).astype(input_dtype)
    actual = triangle_attention.triangle_attention_forward(params, x, starting=starting)
    np.testing.assert_array_equal(actual, x)
    assert len(calls) == int(
        input_dtype == jnp.float32 and kernel_dtype == jnp.bfloat16
    )
    if calls:
        np.testing.assert_array_equal(calls[0], x if starting else x.swapaxes(1, 2))


@pytest.mark.parametrize("module", [msa, pairformer])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_pair_transition_native_norm_does_not_change_single_policy(
    monkeypatch, module, dtype
):
    calls = []

    def transition_stub(params, x, **kwargs):
        calls.append(kwargs)
        return jnp.zeros_like(x)

    monkeypatch.setattr(module, "transition_forward", transition_stub)
    for name in ("triangle_multiplication_forward", "triangle_attention_forward"):
        monkeypatch.setattr(module, name, lambda p, x, *a, **kw: jnp.zeros_like(x))
    params = {
        key: {} for key in ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end")
    }
    params["transition_z"] = {"fc1": {"kernel": jnp.zeros((4, 16), dtype)}}
    z, mask = jnp.ones((1, 2, 2, 4)), jnp.ones((1, 2, 2))
    if module is msa:
        module.pairformer_no_seq_layer_forward(params, z, mask)
    else:
        params.update(
            pre_norm_s={"scale": jnp.ones(4), "bias": jnp.zeros(4)},
            attention={},
            transition_s={},
        )
        monkeypatch.setattr(
            module,
            "attention_pair_bias_forward",
            lambda p, **kw: jnp.zeros_like(kw["s"]),
        )
        module.pairformer_layer_forward(
            params, jnp.ones((1, 2, 4)), z, mask[:, 0], mask
        )
        assert not calls[1].get("native_amp_norm", False)
    assert calls[0]["native_amp_norm"] == (dtype == jnp.bfloat16)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
def test_msa_transition_selects_native_norm_only_for_bf16(monkeypatch, dtype):
    calls = []

    def transition_stub(params, x, **kwargs):
        calls.append(kwargs)
        return jnp.zeros_like(x)

    monkeypatch.setattr(msa, "transition_forward", transition_stub)
    monkeypatch.setattr(
        msa, "pair_weighted_averaging_forward", lambda p, m, *a, **k: jnp.zeros_like(m)
    )
    z = jnp.zeros((1, 2, 2, 128))
    monkeypatch.setattr(msa, "outer_product_mean_forward", lambda *a, **k: z)
    monkeypatch.setattr(msa, "pairformer_no_seq_layer_forward", lambda p, z, *a, **k: z)
    params = {
        "pair_weighted_averaging": {},
        "msa_transition": {"fc1": {"kernel": jnp.zeros((64, 256), dtype)}},
        "outer_product_mean": {},
        "pairformer_layer": {},
    }
    msa.msa_layer_forward(
        params, z, jnp.ones((1, 3, 2, 64)), jnp.ones((1, 2, 2)), jnp.ones((1, 3, 2))
    )
    assert len(calls) == 1
    assert calls[0].get("native_amp_norm", False) == (dtype == jnp.bfloat16)


@pytest.mark.parametrize("width", [8, 16, 64, 128, 256])
def test_cpu_norm_and_affine_fallback(width):
    rng = np.random.default_rng(width)
    args = [
        jnp.asarray(rng.normal(size=shape), jnp.float32)
        for shape in ((2, 3, width), (width,), (width,))
    ]
    with jax.default_device(jax.devices("cpu")[0]):
        actual = jax.jit(native_amp_norm.amp_layer_norm)(*args, 1e-5)
        expected = jax.jit(layer_norm)(*args, 1e-5)
        np.testing.assert_array_equal(actual, expected)
        affine = jax.jit(native_amp_norm.amp_affine)(*args)
        reference = jax.jit(lambda x, s, b: x * s + b)(*args)
        np.testing.assert_array_equal(affine, reference)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
def test_opm_native_norm_selection_preserves_original_affine(monkeypatch, dtype):
    calls = []
    scale, bias = jnp.array([1.0001, 0.9999]), jnp.array([0.0001, -0.0001])

    def native_norm(x, s, b, eps):
        calls.append((x, s, b))
        return x

    monkeypatch.setattr(msa, "amp_layer_norm", native_norm)
    params = {
        "norm": {"scale": scale, "bias": bias},
        "proj_a": {"kernel": jnp.ones((2, 2), dtype)},
        "proj_b": {"kernel": jnp.ones((2, 2), dtype)},
        "proj_o": {
            "kernel": jnp.ones((4, 2), dtype),
            "bias": jnp.zeros(2),
        },
    }
    result = msa.outer_product_mean_forward(
        params, jnp.ones((1, 3, 2, 2)), jnp.ones((1, 3, 2)),
        preserve_native_amp_shape=True,
    )
    assert result.shape == (1, 2, 2, 2)
    assert len(calls) == int(dtype == jnp.bfloat16)
    if calls:
        assert calls[0][0].dtype == jnp.float32
        np.testing.assert_array_equal(calls[0][1], scale)
        np.testing.assert_array_equal(calls[0][2], bias)


def test_context_parallel_does_not_enter_cuda_kernel(monkeypatch):
    monkeypatch.setattr(native_amp_norm, "cp_mesh", lambda: object())

    def forbidden(*args, **kwargs):
        raise AssertionError("CP must use the partitionable JAX path")

    monkeypatch.setattr(native_amp_norm, "_cuda_layer_norm", forbidden)
    monkeypatch.setattr(native_amp_norm, "_cuda_affine", forbidden)
    x, scale, bias = jnp.ones((2, 16)), jnp.ones(16), jnp.zeros(16)
    np.testing.assert_array_equal(
        native_amp_norm.amp_layer_norm(x, scale, bias, 1e-5),
        layer_norm(x, scale, bias, 1e-5),
    )
    np.testing.assert_array_equal(native_amp_norm.amp_affine(x, scale, bias), x)
    np.testing.assert_array_equal(
        native_amp_norm.amp_affine(x, scale, bias, jnp.bfloat16),
        x.astype(jnp.bfloat16),
    )


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("width", [16, 64, 128, 256])
def test_amp_affine_out_dtype_is_one_rounding_not_two(width, jit):
    # Narrowing inside the affine must be the same single round-nearest-even
    # convert the caller would apply to an FP32 result, not a second rounding.
    rng = np.random.default_rng(width)
    args = [
        jnp.asarray(rng.normal(size=shape), jnp.float32)
        for shape in ((3, 5, width), (width,), (width,))
    ]
    wrap = jax.jit if jit else (lambda f: f)
    direct = wrap(lambda x, s, b: native_amp_norm.amp_affine(x, s, b, jnp.bfloat16))(
        *args
    )
    staged = wrap(
        lambda x, s, b: native_amp_norm.amp_affine(x, s, b).astype(jnp.bfloat16)
    )(*args)
    assert direct.dtype == jnp.bfloat16
    np.testing.assert_array_equal(direct, staged)


@pytest.mark.parametrize("out_dtype", [jnp.float32, jnp.bfloat16])
def test_cuda_affine_stores_out_dtype_after_a_single_convert(out_dtype):
    # Tracing needs no GPU: assert the kernel keeps its FP32 FMA and narrows
    # exactly once on the way to the store, so the output buffer is the only
    # thing ``out_dtype`` changes.
    traced = jax.make_jaxpr(
        lambda x, s, b: native_amp_norm._cuda_affine(x, s, b, out_dtype)
    )(
        jnp.zeros((2, 128), jnp.float32),
        jnp.ones(128, jnp.float32),
        jnp.zeros(128, jnp.float32),
    )
    call = next(e for e in traced.jaxpr.eqns if e.primitive.name == "pallas_call")
    assert all(v.aval.dtype == out_dtype for v in call.outvars)
    kernel = call.params["jaxpr"]
    kernel = getattr(kernel, "jaxpr", kernel)
    (fma,) = [
        e for e in kernel.eqns if e.params.get("asm") == "fma.rn.f32 $0, $1, $2, $3;"
    ]
    assert all(v.aval.dtype == jnp.float32 for v in fma.invars)
    assert fma.outvars[0].aval.dtype == jnp.float32
    converts = [e for e in kernel.eqns if e.primitive.name == "convert_element_type"]
    if out_dtype == jnp.float32:
        assert converts == []
    else:
        (convert,) = converts
        assert convert.invars[0] is fma.outvars[0]
        assert convert.params["new_dtype"] == out_dtype


@pytest.mark.parametrize("dtype", [None, jnp.float32, jnp.bfloat16])
def test_conditioning_norm_is_opt_in_and_survives_row_chunks(monkeypatch, dtype):
    calls = []

    def norm(x, scale, bias, eps):
        calls.append(x.shape)
        return layer_norm(x, scale, bias, eps)

    monkeypatch.setattr(transition, "amp_layer_norm", norm)
    params = {
        "norm": {"scale": jnp.ones(16), "bias": jnp.zeros(16)},
        "fc1": {"kernel": jnp.ones((16, 32))},
        "fc2": {"kernel": jnp.ones((16, 32))},
        "fc3": {"kernel": jnp.ones((32, 16))},
    }
    x = jnp.ones((1, 3, 2, 16))
    transition.transition_forward(params, x, compute_dtype=dtype)
    assert not calls
    transition.transition_forward(
        params, x, compute_dtype=dtype, native_amp_norm=True, row_chunk_size=2
    )
    assert calls == ([(1, 2, 2, 16), (1, 1, 2, 16)] if dtype == jnp.bfloat16 else [])


@pytest.mark.parametrize("width", [16, 128])
def test_native_width_multi_layer_lazy_eager_projection_parity(width):
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=(1, 3, 3, width)), jnp.float32)
    params = [
        {
            "norm": {
                "scale": jnp.asarray(rng.normal(size=width), jnp.float32),
                "bias": jnp.asarray(rng.normal(size=width), jnp.float32),
            },
            "linear": {
                "kernel": jnp.asarray(rng.normal(size=(width, 16)), jnp.float32)
            },
        }
        for _ in range(3)
    ]

    @jax.jit
    def run(x, params):
        normalized = _projection_input_norm(x, 1e-5, native_amp=True)
        lazy = jnp.concatenate(
            [
                _projection_layer_forward(
                    p, None, 1e-5, normed_input=normalized, compute_dtype=jnp.bfloat16
                )
                for p in params
            ],
            axis=-1,
        )
        eager = _projection_list_forward(params, x, 1e-5, compute_dtype=jnp.bfloat16)
        return lazy, eager.astype(jnp.float32)

    lazy, eager = run(x, params)
    np.testing.assert_array_equal(lazy, eager)
