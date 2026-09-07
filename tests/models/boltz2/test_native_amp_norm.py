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


@pytest.mark.parametrize("width", [8, 16, 128, 256])
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
