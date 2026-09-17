import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import embedders, trunk


@pytest.mark.parametrize("native", [False, True])
def test_opm_masked_output_retains_post_projection_bias_division(native):
    params = {
        "norm.weight": jnp.ones(8),
        "norm.bias": jnp.zeros(8),
        "W.weight": jnp.ones((4, 8)),
        "Wout.weight": jnp.ones((2, 4)),
        "Wout.bias": jnp.array([1.0037, 0.1037]),
    }
    result = jax.jit(
        lambda x, mask: trunk.outer_product_mean(
            x, params, "", msa_mask=mask, native_autocast=native
        )
    )(jnp.ones((1, 3, 4, 8)), jnp.zeros((1, 3, 4)))
    dtype = jnp.bfloat16 if native else jnp.float32
    assert result.dtype == dtype
    np.testing.assert_array_equal(
        result, jnp.broadcast_to(params["Wout.bias"].astype(dtype), result.shape)
    )


def test_the_outer_product_norm_is_asked_for_float32(monkeypatch):
    """This norm's width is read, not just rounded, so it stays float32.

    The mask multiply immediately after it uses `normalised.dtype` to
    reproduce native's promotion of the bfloat16 projection back to float32.
    Every other native norm in this module hands its result to one linear
    that rounds it, which is what makes `out_dtype=bfloat16` free there and
    a policy change here -- so the width this site asks for is pinned rather
    than left to whoever narrows the next one.
    """
    params = {
        "norm.weight": jnp.ones(8),
        "norm.bias": jnp.zeros(8),
        "W.weight": jnp.ones((4, 8)),
        "Wout.weight": jnp.ones((2, 4)),
    }
    original = trunk._autocast_norm
    asked = []

    def norm(x, p, prefix, eps=1e-5, *, out_dtype=jnp.float32):
        asked.append((prefix, jnp.dtype(out_dtype)))
        return original(x, p, prefix, eps, out_dtype=out_dtype)

    monkeypatch.setattr(trunk, "_autocast_norm", norm)
    projected = []
    linear = trunk._autocast_linear

    def record(x, p, prefix):
        projected.append((prefix, x.dtype))
        return linear(x, p, prefix)

    monkeypatch.setattr(trunk, "_autocast_linear", record)
    trunk.outer_product_mean(
        jnp.ones((1, 3, 4, 8)),
        params,
        "",
        msa_mask=jnp.ones((1, 3, 4)),
        native_autocast=True,
    )

    assert asked == [("norm", jnp.dtype(jnp.float32))]
    # ... and the promotion it exists for is still there: `Wout` reads the
    # bfloat16 outer product, `W` reads the float32 normalisation.
    assert dict(projected)["W"] == jnp.float32


@pytest.mark.parametrize("native", [False, True])
def test_msa_block_passes_original_weights_to_native_updates(monkeypatch, native):
    original = {"norm.weight": jnp.array([1.0037], jnp.float32)}
    other = {"other": jnp.ones(1, jnp.bfloat16)}
    calls = []

    def opm(msa, params, prefix, **kwargs):
        assert params is (original if native else other)
        assert kwargs["native_autocast"] is native
        calls.append(prefix)
        return jnp.zeros((1, 2, 2, 4))

    def pair_update(x, params, *args, **kwargs):
        assert params is (original if native else other)
        assert kwargs["native_autocast"] is native
        return jnp.zeros_like(x)

    monkeypatch.setattr(embedders, "outer_product_mean", opm)
    monkeypatch.setattr(embedders, "triangle_multiplicative", pair_update)
    monkeypatch.setattr(embedders, "transition", pair_update)
    embedders.msa_encoder_block(
        jnp.zeros((1, 2, 3, 4)),
        jnp.zeros((1, 2, 2, 4)),
        other,
        "msa_encoder.blocks.0",
        msa_mask=jnp.ones((1, 2, 3)),
        pair_mask=jnp.ones((1, 2, 2)),
        is_final=True,
        native_opm_params=original if native else None,
    )
    assert calls == ["msa_encoder.blocks.0.outer_product_mean"]
