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


@pytest.mark.parametrize("native", [False, True])
def test_msa_block_passes_original_weights_only_to_opm(monkeypatch, native):
    original = {"norm.weight": jnp.array([1.0037], jnp.float32)}
    other = {"other": jnp.ones(1, jnp.bfloat16)}
    calls = []

    def opm(msa, params, prefix, **kwargs):
        assert params is (original if native else other)
        assert kwargs["native_autocast"] is native
        calls.append(prefix)
        return jnp.zeros((1, 2, 2, 4))

    def unchanged(x, params, *args, **kwargs):
        assert params is other
        return jnp.zeros_like(x)

    monkeypatch.setattr(embedders, "outer_product_mean", opm)
    monkeypatch.setattr(embedders, "triangle_multiplicative", unchanged)
    monkeypatch.setattr(embedders, "transition", unchanged)
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
