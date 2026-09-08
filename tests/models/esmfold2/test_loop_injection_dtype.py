"""Native loop casts, with network boundaries replaced by dtype sentinels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import model


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("encoder", [False, True])
@pytest.mark.parametrize("msa_mode", [None, "add", "overwrite"])
def test_native_loop_injection_casts_after_fp32_dropout(
    monkeypatch, compiled, encoder, msa_mode
):
    shape = (1, 2, 2, 4)
    dtype = jnp.bfloat16
    initial = jnp.full(shape, 0.375, dtype)
    lm = jnp.asarray(np.random.default_rng(9).normal(size=shape), jnp.float32)
    masks = jnp.ones((1, *shape), dtype=bool)
    masks = masks.at[0, 0, 0, 0, 0].set(False)
    refined_offset = jnp.float32(0.00371)
    msa_value = jnp.full(shape, 0.50171, jnp.float32)

    def trunk(value, params, prefix, **kwargs):
        assert value.dtype == dtype
        if prefix == "lm_encoder":
            return value.astype(jnp.float32) + refined_offset
        return value

    def msa(value, *args, **kwargs):
        assert value.dtype == dtype
        return msa_value

    def norm(value, *args, **kwargs):
        assert value.dtype == dtype
        return value

    monkeypatch.setattr(model, "folding_trunk", trunk)
    monkeypatch.setattr(model, "msa_encoder", msa)
    monkeypatch.setattr(model, "layer_norm", norm)
    params = {
        "parcae_log_delta": jnp.zeros(4),
        "parcae_log_a": jnp.zeros(4),
        "parcae_b_cont": jnp.eye(4),
        "parcae_input_norm.weight": jnp.ones(4),
        "parcae_input_norm.bias": jnp.zeros(4),
    }
    settings = model.ModelSettings(
        d_pair=4,
        trunk_n_layers=0,
        lm_encoder_n_layers=1 if encoder else None,
        lm_dropout=0.15,
        msa_n_layers=1 if msa_mode else None,
        msa_encoder_overwrite=msa_mode == "overwrite",
    )
    msa_inputs = None
    if msa_mode:
        msa_inputs = {
            "msa_one_hot": jnp.zeros((1, 2, 1, 32)),
            "msa_mask": jnp.ones((1, 2, 1)),
            "has_deletion": jnp.zeros((1, 2, 1)),
            "deletion_value": jnp.zeros((1, 2, 1)),
            "x_inputs": jnp.zeros((1, 2, 4)),
        }

    def run(hidden):
        return model.run_loops(
            jax.random.key(3),
            jnp.zeros_like(initial),
            initial,
            hidden,
            msa_inputs,
            jnp.ones(shape[:-1]),
            params,
            settings=settings,
            total_steps=1,
            lm_dropout_masks=masks,
        )

    dropped = jnp.where(masks[0], lm / 0.85, 0).astype(dtype)
    # A premature BF16 conversion before dropout changes this exact sentinel.
    premature = jnp.where(
        masks[0], lm.astype(dtype).astype(jnp.float32) / 0.85, 0
    ).astype(dtype)
    assert np.any(np.asarray(dropped) != np.asarray(premature))
    injected = initial if encoder else initial + dropped
    if msa_mode:
        injected = (
            msa_value.astype(dtype)
            if msa_mode == "overwrite"
            else injected + msa_value.astype(dtype)
        )
    if encoder:
        injected = injected + (
            dropped.astype(jnp.float32) + refined_offset
        ).astype(dtype)
    matrix = (jax.nn.softplus(jnp.zeros(4))[:, None] * jnp.eye(4)).astype(dtype)
    expected = jnp.matmul(injected, matrix.T)
    actual = (jax.jit(run) if compiled else run)(lm)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected)
