"""Eval dropout is numerically inactive but promotes native MSA residuals."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.trunk_blocks import msa


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("compiled", [False, True])
def test_eval_msa_add_preserves_native_fp32_dropout_promotion(
    monkeypatch, dtype, compiled
):
    m = jnp.ones((1, 2, 3, 4), dtype)
    update = jnp.full_like(m, 1 / 1024)
    seen = []
    monkeypatch.setattr(msa, "pair_weighted_averaging_forward", lambda *a, **k: update)

    def transition(params, value, **kwargs):
        seen.append(value.dtype)
        return update

    def opm(params, value, *args, **kwargs):
        seen.append(value.dtype)
        return jnp.zeros((1, 3, 3, 2), jnp.float32)

    monkeypatch.setattr(msa, "transition_forward", transition)
    monkeypatch.setattr(msa, "outer_product_mean_forward", opm)
    monkeypatch.setattr(msa, "pairformer_no_seq_layer_forward", lambda p, z, *a, **k: z)
    params = {
        "pair_weighted_averaging": {},
        "msa_transition": {"fc1": {"kernel": jnp.zeros((4, 4), dtype)}},
        "outer_product_mean": {},
        "pairformer_layer": {},
    }
    fn = jax.jit(msa.msa_layer_forward) if compiled else msa.msa_layer_forward
    _, actual = fn(
        params, jnp.zeros((1, 3, 3, 2)), m, jnp.ones((1, 3, 3)), jnp.ones((1, 2, 3))
    )
    # Native get_dropout_mask produces FP32 ones even when training=False.
    expected = (m.astype(jnp.float32) + update.astype(jnp.float32)) + update.astype(
        jnp.float32
    )
    assert seen == [jnp.float32, jnp.float32]
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("scanned", [False, True])
def test_msa_stack_can_carry_promoted_residual(monkeypatch, scanned):
    dtype = jnp.bfloat16
    params = {
        "msa_proj": {"kernel": jnp.zeros((36, 4), dtype)},
        "s_proj": {"kernel": jnp.zeros((2, 4), dtype)},
        "layers": [{"index": jnp.asarray(0)}, {"index": jnp.asarray(1)}],
    }
    feats = {
        k: jnp.ones((1, 2, 3))
        for k in ("has_deletion", "deletion_value", "msa_paired", "msa_mask")
    }
    feats.update(msa=jnp.zeros((1, 2, 3), jnp.int32), token_pad_mask=jnp.ones((1, 3)))

    def layer(p, z, m, *args, **kwargs):
        return z, m.astype(jnp.float32) + 0.125

    monkeypatch.setattr(msa, "msa_layer_forward", layer)
    def fn(p, z, e, f):
        return msa.msa_module_forward(p, z, e, f, use_scan=scanned)

    result = jax.jit(fn)(params, jnp.ones((1, 3, 3, 2)), jnp.ones((1, 3, 2)), feats)
    np.testing.assert_array_equal(result, np.ones((1, 3, 3, 2)))
