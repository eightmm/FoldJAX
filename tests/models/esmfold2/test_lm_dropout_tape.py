import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import model


def _loop_inputs():
    shape = (1, 2, 2, 4)
    z = jnp.arange(16, dtype=jnp.float32).reshape(shape) / 10
    params = {
        "parcae_log_delta": jnp.zeros(4),
        "parcae_log_a": jnp.zeros(4),
        "parcae_b_cont": jnp.eye(4),
        "parcae_input_norm.weight": jnp.ones(4),
        "parcae_input_norm.bias": jnp.zeros(4),
    }
    settings = model.ModelSettings(
        d_pair=4, trunk_n_layers=0, lm_encoder_n_layers=None, lm_dropout=0.25
    )
    return z, params, settings


def test_dropout_no_tape_preserves_historical_draw_and_key_mapping():
    key = jax.random.key(3)
    x = jnp.arange(32, dtype=jnp.float32).reshape(2, 4, 4)
    keep = jax.random.bernoulli(key, 0.75, x.shape)
    expected = jnp.where(keep, x / 0.75, 0.0)
    np.testing.assert_array_equal(model._dropout(key, x, 0.25), expected)
    np.testing.assert_array_equal(
        model._dropout(jax.random.key(99), x, 0.25, keep_mask=keep), expected
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_loop_mask_tape_is_key_independent_and_consumes_each_loop(compiled):
    z, params, settings = _loop_inputs()
    masks = (jnp.arange(48).reshape(3, *z.shape) % 3) != 0

    def run(key, tape):
        return model.run_loops(
            key,
            z,
            z / 2,
            z + 1,
            None,
            jnp.ones(z.shape[:-1]),
            params,
            settings=settings,
            total_steps=3,
            lm_dropout_masks=tape,
        )

    fn = jax.jit(run) if compiled else run
    first = fn(jax.random.key(0), masks)
    np.testing.assert_array_equal(first, fn(jax.random.key(99), masks))
    assert not np.array_equal(first, fn(jax.random.key(0), masks[::-1]))


@pytest.mark.parametrize(
    "bad", ["dtype", "shape", "absent_lm", "disabled", "zero_rate"]
)
def test_lm_tape_invalid_contract_rejected_before_parameters(bad):
    z, _, settings = _loop_inputs()
    masks = jnp.ones((3, *z.shape), dtype=bool)
    lm = z
    if bad == "dtype":
        masks = masks.astype(jnp.float32)
    elif bad == "shape":
        masks = masks[:2]
    elif bad == "absent_lm":
        lm = None
    elif bad == "disabled":
        settings = replace(settings, per_loop_lm_dropout=False)
    elif bad == "zero_rate":
        settings = replace(settings, lm_dropout=0)
    with pytest.raises(ValueError, match="LM dropout tape"):
        model.run_loops(
            jax.random.key(0),
            z,
            z,
            lm,
            None,
            jnp.ones(z.shape[:-1]),
            {},
            settings=settings,
            total_steps=3,
            lm_dropout_masks=masks,
        )


@pytest.mark.parametrize("rate", [0.15, 0.2, 0.25, 0.3])
@pytest.mark.parametrize("compiled", [False, True])
def test_bf16_dropout_cuda_source_arithmetic(rate, compiled):
    """Source-derived CUDA opmath oracle, not a claim of GPU execution.

    PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb Dropout.cu
    lines 57,110,230-231; AccumulateType.h maps CUDA BFloat16 to float.
    CPU F.dropout is not a CUDA rounding oracle.
    """
    x = jnp.asarray(np.random.default_rng(9).normal(size=4096), jnp.bfloat16)
    key = jax.random.key(0)
    keep = jax.random.bernoulli(key, 1 - rate, x.shape)
    scale = np.float32(1.0 / float(np.float32(1 - rate)))
    expected = jnp.asarray(
        np.asarray(x, np.float32) * np.asarray(keep, np.float32) * scale,
        jnp.bfloat16,
    )

    def fn(a, m):
        return model._dropout(key, a, rate, keep_mask=m)

    if compiled:
        fn = jax.jit(fn)
    np.testing.assert_array_equal(fn(x, keep), expected)
    np.testing.assert_array_equal(model._dropout(key, x, rate), expected)
    if rate != 0.25:
        assert not np.array_equal(jnp.where(keep, x / (1 - rate), 0), expected)


@pytest.mark.skipif(
    os.environ.get("FOLDJAX_RUN_NATIVE_DROPOUT_CUDA") != "1",
    reason="explicit queued CUDA operator probe only",
)
@pytest.mark.parametrize("rate", [0.15, 0.2, 0.25, 0.3])
def test_actual_native_cuda_bf16_dropout_same_operands(rate):
    """Run only under the GPU queue; JAX may remain on CPU for this op gate."""
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available()
    rng = np.random.default_rng(9)
    values = rng.normal(size=4096).astype(np.float32)
    values[values == 0] = 1
    x = torch.from_numpy(values).cuda().to(torch.bfloat16)
    torch.cuda.manual_seed_all(17)
    before = torch.cuda.get_rng_state()
    expected = torch.nn.functional.dropout(x, p=rate, training=True)
    torch.cuda.set_rng_state(before)
    repeated = torch.nn.functional.dropout(x, p=rate, training=True)
    assert torch.equal(expected, repeated)
    keep = (expected != 0).cpu().numpy()
    jax_x = jnp.asarray(x.float().cpu().numpy(), jnp.bfloat16)
    actual = jax.jit(
        lambda a, m: model._dropout(jax.random.key(0), a, rate, keep_mask=m)
    )(jax_x, jnp.asarray(keep))
    np.testing.assert_array_equal(
        np.asarray(actual, np.float32), expected.float().cpu().numpy()
    )
