"""Pin the small non-v2 publisher runner's FP32 confidence exception.

At publisher commit 4c355be4553512f72453ecbfb65e69f4c35d1413,
``runner.inference.update_inference_configs`` overrides the base configuration:
non-v2 inputs with at most 2,560 tokens disable confidence autocast. This is
not evidence for v2 or longer inputs, whose runner keeps confidence AMP enabled.

Since the port's default moved, ``--amp-policy auto`` no longer selects this
route below 2,560 tokens -- ``upstream`` and ``fp32`` do. What is pinned here
is the ``confidence_autocast=False`` program itself, which those two spellings
have to keep reaching; the released default's own realised dtypes are pinned
in ``test_amp_policy.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.models import model
from foldjax.models.protenix.models.heads import confidence
from foldjax.models.protenix.models.trunk_blocks.pairformer import PairformerStackParams

from .test_model import _toy_features, _toy_params
from .test_trunk import _zero_pairformer_block


@pytest.mark.parametrize("trunk_dtype", [None, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("num_samples", [1, 5])
def test_released_small_input_confidence_keeps_fp32_operators(
    monkeypatch, trunk_dtype, num_samples
) -> None:
    params = _toy_params()
    params = params._replace(
        confidence=params.confidence._replace(
            pairformer_stack=PairformerStackParams(
                blocks=(_zero_pairformer_block(2, 2),)
            )
        )
    )
    original_confidence = params.confidence
    if trunk_dtype is not None:
        params = model.cast_trunk_params(params, trunk_dtype)
    assert params.confidence is original_confidence

    seen = {"entry": [], "stack": [], "linear": []}
    original_head = model.confidence_head
    original_stack = confidence.pairformer_stack
    original_linear = confidence.linear

    def head(
        features, s_inputs, s_trunk, z_trunk, pair_mask, coords, head_params, **kw
    ):
        seen["entry"].append(
            (s_inputs.dtype, s_trunk.dtype, z_trunk.dtype, coords.dtype)
        )
        assert head_params is original_confidence
        return original_head(
            features, s_inputs, s_trunk, z_trunk, pair_mask, coords, head_params, **kw
        )

    def stack(s, z, pair_mask, stack_params, **kw):
        seen["stack"].append((s.dtype, z.dtype))
        return original_stack(s, z, pair_mask, stack_params, **kw)

    def linear(x, linear_params):
        result = original_linear(x, linear_params)
        seen["linear"].append((x.dtype, linear_params.weight.dtype, result.dtype))
        return result

    monkeypatch.setattr(model, "confidence_head", head)
    monkeypatch.setattr(confidence, "pairformer_stack", stack)
    monkeypatch.setattr(confidence, "linear", linear)
    noise = jnp.ones((num_samples, 3, 3), dtype=jnp.float32)
    output = model.protenix_infer_static(
        _toy_features(),
        params,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        key=None,
        num_samples=num_samples,
        init_noise=noise,
        step_noises=(jnp.zeros_like(noise),),
        num_recycles=1,
        input_atom_heads=1,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
        centre_each_step=False,
        run_confidence_scores=False,
        trunk_dtype=trunk_dtype,
    )
    jax.block_until_ready(output)

    # Observe execution boundaries, not just prepared parameter storage. Both
    # sample routes must enter the head and its nonempty Pairformer stack.
    for boundary, records in seen.items():
        assert records, f"{boundary} was not exercised"
        assert all(dtype == jnp.float32 for row in records for dtype in row)
    for name in ("plddt", "pae", "pde", "resolved"):
        assert output[name].dtype == jnp.float32
        assert output[name].shape[0] == num_samples
        assert np.isfinite(np.asarray(output[name])).all()


def test_mixed_trunk_preserves_raw_fp32_conditioning_tail():
    features = _toy_features()
    features["profile"] = jnp.arange(64, dtype=jnp.float32).reshape(2, 32) / 71
    features["deletion_mean"] = jnp.asarray([0.1234567, 0.7654321], jnp.float32)
    params = model.cast_trunk_params(_toy_params(), jnp.bfloat16)
    output = model.protenix_infer_static(
        features,
        params,
        jnp.asarray([1.0, 0.0], jnp.float32),
        key=None,
        num_samples=1,
        init_noise=jnp.ones((1, 3, 3)),
        step_noises=jnp.zeros((1, 1, 3, 3)),
        input_atom_heads=1,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
        centre_each_step=False,
        run_confidence=False,
        trunk_dtype=jnp.bfloat16,
    )
    expected = jnp.concatenate(
        [features["restype"], features["profile"], features["deletion_mean"][:, None]],
        axis=-1,
    )
    assert output["s_inputs"].dtype == jnp.float32
    np.testing.assert_array_equal(output["s_inputs"][..., -65:], expected)
    assert output["s_trunk"].dtype == jnp.bfloat16
    assert output["z_trunk"].dtype == jnp.bfloat16
    assert output["distogram_logits"].dtype == jnp.bfloat16
