"""ESMC-only native CUDA autocast boundaries; CPU tests do not prove SDPA parity."""

from dataclasses import asdict, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.models import esmc


def _params(dtype):
    rng = np.random.default_rng(9)
    shapes = {"embed.weight": (64, 8), "transformer.norm.weight": (8,)}
    for i in range(2):
        prefix = f"transformer.blocks.{i}."
        for name, shape in {
            "attn.layernorm_qkv.layer_norm_weight": (8,),
            "attn.layernorm_qkv.layer_norm_bias": (8,),
            "attn.layernorm_qkv.weight": (24, 8),
            "attn.q_ln.weight": (8,),
            "attn.k_ln.weight": (8,),
            "attn.out_proj.weight": (8, 8),
            "ffn.layer_norm_weight": (8,),
            "ffn.layer_norm_bias": (8,),
            "ffn.fc1_weight": (32, 8),
            "ffn.fc2_weight": (8, 16),
        }.items():
            shapes[prefix + name] = shape
    return {
        name: jnp.asarray(rng.normal(size=shape) * 0.1, dtype)
        for name, shape in shapes.items()
    }


@pytest.mark.parametrize("compiled", [False, True])
def test_norm_fp32_linear_bf16_and_stack_promotion(compiled):
    x = jnp.asarray(np.arange(16).reshape(2, 8) / 7, jnp.bfloat16)

    def stage(x):
        normalized = esmc._norm(x, jnp.ones(8, jnp.bfloat16), autocast_bfloat16=True)
        return normalized, esmc._matmul(
            normalized, jnp.eye(8, dtype=jnp.bfloat16), autocast_bfloat16=True
        )

    normed, projected = (jax.jit(stage) if compiled else stage)(x)
    assert normed.dtype == jnp.float32 and projected.dtype == jnp.bfloat16
    settings = esmc.ESMCSettings(
        d_model=8, n_heads=2, n_layers=2, autocast_bfloat16=True
    )
    params = _params(jnp.bfloat16)
    ids = jnp.array([[0, 4, 2]], jnp.int32)

    def run(params):
        return esmc.encode(ids, jnp.zeros_like(ids), params, settings=settings)

    hidden = (jax.jit(run) if compiled else run)(params)
    assert hidden.dtype == jnp.float32 and hidden.shape == (3, 1, 3, 8)
    np.testing.assert_array_equal(
        hidden[0], np.asarray(params["embed.weight"][ids], np.float32)
    )
    original = esmc.encode(
        ids,
        jnp.zeros_like(ids),
        params,
        settings=replace(settings, autocast_bfloat16=False),
    )
    assert original.dtype == jnp.bfloat16


def test_qk_rotary_and_attention_outputs_remain_bf16(monkeypatch):
    seen = []
    original = esmc.apply_rotary

    def rotary(x, cos, sin):
        seen.append((x.dtype, cos.dtype, sin.dtype))
        return original(x, cos, sin)

    monkeypatch.setattr(esmc, "apply_rotary", rotary)
    result = esmc.attention(
        jnp.ones((1, 3, 8), jnp.bfloat16),
        _params(jnp.bfloat16),
        "transformer.blocks.0.attn",
        n_heads=2,
        sequence_id=jnp.zeros((1, 3), jnp.int32),
        rope=esmc.rotary_tables(3, 4),
        autocast_bfloat16=True,
    )
    assert result.dtype == jnp.bfloat16
    assert seen == [(jnp.bfloat16,) * 3] * 2


@pytest.mark.parametrize("stage", [False, True])
@pytest.mark.parametrize(
    "dtype,override,expected",
    [
        ("bfloat16", None, True),
        ("float32", None, False),
        ("bfloat16", False, False),
        ("float32", True, True),
    ],
)
def test_actual_load_routes_explicit_policy(
    tmp_path, monkeypatch, stage, dtype, override, expected
):
    (tmp_path / "esmc").mkdir()
    params = {
        name: jnp.ones(1)
        for name in inference._LANGUAGE_MODEL_EMBEDDING_REQUIRED_PARAMETERS
    }
    monkeypatch.setattr(
        inference.structure_checkpoint, "load_parameters", lambda *a, **kw: params
    )
    monkeypatch.setattr(
        inference.structure_checkpoint,
        "load_settings",
        lambda *a: inference.structure_model.ModelSettings(),
    )
    monkeypatch.setattr(
        inference.esmc_checkpoint, "load_parameters", lambda *a, **kw: {}
    )
    monkeypatch.setattr(
        inference.esmc_checkpoint, "load_settings", lambda *a: esmc.ESMCSettings()
    )
    loader = inference.load_language_model_stage if stage else inference.load
    loaded = loader(tmp_path, esmc_dtype=dtype, esmc_autocast_bfloat16=override)
    assert loaded.esmc_settings.autocast_bfloat16 is expected


def test_autocast_setting_is_serialized_and_separates_static_cache_keys():
    plain = esmc.ESMCSettings()
    native = replace(plain, autocast_bfloat16=True)
    assert esmc.ESMCSettings(**asdict(native)) == native
    assert len({plain: "plain", native: "native"}) == 2
    assert not esmc.settings_from_config({}).autocast_bfloat16
