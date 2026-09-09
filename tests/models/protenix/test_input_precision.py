import jax
import jax.numpy as jnp
import pytest

from foldjax.models.protenix.models.input_precision import native_input_autocast_params
from foldjax.models.protenix.models.primitives.primitives import (
    AutocastLinearParams,
    LayerNormParams,
    LinearParams,
)

from .test_model import _toy_params


@pytest.mark.parametrize("dtype", ["bfloat16", jnp.dtype("bfloat16"), jnp.bfloat16])
def test_trunk_cast_dtype_spellings_preserve_geometry(dtype):
    from foldjax.models.protenix.models.model import cast_trunk_params

    original = _toy_params()
    result = cast_trunk_params(original, dtype)
    assert (
        result.input_embedder.atom_encoder.cache.linear_d
        is original.input_embedder.atom_encoder.cache.linear_d
    )
    assert isinstance(result.input_embedder.atom_encoder.linear_q, AutocastLinearParams)


def test_native_input_autocast_preserves_original_geometry_weights():
    original = _toy_params().input_embedder
    result = native_input_autocast_params(original)
    assert result.atom_encoder.cache.linear_d is original.atom_encoder.cache.linear_d
    assert (
        result.atom_encoder.cache.linear_ref_pos
        is original.atom_encoder.cache.linear_ref_pos
    )
    assert isinstance(result.atom_encoder.linear_q, AutocastLinearParams)
    assert result.atom_encoder.linear_q.weight.dtype == jnp.bfloat16
    assert original.atom_encoder.linear_q.weight.dtype == jnp.float32


def test_native_input_autocast_rejects_already_rounded_weights():
    original = _toy_params().input_embedder
    narrowed = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") else x, original
    )
    with pytest.raises(ValueError, match="original FP32"):
        native_input_autocast_params(narrowed)


def test_native_input_autocast_covers_every_projection_and_preserves_norms():
    original = _toy_params().input_embedder
    result = native_input_autocast_params(original)
    def is_parameter(x):
        return isinstance(x, (LinearParams, AutocastLinearParams, LayerNormParams))

    before = jax.tree.leaves(original, is_leaf=is_parameter)
    after = jax.tree.leaves(result, is_leaf=is_parameter)
    assert len(before) == len(after)
    islands = (
        original.atom_encoder.cache.linear_ref_pos,
        original.atom_encoder.cache.linear_d,
    )
    narrowed = norms = 0
    for source, prepared in zip(before, after, strict=True):
        if isinstance(source, LayerNormParams):
            norms += 1
            assert prepared is source
            assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(prepared))
        elif isinstance(source, LinearParams):
            if any(source is island for island in islands):
                assert prepared is source
            else:
                narrowed += 1
                assert isinstance(prepared, AutocastLinearParams)
                assert all(x.dtype == jnp.bfloat16 for x in jax.tree.leaves(prepared))
    assert narrowed > 1
    assert norms > 0


def test_prepared_cli_loader_preserves_native_input_islands(tmp_path):
    from foldjax.models.protenix.bridge.weights_io import save_native_weights
    from foldjax.models.protenix.cli.predict import _load_prepared_params

    original = _toy_params()
    path = tmp_path / "weights.npz"
    save_native_weights(path, original, compress=False)
    loaded = _load_prepared_params(path, "bf16")
    cache = loaded.input_embedder.atom_encoder.cache
    assert cache.linear_ref_pos.weight.dtype == jnp.float32
    assert cache.linear_d.weight.dtype == jnp.float32
    assert isinstance(loaded.input_embedder.atom_encoder.linear_q, AutocastLinearParams)
    assert loaded.input_embedder.atom_encoder.linear_q.weight.dtype == jnp.bfloat16
    for name in ("linear_ref_pos", "linear_d"):
        assert jnp.array_equal(
            getattr(cache, name).weight,
            getattr(original.input_embedder.atom_encoder.cache, name).weight,
        )


def test_jitted_input_stage_does_not_prequantize_native_geometry(monkeypatch):
    from foldjax.models.protenix.models import model

    from .test_model import _toy_features

    params = model.cast_trunk_params(_toy_params(), jnp.bfloat16)
    features = _toy_features()
    seen = []

    def embed(values, encoder_params, **kwargs):
        seen.append(values["ref_pos"].dtype)
        assert (
            encoder_params.atom_encoder.cache.linear_ref_pos.weight.dtype == jnp.float32
        )
        return values["ref_pos"]

    monkeypatch.setattr(model, "input_feature_embedder", embed)
    run = jax.jit(
        lambda values: model.protenix_infer_static(
            values,
            params,
            jnp.ones(2),
            key=None,
            num_samples=1,
            trunk_dtype=jnp.bfloat16,
            stop_after_inputs=True,
        )["single_inputs"]
    )
    result = run(features)
    assert seen == [jnp.float32]
    assert jnp.array_equal(result, features["ref_pos"])


@pytest.mark.parametrize("use_scan", [False, True])
def test_real_encoder_projection_dtypes(monkeypatch, use_scan):
    from foldjax.models.protenix.models.diffusion import atom
    from foldjax.models.protenix.models.trunk_blocks.embedders import (
        input_feature_embedder,
    )

    from .test_model import _toy_features

    params = native_input_autocast_params(_toy_params().input_embedder)
    original_linear = atom.linear
    seen = []

    def observed_linear(x, projection):
        result = original_linear(x, projection)
        if isinstance(projection, AutocastLinearParams):
            assert result.dtype == jnp.bfloat16
            seen.append("ordinary")
        else:
            assert x.dtype == result.dtype == jnp.float32
            seen.append("geometry")
        return result

    monkeypatch.setattr(atom, "linear", observed_linear)
    result = jax.jit(
        lambda features: input_feature_embedder(
            features, params, n_token=2, n_heads=1,
            n_queries=2, n_keys=4, use_scan=use_scan,
        )
    )(_toy_features())
    assert result.dtype == jnp.float32
    assert seen.count("geometry") == 2
    assert seen.count("ordinary") > 2
