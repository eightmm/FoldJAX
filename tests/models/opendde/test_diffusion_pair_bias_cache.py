"""OpenDDE opt-in precomputed token pair-bias cache wiring, parity, and guards."""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends import opendde as backend_impl
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.models._stacking import StackedLayers
from foldjax.models.opendde import runner as opendde_runner
from foldjax.models.opendde.cli import predict as opendde_cli
from foldjax.models.opendde.models import diffusion_module as opendde_diffusion
from foldjax.models.opendde.models import model as opendde_model
from foldjax.models.opendde.models.diffusion_conditioning import (
    DiffusionConditioningParams,
    diffusion_conditioning_prepare_cache,
)
from foldjax.models.opendde.models.diffusion_precision import (
    native_diffusion_autocast_params,
)
from foldjax.models.protenix.models.diffusion import (
    diffusion as protenix_diffusion,
)
from foldjax.models.protenix.models.diffusion.atom import (
    AtomAttentionDecoderParams,
    AtomAttentionEncoderCacheParams,
    AtomAttentionEncoderParams,
    AtomPairMlpParams,
    atom_attention_encoder_prepare_diffusion_cache,
    rearrange_qk_to_dense_trunk,
)
from foldjax.models.protenix.models.diffusion.diffusion import DiffusionModuleParams
from foldjax.models.protenix.models.diffusion.transformer import (
    ConditionedTransitionParams,
    DiffusionTransformerBlockParams,
    DiffusionTransformerStackParams,
    build_diffusion_pair_bias_cache,
    diffusion_transformer_block,
    diffusion_transformer_stack,
)
from foldjax.models.protenix.models.primitives import attention as attention_impl
from foldjax.models.protenix.models.primitives.attention import (
    AttentionPairBiasParams,
    AttentionParams,
    attention_pair_bias,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LayerNormParams,
    LinearParams,
    TransitionParams,
    layer_norm,
)
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    FourierParams,
    RelativePositionParams,
)
from foldjax.schema import PredictionRequest
from tests._native_double import native_module
from tests.models.opendde.toy_params import inference_params


def _random_linear(
    rng: np.random.Generator, out_dim: int, in_dim: int, *, with_bias: bool = False
) -> LinearParams:
    scale = 0.25 / np.sqrt(max(in_dim, 1))
    weight = jnp.asarray(rng.normal(scale=scale, size=(out_dim, in_dim)), jnp.float32)
    bias = (
        jnp.asarray(rng.normal(scale=0.05, size=(out_dim,)), jnp.float32)
        if with_bias
        else None
    )
    return LinearParams(weight=weight, bias=bias)


def _random_norm(
    rng: np.random.Generator, dim: int, *, with_bias: bool = True
) -> LayerNormParams:
    weight = jnp.asarray(1.0 + rng.normal(scale=0.05, size=(dim,)), jnp.float32)
    bias = (
        jnp.asarray(rng.normal(scale=0.02, size=(dim,)), jnp.float32)
        if with_bias
        else None
    )
    return LayerNormParams(weight=weight, bias=bias)


def _random_transition(rng: np.random.Generator, dim: int) -> TransitionParams:
    hidden = 2 * dim
    return TransitionParams(
        layer_norm=_random_norm(rng, dim, with_bias=False),
        linear_a=_random_linear(rng, hidden, dim),
        linear_b=_random_linear(rng, hidden, dim),
        linear_out=_random_linear(rng, dim, hidden),
    )


def _random_transformer_block(
    rng: np.random.Generator,
    *,
    c_a: int,
    c_s: int,
    c_z: int,
    heads: int,
    cross_attention: bool,
    layernorm_z_bias: bool = False,
) -> DiffusionTransformerBlockParams:
    def make_adaln() -> AdaptiveLayerNormParams:
        return AdaptiveLayerNormParams(
            layernorm_a=LayerNormParams(weight=None, bias=None),
            layernorm_s=_random_norm(rng, c_s, with_bias=False),
            linear_s=_random_linear(rng, c_a, c_s, with_bias=True),
            linear_no_bias_s=_random_linear(rng, c_a, c_s),
        )

    adaln_a = make_adaln()
    adaln_kv = make_adaln() if cross_attention else None
    return DiffusionTransformerBlockParams(
        attention_pair_bias=AttentionPairBiasParams(
            layernorm_a=adaln_a,
            layernorm_kv=adaln_kv,
            attention=AttentionParams(
                linear_q=_random_linear(rng, c_a, c_a, with_bias=True),
                linear_k=_random_linear(rng, c_a, c_a),
                linear_v=_random_linear(rng, c_a, c_a),
                linear_o=_random_linear(rng, c_a, c_a),
                linear_g=_random_linear(rng, c_a, c_a),
            ),
            layernorm_z=_random_norm(rng, c_z, with_bias=layernorm_z_bias),
            linear_z=_random_linear(rng, heads, c_z),
            linear_a_last=_random_linear(rng, c_a, c_s, with_bias=True),
            has_s=True,
            cross_attention_mode=cross_attention,
        ),
        conditioned_transition=ConditionedTransitionParams(
            adaln=make_adaln(),
            linear_a1=_random_linear(rng, 2 * c_a, c_a),
            linear_a2=_random_linear(rng, 2 * c_a, c_a),
            linear_b=_random_linear(rng, c_a, 2 * c_a),
            linear_s=_random_linear(rng, c_a, c_s, with_bias=True),
        ),
    )


def _opendde_diffusion_fixture(
    seed: int = 31,
    *,
    n_blocks: int = 3,
    layernorm_z_bias: bool = True,
) -> dict[str, Any]:
    """Tiny multi-block OpenDDE denoiser fixture with distinct per-block weights."""
    rng = np.random.default_rng(seed)
    n_samples = 2
    n_struct = 5
    n_atom = 7
    n_queries = 2
    n_window_cols = 4
    c_s_inputs, c_s_trunk, c_z_trunk = 5, 6, 6
    c_s, c_z, c_atom, c_atompair, c_tok = 4, 4, 8, 4, 8
    heads = 2

    conditioning = DiffusionConditioningParams(
        relpe=RelativePositionParams(linear_no_bias=_random_linear(rng, c_z, 4)),
        layernorm_z_trunk=_random_norm(rng, c_z_trunk),
        linear_z_trunk=_random_linear(rng, c_z, c_z_trunk),
        layernorm_z=_random_norm(rng, 2 * c_z),
        linear_z=_random_linear(rng, c_z, 2 * c_z),
        transition_z1=_random_transition(rng, c_z),
        transition_z2=_random_transition(rng, c_z),
        layernorm_s=_random_norm(rng, c_s_trunk + c_s_inputs),
        linear_s=_random_linear(rng, c_s, c_s_trunk + c_s_inputs),
        fourier=FourierParams(
            w=jnp.asarray(rng.normal(size=(4,)), jnp.float32),
            b=jnp.asarray(rng.normal(size=(4,)), jnp.float32),
        ),
        layernorm_n=_random_norm(rng, 4),
        linear_n=_random_linear(rng, c_s, 4),
        transition_s1=_random_transition(rng, c_s),
        transition_s2=_random_transition(rng, c_s),
    )
    enc_block = _random_transformer_block(
        rng, c_a=c_atom, c_s=c_atom, c_z=c_atompair, heads=heads, cross_attention=True
    )
    tok_blocks = tuple(
        _random_transformer_block(
            rng,
            c_a=c_tok,
            c_s=c_s,
            c_z=c_z,
            heads=heads,
            cross_attention=False,
            layernorm_z_bias=layernorm_z_bias,
        )
        for _ in range(n_blocks)
    )
    dec_block = _random_transformer_block(
        rng, c_a=c_atom, c_s=c_atom, c_z=c_atompair, heads=heads, cross_attention=True
    )
    atom_encoder = AtomAttentionEncoderParams(
        cache=AtomAttentionEncoderCacheParams(
            linear_ref_pos=_random_linear(rng, c_atom, 3),
            linear_ref_charge=_random_linear(rng, c_atom, 1),
            linear_f=_random_linear(rng, c_atom, 385),
            linear_d=_random_linear(rng, c_atompair, 3),
            linear_invd=_random_linear(rng, c_atompair, 1),
            linear_v=_random_linear(rng, c_atompair, 1),
        ),
        linear_cl=_random_linear(rng, c_atompair, c_atom),
        linear_cm=_random_linear(rng, c_atompair, c_atom),
        small_mlp=AtomPairMlpParams(
            linear_1=_random_linear(rng, c_atompair, c_atompair),
            linear_2=_random_linear(rng, c_atompair, c_atompair),
            linear_3=_random_linear(rng, c_atompair, c_atompair),
        ),
        atom_transformer=DiffusionTransformerStackParams(blocks=(enc_block,)),
        linear_q=_random_linear(rng, c_tok, c_atom),
        layernorm_s=_random_norm(rng, c_s_trunk),
        linear_s=_random_linear(rng, c_atom, c_s_trunk),
        layernorm_z=_random_norm(rng, c_z),
        linear_z=_random_linear(rng, c_atompair, c_z),
        linear_r=_random_linear(rng, c_atom, 3),
    )
    params = DiffusionModuleParams(
        conditioning=conditioning,
        atom_encoder=atom_encoder,
        layernorm_s=_random_norm(rng, c_s),
        linear_s=_random_linear(rng, c_tok, c_s),
        diffusion_transformer=DiffusionTransformerStackParams(blocks=tok_blocks),
        layernorm_a=_random_norm(rng, c_tok),
        atom_decoder=AtomAttentionDecoderParams(
            linear_a=_random_linear(rng, c_atom, c_tok),
            layernorm_q=_random_norm(rng, c_atom),
            linear_out=_random_linear(rng, 3, c_atom),
            atom_transformer=DiffusionTransformerStackParams(blocks=(dec_block,)),
        ),
    )

    atom_to_struct = jnp.asarray([0, 0, 1, 2, 2, 3, 4], dtype=jnp.int32)
    struct_mask = jnp.asarray([True, True, True, False, False])
    atom_mask = jnp.asarray([True, True, False, True, True, False, False])
    ref_pos = jnp.asarray(rng.normal(size=(n_atom, 3)), jnp.float32)
    ref_charge = jnp.asarray(rng.normal(size=(n_atom,)), jnp.float32)
    ref_mask = atom_mask.astype(jnp.float32)
    ref_atom_name_chars = jnp.zeros((n_atom, 4, 64), dtype=jnp.float32)
    ref_element = jnp.zeros((n_atom, 128), dtype=jnp.float32)
    pos_q, pos_k, pad_info = rearrange_qk_to_dense_trunk(
        ref_pos,
        ref_pos,
        **{"n_queries": n_queries, "n_keys": n_window_cols},
    )
    d_lm = pos_q[:, :, None, :] - pos_k[:, None, :, :]
    id_q, id_k, _ = rearrange_qk_to_dense_trunk(
        atom_to_struct[:, None],
        atom_to_struct[:, None],
        **{
            "n_queries": n_queries,
            "n_keys": n_window_cols,
            "compute_mask": False,
        },
    )
    v_lm = (id_q[:, :, None, 0] == id_k[:, None, :, 0])[..., None].astype(jnp.float32)

    s_inputs = jnp.asarray(rng.normal(size=(n_struct, c_s_inputs)), jnp.float32)
    s_trunk = jnp.asarray(rng.normal(size=(n_struct, c_s_trunk)), jnp.float32)
    z_trunk = jnp.asarray(rng.normal(size=(n_struct, n_struct, c_z_trunk)), jnp.float32)
    relp_encoding = jnp.asarray(rng.normal(size=(n_struct, n_struct, c_z)), jnp.float32)
    structural_pair_bias = jnp.asarray(
        rng.normal(scale=0.8, size=(n_struct, n_struct)), jnp.float32
    )
    x_noisy = (
        jnp.asarray(rng.normal(size=(n_samples, n_atom, 3)), jnp.float32)
        * atom_mask[None, :, None].astype(jnp.float32)
    )
    t_hat = jnp.asarray([2.5, 1.2], dtype=jnp.float32)

    pair_mask = struct_mask[:, None] & struct_mask[None, :]
    pair_z = diffusion_conditioning_prepare_cache(
        None,
        z_trunk,
        params.conditioning,
        relp_encoding=relp_encoding,
    ) * pair_mask.astype(jnp.float32)[..., None]
    transformer_z = layer_norm(
        jnp.expand_dims(pair_z, axis=-4).astype(jnp.float32),
        LayerNormParams(),
    )
    p_lm, c_l = atom_attention_encoder_prepare_diffusion_cache(
        atom_to_struct,
        ref_pos,
        ref_charge,
        ref_mask,
        ref_element,
        ref_atom_name_chars,
        d_lm,
        v_lm,
        pad_info,
        z=jnp.expand_dims(pair_z, axis=-4),
        params=params.atom_encoder,
        **{"n_queries": n_queries, "n_keys": n_window_cols},
    )

    return {
        "positional": (
            atom_to_struct,
            ref_pos,
            ref_charge,
            ref_mask,
            ref_atom_name_chars,
            ref_element,
            d_lm,
            v_lm,
            pad_info,
        ),
        "x_noisy": x_noisy,
        "t_hat": t_hat,
        "s_inputs": s_inputs,
        "s_trunk": s_trunk,
        "z_trunk": z_trunk,
        "relp_encoding": relp_encoding,
        "params": params,
        "pair_z": pair_z,
        "transformer_z": transformer_z,
        "p_lm": p_lm,
        "c_l": c_l,
        "extra_attn_bias": structural_pair_bias,
        "struct_mask": struct_mask,
        "atom_mask": atom_mask,
        "n_struct": n_struct,
        "n_queries": n_queries,
        "n_window_cols": n_window_cols,
        "heads": heads,
        "n_blocks": n_blocks,
    }


def _call_denoiser(
    fixture: dict[str, Any],
    *,
    use_scan: bool = False,
    use_efficient_fusion: bool = True,
    extra_attn_bias: jnp.ndarray | None = ...,  # type: ignore[assignment]
    precomputed_pair_bias: jnp.ndarray | None = None,
    include_precomputed_kwarg: bool = True,
) -> jnp.ndarray:
    bias = (
        fixture["extra_attn_bias"] if extra_attn_bias is ... else extra_attn_bias
    )
    kw: dict[str, Any] = {
        "x_noisy": fixture["x_noisy"],
        "t_hat_noise_level": fixture["t_hat"],
        "relp_feature": None,
        "s_inputs": fixture["s_inputs"],
        "s_trunk": fixture["s_trunk"],
        "z_trunk": fixture["z_trunk"],
        "params": fixture["params"],
        "pair_z": fixture["pair_z"],
        "p_lm": fixture["p_lm"],
        "c_l": fixture["c_l"],
        "transformer_z": (
            fixture["transformer_z"] if use_efficient_fusion else None
        ),
        "extra_attn_bias": bias,
        "n_token": fixture["n_struct"],
        "atom_encoder_heads": fixture["heads"],
        "token_heads": fixture["heads"],
        "atom_decoder_heads": fixture["heads"],
        "n_queries": fixture["n_queries"],
        "n_keys": fixture["n_window_cols"],
        "sigma_data": 4.0,
        "use_scan": use_scan,
        "use_efficient_fusion": use_efficient_fusion,
        "attention_backend": "xla",
        "token_mask": fixture["struct_mask"],
        "atom_mask": fixture["atom_mask"],
    }
    if include_precomputed_kwarg:
        kw["precomputed_pair_bias"] = precomputed_pair_bias
    return opendde_diffusion.diffusion_module_forward(*fixture["positional"], **kw)


def _request(tmp_path: Path, **options: object) -> PredictionRequest:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "job.json"
    input_path.write_text(
        json.dumps([{"name": "tiny", "modelSeeds": [101], "sequences": []}]),
        encoding="utf-8",
    )
    weights = tmp_path / "opendde.jax"
    weights.write_bytes(b"native fixture")
    return PredictionRequest(
        model="opendde",
        input=input_path,
        input_format="native",
        weights=weights,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        seed=101,
        padding=False,
        options=dict(options),
    )


def _run_cli_to_infer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *extra_argv: str,
) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    """Drive `opendde/cli/predict.py:main` through `runner._predict` to infer."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    job = {"name": "tiny", "modelSeeds": [101], "sequences": []}
    input_path = tmp_path / "tiny.json"
    input_path.write_text(json.dumps([job]), encoding="utf-8")
    weights = tmp_path / "opendde.jax"
    weights.write_bytes(b"native fixture")
    features = {"restype": np.zeros((2, 32), dtype=np.float32)}
    predict_calls: list[dict[str, Any]] = []
    infer_calls: list[tuple[str, dict[str, Any]]] = []

    monkeypatch.setattr(opendde_runner, "_load_jobs", lambda _path: [job])
    monkeypatch.setattr(opendde_runner, "_featurize", lambda *_a, **_k: features)
    monkeypatch.setattr(
        opendde_runner,
        "_load_prepared_params",
        lambda _path, _dtype: inference_params(),
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.geometry.prepare_tfg_features",
        lambda raw, **_k: {**raw, "geometry_unsupported": False},
    )

    real_predict = opendde_runner._predict

    def spy_predict(*args: Any, **kwargs: Any) -> dict[str, Any]:
        predict_calls.append(dict(kwargs))
        return real_predict(*args, **kwargs)

    def fake_compiled(_features: Any, _params: Any, _schedule: Any, **kwargs: Any):
        infer_calls.append(("compiled", dict(kwargs)))
        return {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}

    def fake_static(_features: Any, _params: Any, _schedule: Any, **kwargs: Any):
        infer_calls.append(("static", dict(kwargs)))
        return {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}

    monkeypatch.setattr(opendde_runner, "_predict", spy_predict)
    monkeypatch.setattr(opendde_model, "opendde_infer_compiled", fake_compiled)
    monkeypatch.setattr(opendde_model, "opendde_infer_static", fake_static)
    monkeypatch.setattr(opendde_runner, "_score", lambda out, *_a, **_k: dict(out))
    monkeypatch.setattr(
        opendde_runner,
        "_write",
        lambda root, **kw: [root / kw["job_name"] / "predictions" / "tiny.cif"],
    )

    opendde_cli.main(
        [
            "--input-json",
            str(input_path),
            "--weights",
            str(weights),
            "--out",
            str(tmp_path / "out"),
            "--n-sample",
            "1",
            "--n-step",
            "2",
            "--n-cycle",
            "1",
            *extra_argv,
        ]
    )
    return predict_calls, infer_calls


# --------------------------------------------------- Helper & Per-block Parity


@pytest.mark.parametrize("use_scan", [False, True])
@pytest.mark.parametrize("layernorm_z_bias", [False, True])
def test_helper_cache_matches_uncached_attention_pair_bias_per_block(
    monkeypatch: pytest.MonkeyPatch,
    use_scan: bool,
    layernorm_z_bias: bool,
) -> None:
    """`build_diffusion_pair_bias_cache` matches each block's uncached bias."""
    fixture = _opendde_diffusion_fixture(
        seed=43, n_blocks=3, layernorm_z_bias=layernorm_z_bias
    )
    stack_params = fixture["params"].diffusion_transformer
    transformer_z = fixture["transformer_z"]  # [1, S, S, c_z]
    rng = np.random.default_rng(7)
    n_samples, n_struct, c_tok = 2, fixture["n_struct"], 8
    a = jnp.asarray(rng.normal(size=(n_samples, n_struct, c_tok)), jnp.float32)
    s = jnp.asarray(rng.normal(size=(n_samples, n_struct, 4)), jnp.float32)

    captured_biases: list[jnp.ndarray] = []
    real_attention = attention_impl.attention

    def spy_attention(
        q_x: jnp.ndarray,
        kv_x: jnp.ndarray,
        params: AttentionParams,
        num_heads: int,
        attn_bias: jnp.ndarray | None = None,
        **kwargs: Any,
    ) -> jnp.ndarray:
        assert attn_bias is not None
        captured_biases.append(attn_bias)
        return real_attention(q_x, kv_x, params, num_heads, attn_bias, **kwargs)

    monkeypatch.setattr(attention_impl, "attention", spy_attention)

    diffusion_transformer_stack(
        a,
        s,
        transformer_z,
        stack_params,
        num_heads=fixture["heads"],
        use_scan=False,
        z_is_normalized=True,
        extra_attn_bias=fixture["extra_attn_bias"],
        sequence_mask=fixture["struct_mask"],
    )
    assert len(captured_biases) == fixture["n_blocks"]

    cache = build_diffusion_pair_bias_cache(
        transformer_z,
        stack_params,
        extra_attn_bias=fixture["extra_attn_bias"],
        sequence_mask=fixture["struct_mask"],
        z_is_normalized=True,
        use_scan=use_scan,
    )
    assert cache.shape == (
        fixture["n_blocks"],
        1,
        fixture["heads"],
        n_struct,
        n_struct,
    )
    assert cache.dtype == jnp.float32

    # Distinct per-block weights must produce distinct per-block biases.
    assert not np.allclose(np.asarray(cache[0]), np.asarray(cache[1]))
    assert not np.allclose(np.asarray(cache[1]), np.asarray(cache[2]))

    for block_idx, expected_bias in enumerate(captured_biases):
        np.testing.assert_allclose(
            np.asarray(cache[block_idx]),
            np.asarray(expected_bias),
            rtol=1e-6,
            atol=1e-6,
        )

    # StackedLayers input (the native checkpoint layout) produces identical output.
    stacked_params = DiffusionTransformerStackParams(
        blocks=StackedLayers.from_layers(stack_params.blocks)
    )
    stacked_cache = build_diffusion_pair_bias_cache(
        transformer_z,
        stacked_params,
        extra_attn_bias=fixture["extra_attn_bias"],
        sequence_mask=fixture["struct_mask"],
        z_is_normalized=True,
        use_scan=use_scan,
    )
    np.testing.assert_array_equal(np.asarray(stacked_cache), np.asarray(cache))


@pytest.mark.parametrize("use_scan", [False, True])
def test_cached_denoiser_matches_uncached_and_none_is_bitwise_equal(
    use_scan: bool,
) -> None:
    """Cached vs uncached denoiser agrees to 1e-6; `None` matches omitted bitwise."""
    fixture = _opendde_diffusion_fixture(seed=67, n_blocks=3)
    cache = build_diffusion_pair_bias_cache(
        fixture["transformer_z"],
        fixture["params"].diffusion_transformer,
        extra_attn_bias=fixture["extra_attn_bias"],
        sequence_mask=fixture["struct_mask"],
        z_is_normalized=True,
        use_scan=use_scan,
    )

    baseline = _call_denoiser(
        fixture,
        use_scan=use_scan,
        use_efficient_fusion=True,
        include_precomputed_kwarg=False,
    )
    explicit_none = _call_denoiser(
        fixture,
        use_scan=use_scan,
        use_efficient_fusion=True,
        precomputed_pair_bias=None,
    )
    np.testing.assert_array_equal(np.asarray(explicit_none), np.asarray(baseline))

    cached = _call_denoiser(
        fixture,
        use_scan=use_scan,
        use_efficient_fusion=True,
        extra_attn_bias=None,
        precomputed_pair_bias=cache,
    )
    assert not bool(jnp.isnan(cached).any())
    np.testing.assert_allclose(
        np.asarray(cached), np.asarray(baseline), rtol=1e-6, atol=1e-6
    )

    # Confirm bias actually affects the output (a no-op bias path would fail here).
    no_extra_cache = build_diffusion_pair_bias_cache(
        fixture["transformer_z"],
        fixture["params"].diffusion_transformer,
        extra_attn_bias=None,
        sequence_mask=fixture["struct_mask"],
        z_is_normalized=True,
        use_scan=use_scan,
    )
    without_extra = _call_denoiser(
        fixture,
        use_scan=use_scan,
        use_efficient_fusion=True,
        extra_attn_bias=None,
        precomputed_pair_bias=no_extra_cache,
    )
    valid_atoms = np.asarray(fixture["atom_mask"])
    assert not np.allclose(
        np.asarray(cached)[:, valid_atoms, :],
        np.asarray(without_extra)[:, valid_atoms, :],
    )
    np.testing.assert_array_equal(np.asarray(cached)[:, ~valid_atoms, :], 0.0)


@pytest.mark.parametrize("use_scan", [False, True])
def test_opendde_infer_static_builds_cache_once_and_matches_uncached(
    monkeypatch: pytest.MonkeyPatch, use_scan: bool
) -> None:
    """`opendde_infer_static` builds the cache once outside the multi-step sampler."""
    fixture = _opendde_diffusion_fixture(seed=83, n_blocks=3)
    n_res = 3
    n_struct = fixture["n_struct"]
    n_atom = int(fixture["atom_mask"].shape[0])

    features = {
        "restype": jnp.zeros((n_res, 32), dtype=jnp.float32),
        "ref_pos": fixture["positional"][1],
        "ref_charge": fixture["positional"][2],
        "ref_mask": fixture["positional"][3],
        "ref_atom_name_chars": fixture["positional"][4],
        "ref_element": fixture["positional"][5],
        "d_lm": fixture["positional"][6],
        "v_lm": fixture["positional"][7],
        "pad_info": fixture["positional"][8],
        "token_index": jnp.arange(n_res).astype(jnp.int32),
        "asym_id": jnp.zeros((n_res,), dtype=jnp.int32),
        "residue_index": jnp.arange(n_res, dtype=jnp.int32),
        "entity_id": jnp.zeros((n_res,), dtype=jnp.int32),
        "sym_id": jnp.zeros((n_res,), dtype=jnp.int32),
        "atom_to_token_idx": jnp.asarray([0, 0, 1, 1, 2, 2, 2], jnp.int32),
        "atom_to_tokatom_idx": jnp.asarray([0, 1, 0, 1, 0, 1, 2], jnp.int32),
        "has_frame": jnp.asarray([True, True, True]),
        "frame_atom_index": jnp.zeros((n_res, 3), dtype=jnp.int32),
        "pae_rep_atom_mask": jnp.asarray([1, 0, 1, 0, 1, 0, 0], dtype=jnp.int32),
        "distogram_rep_atom_mask": jnp.asarray([1, 0, 1, 0, 1, 0, 0], dtype=jnp.int32),
        "parent_residue_idx": jnp.asarray([0, 0, 1, 2, 2], dtype=jnp.int32),
        "subtoken_role_id": jnp.asarray([1, 2, 1, 1, 2], jnp.int32),
        "structural_token_index": jnp.arange(n_struct).astype(jnp.int32),
        "atom_to_structural_token_idx": fixture["positional"][0],
        "atom_to_structural_tokatom_idx": jnp.asarray(
            [0, 1, 0, 0, 1, 0, 0], dtype=jnp.int32
        ),
        "structural_distogram_rep_atom_mask": jnp.asarray(
            [1, 0, 1, 1, 0, 1, 1], dtype=jnp.int32
        ),
        "structural_pae_rep_atom_mask": jnp.asarray(
            [1, 0, 1, 1, 0, 1, 1], dtype=jnp.int32
        ),
        "structural_has_frame": jnp.asarray([True, False, True, False, False]),
        "structural_frame_atom_index": jnp.zeros((n_struct, 3), dtype=jnp.int32),
        "structural_token_padding_mask": fixture["struct_mask"],
        "atom_padding_mask": fixture["atom_mask"],
    }

    monkeypatch.setattr(
        opendde_model,
        "input_feature_embedder",
        lambda *_a, **_k: jnp.ones((n_res, 5), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        opendde_model,
        "pairformer_output_from_s_inputs",
        lambda *_a, **_k: (
            jnp.ones((n_res, 5), dtype=jnp.float32),
            jnp.ones((n_res, 6), dtype=jnp.float32),
            jnp.ones((n_res, n_res, 6), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        opendde_model,
        "structural_token_expand",
        lambda *_a, **_k: (
            fixture["s_inputs"],
            fixture["s_trunk"],
            fixture["z_trunk"],
            {"structural_pair_attn_bias": fixture["extra_attn_bias"]},
        ),
    )
    monkeypatch.setattr(
        opendde_model,
        "prepare_structural_features",
        lambda raw, extra, **_k: {
            **raw,
            **extra,
            "atom_to_token_idx": raw["atom_to_structural_token_idx"],
        },
    )
    monkeypatch.setattr(
        opendde_model,
        "structural_refiner_stack",
        lambda s, z, *_a, **_k: (s, z),
    )
    monkeypatch.setattr(
        opendde_model,
        "relative_position_encoding_from_features",
        lambda *_a, **_k: fixture["relp_encoding"],
    )
    monkeypatch.setattr(
        opendde_model,
        "distogram_head",
        lambda z, _p: jnp.zeros((n_res, n_res, 2), dtype=z.dtype),
    )

    build_calls = 0
    real_build = opendde_model.build_diffusion_pair_bias_cache

    def spy_build(*args: Any, **kwargs: Any) -> jnp.ndarray:
        nonlocal build_calls
        build_calls += 1
        return real_build(*args, **kwargs)

    monkeypatch.setattr(opendde_model, "build_diffusion_pair_bias_cache", spy_build)

    params = SimpleNamespace(
        input_embedder=object(),
        pairformer_output=object(),
        structural_expander=object(),
        structural_refiner=object(),
        diffusion=fixture["params"],
        distogram=object(),
        confidence=None,
    )
    schedule = jnp.asarray([2.0, 1.0, 0.5, 0.0], dtype=jnp.float32)  # 3 steps
    init_noise = fixture["x_noisy"]
    step_noises = tuple(
        jnp.zeros((2, n_atom, 3), dtype=jnp.float32) for _ in range(3)
    )
    rotations = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (3, 2, 3, 3))
    translations = jnp.zeros((3, 2, 3), dtype=jnp.float32)

    def run_infer(*, cache_on: bool, autocast: bool = False) -> jnp.ndarray:
        model_params = (
            SimpleNamespace(
                **{
                    **vars(params),
                    "diffusion": native_diffusion_autocast_params(fixture["params"]),
                }
            )
            if autocast
            else params
        )
        call_kw: dict[str, Any] = {
            "key": None,
            "num_samples": 2,
            "init_noise": init_noise,
            "step_noises": step_noises,
            "rotations": rotations,
            "translations": translations,
            "num_recycles": 1,
            "atom_encoder_heads": fixture["heads"],
            "token_heads": fixture["heads"],
            "atom_decoder_heads": fixture["heads"],
            "n_queries": fixture["n_queries"],
            "n_keys": fixture["n_window_cols"],
            "sigma_data": 4.0,
            "use_diffusion_scan": use_scan,
            "use_sampler_scan": use_scan,
            "use_diffusion_efficient_fusion": True,
            "diffusion_pair_bias_cache": cache_on,
            "diffusion_attention_backend": "xla",
            "diffusion_autocast": autocast,
            "run_confidence": False,
        }
        return opendde_model.opendde_infer_static(
            features, model_params, schedule, **call_kw
        )["coordinate"]

    uncached = run_infer(cache_on=False)
    assert build_calls == 0

    cached = run_infer(cache_on=True)
    assert build_calls == 1
    np.testing.assert_allclose(
        np.asarray(cached), np.asarray(uncached), rtol=1e-6, atol=1e-6
    )

    # Also verify under bf16 diffusion autocast (linear_z delivers fp32 bias).
    uncached_bf16 = run_infer(cache_on=False, autocast=True)
    cached_bf16 = run_infer(cache_on=True, autocast=True)
    assert build_calls == 2
    assert cached_bf16.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(cached_bf16), np.asarray(uncached_bf16), rtol=1e-5, atol=1e-5
    )


# ------------------------------------------------------------ Refusals & Edges


def test_refusals_and_malformed_inputs_across_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse efficient-fusion-off, local attention, CP, guidance, and bad shapes."""
    fixture = _opendde_diffusion_fixture(seed=97, n_blocks=2)
    stack_params = fixture["params"].diffusion_transformer
    block0 = stack_params.blocks[0]
    cache = build_diffusion_pair_bias_cache(
        fixture["transformer_z"],
        stack_params,
        extra_attn_bias=fixture["extra_attn_bias"],
        sequence_mask=fixture["struct_mask"],
    )
    a = jnp.zeros((2, fixture["n_struct"], 8), dtype=jnp.float32)
    s = jnp.zeros((2, fixture["n_struct"], 4), dtype=jnp.float32)

    # Helper edges: empty stack, z_is_normalized=False, malformed shapes.
    with pytest.raises(ValueError, match="at least one block"):
        build_diffusion_pair_bias_cache(
            fixture["transformer_z"], DiffusionTransformerStackParams(blocks=())
        )
    with pytest.raises(ValueError, match="z_is_normalized=True"):
        build_diffusion_pair_bias_cache(
            fixture["transformer_z"], stack_params, z_is_normalized=False
        )
    with pytest.raises(ValueError, match="at least 3 axes"):
        build_diffusion_pair_bias_cache(
            jnp.zeros((5, 4), dtype=jnp.float32), stack_params
        )
    with pytest.raises(ValueError, match="must be square"):
        build_diffusion_pair_bias_cache(
            jnp.zeros((1, 5, 4, 4), dtype=jnp.float32), stack_params
        )
    with pytest.raises(ValueError, match="sequence_mask must have shape"):
        build_diffusion_pair_bias_cache(
            fixture["transformer_z"],
            stack_params,
            sequence_mask=jnp.ones((3,), dtype=bool),
        )
    with pytest.raises(ValueError, match="pair axes must match"):
        build_diffusion_pair_bias_cache(
            fixture["transformer_z"],
            stack_params,
            extra_attn_bias=jnp.zeros((3, 3), dtype=jnp.float32),
        )

    # attention_pair_bias edges.
    with pytest.raises(ValueError, match="extra_attn_bias must be None"):
        attention_pair_bias(
            a,
            s,
            None,
            block0.attention_pair_bias,
            num_heads=fixture["heads"],
            extra_attn_bias=fixture["extra_attn_bias"],
            precomputed_pair_bias=cache[0],
        )
    with pytest.raises(ValueError, match="at least 3 axes"):
        attention_pair_bias(
            a,
            s,
            None,
            block0.attention_pair_bias,
            num_heads=fixture["heads"],
            precomputed_pair_bias=jnp.zeros((5, 5), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="trailing shape must match"):
        attention_pair_bias(
            a,
            s,
            None,
            block0.attention_pair_bias,
            num_heads=fixture["heads"],
            precomputed_pair_bias=jnp.zeros((1, 1, 5, 5), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="exceeds target logits rank"):
        attention_pair_bias(
            a[0],  # rank 2 -> target_ndim 3
            s[0],
            None,
            block0.attention_pair_bias,
            num_heads=fixture["heads"],
            precomputed_pair_bias=cache[0],  # rank 4
        )

    # Local attention refused in both block and stack.
    with pytest.raises(ValueError, match="only supported by global token attention"):
        diffusion_transformer_block(
            a,
            s,
            None,
            block0,
            num_heads=fixture["heads"],
            n_queries=2,
            **{"n_keys": 4},
            z_is_normalized=True,
            precomputed_pair_bias=cache[0],
        )
    with pytest.raises(ValueError, match="only supported by global token attention"):
        diffusion_transformer_stack(
            a,
            s,
            None,
            stack_params,
            num_heads=fixture["heads"],
            n_queries=2,
            **{"n_keys": 4},
            z_is_normalized=True,
            precomputed_pair_bias=cache,
        )

    # Efficient fusion off refused in block, stack, denoiser, and opendde_infer_static.
    with pytest.raises(ValueError, match="z_is_normalized=True"):
        diffusion_transformer_block(
            a,
            s,
            None,
            block0,
            num_heads=fixture["heads"],
            z_is_normalized=False,
            precomputed_pair_bias=cache[0],
        )
    with pytest.raises(ValueError, match="z_is_normalized=True"):
        diffusion_transformer_stack(
            a,
            s,
            None,
            stack_params,
            num_heads=fixture["heads"],
            z_is_normalized=False,
            precomputed_pair_bias=cache,
        )
    with pytest.raises(ValueError, match="extra_attn_bias must be None"):
        diffusion_transformer_stack(
            a,
            s,
            None,
            stack_params,
            num_heads=fixture["heads"],
            z_is_normalized=True,
            extra_attn_bias=fixture["extra_attn_bias"],
            precomputed_pair_bias=cache,
        )
    with pytest.raises(ValueError, match="leading axis 2"):
        diffusion_transformer_stack(
            a,
            s,
            None,
            stack_params,
            num_heads=fixture["heads"],
            z_is_normalized=True,
            precomputed_pair_bias=cache[:1],
        )
    with pytest.raises(ValueError, match="use_efficient_fusion=True"):
        _call_denoiser(
            fixture,
            use_efficient_fusion=False,
            extra_attn_bias=None,
            precomputed_pair_bias=cache,
        )

    # CP refusal in shared denoiser.
    with monkeypatch.context() as mp:
        mp.setattr(protenix_diffusion, "cp_mesh", lambda: object())
        with pytest.raises(ValueError, match="not supported under context parallelism"):
            _call_denoiser(
                fixture,
                use_efficient_fusion=True,
                extra_attn_bias=None,
                precomputed_pair_bias=cache,
            )

    # opendde_infer_static refusals: efficient fusion off (including default), CP, TFG.
    schedule = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    static_kw: dict[str, Any] = {"key": None, "num_samples": 1}
    with pytest.raises(ValueError, match="use_diffusion_efficient_fusion=True"):
        opendde_model.opendde_infer_static(
            {},
            None,
            schedule,
            diffusion_pair_bias_cache=True,
            **static_kw,
        )
    with pytest.raises(ValueError, match="not supported under context parallelism"):
        opendde_model.opendde_infer_static(
            {},
            None,
            schedule,
            use_diffusion_efficient_fusion=True,
            diffusion_pair_bias_cache=True,
            cp_shards=2,
            **static_kw,
        )
    with pytest.raises(ValueError, match="not supported with TFG guidance"):
        opendde_model.opendde_infer_static(
            {},
            None,
            schedule,
            use_diffusion_efficient_fusion=True,
            diffusion_pair_bias_cache=True,
            guidance_config={"enable": True},
            **static_kw,
        )
    # The compiled entry refuses efficient-fusion-off, CP, and TFG on concrete
    # values, before any feature validation, mesh replication or tracing.
    for refused, match in (
        (
            {"use_diffusion_efficient_fusion": False},
            "use_diffusion_efficient_fusion=True",
        ),
        (
            {"use_diffusion_efficient_fusion": True, "cp_shards": 2},
            "not supported under context parallelism",
        ),
        (
            {
                "use_diffusion_efficient_fusion": True,
                "guidance_config": {"enable": True},
            },
            "not supported with TFG guidance",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            opendde_model.opendde_infer_compiled(
                {},
                None,
                schedule,
                diffusion_pair_bias_cache=True,
                **refused,
                **static_kw,
            )


# ----------------------------------------- Backend, CLI & Runner Option Wiring


def test_backend_option_defaults_false_and_true_forwards_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off-GPU default is False with unchanged cache namespace; True forwards."""
    # Off a GPU an omitted pair-bias cache realises the released False; serial
    # GPU resolution to True is covered in its own test below.
    monkeypatch.setattr(backend_impl, "_gpu_process", lambda: False)
    backend = OpenDDEBackend()
    assert backend_impl._RELEASED_COMPILE_DEFAULTS["diffusion_pair_bias_cache"] is False
    assert backend_impl._PARSER_DEFAULTS["diffusion_pair_bias_cache"] is False
    assert (
        inspect.signature(opendde_model.opendde_infer_static)
        .parameters["diffusion_pair_bias_cache"]
        .default
        is False
    )
    assert "diffusion_pair_bias_cache" in opendde_model.GRAPH_STATIC_ARGNAMES

    configs: list[opendde_runner.PredictionConfig] = []

    def record_run(config: opendde_runner.PredictionConfig, **_kw: Any) -> list[Path]:
        configs.append(config)
        out = Path(config.out)
        out.mkdir(parents=True, exist_ok=True)
        cif = out / "tiny_sample_0.cif"
        cif.write_text("data_x\n", encoding="utf-8")
        return [cif]

    with monkeypatch.context() as mp:
        mp.setattr(
            backend_impl,
            "import_module",
            lambda _name: native_module("opendde", record_run),
        )

        omitted = _request(tmp_path)
        omitted_result = backend.predict(omitted)
        assert "--diffusion-pair-bias-cache" not in omitted_result.raw["argv"]
        assert configs[-1].diffusion_pair_bias_cache is False
        assert "diffusion_pair_bias_cache" not in backend.cache_profile(omitted)

        explicit_false = dataclasses.replace(
            omitted,
            output_dir=tmp_path / "out-false",
            options={"diffusion_pair_bias_cache": False},
        )
        backend.validate_request(explicit_false)
        assert backend.cache_profile(explicit_false) == backend.cache_profile(omitted)
        assert resolve_cache_dir(explicit_false, backend) == resolve_cache_dir(
            omitted, backend
        )

        enabled = dataclasses.replace(
            omitted,
            output_dir=tmp_path / "out-cached",
            options={"diffusion_pair_bias_cache": True},
        )
        backend.validate_request(enabled)
        enabled_result = backend.predict(enabled)
        argv = list(enabled_result.raw["argv"])
        assert argv[argv.index("--diffusion-pair-bias-cache") + 1] == "True"
        assert configs[-1].diffusion_pair_bias_cache is True
        assert backend.cache_profile(enabled)["diffusion_pair_bias_cache"] is True
        assert resolve_cache_dir(enabled, backend) != resolve_cache_dir(
            omitted, backend
        )

    # Full backend -> runner.run_prediction -> opendde_infer_compiled.
    with monkeypatch.context() as mp:
        backend_infer_calls: list[dict[str, Any]] = []
        job = {"name": "tiny", "modelSeeds": [101], "sequences": []}
        mp.setattr(opendde_runner, "_load_jobs", lambda _p: [job])
        mp.setattr(
            opendde_runner,
            "_featurize",
            lambda *_a, **_k: {"restype": np.zeros((2, 32), dtype=np.float32)},
        )
        mp.setattr(
            opendde_runner,
            "_load_prepared_params",
            lambda _p, _d: inference_params(),
        )
        mp.setattr(
            opendde_model,
            "opendde_infer_compiled",
            lambda _f, _p, _s, **kw: (
                backend_infer_calls.append(dict(kw))
                or {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}
            ),
        )
        mp.setattr(opendde_runner, "_score", lambda out, *_a, **_k: dict(out))
        mp.setattr(
            opendde_runner,
            "_write",
            lambda root, **kw: [
                d := root / kw["job_name"] / "predictions",
                d.mkdir(parents=True, exist_ok=True),
                cif := d / "tiny_sample_0.cif",
                cif.write_text("data_x\n", encoding="utf-8"),
                (cif.parent / "tiny_summary_confidences_0.json").write_text(
                    json.dumps({"ranking_score": 0.9}), encoding="utf-8"
                ),
            ][2:3],
        )
        backend.predict(
            _request(
                tmp_path / "full-default",
                num_samples=1,
                num_steps=2,
                num_recycles=1,
            )
        )
        assert backend_infer_calls[-1]["diffusion_pair_bias_cache"] is False
        assert "use_diffusion_efficient_fusion" not in backend_infer_calls[-1]

        backend.predict(
            _request(
                tmp_path / "full-enabled",
                diffusion_pair_bias_cache=True,
                num_samples=1,
                num_steps=2,
                num_recycles=1,
            )
        )
        assert backend_infer_calls[-1]["diffusion_pair_bias_cache"] is True
        assert backend_infer_calls[-1]["use_diffusion_efficient_fusion"] is True

    # Native CLI -> runner._predict -> opendde_infer_compiled.
    default_pred, default_infer = _run_cli_to_infer(tmp_path / "cli-def", monkeypatch)
    assert default_pred[0]["diffusion_pair_bias_cache"] is False
    assert default_infer[0][0] == "compiled"
    assert default_infer[0][1]["diffusion_pair_bias_cache"] is False
    assert "use_diffusion_efficient_fusion" not in default_infer[0][1]

    cached_pred, cached_infer = _run_cli_to_infer(
        tmp_path / "cli-cached",
        monkeypatch,
        "--diffusion-pair-bias-cache",
        "true",
    )
    assert cached_pred[0]["diffusion_pair_bias_cache"] is True
    assert cached_infer[0][0] == "compiled"
    assert cached_infer[0][1]["diffusion_pair_bias_cache"] is True
    assert cached_infer[0][1]["use_diffusion_efficient_fusion"] is True


def test_compiled_pool_identity_separates_pair_bias_cache_arms() -> None:
    args = ({}, (), jnp.zeros((2,), dtype=jnp.float32))
    base = {
        "params_treedef": (),
        "params_flags": (),
        "cp_shards": 1,
        "cp_layout": "1d",
        "num_samples": 1,
        "num_recycles": 1,
        "n_queries": 32,
        "n_keys": 128,
        "diffusion_attention_backend": "xla_jit",
        "trunk_single_attention_backend": "xla_jit",
        "structural_single_attention_backend": "xla_jit",
    }
    id_off = opendde_model._compiled_opendde_infer._identity(
        args, {**base, "diffusion_pair_bias_cache": False}
    )
    id_on = opendde_model._compiled_opendde_infer._identity(
        args, {**base, "diffusion_pair_bias_cache": True}
    )
    assert id_off != id_on


def _stub_full_backend_run(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, dict[str, Any]]]:
    """Stub I/O and model entries so OpenDDEBackend.predict runs end-to-end on CPU."""
    infer_calls: list[tuple[str, dict[str, Any]]] = []
    job = {"name": "tiny", "modelSeeds": [101], "sequences": []}
    monkeypatch.setattr(opendde_runner, "_load_jobs", lambda _p: [job])
    monkeypatch.setattr(
        opendde_runner,
        "_featurize",
        lambda *_a, **_k: {"restype": np.zeros((2, 32), dtype=np.float32)},
    )
    monkeypatch.setattr(
        opendde_runner,
        "_load_prepared_params",
        lambda _p, _d: inference_params(),
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.geometry.prepare_tfg_features",
        lambda raw, **_k: {**raw, "geometry_unsupported": False},
    )
    monkeypatch.setattr(
        opendde_model,
        "opendde_infer_compiled",
        lambda _f, _p, _s, **kw: (
            infer_calls.append(("compiled", dict(kw)))
            or {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}
        ),
    )
    monkeypatch.setattr(
        opendde_model,
        "opendde_infer_static",
        lambda _f, _p, _s, **kw: (
            infer_calls.append(("static", dict(kw)))
            or {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}
        ),
    )
    monkeypatch.setattr(opendde_runner, "_score", lambda out, *_a, **_k: dict(out))

    def fake_write(root: Path, **kw: Any) -> list[Path]:
        pred_dir = root / kw["job_name"] / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        cif = pred_dir / "tiny_sample_0.cif"
        cif.write_text("data_x\n", encoding="utf-8")
        (pred_dir / "tiny_summary_confidences_0.json").write_text(
            json.dumps({"ranking_score": 0.9}), encoding="utf-8"
        )
        return [cif]

    monkeypatch.setattr(opendde_runner, "_write", fake_write)
    return infer_calls


def test_realised_diffusion_pair_bias_cache_resolver_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolver table: GPU/CPU x CP x TFG x explicit."""
    # Serial unguided GPU -> True.
    monkeypatch.setattr(backend_impl, "_gpu_process", lambda: True)
    assert backend_impl.realised_diffusion_pair_bias_cache(None) is True
    for cp_serial in (1, "1", np.int64(1)):
        assert (
            backend_impl.realised_diffusion_pair_bias_cache(None, cp_devices=cp_serial)
            is True
        )
    for tfg_off in (False, "false", " FALSE ", np.bool_(False)):
        assert (
            backend_impl.realised_diffusion_pair_bias_cache(
                None, use_tfg_guidance=tfg_off
            )
            is True
        )

    # Off a GPU (CPU / TPU) -> False.
    monkeypatch.setattr(backend_impl, "_gpu_process", lambda: False)
    assert backend_impl.realised_diffusion_pair_bias_cache(None) is False

    # Under CP, TFG, or any explicit value, _gpu_process is never probed.
    monkeypatch.setattr(
        backend_impl,
        "_gpu_process",
        lambda: (_ for _ in ()).throw(AssertionError("device probed unexpectedly")),
    )
    for cp_non_serial in (2, "2", 4, np.int64(2)):
        assert (
            backend_impl.realised_diffusion_pair_bias_cache(
                None, cp_devices=cp_non_serial
            )
            is False
        )
    for tfg_on in (True, "true", " TRUE ", np.bool_(True)):
        assert (
            backend_impl.realised_diffusion_pair_bias_cache(
                None, use_tfg_guidance=tfg_on
            )
            is False
        )
    # Any explicit value is returned unchanged.
    for explicit in (True, False, "true", "false", "malformed", 0, ""):
        assert backend_impl.realised_diffusion_pair_bias_cache(explicit) is explicit


def test_omitted_pair_bias_cache_resolves_true_on_serial_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serial unguided GPU resolves omitted -> True across profile, argv, predict."""
    from tests._parser_capture import capture_parser

    monkeypatch.setattr(backend_impl, "_gpu_process", lambda: True)
    backend = OpenDDEBackend()
    omitted = _request(
        tmp_path / "gpu-omitted", num_samples=1, num_steps=2, num_recycles=1
    )
    explicit_true = dataclasses.replace(
        omitted,
        output_dir=tmp_path / "gpu-true",
        options={**omitted.options, "diffusion_pair_bias_cache": True},
    )
    explicit_false = dataclasses.replace(
        omitted,
        output_dir=tmp_path / "gpu-false",
        options={**omitted.options, "diffusion_pair_bias_cache": False},
    )

    # Cache profile records True and aliases explicit True; explicit False stays
    # stripped and distinct.
    omitted_profile = backend.cache_profile(omitted)
    assert omitted_profile["diffusion_pair_bias_cache"] is True
    assert omitted_profile == backend.cache_profile(explicit_true)
    assert resolve_cache_dir(omitted, backend) == resolve_cache_dir(
        explicit_true, backend
    )
    assert "diffusion_pair_bias_cache" not in backend.cache_profile(explicit_false)
    assert resolve_cache_dir(omitted, backend) != resolve_cache_dir(
        explicit_false, backend
    )

    # Invocation renders `--diffusion-pair-bias-cache True` matching the native
    # parser namespace.
    invocation = backend._native_invocation(omitted)
    assert invocation.config_fields["diffusion_pair_bias_cache"] is True
    assert (
        invocation.argv[invocation.argv.index("--diffusion-pair-bias-cache") + 1]
        == "True"
    )
    parsed = capture_parser(opendde_cli.main).parse_args(invocation.argv)
    assert parsed.diffusion_pair_bias_cache is True

    # Full backend.predict -> runner.run_prediction -> opendde_infer_compiled
    # enables both the cache and use_diffusion_efficient_fusion=True.
    infer_calls = _stub_full_backend_run(monkeypatch)
    result = backend.predict(omitted)
    argv = list(result.raw["argv"])
    assert argv[argv.index("--diffusion-pair-bias-cache") + 1] == "True"
    assert infer_calls[-1][0] == "compiled"
    assert infer_calls[-1][1]["diffusion_pair_bias_cache"] is True
    assert infer_calls[-1][1]["use_diffusion_efficient_fusion"] is True

    # Explicit False on a serial GPU disables the cache without forcing fusion.
    false_result = backend.predict(explicit_false)
    false_argv = list(false_result.raw["argv"])
    assert false_argv[false_argv.index("--diffusion-pair-bias-cache") + 1] == "False"
    assert infer_calls[-1][0] == "compiled"
    assert infer_calls[-1][1]["diffusion_pair_bias_cache"] is False
    assert "use_diffusion_efficient_fusion" not in infer_calls[-1][1]

    # Native opendde-jax-predict CLI keeps False even on a GPU process.
    cli_pred, cli_infer = _run_cli_to_infer(tmp_path / "cli-gpu-default", monkeypatch)
    assert cli_pred[0]["diffusion_pair_bias_cache"] is False
    assert cli_infer[0][0] == "compiled"
    assert cli_infer[0][1]["diffusion_pair_bias_cache"] is False
    assert "use_diffusion_efficient_fusion" not in cli_infer[0][1]


def test_omitted_pair_bias_cache_stays_false_under_cp_tfg_and_off_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitted stays False under CP, TFG, and off-GPU; explicit True refuses."""
    backend = OpenDDEBackend()
    infer_calls = _stub_full_backend_run(monkeypatch)

    # 1. Off a GPU (CPU): omitted resolves False and aliases explicit False.
    monkeypatch.setattr(backend_impl, "_gpu_process", lambda: False)
    cpu_omitted = _request(
        tmp_path / "cpu-omitted", num_samples=1, num_steps=2, num_recycles=1
    )
    cpu_false = dataclasses.replace(
        cpu_omitted,
        options={**cpu_omitted.options, "diffusion_pair_bias_cache": False},
    )
    assert "diffusion_pair_bias_cache" not in backend.cache_profile(cpu_omitted)
    assert resolve_cache_dir(cpu_omitted, backend) == resolve_cache_dir(
        cpu_false, backend
    )
    cpu_res = backend.predict(cpu_omitted)
    assert "--diffusion-pair-bias-cache" not in cpu_res.raw["argv"]
    assert infer_calls[-1][0] == "compiled"
    assert infer_calls[-1][1]["diffusion_pair_bias_cache"] is False
    assert "use_diffusion_efficient_fusion" not in infer_calls[-1][1]

    # 2. On a GPU under CP: omitted -> False (not refused); explicit True refuses.
    monkeypatch.setattr(backend_impl, "_gpu_process", lambda: True)
    cp_omitted = _request(
        tmp_path / "gpu-cp-omitted",
        cp_devices=2,
        num_samples=1,
        num_steps=2,
        num_recycles=1,
    )
    cp_false = dataclasses.replace(
        cp_omitted,
        options={**cp_omitted.options, "diffusion_pair_bias_cache": False},
    )
    cp_true = dataclasses.replace(
        cp_omitted,
        options={**cp_omitted.options, "diffusion_pair_bias_cache": True},
    )
    assert "diffusion_pair_bias_cache" not in backend.cache_profile(cp_omitted)
    assert resolve_cache_dir(cp_omitted, backend) == resolve_cache_dir(
        cp_false, backend
    )
    cp_res = backend.predict(cp_omitted)
    assert "--diffusion-pair-bias-cache" not in cp_res.raw["argv"]
    assert infer_calls[-1][0] == "compiled"
    assert infer_calls[-1][1]["diffusion_pair_bias_cache"] is False
    for fn in (
        backend.validate_request,
        backend.cache_profile,
        backend._native_invocation,
    ):
        with pytest.raises(ValueError, match="not supported under context parallelism"):
            fn(cp_true)

    # 3. On a GPU under TFG: omitted -> False (not refused); explicit True refuses.
    tfg_omitted = _request(
        tmp_path / "gpu-tfg-omitted",
        use_tfg_guidance=True,
        num_samples=1,
        num_steps=2,
        num_recycles=1,
    )
    tfg_false = dataclasses.replace(
        tfg_omitted,
        options={**tfg_omitted.options, "diffusion_pair_bias_cache": False},
    )
    tfg_true = dataclasses.replace(
        tfg_omitted,
        options={**tfg_omitted.options, "diffusion_pair_bias_cache": True},
    )
    assert "diffusion_pair_bias_cache" not in backend.cache_profile(tfg_omitted)
    assert resolve_cache_dir(tfg_omitted, backend) == resolve_cache_dir(
        tfg_false, backend
    )
    tfg_res = backend.predict(tfg_omitted)
    assert "--diffusion-pair-bias-cache" not in tfg_res.raw["argv"]
    assert infer_calls[-1][0] == "static"
    assert infer_calls[-1][1]["diffusion_pair_bias_cache"] is False
    for fn in (
        backend.validate_request,
        backend.cache_profile,
        backend._native_invocation,
    ):
        with pytest.raises(ValueError, match="not supported with use_tfg_guidance"):
            fn(tfg_true)
