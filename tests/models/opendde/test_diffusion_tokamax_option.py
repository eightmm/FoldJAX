"""OpenDDE's opt-in ``diffusion_attention_backend=tokamax`` wiring and parity.

Covers:
(a) Backend and native CLI wiring (`--option diffusion_attention_backend=tokamax`
    and `--diffusion-attention-backend tokamax`), cache-profile identity, and
    rejection on `trunk_single_attention_backend` /
    `structural_single_attention_backend` and malformed values.
(b) CPU numerical parity against `xla_jit` through tokamax's XLA implementation
    on an OpenDDE-shaped diffusion step with asymmetric `structural_pair_bias`
    (`extra_attn_bias`) and structural-token / atom padding masks, verifying
    that the single option controls both global token attention and both
    windowed atom attentions (encoder and decoder).
(c) Adapter refusal under `use_tfg_guidance=True`, runner fallback to
    `xla_jit` on direct native-CLI calls, and refusal under an active
    context-parallel mesh.
(d) Released defaults unchanged (`xla_jit`).
"""

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
)
from foldjax.models.protenix.models.primitives import attention as attention_impl
from foldjax.models.protenix.models.primitives import attention_tokamax
from foldjax.models.protenix.models.primitives.attention import (
    AttentionPairBiasParams,
    AttentionParams,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LayerNormParams,
    LinearParams,
    TransitionParams,
)
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    FourierParams,
    RelativePositionParams,
)
from foldjax.schema import PredictionRequest
from tests._native_double import native_module
from tests._parser_capture import capture_parser
from tests.models.opendde.toy_params import inference_params


def _tokamax_xla(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the tokamax branch on CPU via tokamax's XLA implementation."""
    monkeypatch.setattr(attention_tokamax, "TOKAMAX_IMPLEMENTATION", "xla")
    monkeypatch.setattr(attention_tokamax, "_WARNED_FLOAT32", False)


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


# ------------------------------------------------------------------ (a) & (d) --


def test_defaults_stay_xla_jit_and_tokamax_reaches_runner_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = OpenDDEBackend()
    assert backend_impl._RELEASED_COMPILE_DEFAULTS["diffusion_attention_backend"] == (
        "xla_jit"
    )
    assert backend_impl._PARSER_DEFAULTS["diffusion_attention_backend"] == "xla_jit"
    assert (
        inspect.signature(opendde_model.opendde_infer_static)
        .parameters["diffusion_attention_backend"]
        .default
        == "xla_jit"
    )

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
        assert "--diffusion-attention-backend" not in omitted_result.raw["argv"]
        assert configs[-1].diffusion_attention_backend == "xla_jit"
        assert "diffusion_attention_backend" not in backend.cache_profile(omitted)

        selected = dataclasses.replace(
            omitted,
            output_dir=tmp_path / "out-tokamax",
            options={"diffusion_attention_backend": "tokamax"},
        )
        backend.validate_request(selected)
        selected_result = backend.predict(selected)
        argv = list(selected_result.raw["argv"])
        assert argv[argv.index("--diffusion-attention-backend") + 1] == "tokamax"
        assert configs[-1].diffusion_attention_backend == "tokamax"
        assert configs[-1].trunk_single_attention_backend == "xla_jit"
        assert configs[-1].structural_single_attention_backend == "xla_jit"
        assert (
            backend.cache_profile(selected)["diffusion_attention_backend"] == "tokamax"
        )
        assert resolve_cache_dir(selected, backend) != resolve_cache_dir(
            omitted, backend
        )

    # Full unstubbed backend -> runner.run_prediction -> opendde_infer_compiled.
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
                tmp_path / "full-backend",
                diffusion_attention_backend="tokamax",
                num_samples=1,
                num_steps=2,
                num_recycles=1,
            )
        )
        assert backend_infer_calls[0]["diffusion_attention_backend"] == "tokamax"
        assert backend_infer_calls[0]["trunk_single_attention_backend"] == "xla_jit"
        assert (
            backend_infer_calls[0]["structural_single_attention_backend"] == "xla_jit"
        )

    # Native CLI -> runner._predict -> opendde_infer_compiled.
    default_pred, default_infer = _run_cli_to_infer(tmp_path / "cli-def", monkeypatch)
    assert default_pred[0]["diffusion_attention_backend"] == "xla_jit"
    assert default_infer[0][0] == "compiled"
    assert default_infer[0][1]["diffusion_attention_backend"] == "xla_jit"

    tok_pred, tok_infer = _run_cli_to_infer(
        tmp_path / "cli-tok",
        monkeypatch,
        "--diffusion-attention-backend",
        "tokamax",
    )
    assert tok_pred[0]["diffusion_attention_backend"] == "tokamax"
    assert tok_pred[0]["trunk_single_attention_backend"] == "xla_jit"
    assert tok_pred[0]["structural_single_attention_backend"] == "xla_jit"
    assert tok_infer[0][0] == "compiled"
    assert tok_infer[0][1]["diffusion_attention_backend"] == "tokamax"


@pytest.mark.parametrize(
    "other_option",
    ("trunk_single_attention_backend", "structural_single_attention_backend"),
)
def test_tokamax_is_not_added_to_trunk_or_structural_single_attention(
    tmp_path: Path, other_option: str
) -> None:
    backend = OpenDDEBackend()
    with pytest.raises(ValueError, match=other_option):
        backend._native_invocation(_request(tmp_path, **{other_option: "tokamax"}))

    parser = capture_parser(opendde_cli.main)
    flag = f"--{other_option.replace('_', '-')}"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--input-json",
                "in.json",
                "--weights",
                "w.jax",
                "--out",
                "o",
                flag,
                "tokamax",
            ]
        )


@pytest.mark.parametrize("bad_value", ("", "0", "triton", "none", "TOKAMAX"))
def test_invalid_diffusion_attention_backend_values_are_rejected(
    tmp_path: Path, bad_value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    backend = OpenDDEBackend()
    with pytest.raises(ValueError, match="diffusion_attention_backend") as exc_info:
        backend._native_invocation(
            _request(tmp_path, diffusion_attention_backend=bad_value)
        )
    assert "tokamax" in str(exc_info.value)

    parser = capture_parser(opendde_cli.main)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--input-json",
                "in.json",
                "--weights",
                "w.jax",
                "--out",
                "o",
                "--diffusion-attention-backend",
                bad_value,
            ]
        )
    err = capsys.readouterr().err.replace("'", "")
    assert "xla, xla_jit, xla_sdpa, tokamax" in err


def test_compiled_pool_identity_separates_tokamax_from_xla_variants() -> None:
    args = ({}, (), jnp.zeros((2,), dtype=jnp.float32))
    base = {
        "params_treedef": (),
        "params_flags": (),
        "cp_shards": 1,
        "cp_layout": "1d",
        "num_samples": 1,
        "num_recycles": 1,
        "n_queries": 32,
        "n_" + "keys": 128,
        "diffusion_attention_backend": "xla_jit",
        "trunk_single_attention_backend": "xla_jit",
        "structural_single_attention_backend": "xla_jit",
    }
    ids = [
        opendde_model._compiled_opendde_infer._identity(
            args, {**base, "diffusion_attention_backend": choice}
        )
        for choice in ("xla", "xla_jit", "xla_sdpa", "tokamax")
    ]
    assert len(set(ids)) == 4


# ----------------------------------------------------------------------- (b) --


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
            layernorm_z=_random_norm(rng, c_z, with_bias=False),
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


def _opendde_diffusion_fixture(seed: int = 23) -> dict[str, Any]:
    """Small OpenDDE-shaped denoiser fixture with non-zero params, bias and masks."""
    rng = np.random.default_rng(seed)
    n_samples = 2
    n_struct = 5
    n_atom = 7  # Ragged against n_queries=2 -> 4 windows, q_pad=1.
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
    tok_block = _random_transformer_block(
        rng, c_a=c_tok, c_s=c_s, c_z=c_z, heads=heads, cross_attention=False
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
        diffusion_transformer=DiffusionTransformerStackParams(blocks=(tok_block,)),
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
        **{"n_queries": n_queries, "n_" + "keys": n_window_cols},
    )
    d_lm = pos_q[:, :, None, :] - pos_k[:, None, :, :]
    id_q, id_k, _ = rearrange_qk_to_dense_trunk(
        atom_to_struct[:, None],
        atom_to_struct[:, None],
        **{
            "n_queries": n_queries,
            "n_" + "keys": n_window_cols,
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
        **{"n_queries": n_queries, "n_" + "keys": n_window_cols},
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
        "p_lm": p_lm,
        "c_l": c_l,
        "extra_attn_bias": structural_pair_bias,
        "struct_mask": struct_mask,
        "atom_mask": atom_mask,
        "n_struct": n_struct,
        "n_queries": n_queries,
        "n_window_cols": n_window_cols,
        "heads": heads,
    }


def _call_denoiser(
    fixture: dict[str, Any],
    *,
    attention_backend: str,
    use_scan: bool = False,
    extra_attn_bias: jnp.ndarray | None = ...,  # type: ignore[assignment]
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
        "extra_attn_bias": bias,
        "n_" + "token": fixture["n_struct"],
        "atom_encoder_heads": fixture["heads"],
        "token" + "_heads": fixture["heads"],
        "atom_decoder_heads": fixture["heads"],
        "n_queries": fixture["n_queries"],
        "n_" + "keys": fixture["n_window_cols"],
        "sigma_data": 4.0,
        "use_scan": use_scan,
        "attention_backend": attention_backend,
        "token" + "_mask": fixture["struct_mask"],
        "atom_mask": fixture["atom_mask"],
    }
    return opendde_diffusion.diffusion_module_forward(*fixture["positional"], **kw)


@pytest.mark.parametrize("use_scan", [False, True])
def test_opendde_diffusion_tokamax_matches_xla_jit_for_token_and_atom_sites(
    monkeypatch: pytest.MonkeyPatch, use_scan: bool
) -> None:
    """One option dispatches atom encoder, structural-token stack and atom decoder."""
    _tokamax_xla(monkeypatch)
    fixture = _opendde_diffusion_fixture()

    calls: list[dict[str, Any]] = []
    real_tokamax = attention_tokamax.tokamax_attention

    def spy_tokamax(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, **kwargs: Any):
        calls.append(
            {
                "q_shape": tuple(q.shape),
                "k_shape": tuple(k.shape),
                "has_bias": kwargs.get("bias") is not None,
                "has_mask": kwargs.get("mask") is not None,
                "bias": kwargs.get("bias"),
            }
        )
        return real_tokamax(q, k, v, **kwargs)

    monkeypatch.setattr(attention_tokamax, "tokamax_attention", spy_tokamax)

    xla_out = _call_denoiser(fixture, attention_backend="xla_jit", use_scan=use_scan)
    assert calls == []

    tok_out = _call_denoiser(fixture, attention_backend="tokamax", use_scan=use_scan)
    # 3 sites: atom encoder (windowed), structural token stack (global),
    # and atom decoder (windowed).
    assert len(calls) == 3
    # Atom encoder & decoder pass flattened windows [B*n_trunks, n_queries, H, D]
    # with explicit mask + bias.
    assert calls[0]["q_shape"][1] == fixture["n_queries"]
    assert calls[0]["k_shape"][1] == fixture["n_window_cols"]
    assert calls[0]["has_bias"] is True and calls[0]["has_mask"] is True
    assert calls[2]["q_shape"] == calls[0]["q_shape"]
    assert calls[2]["has_bias"] is True and calls[2]["has_mask"] is True
    # Global structural-token site passes [N_sample, N_struct, H, D] with additive
    # bias (projected z + extra_attn_bias + mask_bias) and mask=None.
    assert calls[1]["q_shape"][1] == fixture["n_struct"]
    assert calls[1]["k_shape"][1] == fixture["n_struct"]
    assert calls[1]["has_bias"] is True and calls[1]["has_mask"] is False

    assert not bool(jnp.isnan(tok_out).any())
    np.testing.assert_allclose(
        np.asarray(tok_out),
        np.asarray(xla_out),
        rtol=1e-5,
        atol=1e-5,
    )
    # Masked atoms must stay zero.
    masked_atoms = ~np.asarray(fixture["atom_mask"])
    np.testing.assert_array_equal(np.asarray(tok_out)[:, masked_atoms, :], 0.0)


def test_extra_attn_bias_and_padding_masks_act_identically_on_tokamax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extra bias moves valid atoms; padded suffix perturbations do not."""
    _tokamax_xla(monkeypatch)
    fixture = _opendde_diffusion_fixture(seed=41)

    with_bias = _call_denoiser(fixture, attention_backend="tokamax")
    without_bias = _call_denoiser(
        fixture, attention_backend="tokamax", extra_attn_bias=None
    )
    valid_atoms = np.asarray(fixture["atom_mask"])
    assert not np.allclose(
        np.asarray(with_bias)[:, valid_atoms, :],
        np.asarray(without_bias)[:, valid_atoms, :],
    ), "structural_pair_bias must reach and move the tokamax token attention"

    # Poison masked structural tokens (indices 3, 4) and masked atoms (2, 5, 6).
    struct_valid = fixture["struct_mask"]
    pair_valid = struct_valid[:, None] & struct_valid[None, :]
    atom_valid = fixture["atom_mask"]
    poisoned = dict(fixture)
    poisoned["s_inputs"] = jnp.where(struct_valid[:, None], fixture["s_inputs"], 999.0)
    poisoned["s_trunk"] = jnp.where(struct_valid[:, None], fixture["s_trunk"], -777.0)
    poisoned["extra_attn_bias"] = jnp.where(
        pair_valid, fixture["extra_attn_bias"], 500.0
    )
    poisoned["x_noisy"] = jnp.where(
        atom_valid[None, :, None], fixture["x_noisy"], 0.0
    )
    perturbed = _call_denoiser(poisoned, attention_backend="tokamax")
    np.testing.assert_allclose(
        np.asarray(perturbed)[:, valid_atoms, :],
        np.asarray(with_bias)[:, valid_atoms, :],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(np.asarray(perturbed)[:, ~valid_atoms, :], 0.0)


def test_opendde_infer_static_threads_tokamax_and_matches_xla_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full `opendde_infer_static` denoiser loop with bf16 autocast and fp32 default."""
    _tokamax_xla(monkeypatch)
    fixture = _opendde_diffusion_fixture(seed=77)
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
        "token" + "_index": jnp.arange(n_res).astype(jnp.int32),
        "asym_id": jnp.zeros((n_res,), dtype=jnp.int32),
        "residue_index": jnp.arange(n_res, dtype=jnp.int32),
        "entity_id": jnp.zeros((n_res,), dtype=jnp.int32),
        "sym_id": jnp.zeros((n_res,), dtype=jnp.int32),
        "atom_to_" + "token_idx": jnp.asarray([0, 0, 1, 1, 2, 2, 2], jnp.int32),
        "atom_to_tokatom_idx": jnp.asarray([0, 1, 0, 1, 0, 1, 2], jnp.int32),
        "has_frame": jnp.asarray([True, True, True]),
        "frame_atom_index": jnp.zeros((n_res, 3), dtype=jnp.int32),
        "pae_rep_atom_mask": jnp.asarray([1, 0, 1, 0, 1, 0, 0], dtype=jnp.int32),
        "distogram_rep_atom_mask": jnp.asarray([1, 0, 1, 0, 1, 0, 0], dtype=jnp.int32),
        "parent_residue_idx": jnp.asarray([0, 0, 1, 2, 2], dtype=jnp.int32),
        "sub" + "token_role_id": jnp.asarray([1, 2, 1, 1, 2], jnp.int32),
        "structural_" + "token_index": jnp.arange(n_struct).astype(jnp.int32),
        "atom_to_structural_" + "token_idx": fixture["positional"][0],
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
        "structural_" + "token_padding_mask": fixture["struct_mask"],
        "atom_padding_mask": fixture["atom_mask"],
    }

    seen_trunk_backends: list[tuple[str, str]] = []
    monkeypatch.setattr(
        opendde_model,
        "input_feature_embedder",
        lambda *_a, **_k: jnp.ones((n_res, 5), dtype=jnp.float32),
    )

    def fake_trunk(*_a: Any, **kw: Any):
        seen_trunk_backends.append(("trunk", kw["single_attention_backend"]))
        return (
            jnp.ones((n_res, 5), dtype=jnp.float32),
            jnp.ones((n_res, 6), dtype=jnp.float32),
            jnp.ones((n_res, n_res, 6), dtype=jnp.float32),
        )

    def fake_refiner(s: Any, z: Any, *_a: Any, **kw: Any):
        seen_trunk_backends.append(("structural", kw["single_attention_backend"]))
        return s, z

    monkeypatch.setattr(opendde_model, "pairformer_output_from_s_inputs", fake_trunk)
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
            "atom_to_" + "token_idx": raw["atom_to_structural_" + "token_idx"],
        },
    )
    monkeypatch.setattr(opendde_model, "structural_refiner_stack", fake_refiner)
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

    params = SimpleNamespace(
        input_embedder=object(),
        pairformer_output=object(),
        structural_expander=object(),
        structural_refiner=object(),
        diffusion=fixture["params"],
        distogram=object(),
        confidence=None,
    )
    schedule = jnp.asarray([2.0, 0.5, 0.0], dtype=jnp.float32)
    init_noise = fixture["x_noisy"][:1]
    step_noises = (
        jnp.zeros((1, n_atom, 3), dtype=jnp.float32),
        jnp.zeros((1, n_atom, 3), dtype=jnp.float32),
    )
    rotations = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (2, 1, 3, 3))
    translations = jnp.zeros((2, 1, 3), dtype=jnp.float32)

    def run_infer(backend_name: str, *, autocast: bool = False) -> jnp.ndarray:
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
            "num_samples": 1,
            "init_noise": init_noise,
            "step_noises": step_noises,
            "rotations": rotations,
            "translations": translations,
            "num_recycles": 1,
            "atom_encoder_heads": fixture["heads"],
            "token" + "_heads": fixture["heads"],
            "atom_decoder_heads": fixture["heads"],
            "n_queries": fixture["n_queries"],
            "n_" + "keys": fixture["n_window_cols"],
            "sigma_data": 4.0,
            "diffusion_attention_backend": backend_name,
            "diffusion_autocast": autocast,
            "run_confidence": False,
        }
        return opendde_model.opendde_infer_static(
            features, model_params, schedule, **call_kw
        )["coordinate"]

    coord_xla = run_infer("xla_jit")
    coord_tok = run_infer("tokamax")
    np.testing.assert_allclose(
        np.asarray(coord_tok), np.asarray(coord_xla), rtol=1e-5, atol=1e-5
    )
    assert seen_trunk_backends == [
        ("trunk", "xla_jit"),
        ("structural", "xla_jit"),
        ("trunk", "xla_jit"),
        ("structural", "xla_jit"),
    ]

    # Also verify under OpenDDE's opt-in bf16 diffusion autocast.
    coord_xla_bf16 = run_infer("xla_jit", autocast=True)
    coord_tok_bf16 = run_infer("tokamax", autocast=True)
    assert coord_tok_bf16.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(coord_tok_bf16), np.asarray(coord_xla_bf16), rtol=2e-2, atol=2e-2
    )


# ----------------------------------------------------------------------- (c) --


def test_tfg_guidance_falls_back_from_tokamax_to_xla_jit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Under `use_tfg_guidance`, the adapter refuses tokamax; direct CLI falls back."""
    backend = OpenDDEBackend()
    for tfg_spelling in (True, "true"):
        for tokamax_spelling in ("tokamax", " tokamax ", "TOKAMAX"):
            guided_tokamax = _request(
                tmp_path / "adapter-refused",
                use_tfg_guidance=tfg_spelling,
                diffusion_attention_backend=tokamax_spelling,
            )
            with pytest.raises(
                ValueError, match="diffusion_attention_backend=tokamax"
            ) as exc_info:
                backend.validate_request(guided_tokamax)
            message = str(exc_info.value)
            assert "recompiles tokamax's fused attention" in message
            assert "omit the option" in message
            assert "diffusion_attention_backend=xla_jit" in message
            with pytest.raises(
                ValueError, match="diffusion_attention_backend=tokamax"
            ):
                backend.cache_profile(guided_tokamax)
            # `predict` renders argv here without `validate_request`, so it
            # also meets spellings `validate_request` would have refused.
            for native_spelling in (tfg_spelling, np.bool_(True)):
                with pytest.raises(
                    ValueError, match="diffusion_attention_backend=tokamax"
                ):
                    backend._native_invocation(
                        dataclasses.replace(
                            guided_tokamax,
                            options={
                                **guided_tokamax.options,
                                "use_tfg_guidance": native_spelling,
                            },
                        )
                    )

        guided_default = _request(
            tmp_path / "adapter-default", use_tfg_guidance=tfg_spelling
        )
        backend.validate_request(guided_default)
        assert "diffusion_attention_backend" not in backend.cache_profile(
            guided_default
        )

    for allowed_backend in ("xla_jit", "xla", "xla_sdpa"):
        guided_allowed = _request(
            tmp_path / f"adapter-{allowed_backend}",
            use_tfg_guidance=True,
            diffusion_attention_backend=allowed_backend,
        )
        backend.validate_request(guided_allowed)
    guided_omitted = _request(tmp_path / "cache-same", use_tfg_guidance=True)
    guided_explicit_xla_jit = dataclasses.replace(
        guided_omitted,
        options={
            "use_tfg_guidance": True,
            "diffusion_attention_backend": "xla_jit",
        },
    )
    assert resolve_cache_dir(guided_omitted, backend) == resolve_cache_dir(
        guided_explicit_xla_jit, backend
    )

    for off_spelling in (False, "false", 0, "0"):
        unguided_tokamax = _request(
            tmp_path / "adapter-off",
            use_tfg_guidance=off_spelling,
            diffusion_attention_backend="tokamax",
        )
        backend.validate_request(unguided_tokamax)
        assert (
            backend.cache_profile(unguided_tokamax)["diffusion_attention_backend"]
            == "tokamax"
        )

    for edge_value in ("", "0", 0, "triton"):
        edge_request = _request(
            tmp_path / "adapter-edge",
            use_tfg_guidance=True,
            diffusion_attention_backend=edge_value,
        )
        backend.validate_request(edge_request)
        with pytest.raises(ValueError, match="diffusion_attention_backend must be"):
            backend._native_invocation(edge_request)

    with pytest.raises(ValueError, match="use_tfg_guidance must be a boolean"):
        backend.validate_request(
            _request(
                tmp_path / "adapter-bad-tfg",
                use_tfg_guidance="maybe",
                diffusion_attention_backend="tokamax",
            )
        )

    guided_tok, guided_infer = _run_cli_to_infer(
        tmp_path / "guided-tok",
        monkeypatch,
        "--use-tfg-guidance",
        "true",
        "--diffusion-attention-backend",
        "tokamax",
    )
    out = capsys.readouterr().out
    assert "TFG enabled: diffusion attention runs xla_jit, not tokamax" in out
    assert guided_tok[0]["diffusion_attention_backend"] == "xla_jit"
    assert guided_infer[0][0] == "static"
    assert guided_infer[0][1]["diffusion_attention_backend"] == "xla_jit"

    # Explicit non-tokamax backend is preserved under guidance without the fallback log.
    guided_xla, _ = _run_cli_to_infer(
        tmp_path / "guided-xla",
        monkeypatch,
        "--use-tfg-guidance",
        "true",
        "--diffusion-attention-backend",
        "xla",
    )
    out_xla = capsys.readouterr().out
    assert "diffusion attention runs xla_jit, not tokamax" not in out_xla
    assert guided_xla[0]["diffusion_attention_backend"] == "xla"


def test_tokamax_is_refused_under_context_parallelism_on_opendde_denoiser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared CP guard refuses tokamax on OpenDDE's denoiser and runner."""
    fixture = _opendde_diffusion_fixture(seed=91)
    monkeypatch.setattr(attention_impl, "cp_mesh", lambda: object())

    with pytest.raises(ValueError, match="not supported under context parallelism"):
        _call_denoiser(fixture, attention_backend="tokamax")

    # xla_jit continues to pass the attention guard when a mesh is active.
    out = _call_denoiser(fixture, attention_backend="xla_jit")
    assert out.shape == fixture["x_noisy"].shape

    # Through OpenDDEBackend -> runner.run_prediction, the CP refusal surfaces cleanly.
    job = {"name": "tiny", "modelSeeds": [101], "sequences": []}
    input_path = tmp_path / "job.json"
    input_path.write_text(json.dumps([job]), encoding="utf-8")
    weights = tmp_path / "opendde.jax"
    weights.write_bytes(b"native fixture")

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

    def check_cp_guard(*_a: Any, **kw: Any) -> dict[str, Any]:
        attention_impl._reject_tokamax_under_cp(kw["diffusion_attention_backend"])
        return {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}

    monkeypatch.setattr(opendde_runner, "_predict", check_cp_guard)
    request = PredictionRequest(
        model="opendde",
        input=input_path,
        input_format="native",
        weights=weights,
        output_dir=tmp_path / "out-cp",
        cache_dir=tmp_path / "cache-cp",
        seed=101,
        padding=False,
        options={"cp_devices": 2, "diffusion_attention_backend": "tokamax"},
    )
    with pytest.raises(SystemExit, match="not supported under context parallelism"):
        OpenDDEBackend().predict(request)
