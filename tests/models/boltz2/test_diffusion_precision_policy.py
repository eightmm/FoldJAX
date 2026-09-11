"""Scope gates for the opt-in BF16 diffusion score model and its attention.

Boltz-2 ships the whole score model in FP32 -- upstream wraps
``structure_module.sample`` in ``torch.autocast(enabled=False)`` -- so the two
knobs here are experiments, not defaults. What they must preserve is the shape
of the AlphaFold 3 mixed-precision cell: low-precision GEMMs and low-precision
q/k/v/bias into a fused kernel, an FP32 residual stream, and FP32 coordinates
in and out. These tests assert those properties on the arrays themselves
rather than on a config spelling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.boltz2.models.diffusion.atom as atom_module
import foldjax.models.boltz2.models.diffusion.diffusion_transformer as dt_module
import foldjax.models.boltz2.models.predict as predict_module
import foldjax.models.boltz2.models.trunk_blocks.trunk as trunk_module
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.models.boltz2 import api
from tests.models.boltz2.test_atom_cp_model_integration import (
    _model_inputs,
    _native_params,
)


def _sampler_case():
    native = _native_params()
    inputs = _model_inputs()
    params = {"trunk": {}, "conditioned_diffusion": native}
    trunk = {
        "s_inputs": inputs["s_inputs"],
        "s": inputs["s_trunk"],
        "z": inputs["z_trunk"],
        "relative_position_encoding": inputs["relative_position_encoding"],
    }
    return params, dict(inputs["feats"]), trunk


def _sample(params, feats, trunk, **kwargs):
    return trunk_module.boltz2_sample_forward(
        params,
        feats,
        jax.random.PRNGKey(0),
        trunk=trunk,
        num_sampling_steps=3,
        multiplicity=1,
        token_layers=1,
        # Both are the sampler's own stochastic/alignment machinery, and both
        # only add FP32 geometry around the score model this measures.
        augmentation=False,
        alignment_reverse_diff=False,
        **kwargs,
    )


def _record_attention(monkeypatch) -> list[tuple[jnp.dtype, ...]]:
    """Capture the operand dtypes each fused-attention call receives."""

    seen: list[tuple[jnp.dtype, ...]] = []

    def fake(q, k, v, bias, mask, **kwargs):
        seen.append((q.dtype, k.dtype, v.dtype, bias.dtype))
        return jnp.zeros_like(q)

    monkeypatch.setattr(dt_module, "tokamax_dot_product_attention", fake)
    return seen


def test_score_cast_moves_only_kernels_outside_the_coordinate_islands() -> None:
    native = _native_params()
    cast = trunk_module._cast_score_params(native["score_model"], jnp.bfloat16)

    moved = set()
    originals = jax.tree_util.tree_flatten_with_path(native["score_model"])[0]
    casts = jax.tree_util.tree_flatten_with_path(cast)[0]
    for (path, original), (_, value) in zip(originals, casts, strict=True):
        # Stacked layers are lists, whose path entries carry `idx` not `key`.
        keys = tuple(getattr(entry, "key", None) for entry in path)
        if not hasattr(original, "dtype"):
            # e.g. the `num_heads` int beside the projection kernels.
            assert value is original, keys
            continue
        island = any(
            keys[: len(prefix)] == prefix
            for prefix in trunk_module._FP32_SCORE_ISLANDS
        )
        expected = jnp.dtype(
            jnp.bfloat16 if keys[-1] == "kernel" and not island else jnp.float32
        )
        assert value.dtype == expected, keys
        np.testing.assert_array_equal(value, original.astype(value.dtype))
        if value.dtype == jnp.dtype(jnp.bfloat16):
            moved.add(keys[0])

    # The cast has to reach the modules the knob exists for, not just parse.
    assert {"token_transformer", "atom_attention_encoder", "s_to_a_linear"} <= moved
    # And it must leave both coordinate boundaries alone.
    assert (
        cast["atom_attention_encoder"]["r_to_q_trans"]["kernel"].dtype
        == jnp.dtype(jnp.float32)
    )
    assert (
        cast["atom_attention_decoder"]["atom_feat_to_atom_pos_update"]["linear"][
            "kernel"
        ].dtype
        == jnp.dtype(jnp.float32)
    )


@pytest.mark.parametrize("trunk_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("lazy_token_trans_bias", [True, False])
def test_bfloat16_diffusion_is_bf16_gemms_around_an_fp32_residual(
    monkeypatch, trunk_dtype: str, lazy_token_trans_bias: bool
) -> None:
    """The AF3 cell: bf16 into the kernel, fp32 carry, fp32 coordinates."""

    params, feats, trunk = _sampler_case()
    operands = _record_attention(monkeypatch)

    residuals: list[jnp.dtype] = []
    apply_layer = atom_module.diffusion_transformer_layer_apply

    def record_residual(layer_params, a, *args, **kwargs):
        residuals.append(a.dtype)
        return apply_layer(layer_params, a, *args, **kwargs)

    monkeypatch.setattr(
        atom_module, "diffusion_transformer_layer_apply", record_residual
    )

    kernels: dict[str, jnp.dtype] = {}
    score_forward = trunk_module._preconditioned_score_forward

    def record_score(score_params, **kwargs):
        kernels["s_to_a"] = score_params["s_to_a_linear"]["linear"]["kernel"].dtype
        kernels["r_to_q"] = score_params["atom_attention_encoder"]["r_to_q_trans"][
            "kernel"
        ].dtype
        kernels["pos_update"] = score_params["atom_attention_decoder"][
            "atom_feat_to_atom_pos_update"
        ]["linear"]["kernel"].dtype
        kernels["r_noisy"] = kwargs["r_noisy"].dtype
        return score_forward(score_params, **kwargs)

    monkeypatch.setattr(trunk_module, "_preconditioned_score_forward", record_score)

    out = _sample(
        params,
        feats,
        trunk,
        compute_dtype=jnp.dtype(trunk_dtype),
        lazy_token_trans_bias=lazy_token_trans_bias,
        diffusion_compute_dtype="bfloat16",
        diffusion_attention_backend="tokamax",
    )

    assert operands, "the fused attention backend never ran"
    assert set(operands) == {(jnp.dtype(jnp.bfloat16),) * 4}
    assert set(residuals) == {jnp.dtype(jnp.float32)}
    assert out["sample_atom_coords"].dtype == jnp.dtype(jnp.float32)
    assert kernels["s_to_a"] == jnp.dtype(jnp.bfloat16)
    assert kernels["r_to_q"] == jnp.dtype(jnp.float32)
    assert kernels["pos_update"] == jnp.dtype(jnp.float32)
    assert kernels["r_noisy"] == jnp.dtype(jnp.float32)


def test_bfloat16_diffusion_adaln_normalizations_only_see_fp32(monkeypatch) -> None:
    """`_layer_norm_scale`/`_layer_norm_no_affine` have no FP32 upcast.

    They accumulate their mean and variance in whatever they are handed, so
    the knob has to keep both the residual and the conditioning stream FP32.
    """

    params, feats, trunk = _sampler_case()
    _record_attention(monkeypatch)
    seen: list[tuple[str, jnp.dtype]] = []

    no_affine = dt_module._layer_norm_no_affine
    scale = dt_module._layer_norm_scale

    def record_no_affine(x, eps):
        seen.append(("no_affine", x.dtype))
        return no_affine(x, eps)

    def record_scale(x, scale_param, eps):
        seen.append(("scale", x.dtype))
        return scale(x, scale_param, eps)

    monkeypatch.setattr(dt_module, "_layer_norm_no_affine", record_no_affine)
    monkeypatch.setattr(dt_module, "_layer_norm_scale", record_scale)

    _sample(
        params,
        feats,
        trunk,
        diffusion_compute_dtype="bfloat16",
        diffusion_attention_backend="tokamax",
    )

    assert {name for name, _ in seen} == {"no_affine", "scale"}
    assert {dtype for _, dtype in seen} == {jnp.dtype(jnp.float32)}


def test_released_diffusion_still_hands_the_kernel_fp32_operands(monkeypatch) -> None:
    """Without the dtype knob the fused backend gets the FP32 island it always did."""

    params, feats, trunk = _sampler_case()
    operands = _record_attention(monkeypatch)

    _sample(params, feats, trunk, diffusion_attention_backend="tokamax")

    assert operands
    assert set(operands) == {(jnp.dtype(jnp.float32),) * 4}


def test_diffusion_attention_backend_leaves_the_trunk_alone(monkeypatch) -> None:
    seen = {}

    def fake_trunk(params, feats, **kwargs):
        seen["trunk"] = kwargs["attention_backend"]
        return {
            "s_inputs": jnp.zeros((1, 1, 1)),
            "s": jnp.zeros((1, 1, 1)),
            "z": jnp.zeros((1, 1, 1, 1)),
        }

    def fake_sample(params, feats, key, *, trunk, **kwargs):
        seen["sample"] = (
            kwargs["attention_backend"],
            kwargs["diffusion_attention_backend"],
            kwargs["diffusion_compute_dtype"],
        )
        return {"sample_atom_coords": jnp.zeros((1, 1, 3))}

    monkeypatch.setattr(predict_module, "boltz2_trunk_forward", fake_trunk)
    monkeypatch.setattr(predict_module, "boltz2_sample_forward", fake_sample)

    predict_module.boltz2_predict(
        {"trunk": {}},
        {},
        jax.random.PRNGKey(0),
        run_confidence=False,
        run_distogram=False,
        attention_backend="xla",
        diffusion_attention_backend="tokamax",
        diffusion_compute_dtype="bfloat16",
    )

    assert seen == {"trunk": "xla", "sample": ("xla", "tokamax", "bfloat16")}


def test_predict_canonicalizes_an_inherited_diffusion_backend(monkeypatch) -> None:
    """An explicit value equal to the global one must not be a second program."""

    seen = {}

    def fake_trunk(params, feats, **kwargs):
        return {
            "s_inputs": jnp.zeros((1, 1, 1)),
            "s": jnp.zeros((1, 1, 1)),
            "z": jnp.zeros((1, 1, 1, 1)),
        }

    def fake_sample(params, feats, key, *, trunk, **kwargs):
        seen["diffusion"] = kwargs["diffusion_attention_backend"]
        return {"sample_atom_coords": jnp.zeros((1, 1, 3))}

    monkeypatch.setattr(predict_module, "boltz2_trunk_forward", fake_trunk)
    monkeypatch.setattr(predict_module, "boltz2_sample_forward", fake_sample)

    predict_module.boltz2_predict(
        {"trunk": {}},
        {},
        jax.random.PRNGKey(0),
        run_confidence=False,
        run_distogram=False,
        attention_backend="tokamax",
        diffusion_attention_backend="tokamax",
    )

    assert seen == {"diffusion": None}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"diffusion_attention_backend": "unknown"}, "must be 'tokamax'"),
        ({"diffusion_compute_dtype": "float16"}, "must be float32 or bfloat16"),
        (
            {"diffusion_attention_backend": "triton"},
            "requires diffusion_compute_dtype='bfloat16'",
        ),
    ],
)
def test_sampler_rejects_unsupported_diffusion_policy(
    kwargs: dict[str, object], match: str
) -> None:
    params, feats, trunk = _sampler_case()
    with pytest.raises(ValueError, match=match):
        _sample(params, feats, trunk, **kwargs)


@pytest.mark.parametrize("backend", ["tokamax", "triton"])
def test_context_parallelism_rejects_a_fused_diffusion_backend(
    monkeypatch, backend: str
) -> None:
    """The 2-D grid routes pair-bias attention past the backend switch."""

    params, feats, trunk = _sampler_case()
    monkeypatch.setattr(trunk_module, "_cp_mesh", lambda: object())
    with pytest.raises(ValueError, match="context parallelism requires"):
        _sample(
            params,
            feats,
            trunk,
            diffusion_attention_backend=backend,
            diffusion_compute_dtype="bfloat16",
        )


def test_context_parallelism_accepts_an_inherited_diffusion_backend(
    monkeypatch,
) -> None:
    seen = {}

    def fake_sample(params, feats, key, *, trunk, **kwargs):
        seen["diffusion"] = kwargs["diffusion_attention_backend"]
        return {"sample_atom_coords": jnp.zeros((1, 1, 3))}

    monkeypatch.setattr(predict_module, "cp_mesh", lambda: object())
    monkeypatch.setattr(
        predict_module,
        "boltz2_trunk_forward",
        lambda params, feats, **kwargs: {
            "s_inputs": jnp.zeros((1, 1, 1)),
            "s": jnp.zeros((1, 1, 1)),
            "z": jnp.zeros((1, 1, 1, 1)),
        },
    )
    monkeypatch.setattr(predict_module, "boltz2_sample_forward", fake_sample)

    predict_module.boltz2_predict(
        {"trunk": {}},
        {},
        jax.random.PRNGKey(0),
        run_confidence=False,
        run_distogram=False,
        attention_backend="tokamax",
        diffusion_attention_backend="tokamax",
    )

    assert seen == {"diffusion": None}


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"diffusion_attention_backend": "unknown"}, "must be one of"),
        ({"diffusion_compute_dtype": "float16"}, "must be one of"),
        (
            {"diffusion_attention_backend": "triton"},
            "requires diffusion_compute_dtype='bfloat16'",
        ),
        (
            {
                "diffusion_attention_backend": "tokamax",
                "diffusion_compute_dtype": "bfloat16",
                "cp_devices": 2,
            },
            "context parallelism requires",
        ),
    ],
)
def test_backend_rejects_unsupported_diffusion_policy(
    options: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        Boltz2Backend().validate_native_options(options)


def test_backend_accepts_an_inherited_diffusion_backend_with_cp() -> None:
    Boltz2Backend().validate_native_options(
        {
            "attention_backend": "tokamax",
            "diffusion_attention_backend": "tokamax",
            "cp_devices": 2,
        }
    )


def test_backend_option_table_carries_both_diffusion_knobs() -> None:
    backend = Boltz2Backend()
    for name in ("diffusion_attention_backend", "diffusion_compute_dtype"):
        assert name in backend.native_options
        assert name in backend.compile_options


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"diffusion_attention_backend": "unknown"}, "must be null or one of"),
        ({"diffusion_compute_dtype": "float16"}, "must be one of"),
        (
            {"diffusion_attention_backend": "triton"},
            "requires diffusion_compute_dtype='bfloat16'",
        ),
        (
            {"diffusion_attention_backend": "tokamax", "cp_devices": 2},
            "context parallelism requires",
        ),
    ],
)
def test_high_level_predict_rejects_unsupported_diffusion_policy(
    tmp_path, kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        api.predict(
            seq=["ACD"], weights=tmp_path / "unused", mols=tmp_path, **kwargs
        )


@pytest.mark.parametrize("attention", ["xla", "tokamax"])
def test_bfloat16_diffusion_stays_close_to_the_fp32_arm(attention: str) -> None:
    """A loose BF16 tolerance, read against the structure's own extent."""

    params, feats, trunk = _sampler_case()
    reference = np.asarray(_sample(params, feats, trunk)["sample_atom_coords"])
    low = np.asarray(
        _sample(
            params,
            feats,
            trunk,
            diffusion_compute_dtype="bfloat16",
            diffusion_attention_backend=attention,
        )["sample_atom_coords"]
    )

    assert np.isfinite(low).all()
    real = np.asarray(feats["atom_pad_mask"]).astype(bool)[0]
    extent = float(reference[0][real].std())
    rmsd = float(
        np.sqrt(np.mean(np.sum((low[0][real] - reference[0][real]) ** 2, axis=-1)))
    )
    # Not a parity bound: bf16 carries ~3 decimal digits, so the arms are
    # different draws of the same trajectory. The claim is that the knob does
    # not move the structure on the scale of the structure.
    assert rmsd < 0.01 * extent
