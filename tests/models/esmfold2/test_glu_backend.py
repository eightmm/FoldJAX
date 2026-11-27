"""The opt-in fused SwiGLU in the diffusion token transformer.

Everything here runs on CPU, where the Triton kernel cannot. What that leaves
provable is the part that is actually easy to get wrong: which half of the
packed `lin_swish` projection the fused entry point activates. `lin_swish` is
one `nn.Linear(d_model, 2 * hidden)`, so its transpose is `[K, 2N]` and
tokamax wants `[K, 2, N]` with index 0 activated -- a claim about C order that
is either exactly right or silently swaps the gate and the value.

So `_glu._fused` is replaced with tokamax's own `implementation="xla"` path.
That keeps tokamax's semantics for the fused-weights layout while dropping
only the requirement for a GPU, and it counts its calls, because a
numbers-only test passes just as well when the branch is dead. The swapped
control in `test_the_packed_orientation_is_not_symmetric` is what gives the
agreement its meaning.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from foldjax.models._glu import GLU_BACKENDS
from foldjax.models.esmfold2.models import diffusion

D_TOKEN, HIDDEN, N_TOKENS = 16, 24, 5

#: Zero on CPU float32 in the measured run: tokamax's XLA implementation and
#: the split path issue the same two matmuls. Stated as a tolerance rather
#: than asserted exact, because bit-identity across the two is not the
#: contract -- `foldjax.models._glu` says to read a backend change as a
#: numerics change.
ATOL = 1e-5


def _params(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "pre_norm.weight": rng.standard_normal(D_TOKEN).astype(np.float32),
        "pre_norm.bias": rng.standard_normal(D_TOKEN).astype(np.float32),
        "lin_swish.weight": rng.standard_normal((2 * HIDDEN, D_TOKEN)).astype(
            np.float32
        ),
        "lin_out.weight": rng.standard_normal((D_TOKEN, HIDDEN)).astype(np.float32),
    }


def _activations(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, N_TOKENS, D_TOKEN)).astype(np.float32)


@pytest.fixture
def fused_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    """Record every fused call and answer it on the CPU, via tokamax itself."""

    tokamax = pytest.importorskip("tokamax")
    seen: list[tuple[int, ...]] = []

    def stand_in(x, weights, activation):
        seen.append(tuple(weights.shape))
        return tokamax.gated_linear_unit(
            x=x, weights=weights, activation=activation, implementation="xla"
        )

    monkeypatch.setattr("foldjax.models._glu._fused", stand_in)
    return seen


def test_the_fused_backend_matches_the_split_path(fused_calls) -> None:
    params, a = _params(), _activations()

    expected = diffusion.conditioned_transition_block(a, None, params)
    assert fused_calls == [], "the xla path must not reach the fused kernel"

    actual = diffusion.conditioned_transition_block(
        a, None, params, glu_backend="tokamax"
    )
    # The tripwire: without it this file passes with the branch removed.
    assert fused_calls == [(D_TOKEN, 2, HIDDEN)]
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=ATOL, rtol=ATOL
    )


def test_the_packed_orientation_is_not_symmetric(fused_calls) -> None:
    """Swapping the two halves must change the answer, or nothing is proved."""

    params, a = _params(), _activations()
    expected = diffusion.conditioned_transition_block(a, None, params)

    swapped = dict(params)
    kernel = params["lin_swish.weight"]
    swapped["lin_swish.weight"] = np.concatenate(
        [kernel[HIDDEN:], kernel[:HIDDEN]], axis=0
    )
    wrong = diffusion.conditioned_transition_block(
        a, None, swapped, glu_backend="tokamax"
    )

    assert fused_calls, "the swapped run must still take the fused branch"
    difference = float(np.abs(np.asarray(wrong) - np.asarray(expected)).max())
    assert difference > 1.0, difference


def test_the_conditioned_path_also_matches(fused_calls) -> None:
    """The released blocks all take `s`, which is the adaLN + output gate."""

    rng = np.random.default_rng(2)
    params = _params()
    del params["pre_norm.weight"], params["pre_norm.bias"]
    params.update(
        {
            "adaln.s_scale": rng.standard_normal(D_TOKEN).astype(np.float32),
            "adaln.s_gate.weight": rng.standard_normal((D_TOKEN, D_TOKEN)).astype(
                np.float32
            ),
            "adaln.s_shift.weight": rng.standard_normal((D_TOKEN, D_TOKEN)).astype(
                np.float32
            ),
            "output_gate.weight": rng.standard_normal((D_TOKEN, D_TOKEN)).astype(
                np.float32
            ),
            "output_gate.bias": rng.standard_normal(D_TOKEN).astype(np.float32),
        }
    )
    a, s = _activations(), _activations(3)

    expected = diffusion.conditioned_transition_block(a, s, params)
    actual = diffusion.conditioned_transition_block(a, s, params, glu_backend="tokamax")

    assert fused_calls == [(D_TOKEN, 2, HIDDEN)]
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=ATOL, rtol=ATOL
    )


def test_the_released_default_never_reaches_the_fused_function(
    fused_calls,
) -> None:
    """The behavioural pin, not the spelling of the default.

    A test that asserted `DiffusionSettings().glu_backend == "xla"` would pass
    on a port whose call site ignored the field entirely.
    """

    params, a = _params(), _activations()

    diffusion.conditioned_transition_block(a, None, params)
    diffusion.conditioned_transition_block(a, None, params, glu_backend="xla")
    diffusion.diffusion_transformer(
        a,
        None,
        np.zeros((1, N_TOKENS, N_TOKENS), dtype=np.float32),
        {f"transition_blocks.0.{key}": value for key, value in params.items()}
        | {
            "attn_blocks.0.pre_norm.weight": params["pre_norm.weight"],
            "attn_blocks.0.pre_norm.bias": params["pre_norm.bias"],
            "attn_blocks.0.q_proj.weight": np.eye(D_TOKEN, dtype=np.float32),
            "attn_blocks.0.kv_proj.weight": np.eye(
                2 * D_TOKEN, D_TOKEN, dtype=np.float32
            ),
            "attn_blocks.0.g_proj.weight": np.eye(D_TOKEN, dtype=np.float32),
            "attn_blocks.0.out_proj.weight": np.eye(D_TOKEN, dtype=np.float32),
        },
        n_blocks=1,
        n_heads=2,
    )

    assert fused_calls == []


def test_the_default_settings_and_the_checkpoint_config_stay_on_xla() -> None:
    assert diffusion.DiffusionSettings().glu_backend == "xla"
    # No `config.json` key reaches this field, so a released checkpoint can
    # never select the fused kernel by accident.
    assert diffusion.settings_from_config({}).glu_backend == "xla"


def test_an_unknown_backend_is_refused_naming_the_allowed_values() -> None:
    params, a = _params(), _activations()

    with pytest.raises(ValueError) as excinfo:
        diffusion.conditioned_transition_block(a, None, params, glu_backend="fused")

    message = str(excinfo.value)
    assert all(name in message for name in GLU_BACKENDS), message


def test_a_biased_projection_is_refused_rather_than_silently_unfused() -> None:
    """Upstream builds `lin_swish` with `bias=False`; this is the tripwire."""

    params, a = _params(), _activations()
    params["lin_swish.bias"] = np.zeros(2 * HIDDEN, dtype=np.float32)

    with pytest.raises(ValueError, match="bias"):
        diffusion.conditioned_transition_block(a, None, params, glu_backend="tokamax")


def test_the_transformer_hands_its_backend_to_every_transition(
    fused_calls,
) -> None:
    """`diffusion_transformer` -> `conditioned_transition_block`.

    The counter in the released-default test proves the default reaches
    nothing. This proves a non-default reaches everything: two blocks, two
    fused calls.
    """

    params, a = _params(), _activations()
    blocks = {
        f"{stack}_blocks.{index}.{key}": value
        for index in (0, 1)
        for stack, keys in (
            ("transition", params),
            (
                "attn",
                {
                    "pre_norm.weight": params["pre_norm.weight"],
                    "pre_norm.bias": params["pre_norm.bias"],
                    "q_proj.weight": np.eye(D_TOKEN, dtype=np.float32),
                    "kv_proj.weight": np.eye(2 * D_TOKEN, D_TOKEN, dtype=np.float32),
                    "g_proj.weight": np.eye(D_TOKEN, dtype=np.float32),
                    "out_proj.weight": np.eye(D_TOKEN, dtype=np.float32),
                },
            ),
        )
        for key, value in keys.items()
    }

    diffusion.diffusion_transformer(
        a,
        None,
        np.zeros((1, N_TOKENS, N_TOKENS), dtype=np.float32),
        blocks,
        n_blocks=2,
        n_heads=2,
        glu_backend="tokamax",
    )

    assert fused_calls == [(D_TOKEN, 2, HIDDEN)] * 2


def test_the_settings_field_reaches_the_token_transformer(monkeypatch) -> None:
    """`DiffusionSettings.glu_backend` -> `diffusion_transformer`.

    Deleting the one line in `diffusion_module` that forwards the field
    leaves every other test in this file green, because they all pass the
    keyword by hand. This is the only test that fails.
    """

    seen: dict[str, object] = {}
    settings = diffusion.DiffusionSettings(c_token=D_TOKEN, glu_backend="tokamax")
    tokens = np.zeros((1, N_TOKENS, D_TOKEN), dtype=np.float32)

    monkeypatch.setattr(
        diffusion,
        "condition_single",
        lambda *args, **kwargs: np.zeros((1, N_TOKENS, D_TOKEN), dtype=np.float32),
    )
    monkeypatch.setattr(
        diffusion,
        "atom_encoder",
        lambda *args, **kwargs: (tokens, None, None, None),
    )
    monkeypatch.setattr(
        diffusion,
        "atom_decoder",
        lambda *args, **kwargs: np.zeros((1, 4, 3), dtype=np.float32),
    )

    def recorder(*args, **kwargs):
        seen.update(kwargs)
        return args[0]

    monkeypatch.setattr(diffusion, "diffusion_transformer", recorder)

    identity = np.eye(D_TOKEN, dtype=np.float32)
    ones = np.ones(D_TOKEN, dtype=np.float32)
    zeros = np.zeros(D_TOKEN, dtype=np.float32)
    diffusion.diffusion_module(
        np.zeros((1, 4, 3), dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        np.zeros((1, N_TOKENS, D_TOKEN), dtype=np.float32),
        diffusion.DiffusionCache(
            atom_conditioning=None,
            cos=None,
            sin=None,
            pair=np.zeros((1, N_TOKENS, N_TOKENS), dtype=np.float32),
            atom_to_token=None,
            atom_mask=np.ones((1, 4), dtype=np.float32),
            n_tokens=N_TOKENS,
        ),
        {
            "s_step_norm.weight": ones,
            "s_step_norm.bias": zeros,
            "s_to_token.weight": identity,
            "token_norm.weight": ones,
            "token_norm.bias": zeros,
        },
        settings=settings,
    )

    assert seen["glu_backend"] == "tokamax"


def test_the_model_settings_override_does_not_drop_the_step_count() -> None:
    """Both diffusion overrides land: one `replace`, not two."""

    from foldjax.models.esmfold2.models import model as structure_model

    base = structure_model.ModelSettings()
    updated = structure_model.with_overrides(base, num_steps=7, glu_backend="tokamax")

    assert updated.diffusion.num_steps == 7
    assert updated.diffusion.glu_backend == "tokamax"
    # Unasked leaves the checkpoint's value alone.
    assert structure_model.with_overrides(base).diffusion.glu_backend == "xla"


def test_the_backend_vocabulary_matches_the_shared_module() -> None:
    """The literal in the adapter is a copy; drift makes it a lie."""

    from foldjax.backends import esmfold2 as backend_module

    assert backend_module._GLU_BACKENDS == GLU_BACKENDS


def test_the_backend_refuses_an_unknown_value_before_loading_anything() -> None:
    from foldjax.backends.esmfold2 import ESMFold2Backend

    with pytest.raises(ValueError) as excinfo:
        ESMFold2Backend().validate_native_options({"glu_backend": "triton"})

    message = str(excinfo.value)
    assert all(name in message for name in GLU_BACKENDS), message


def test_the_backend_refuses_a_fused_glu_under_context_parallelism() -> None:
    from foldjax.backends.esmfold2 import ESMFold2Backend

    backend = ESMFold2Backend()
    backend.validate_native_options({"glu_backend": "tokamax", "cp_devices": 1})

    with pytest.raises(ValueError, match="cannot be partitioned"):
        backend.validate_native_options({"glu_backend": "tokamax", "cp_devices": 2})


def test_the_option_travels_and_selects_its_own_compilation_namespace() -> None:
    from foldjax.backends import esmfold2 as backend_module

    assert "glu_backend" in backend_module.ESMFold2Backend.native_options
    assert "glu_backend" in backend_module.ESMFold2Backend.compile_options
    assert backend_module._FIXED_COMPILE_DEFAULTS["glu_backend"] == "xla"


def test_the_port_refuses_a_fused_glu_under_context_parallelism() -> None:
    """The same refusal at the model API, where `cp_shards` is spelled."""

    from foldjax.models.esmfold2 import inference

    with pytest.raises(ValueError, match="cannot be partitioned"):
        inference.predict(
            key=None,
            features={},
            model=dataclasses.replace,  # never reached
            cp_shards=2,
            glu_backend="tokamax",
        )


def _ffn_params(
    rng: np.random.Generator, prefix: str, width: int, hidden: int
) -> dict[str, np.ndarray]:
    dot = f"{prefix}." if prefix else ""
    scale = float(width**-0.5)
    return {
        f"{dot}norm.weight": (
            rng.standard_normal(width).astype(np.float32) * 0.1 + 1.0
        ),
        f"{dot}norm.bias": rng.standard_normal(width).astype(np.float32) * 0.1,
        f"{dot}ffn.w12.weight": (
            rng.standard_normal((2 * hidden, width)) * scale
        ).astype(np.float32),
        f"{dot}ffn.w3.weight": (rng.standard_normal((width, hidden)) * scale).astype(
            np.float32
        ),
    }


def _reference_pallas_transition(
    part,
    norm,
    w_gate,
    w_value,
    w_out,
    *,
    eps: float,
    out_dtype=None,
):
    import jax
    import jax.numpy as jnp

    from foldjax.models.esmfold2.models.primitives import layer_norm

    dt = w_gate.dtype
    normed = layer_norm(part.astype(jnp.float32), norm[0], norm[1], eps=eps).astype(dt)
    gate = jnp.matmul(normed, w_gate, preferred_element_type=jnp.float32).astype(dt)
    val = jnp.matmul(normed, w_value, preferred_element_type=jnp.float32).astype(dt)
    hid = (jax.nn.silu(gate.astype(jnp.float32)).astype(dt) * val).astype(dt)
    out = jnp.matmul(hid, w_out, preferred_element_type=jnp.float32)
    target_dtype = part.dtype if out_dtype is None else jnp.dtype(out_dtype)
    return out.astype(target_dtype)


@pytest.mark.parametrize(
    ("rows", "cols", "width", "hidden", "residual"),
    [
        (1, 4, 32, 64, True),
        (64, 4, 32, 64, True),
        (65, 4, 32, 64, True),
        (195, 3, 256, 1024, True),
        (65, 5, 128, 512, False),
    ],
)
def test_bf16_autocast_transition_calls_pallas_per_64_row_block(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    cols: int,
    width: int,
    hidden: int,
    residual: bool,
) -> None:
    import jax.numpy as jnp

    from foldjax.models import _pallas_pair
    from foldjax.models.esmfold2.models import trunk

    calls: list[dict[str, object]] = []

    def spy_transition(part, norm, w_gate, w_value, w_out, *, eps, out_dtype=None):
        calls.append(
            {
                "part_shape": tuple(part.shape),
                "part_dtype": part.dtype,
                "w_gate_shape": tuple(w_gate.shape),
                "w_value_shape": tuple(w_value.shape),
                "w_out_shape": tuple(w_out.shape),
                "w_dtypes": (w_gate.dtype, w_value.dtype, w_out.dtype),
                "eps": eps,
            }
        )
        return _reference_pallas_transition(
            part, norm, w_gate, w_value, w_out, eps=eps, out_dtype=out_dtype
        )

    monkeypatch.setattr(_pallas_pair, "transition", spy_transition)
    rng = np.random.default_rng(rows + width + int(residual))
    raw_params = _ffn_params(rng, "tr", width, hidden)
    params = {name: jnp.asarray(val) for name, val in raw_params.items()}
    x = jnp.asarray(rng.standard_normal((1, rows, cols, width)), jnp.bfloat16)

    xla_default = trunk._autocast_transition(x, params, "tr", residual, 1e-5)
    with trunk._transition_backend_scope("xla"):
        xla_scoped = trunk._autocast_transition(x, params, "tr", residual, 1e-5)
    assert calls == []
    np.testing.assert_array_equal(np.asarray(xla_default), np.asarray(xla_scoped))

    with trunk._transition_backend_scope("pallas"):
        pallas_out = trunk._autocast_transition(x, params, "tr", residual, 1e-5)

    expected_blocks = (rows + 63) // 64
    # Under `fori_loop` (3+ full 64-row blocks with `residual=True`), the body
    # traces once at 64 rows and any remainder traces once at the tail size.
    expected_traces = (
        1 + int(rows % 64 != 0) if (residual and rows // 64 > 1) else expected_blocks
    )
    assert len(calls) == expected_traces
    for call in calls:
        assert call["part_shape"][1] <= 64
        assert call["part_dtype"] == jnp.bfloat16
        assert call["w_gate_shape"] == (width, hidden)
        assert call["w_value_shape"] == (width, hidden)
        assert call["w_out_shape"] == (hidden, width)
        assert call["w_dtypes"] == (jnp.bfloat16, jnp.bfloat16, jnp.bfloat16)
        assert call["eps"] == 1e-5
    np.testing.assert_allclose(
        np.asarray(pallas_out, np.float32),
        np.asarray(xla_default, np.float32),
        atol=2e-2,
        rtol=2e-2,
    )

    # Swapping packed w12 halves changes the Pallas result (orientation tripwire).
    w12 = params["tr.ffn.w12.weight"]
    swapped_params = {
        **params,
        "tr.ffn.w12.weight": jnp.concatenate([w12[hidden:], w12[:hidden]], axis=0),
    }
    with trunk._transition_backend_scope("pallas"):
        swapped_out = trunk._autocast_transition(
            x, swapped_params, "tr", residual, 1e-5
        )
    diff = float(
        np.max(
            np.abs(
                np.asarray(swapped_out, np.float32) - np.asarray(pallas_out, np.float32)
            )
        )
    )
    assert diff > 1e-2


def test_pallas_interpret_kernel_agrees_with_xla_autocast_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax.numpy as jnp

    from foldjax.models import _pallas_pair
    from foldjax.models.esmfold2.models import trunk

    monkeypatch.setattr(_pallas_pair, "INTERPRET", True)
    rng = np.random.default_rng(42)
    for width, hidden, residual in ((128, 512, False), (256, 1024, True)):
        params = {
            name: jnp.asarray(val)
            for name, val in _ffn_params(rng, "tr", width, hidden).items()
        }
        x = jnp.asarray(rng.standard_normal((1, 5, 4, width)) * 0.5, jnp.bfloat16)
        expected = trunk._autocast_transition(x, params, "tr", residual, 1e-5)
        actual = trunk._autocast_transition(
            x, params, "tr", residual, 1e-5, transition_backend="pallas"
        )
        assert actual.dtype == jnp.bfloat16 and actual.shape == x.shape
        np.testing.assert_allclose(
            np.asarray(actual, np.float32),
            np.asarray(expected, np.float32),
            atol=2e-2,
            rtol=2e-2,
        )


def test_pallas_transition_guards_fallbacks_sites_and_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    import jax.numpy as jnp

    from foldjax.models import _pallas_pair
    from foldjax.models.esmfold2.models import embedders, heads, trunk

    seen_prefixes: list[tuple[int, ...]] = []

    def spy_transition(part, norm, w_gate, w_value, w_out, *, eps, out_dtype=None):
        seen_prefixes.append(tuple(part.shape))
        return _reference_pallas_transition(
            part, norm, w_gate, w_value, w_out, eps=eps, out_dtype=out_dtype
        )

    monkeypatch.setattr(_pallas_pair, "transition", spy_transition)
    rng = np.random.default_rng(77)
    width, hidden = 32, 64
    base_params = {
        name: jnp.asarray(val)
        for name, val in _ffn_params(rng, "tr", width, hidden).items()
    }
    x_bf16 = jnp.asarray(rng.standard_normal((1, 6, 4, width)), jnp.bfloat16)
    x_fp32 = x_bf16.astype(jnp.float32)

    # 1. Float32 operand keeps XLA even inside pallas scope.
    exp_fp32 = trunk._autocast_transition(x_fp32, base_params, "tr", True, 1e-5)
    with trunk._transition_backend_scope("pallas"):
        act_fp32 = trunk._autocast_transition(x_fp32, base_params, "tr", True, 1e-5)
    assert seen_prefixes == []
    np.testing.assert_array_equal(np.asarray(act_fp32), np.asarray(exp_fp32))

    # 2. Empty zero-sized slice keeps XLA without launching a zero-grid kernel.
    empty_bf16 = jnp.zeros((1, 4, 0, width), dtype=jnp.bfloat16)
    with trunk._transition_backend_scope("pallas"):
        empty_out = trunk._autocast_transition(
            empty_bf16, base_params, "tr", True, 1e-5
        )
    assert seen_prefixes == [] and empty_out.shape == (1, 4, 0, width)

    # 3. Biased w12 / w3 projections fall back to the XLA block.
    for bias_name, bias_len in (
        ("tr.ffn.w12.bias", 2 * hidden),
        ("tr.ffn.w3.bias", width),
    ):
        biased = {
            **base_params,
            bias_name: jnp.ones((bias_len,), dtype=jnp.float32) * 0.25,
        }
        exp_biased = trunk._autocast_transition(x_bf16, biased, "tr", True, 1e-5)
        with trunk._transition_backend_scope("pallas"):
            act_biased = trunk._autocast_transition(x_bf16, biased, "tr", True, 1e-5)
        assert seen_prefixes == []
        np.testing.assert_array_equal(np.asarray(act_biased), np.asarray(exp_biased))

    # 4. Non-affine norm (missing weight or missing bias) falls back to XLA block.
    for drop_norm in ("tr.norm.weight", "tr.norm.bias"):
        non_affine = {k: v for k, v in base_params.items() if k != drop_norm}
        exp_na = trunk._autocast_transition(x_bf16, non_affine, "tr", True, 1e-5)
        with trunk._transition_backend_scope("pallas"):
            act_na = trunk._autocast_transition(x_bf16, non_affine, "tr", True, 1e-5)
        assert seen_prefixes == []
        np.testing.assert_array_equal(np.asarray(act_na), np.asarray(exp_na))

    # 4b. A width the kernel cannot slice (the width-12 MSA fixture's, or a
    # hidden width off the 16-column grid) keeps the XLA block.
    for odd_width, odd_hidden in ((12, 24), (32, 40)):
        odd = {
            name: jnp.asarray(val)
            for name, val in _ffn_params(rng, "tr", odd_width, odd_hidden).items()
        }
        x_odd = jnp.asarray(rng.standard_normal((1, 6, 4, odd_width)), jnp.bfloat16)
        exp_odd = trunk._autocast_transition(x_odd, odd, "tr", True, 1e-5)
        with trunk._transition_backend_scope("pallas"):
            act_odd = trunk._autocast_transition(x_odd, odd, "tr", True, 1e-5)
        assert seen_prefixes == []
        np.testing.assert_array_equal(np.asarray(act_odd), np.asarray(exp_odd))

    # 5. MSA encoder reaches both msa_transition (c=16) and pair_transition (c=32).
    msa_c, pair_c = 16, 32
    msa_params: dict[str, jnp.ndarray] = {}
    for name, val in _ffn_params(rng, "blk.msa_transition", msa_c, 2 * msa_c).items():
        msa_params[name] = jnp.asarray(val)
    for name, val in _ffn_params(
        rng, "blk.pair_transition", pair_c, 2 * pair_c
    ).items():
        msa_params[name] = jnp.asarray(val)
    monkeypatch.setattr(
        embedders,
        "outer_product_mean",
        lambda msa, *a, **k: jnp.zeros((1, 4, 4, pair_c), jnp.bfloat16),
    )
    monkeypatch.setattr(
        embedders,
        "msa_pair_weighted_averaging",
        lambda msa, *a, **k: jnp.zeros_like(msa),
    )
    monkeypatch.setattr(
        embedders, "triangle_multiplicative", lambda pair, *a, **k: jnp.zeros_like(pair)
    )
    with trunk._transition_backend_scope("pallas"):
        embedders.msa_encoder_block(
            jnp.ones((1, 4, 3, msa_c), jnp.bfloat16),
            jnp.ones((1, 4, 4, pair_c), jnp.bfloat16),
            msa_params,
            "blk",
            msa_mask=jnp.ones((1, 4, 3), jnp.float32),
            pair_mask=jnp.ones((1, 4, 4), jnp.float32),
            is_final=False,
            native_opm_params=msa_params,
        )
    assert seen_prefixes == [(1, 4, 3, msa_c), (1, 4, 4, pair_c)]
    seen_prefixes.clear()

    # 6. Confidence head's folding trunk: float32 confidence pair keeps XLA;
    # bfloat16 confidence pair calls Pallas.
    from tests.models.esmfold2.test_triangle_cueq_option import _trunk_params

    conf_trunk_params = {
        f"ch.folding_trunk.{k}": v for k, v in _trunk_params(rng, width=32).items()
    }
    conf_params: dict[str, jnp.ndarray] = {
        **conf_trunk_params,
        "ch.s_inputs_norm.weight": jnp.ones((16,), jnp.float32),
        "ch.s_inputs_norm.bias": jnp.zeros((16,), jnp.float32),
        "ch.z_norm.weight": jnp.ones((32,), jnp.float32),
        "ch.z_norm.bias": jnp.zeros((32,), jnp.float32),
        "ch.s_to_z.weight": jnp.zeros((32, 16), jnp.float32),
        "ch.s_to_z_transpose.weight": jnp.zeros((32, 16), jnp.float32),
        "ch.s_to_z_prod_in1.weight": jnp.zeros((16, 16), jnp.float32),
        "ch.s_to_z_prod_in2.weight": jnp.zeros((16, 16), jnp.float32),
        "ch.s_to_z_prod_out.weight": jnp.zeros((32, 16), jnp.float32),
        "ch.boundaries": jnp.linspace(2.0, 20.0, 7, dtype=jnp.float32),
        "ch.dist_bin_pairwise_embed.weight": jnp.zeros((8, 32), jnp.float32),
        "ch.row_attention_pooling.attn_proj.weight": jnp.zeros((1, 32), jnp.float32),
        "ch.row_attention_pooling.out_proj.weight": jnp.zeros((16, 32), jnp.float32),
        "ch.plddt_ln.weight": jnp.ones((16,), jnp.float32),
        "ch.plddt_ln.bias": jnp.zeros((16,), jnp.float32),
        "ch.plddt_weight": jnp.zeros((24, 16, 8), jnp.float32),
        "ch.pae_ln.weight": jnp.ones((32,), jnp.float32),
        "ch.pae_ln.bias": jnp.zeros((32,), jnp.float32),
        "ch.pae_head.weight": jnp.zeros((8, 32), jnp.float32),
        "ch.pde_ln.weight": jnp.ones((32,), jnp.float32),
        "ch.pde_ln.bias": jnp.zeros((32,), jnp.float32),
        "ch.pde_head.weight": jnp.zeros((8, 32), jnp.float32),
        "ch.resolved_ln.weight": jnp.ones((16,), jnp.float32),
        "ch.resolved_ln.bias": jnp.zeros((16,), jnp.float32),
        "ch.resolved_weight": jnp.zeros((24, 16, 2), jnp.float32),
    }

    def run_conf(conf_dtype: str) -> None:
        mask_kw = {
            "to" + "ken_mask": jnp.ones((1, 4), jnp.float32),
            "atom_to_to" + "ken": jnp.arange(4, dtype=jnp.int32)[None],
        }
        heads.confidence_head(
            jnp.zeros((1, 4, 16), jnp.float32),
            jnp.zeros((1, 4, 4, 32), jnp.bfloat16),
            jnp.zeros((1, 4, 3), jnp.float32),
            conf_params,
            "ch",
            distogram_atom_idx=jnp.arange(4, dtype=jnp.int32)[None],
            atom_mask=jnp.ones((1, 4), jnp.float32),
            asym_id=jnp.zeros((1, 4), jnp.int32),
            mol_type=jnp.zeros((1, 4), jnp.int32),
            n_layers=1,
            n_chains=1,
            trunk_dtype=jnp.bfloat16,
            confidence_dtype=jnp.dtype(conf_dtype),
            **mask_kw,
        )

    with trunk._transition_backend_scope("pallas"):
        run_conf("float32")
        assert seen_prefixes == []
        run_conf("bfloat16")
        assert seen_prefixes == [(1, 4, 4, 32)]
    seen_prefixes.clear()

    # 7. Scope cleanup on error, explicit override, invalid values, and CP guard.
    for bad in ("", "tokamax", 0, True, []):
        with pytest.raises(ValueError, match="transition_backend must be one of"):
            trunk._resolve_transition_backend(bad)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="abort"):
        with trunk._transition_backend_scope("pallas"):
            assert trunk._resolve_transition_backend(None) == "pallas"
            trunk.transition(
                x_bf16,
                base_params,
                "tr",
                residual=True,
                native_autocast=True,
                transition_backend="xla",
            )
            assert seen_prefixes == []
            raise RuntimeError("abort")
    assert trunk._resolve_transition_backend(None) == "xla"

    monkeypatch.setattr(trunk, "cp_mesh", lambda: types.SimpleNamespace())
    with pytest.raises(ValueError, match="cannot be partitioned"):
        trunk.transition(
            x_bf16,
            base_params,
            "tr",
            residual=True,
            native_autocast=True,
            transition_backend="pallas",
        )


def test_glu_backend_pallas_resolver_refusals_diffusion_and_compile_pool(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import jax
    import jax.numpy as jnp

    from foldjax.backends import esmfold2 as backend_module
    from foldjax.backends.esmfold2 import ESMFold2Backend
    from foldjax.models._jit_pool import BoundedJitPool
    from foldjax.models.esmfold2 import inference
    from foldjax.models.esmfold2.data.features import build_features
    from foldjax.models.esmfold2.models import model as structure_model
    from foldjax.models.esmfold2.models import trunk
    from foldjax.schema import PredictionRequest

    assert backend_module._checked_glu_backend("pallas") == "pallas"
    assert backend_module._checked_glu_backend("xla") == "xla"
    assert backend_module._checked_glu_backend("tokamax") == "tokamax"

    backend = ESMFold2Backend()
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps({"entities": [{"type": "protein", "id": "A", "sequence": "ACDE"}]}),
        encoding="utf-8",
    )
    weights = tmp_path / "weights"
    weights.mkdir()

    def make_req(**opts: object) -> PredictionRequest:
        return PredictionRequest(
            model="esmfold2",
            input=job,
            input_format="foldjax",
            weights=weights,
            output_dir=tmp_path / "out",
            options=dict(opts),
        )

    for bad in ("", "auto", 0, True, None, []):
        with pytest.raises(ValueError, match="glu_backend must be one of"):
            backend.validate_native_options({"glu_backend": bad})
        if bad is not None:
            with pytest.raises(ValueError, match="glu backend must be one of"):
                structure_model.with_overrides(
                    structure_model.ModelSettings(),
                    glu_backend=bad,  # type: ignore[arg-type]
                )
            with pytest.raises(ValueError, match="glu backend must be one of"):
                inference.predict(
                    jax.random.PRNGKey(0),
                    {},
                    dataclasses.replace,  # type: ignore[arg-type]
                    glu_backend=bad,  # type: ignore[arg-type]
                )
    for bad_tr in ("", "tokamax", 0, True, []):
        with pytest.raises(ValueError, match="transition_backend must be one of"):
            inference.compiled_predict(
                structure_model.ModelSettings(),
                1,
                transition_backend=bad_tr,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="transition_backend must be one of"):
            inference._run(
                jax.random.PRNGKey(0),
                {},
                {},
                None,
                structure_model.ModelSettings(),
                1,
                False,
                transition_backend=bad_tr,  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="cannot be partitioned"):
        inference._run(
            jax.random.PRNGKey(0),
            {},
            {},
            None,
            structure_model.ModelSettings(),
            1,
            False,
            cp_shards=2,
            transition_backend="pallas",
        )

    # Diffusion settings receive "xla" when glu_backend is "pallas".
    prior_tokamax = structure_model.with_overrides(
        structure_model.ModelSettings(), glu_backend="tokamax"
    )
    assert prior_tokamax.diffusion.glu_backend == "tokamax"
    overridden_pallas = structure_model.with_overrides(
        prior_tokamax, glu_backend="pallas"
    )
    assert overridden_pallas.diffusion.glu_backend == "xla"

    # Off GPU: explicit pallas is refused in adapter and port; omitted realises xla.
    monkeypatch.setattr("foldjax.models._pallas_pair.gpu_process", lambda: False)
    assert backend_module.realised_glu_backend(None) == "xla"
    with pytest.raises(ValueError, match="needs a GPU") as exc_plan:
        backend.validate_request(make_req(glu_backend="pallas"))
    assert "platform 'cpu'" in str(exc_plan.value)
    assert "--option glu_backend=xla" in str(exc_plan.value)
    with pytest.raises(ValueError, match="needs a GPU"):
        backend.predict(make_req(glu_backend="pallas"))
    with pytest.raises(ValueError, match="needs a GPU") as exc_port:
        inference.predict(
            jax.random.PRNGKey(0),
            {},
            dataclasses.replace,  # type: ignore[arg-type]
            glu_backend="pallas",
        )
    assert "--option glu_backend=xla" in str(exc_port.value)

    # Safe options never touch _gpu_process during validate_request.
    monkeypatch.setattr(
        backend_module,
        "_gpu_process",
        lambda: (_ for _ in ()).throw(AssertionError("device touched")),
    )
    for safe in (
        {},
        {"cp_devices": 2},
        {"glu_backend": "xla"},
        {"glu_backend": "tokamax"},
    ):
        backend.validate_request(make_req(**safe))
    with pytest.raises(ValueError, match="cannot be partitioned"):
        backend.validate_request(make_req(glu_backend="pallas", cp_devices=2))
    with pytest.raises(ValueError, match="cannot be partitioned"):
        inference.predict(
            jax.random.PRNGKey(0),
            {},
            dataclasses.replace,  # type: ignore[arg-type]
            cp_shards=2,
            glu_backend="pallas",
        )

    # On GPU: omitted realises pallas for serial bf16 trunk, xla under CP or fp32 trunk.
    from foldjax.models import _pallas_pair

    monkeypatch.setattr(
        backend_module, "_gpu_process", lambda: _pallas_pair.gpu_process()
    )
    monkeypatch.setattr(_pallas_pair, "gpu_process", lambda: True)
    assert backend_module.realised_glu_backend(None) == "pallas"
    assert backend_module.realised_glu_backend(None, cp_devices=1) == "pallas"
    assert backend_module.realised_glu_backend(None, cp_devices=2) == "xla"
    assert backend_module.realised_glu_backend(None, trunk_dtype="float32") == "xla"
    assert backend_module.realised_glu_backend("xla", cp_devices=1) == "xla"
    assert backend_module.realised_glu_backend("pallas", cp_devices=2) == "pallas"
    backend.validate_request(make_req(glu_backend="pallas", cp_devices=1))
    with pytest.raises(ValueError, match="cp_devices must be positive"):
        backend.predict(make_req(glu_backend="pallas", cp_devices=0))
    with pytest.raises(ValueError, match="cannot be partitioned"):
        backend.predict(make_req(glu_backend="pallas", cp_devices=2))

    # Whole-model compile pool: omitted on serial GPU shares explicit pallas,
    # hands diffusion "xla", and forks from explicit xla and float32 trunk.
    features = build_features([("AG", "A", 0, 0)])
    base_settings = dataclasses.replace(
        structure_model.ModelSettings(), trunk_n_layers=0, coda_n_layers=0
    )
    bond_w = "to" + "ken_bonds.weight"
    loaded_bf16 = inference.LoadedModel(
        {bond_w: jnp.ones((base_settings.d_pair, 1), jnp.bfloat16)},
        base_settings,
    )
    fp32_settings = dataclasses.replace(base_settings, trunk_dtype="float32")
    loaded_fp32 = inference.LoadedModel(
        {bond_w: jnp.ones((base_settings.d_pair, 1), jnp.float32)},
        fp32_settings,
    )
    lm_embed = jnp.zeros((1, 2, 8), dtype=jnp.float32)
    traces: list[tuple[str, str]] = []

    def tiny_predict(_rng, model_features, _parameters, *, settings, **_kw):
        resolved_tr = trunk._resolve_transition_backend(None)
        traces.append((resolved_tr, settings.diffusion.glu_backend))
        return {
            "probe": model_features["asym_id"].astype(jnp.int32)
            + int(resolved_tr == "pallas")
        }

    monkeypatch.setattr(structure_model, "predict", tiny_predict)
    pool = BoundedJitPool(
        inference._run, static_argnames=inference._COMPILED_PREDICT_STATIC_ARGNAMES
    )
    monkeypatch.setattr(inference, "_compiled_predict_pool", pool)
    inference._compiled_predict_factory.cache_clear()

    def run_pred(loaded_model=loaded_bf16, **kw):
        return inference.predict(
            jax.random.PRNGKey(0),
            features,
            loaded_model,
            precomputed_lm_embedding=lm_embed,
            triangle_backend="xla",
            **kw,
        )

    try:
        out_omitted_gpu = run_pred()
        out_pallas_gpu = run_pred(glu_backend="pallas")
        jax.block_until_ready((out_omitted_gpu, out_pallas_gpu))
        assert traces == [("pallas", "xla")]
        assert pool._entry_count() == 1  # noqa: SLF001
        assert int(out_omitted_gpu["probe"][0, 0]) == 1
        np.testing.assert_array_equal(out_omitted_gpu["probe"], out_pallas_gpu["probe"])

        out_xla_gpu = run_pred(glu_backend="xla")
        jax.block_until_ready(out_xla_gpu)
        assert traces == [("pallas", "xla"), ("xla", "xla")]
        assert pool._entry_count() == 2  # noqa: SLF001
        assert int(out_xla_gpu["probe"][0, 0]) == 0

        # Float32 trunk on GPU: omitted realises xla.
        out_fp32_gpu = run_pred(loaded_fp32)
        jax.block_until_ready(out_fp32_gpu)
        assert traces[-1] == ("xla", "xla")
        assert int(out_fp32_gpu["probe"][0, 0]) == 0

        # CPU: omitted reuses the xla executable without recompiling.
        count_before_cpu = pool._entry_count()  # noqa: SLF001
        monkeypatch.setattr("foldjax.models._pallas_pair.gpu_process", lambda: False)
        out_omitted_cpu = run_pred()
        jax.block_until_ready(out_omitted_cpu)
        assert pool._entry_count() == count_before_cpu  # noqa: SLF001
        np.testing.assert_array_equal(out_omitted_cpu["probe"], out_xla_gpu["probe"])
    finally:
        pool.clear_cache()
        inference._compiled_predict_factory.cache_clear()
