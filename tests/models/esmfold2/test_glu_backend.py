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
