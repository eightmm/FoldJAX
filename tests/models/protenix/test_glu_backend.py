"""The opt-in fused gated linear unit on Protenix's two gated transitions.

Both sites compute ``silu(x @ Wa) * (x @ Wb)``. Spelled as two matmuls and a
product, XLA writes both widened branches before multiplying them; the
``tokamax`` backend runs the same arithmetic in one Triton kernel and writes
neither. That kernel does not exist on CPU, so every test here that needs the
fused route to *run* substitutes a plain-JAX stand-in for
``foldjax.models._glu._fused`` -- the one seam between the dispatch this port
owns and the kernel it does not. What is under test is the dispatch: that the
value reaches both call sites, that it changes the compiled program, that the
released default does not take the branch, and that the two refusals fire.

Nothing here is a measurement. Neither Protenix nor OpenDDE has a GLU number
on a GPU; see `docs/cli.md`.
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models import _glu
from foldjax.models.protenix.models.diffusion.transformer import (
    ConditionedTransitionParams,
    _compiled_conditioned_transition_block,
    conditioned_transition_block,
)
from foldjax.models.protenix.models.primitives import primitives
from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    AutocastLinearParams,
    Fp32PrecisionLinearParams,
    LayerNormParams,
    LinearParams,
    TransitionParams,
    compiled_transition,
    transition,
)


def _rng(seed: int, *shape: int) -> jnp.ndarray:
    return jnp.asarray(
        np.random.default_rng(seed).standard_normal(shape) * 0.5, dtype=jnp.float32
    )


def _transition_params(channels: int = 8, hidden: int = 16) -> TransitionParams:
    """PyTorch layout throughout: every kernel is ``[out, in]``, bias-free."""
    return TransitionParams(
        layer_norm=LayerNormParams(
            weight=_rng(1, channels) + 1.0, bias=_rng(2, channels)
        ),
        linear_a=LinearParams(weight=_rng(3, hidden, channels)),
        linear_b=LinearParams(weight=_rng(4, hidden, channels)),
        linear_out=LinearParams(weight=_rng(5, channels, hidden)),
    )


def _conditioned_params(
    channels: int = 8, hidden: int = 16
) -> ConditionedTransitionParams:
    return ConditionedTransitionParams(
        adaln=AdaptiveLayerNormParams(
            layernorm_a=LayerNormParams(),
            layernorm_s=LayerNormParams(
                weight=_rng(6, channels) + 1.0, bias=_rng(7, channels)
            ),
            linear_s=LinearParams(weight=_rng(8, channels, channels)),
            linear_no_bias_s=LinearParams(weight=_rng(9, channels, channels)),
        ),
        linear_a1=LinearParams(weight=_rng(10, hidden, channels)),
        linear_a2=LinearParams(weight=_rng(11, hidden, channels)),
        linear_b=LinearParams(weight=_rng(12, channels, hidden)),
        linear_s=LinearParams(
            weight=_rng(13, channels, channels), bias=_rng(14, channels)
        ),
    )


def _stand_in(scale: float):
    """A plain-JAX ``_fused`` that is visible in the output it produces.

    ``scale=2.0`` is the tripwire: a recorder flag only proves the patched
    function was *traced*, while a doubled result proves its value is what the
    call site returned. ``scale=1.0`` is the same arithmetic the kernel claims
    to perform, for comparing the two routes.
    """

    calls: list[tuple[int, ...]] = []

    def fused(x, weights, activation):
        calls.append(tuple(weights.shape))
        return scale * activation(x @ weights[..., 0, :]) * (x @ weights[..., 1, :])

    return fused, calls


@pytest.fixture(autouse=True)
def _forget_traces():
    """Each test traces its own programs.

    The two entry points under test are module-level `jax.jit` objects, so a
    trace captured while another test's stand-in was installed would be
    replayed here with that stand-in baked in -- and a patch that never fires
    is exactly what these tests exist to catch.
    """
    jax.clear_caches()
    yield
    jax.clear_caches()


def test_the_fused_route_actually_runs_at_the_chunked_transition(monkeypatch):
    monkeypatch.setattr(primitives, "_WARNED_FUSED_UNCHUNKABLE", False)
    fused, calls = _stand_in(2.0)
    monkeypatch.setattr(_glu, "_fused", fused)
    params = _transition_params()
    x = _rng(20, 6, 8)

    baseline = transition(x, params)
    doubled = transition(x, params, glu_backend="tokamax")

    assert calls, "the fused path never reached foldjax.models._glu._fused"
    # [in, 2, out]: the stacked form the kernel takes, with the gate first.
    assert calls[0] == (8, 2, 16)
    np.testing.assert_allclose(
        np.asarray(doubled), 2.0 * np.asarray(baseline), rtol=2e-6, atol=2e-6
    )


def test_the_fused_route_actually_runs_at_the_conditioned_transition(monkeypatch):
    fused, calls = _stand_in(2.0)
    monkeypatch.setattr(_glu, "_fused", fused)
    params = _conditioned_params()
    a = _rng(21, 5, 8)
    s = _rng(22, 5, 8)

    baseline = conditioned_transition_block(a, s, params)
    doubled = conditioned_transition_block(a, s, params, glu_backend="tokamax")

    assert calls, "the fused path never reached foldjax.models._glu._fused"
    assert calls[0] == (8, 2, 16)
    # Only the gated half is doubled; the output gate and projection are not.
    np.testing.assert_allclose(
        np.asarray(doubled), 2.0 * np.asarray(baseline), rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("chunk_size", [None, 0, 2])
def test_the_two_backends_agree_on_float32(monkeypatch, chunk_size):
    """Same arithmetic, three block sizes, one tolerance.

    Not bitwise, and not only because of the kernel: the blocked XLA route
    already differs from the unblocked one at 1e-7 because XLA tiles the
    smaller GEMM differently, which is the same order as the activation's two
    spellings. Asserting equality here would be asserting that chunking is
    bitwise, which the module docstring says it is not.
    """
    monkeypatch.setattr(primitives, "_WARNED_FUSED_UNCHUNKABLE", False)
    fused, _ = _stand_in(1.0)
    monkeypatch.setattr(_glu, "_fused", fused)
    params = _transition_params()
    x = _rng(23, 7, 8)

    blocked = transition(x, params, chunk_size=chunk_size)
    with pytest.warns(RuntimeWarning) if chunk_size == 2 else nullcontext():
        fused_out = transition(x, params, chunk_size=chunk_size, glu_backend="tokamax")

    np.testing.assert_allclose(
        np.asarray(fused_out), np.asarray(blocked), rtol=1e-5, atol=1e-6
    )


def test_the_two_backends_agree_at_the_conditioned_transition(monkeypatch):
    fused, _ = _stand_in(1.0)
    monkeypatch.setattr(_glu, "_fused", fused)
    params = _conditioned_params()
    a = _rng(24, 5, 8)
    s = _rng(25, 5, 8)

    np.testing.assert_allclose(
        np.asarray(conditioned_transition_block(a, s, params, glu_backend="tokamax")),
        np.asarray(conditioned_transition_block(a, s, params)),
        rtol=1e-5,
        atol=1e-6,
    )


def test_the_released_default_never_takes_the_fused_branch(monkeypatch):
    """The shipped run must be the historical program, not a renamed one."""

    def explode(*args, **kwargs):
        raise AssertionError("the released default reached the fused kernel")

    monkeypatch.setattr(_glu, "_fused", explode)
    params = _transition_params()
    conditioned = _conditioned_params()
    x = _rng(26, 6, 8)
    s = _rng(27, 6, 8)

    implicit = transition(x, params)
    explicit = transition(x, params, glu_backend="xla")
    np.testing.assert_array_equal(np.asarray(implicit), np.asarray(explicit))

    implicit_c = conditioned_transition_block(x, s, conditioned)
    explicit_c = conditioned_transition_block(x, s, conditioned, glu_backend="xla")
    np.testing.assert_array_equal(np.asarray(implicit_c), np.asarray(explicit_c))


def test_a_chunk_size_and_the_fused_kernel_warn_once(monkeypatch):
    """Said once per process, and said only when a caller asked for both.

    The automatic budget inside `_transition_chunk_rows` is this module's own
    choice rather than a request, so it must stay silent; only a `chunk_size`
    the caller passed is a second thing to honour.
    """
    monkeypatch.setattr(primitives, "_WARNED_FUSED_UNCHUNKABLE", False)
    fused, _ = _stand_in(1.0)
    monkeypatch.setattr(_glu, "_fused", fused)
    params = _transition_params()
    x = _rng(28, 6, 8)

    with pytest.warns(RuntimeWarning, match="is not used by the 'tokamax' GLU"):
        transition(x, params, chunk_size=2, glu_backend="tokamax")

    with warnings.catch_warnings(record=True) as later:
        warnings.simplefilter("always")
        transition(x, params, chunk_size=2, glu_backend="tokamax")
        transition(x, params, chunk_size=3, glu_backend="tokamax")
        # No caller-supplied size: nothing to report, whatever the budget does.
        transition(x, params, glu_backend="tokamax")
    assert [w for w in later if issubclass(w.category, RuntimeWarning)] == []


def test_a_chunk_size_the_xla_route_honours_does_not_warn(monkeypatch):
    monkeypatch.setattr(primitives, "_WARNED_FUSED_UNCHUNKABLE", False)
    params = _transition_params()
    x = _rng(29, 6, 8)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        transition(x, params, chunk_size=2)
    assert [w for w in seen if issubclass(w.category, RuntimeWarning)] == []


def test_the_backend_is_part_of_the_compiled_program_identity(monkeypatch):
    """Two backends, two executables -- and the xla one is not reused.

    `compiled_transition` is keyed on a topology tuple, and a second key that
    collided with it would show up exactly here: the cached xla trace would be
    replayed and the doubled stand-in would never appear in the result.
    """
    fused, calls = _stand_in(2.0)
    monkeypatch.setattr(_glu, "_fused", fused)
    params = _transition_params()
    x = _rng(30, 6, 8)

    plain = compiled_transition(x, params)
    again = compiled_transition(x, params, glu_backend="xla")
    doubled = compiled_transition(x, params, glu_backend="tokamax")

    np.testing.assert_array_equal(np.asarray(plain), np.asarray(again))
    assert calls, "the compiled route never reached the fused kernel"
    np.testing.assert_allclose(
        np.asarray(doubled), 2.0 * np.asarray(plain), rtol=2e-6, atol=2e-6
    )

    # The diffusion site has its own `jax.jit`, reached whenever the attention
    # backend ends in `_jit`. A string argument that is not declared static
    # fails at trace rather than quietly, so this is also what proves the
    # declaration is there.
    conditioned = _conditioned_params()
    a = _rng(34, 5, 8)
    s = _rng(35, 5, 8)
    jit_plain = _compiled_conditioned_transition_block(a, s, conditioned)
    jit_doubled = _compiled_conditioned_transition_block(
        a, s, conditioned, glu_backend="tokamax"
    )
    assert not np.allclose(np.asarray(jit_plain), np.asarray(jit_doubled))


def test_an_autocast_projection_is_fusable_and_an_fp32_one_is_refused(monkeypatch):
    """The BF16 stages narrow their operands; the FP32 exemptions do not.

    `input_precision._autocast_linears` rewrites every `LinearParams` in the
    confidence and diffusion subtrees, transitions included, so the fused
    kernel has to accept the autocast node. The two FP32 node types widen the
    operand and round the *result*, which the kernel cannot express -- refused
    by name rather than run at a width nobody asked for.
    """
    fused, calls = _stand_in(1.0)
    monkeypatch.setattr(_glu, "_fused", fused)
    base = _transition_params()
    autocast = base._replace(
        linear_a=AutocastLinearParams(base.linear_a.weight.astype(jnp.bfloat16)),
        linear_b=AutocastLinearParams(base.linear_b.weight.astype(jnp.bfloat16)),
    )
    x = _rng(31, 6, 8)
    assert transition(x, autocast, glu_backend="tokamax").dtype == jnp.float32
    assert calls

    exempt = base._replace(
        linear_a=Fp32PrecisionLinearParams(base.linear_a.weight),
    )
    with pytest.raises(ValueError, match="Fp32PrecisionLinearParams linear_a"):
        transition(x, exempt, glu_backend="tokamax")

    biased = base._replace(
        linear_a=LinearParams(base.linear_a.weight, jnp.zeros((16,), jnp.float32)),
    )
    with pytest.raises(ValueError, match="bias-free linear_a"):
        transition(x, biased, glu_backend="tokamax")


# --- The option's public surface: the wrapper, the adapter, and the parser ---


def test_the_wrapper_refuses_an_unknown_value_and_names_the_allowed_ones():
    from foldjax.models.protenix.models.predict import protenix_predict_static

    with pytest.raises(ValueError, match=r"must be one of \('xla', 'tokamax'\)"):
        protenix_predict_static(None, {}, None, glu_backend="triton")


def test_the_wrapper_refuses_a_fused_glu_under_context_parallelism():
    """Before any device is touched: the operands are never built.

    The kernel is one Triton call over the whole operand, which the SPMD
    partitioner cannot split -- and the widened intermediate it removes is
    already divided across devices, so there is nothing to gain by trying.
    """
    from foldjax.models.protenix.models.predict import protenix_predict_static

    with pytest.raises(ValueError, match="cannot be partitioned"):
        protenix_predict_static(None, {}, None, cp_shards=2, glu_backend="tokamax")

    # The same request with the released value gets past this check and fails
    # later, on the parameters it was not given -- so the refusal above is
    # about the kernel and not about the empty call.
    with pytest.raises(Exception) as fallthrough:
        protenix_predict_static(None, {}, None, cp_shards=2, glu_backend="xla")
    assert "cannot be partitioned" not in str(fallthrough.value)


def test_the_adapter_ships_the_split_matmul_and_knows_the_shared_value_list():
    import foldjax.backends.protenix as backend_impl

    assert backend_impl._RELEASED_COMPILE_DEFAULTS["glu_backend"] == "xla"
    assert "glu_backend" in backend_impl.ProtenixBackend.compile_options
    # The adapter cannot import the shared module -- reading it means importing
    # JAX, and cache planning runs before any model runtime is loaded -- so it
    # keeps a copy. This is what stops the copy from drifting.
    assert backend_impl._GLU_BACKENDS == _glu.GLU_BACKENDS


def test_the_adapter_refuses_an_unknown_value_and_the_cp_combination(tmp_path):
    from foldjax.backends.protenix import ProtenixBackend

    backend = ProtenixBackend()
    with pytest.raises(ValueError, match=r"must be one of \('xla', 'tokamax'\)"):
        backend.validate_native_options({"glu_backend": "triton"})
    with pytest.raises(ValueError, match="cannot be partitioned"):
        backend.validate_native_options({"glu_backend": "tokamax", "cp_devices": 2})
    backend.validate_native_options({"glu_backend": "tokamax", "cp_devices": 1})
    backend.validate_native_options({"glu_backend": "xla", "cp_devices": 4})


def test_the_parser_carries_the_flag_down_to_the_wrapper(monkeypatch):
    """The CLI value reaches `protenix_predict_static`, not just argparse."""
    from foldjax.models.protenix.cli import predict as predict_cli

    parser = None

    def capture(self, args=None, namespace=None):
        nonlocal parser
        parser = self
        raise SystemExit(0)

    import argparse

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(SystemExit):
        predict_cli.main([])
    defaults = {action.dest: action for action in parser._actions}
    assert defaults["glu_backend"].default == "xla"
    assert tuple(defaults["glu_backend"].choices) == _glu.GLU_BACKENDS


def test_the_adapter_renders_the_flag_into_the_native_command(tmp_path, monkeypatch):
    """The adapter drives the parser by writing argv, so the leg is the string.

    Without this the option could sit in the adapter's option set, pass every
    namespace check, and never be written into the command the native CLI
    actually parses.
    """
    import json
    from types import SimpleNamespace

    from foldjax.backends.protenix import ProtenixBackend
    from foldjax.schema import PredictionRequest

    seen: list[str] = []

    def native_main(argv):
        seen.extend(argv)
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        cif = out / "job_sample_0.cif"
        cif.write_text("data_x\n")
        confidence = out / "job_summary_confidence_sample_0.json"
        confidence.write_text(json.dumps({"ptm": 0.5}))
        return [cif, confidence]

    monkeypatch.setattr(
        "foldjax.backends.protenix.import_module",
        lambda name: SimpleNamespace(main=native_main),
    )
    job = tmp_path / "job.json"
    job.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()
    ProtenixBackend().predict(
        PredictionRequest(
            model="protenix",
            input=job,
            weights=weights,
            output_dir=tmp_path / "out",
            seed=5,
            cache_dir=tmp_path / "cache",
            options={"glu_backend": "tokamax"},
        )
    )
    assert seen[seen.index("--glu-backend") + 1] == "tokamax"


def test_the_call_sites_refuse_the_fused_kernel_under_a_mesh(monkeypatch):
    """The wrapper is the entry point, not the only door.

    A library caller can open a mesh and call a transition directly, so the
    refusal is repeated where the kernel would be dispatched. Asserted by
    faking an active mesh rather than by requiring two devices.
    """
    fused, calls = _stand_in(1.0)
    monkeypatch.setattr(_glu, "_fused", fused)
    # Both sites call one refusal, which reads the mesh out of this module --
    # so patching it here covers the diffusion site too, without a second
    # patch that could hide the site having its own copy.
    monkeypatch.setattr(primitives, "cp_mesh", lambda: object())
    params = _transition_params()
    conditioned = _conditioned_params()
    x = _rng(32, 6, 8)
    s = _rng(33, 6, 8)

    with pytest.raises(ValueError, match="cannot be partitioned"):
        transition(x, params, glu_backend="tokamax")
    with pytest.raises(ValueError, match="cannot be partitioned"):
        conditioned_transition_block(x, s, conditioned, glu_backend="tokamax")
    assert not calls

    # The released value is unaffected: context parallelism is a supported
    # route for it, and the pair transition is what runs there today.
    assert transition(x, params).shape == (6, 8)
