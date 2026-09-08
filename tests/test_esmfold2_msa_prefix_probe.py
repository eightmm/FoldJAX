import numpy as np
import pytest

from bench.esmfold2_msa_prefix_probe import msa_inputs


def test_embedding_barriers_are_scoped_and_restore(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import jax

    from bench.esmfold2_msa_prefix_probe import embedding_barrier_control

    original = Mock(return_value="output")
    barrier = Mock(return_value="barrier")
    monkeypatch.setattr(jax.lax, "optimization_barrier", barrier)
    module = SimpleNamespace(linear=original)
    with pytest.raises(RuntimeError, match="stop"):
        with embedding_barrier_control(module):
            for key in ("msa_encoder.embed", "msa_encoder.project_inputs"):
                assert module.linear(None, {}, key) == "barrier"
            assert module.linear(None, {}, "other") == "output"
            raise RuntimeError("stop")
    assert module.linear is original
    assert barrier.call_count == 2


def test_hlo_capture_executes_the_saved_executable(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from bench.esmfold2_msa_prefix_probe import _sha256, execute_capture

    executable = Mock(return_value="result")
    executable.as_text.return_value = "optimized"
    lowered = SimpleNamespace(
        compile=Mock(return_value=executable), as_text=lambda: "lowered"
    )
    run = Mock()
    run.lower.return_value = lowered
    result, bindings = execute_capture(run, (1, 2), tmp_path)
    assert result == "result"
    run.assert_not_called()
    run.lower.assert_called_once_with(1, 2)
    executable.assert_called_once_with(1, 2)
    for name, text in (
        ("lowered.hlo.txt", "lowered"),
        ("compiled.hlo.txt", "optimized"),
    ):
        path = tmp_path / name
        assert path.read_text() == text
        assert bindings[str(path.resolve())] == _sha256(path)


def test_no_hlo_capture_preserves_direct_call():
    from unittest.mock import Mock

    from bench.esmfold2_msa_prefix_probe import execute_capture

    run = Mock(return_value="direct")
    assert execute_capture(run, (3,)) == ("direct", {})
    run.assert_called_once_with(3)
    run.lower.assert_not_called()


@pytest.mark.parametrize("first_only", [False, True])
@pytest.mark.parametrize("capture_inputs", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_stack_propagates_outputs_and_restores_hook(
    failure, first_only, capture_inputs
):
    from types import SimpleNamespace

    import jax.numpy as jnp

    from bench.esmfold2_msa_prefix_probe import candidate_stack

    def block(msa, pair, params, prefix, **kw):
        assert kw["is_final"] is False or not first_only
        if failure:
            raise RuntimeError("stack failed")
        return msa + 1, pair + 2

    embedders = SimpleNamespace(msa_encoder_block=block)

    def encoder(pair, embedding, *a, **kw):
        msa = jnp.zeros_like(pair)
        for i in range(kw["n_layers"]):
            msa, pair = embedders.msa_encoder_block(
                msa,
                pair,
                {},
                f"msa_encoder.blocks.{i}",
                is_final=i == kw["n_layers"] - 1,
            )
        return pair

    embedders.msa_encoder = encoder
    features, tape = fixture()
    args = (
        embedders,
        msa_inputs(features, tape),
        jnp.zeros((1, 2, 3)),
        jnp.zeros((1, 2, 2, 3)),
        {},
        2,
        first_only,
        capture_inputs,
    )
    if failure:
        with pytest.raises(RuntimeError, match="stack failed"):
            candidate_stack(*args)
    else:
        result = candidate_stack(*args)
        assert len(result) == (2 if first_only else 4) + (2 if capture_inputs else 0)
        if capture_inputs:
            np.testing.assert_array_equal(result["blocks.0.msa_input"], 0)
            np.testing.assert_array_equal(result["blocks.0.pair_input"], 0)
        last = 0 if first_only else 1
        np.testing.assert_array_equal(result[f"blocks.{last}.pair"], 2 * (last + 1))
        np.testing.assert_array_equal(result[f"blocks.{last}.msa"], last + 1)
    assert embedders.msa_encoder_block is block


@pytest.mark.parametrize(
    "sizes,width,chunk",
    [
        ([64, 64, 2], 130, 64),
        ([12], 12, 64),
        ([130], 130, None),
    ],
)
def test_transition_chunk_geometry_accepts_native_calls(sizes, width, chunk):
    from bench.esmfold2_msa_prefix_probe import validate_transition_chunks

    validate_transition_chunks([(1, size, 4, 128) for size in sizes], width, chunk)


@pytest.mark.parametrize(
    "sizes,width,chunk",
    [
        ([130], 130, 64),
        ([64, 2, 64], 130, 64),
        ([64, 64], 130, 64),
        ([64, 64, 2, 2], 130, 64),
        ([12], 12, 0),
        ([], 0, 64),
    ],
)
def test_transition_chunk_geometry_rejects_wrong_calls(sizes, width, chunk):
    from bench.esmfold2_msa_prefix_probe import validate_transition_chunks

    with pytest.raises(ValueError, match="chunk"):
        validate_transition_chunks([(1, size, 4, 128) for size in sizes], width, chunk)


@pytest.mark.parametrize("biased", [False, True])
def test_tape_ffi_dispatch_preserves_runtime_bias_path(biased):
    from bench.esmfold2_tape import dispatch_bias_free_ffi

    params = {"W.bias": object()} if biased else {}
    calls = []

    def ffi(x, p, prefix):
        calls.append("ffi")
        assert p is params
        return x

    def fallback(x, p, prefix):
        calls.append("runtime")
        assert p is params
        return x

    x = object()
    assert dispatch_bias_free_ffi(x, params, "W", ffi=ffi, fallback=fallback) is x
    assert calls == ["runtime" if biased else "ffi"]


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("full_opm", [False, True])
@pytest.mark.parametrize("full_block", [False, True])
def test_candidate_probe_restores_hooks_on_missing_boundary_or_failure(
    failure, full_opm, full_block
):
    from types import SimpleNamespace

    import jax.numpy as jnp

    from bench.esmfold2_msa_prefix_probe import candidate_prefix

    def original(*args, **kwargs):
        return None

    def encoder(*args, **kwargs):
        if failure:
            raise RuntimeError("encoder failed")

    embedders = SimpleNamespace(
        linear=original,
        msa_encoder=encoder,
        outer_product_mean=original,
        msa_pair_weighted_averaging=original,
        transition=original,
        triangle_multiplicative=original,
    )
    trunk = SimpleNamespace(
        linear=original,
        layer_norm=original,
        _autocast_linear=original,
        _autocast_norm=original,
    )
    features, tape = fixture()
    with pytest.raises(
        RuntimeError if failure else ValueError,
        match="encoder failed" if failure else "boundary missing",
    ):
        candidate_prefix(
            embedders,
            trunk,
            msa_inputs(features, tape),
            jnp.zeros((1, 2, 3)),
            jnp.zeros((1, 2, 2, 3)),
            {},
            1,
            native_opm=True,
            full_opm=full_opm or full_block,
            full_pwa=full_block,
            full_block=full_block,
        )
    assert embedders.linear is original
    assert embedders.outer_product_mean is original
    assert embedders.msa_pair_weighted_averaging is original
    assert embedders.transition is original
    assert embedders.triangle_multiplicative is original
    assert trunk.linear is original
    assert trunk.layer_norm is original
    assert trunk._autocast_linear is original
    assert trunk._autocast_norm is original


def fixture():
    msa = np.array([[[0, 1], [2, 3], [4, 5]]], np.int64)
    features = {
        "msa": msa,
        "msa_attention_mask": np.ones_like(msa, bool),
        "has_deletion": np.ones_like(msa, bool),
        "deletion_value": np.ones_like(msa, np.float32) * 0.1037,
    }
    tape = {
        "msa_row_choices": np.array([[0, 2]], np.int64),
        "msa_column_keep": np.array([[False, True]]),
    }
    return features, tape


@pytest.mark.parametrize("interchange", [False, True, "bad_shape", "wrong_mode"])
def test_full_opm_probe_runs_actual_jitted_runtime_and_stops_before_triangle(
    interchange,
):
    import jax
    import jax.numpy as jnp

    from bench.esmfold2_msa_prefix_probe import candidate_prefix
    from foldjax.models.esmfold2.models import embedders, trunk

    features, tape = fixture()
    inputs = {k: jnp.asarray(v) for k, v in msa_inputs(features, tape).items()}
    prefix = "msa_encoder.blocks.0.outer_product_mean."
    params = {
        "msa_encoder.embed.weight": jnp.ones((8, 35)) * 0.03,
        "msa_encoder.project_inputs.weight": jnp.ones((8, 3)) * 0.07,
        prefix + "norm.weight": jnp.ones(8),
        prefix + "norm.bias": jnp.zeros(8),
        prefix + "W.weight": jnp.ones((4, 8)),
        prefix + "Wout.weight": jnp.ones((3, 4)),
        prefix + "Wout.bias": jnp.array([0.1037, 0.21, 0.34]),
    }
    original = embedders.outer_product_mean
    control = jnp.full((1, 2, 2, 4), 0.25) if interchange else None
    if interchange == "bad_shape":
        control = control[:, :1]
    run = jax.jit(
        lambda p: candidate_prefix(
            embedders,
            trunk,
            inputs,
            jnp.ones((1, 2, 3)),
            jnp.zeros((1, 2, 2, 3)),
            p,
            1,
            native_opm=True,
            full_opm=interchange != "wrong_mode",
            wout_input=control,
        )
    )
    if interchange in ("bad_shape", "wrong_mode"):
        with pytest.raises(ValueError, match="shape differs|requires full OPM"):
            run(params)
        assert embedders.outer_product_mean is original
        return
    result = run(params)
    assert len(result) == 11
    assert result["blocks.0.outer_product_mean.Wout.input"].shape == (1, 2, 2, 4)
    assert result["blocks.0.outer_product_mean.output"].shape == (1, 2, 2, 3)
    assert result["blocks.0.outer_product_mean.output"].dtype == jnp.bfloat16
    assert embedders.outer_product_mean is original
    if interchange:
        np.testing.assert_array_equal(
            result["blocks.0.outer_product_mean.Wout.input"], control
        )


def test_column_mask_preserves_query_then_selects_rows_without_masking_deletions():
    features, tape = fixture()
    result = msa_inputs(features, tape)
    np.testing.assert_array_equal(result["msa_attention_mask"], [[[1, 0], [1, 1]]])
    assert result["msa_oh"].shape == (1, 2, 2, 33)
    assert result["msa_oh"][0, 0, 0, 0] == 1
    assert result["msa_oh"][0, 0, 1].sum() == 0
    assert result["msa_oh"][0, 1, 1, 5] == 1
    assert result["has_deletion"][0, 0, 1] == 1
    assert result["deletion_value"][0, 0, 1] == np.float32(0.1037)
    assert all(v.dtype == np.float32 for v in result.values())
    assert features["msa_attention_mask"].all()


@pytest.mark.parametrize(
    "change", ["duplicate", "query", "range", "empty", "mask", "nan", "tokens"]
)
def test_invalid_msa_tape_or_features_fail(change):
    features, tape = fixture()
    if change == "duplicate":
        tape["msa_row_choices"] = np.array([[0, 0]])
    elif change == "query":
        tape["msa_row_choices"] = np.array([[1, 2]])
    elif change == "range":
        tape["msa_row_choices"] = np.array([[0, 3]])
    elif change == "empty":
        tape["msa_row_choices"] = np.empty((0, 2), np.int64)
    elif change == "mask":
        tape["msa_column_keep"] = np.array([[0.0, 1.0]])
    elif change == "nan":
        features["deletion_value"].flat[0] = np.nan
    else:
        features["msa"].flat[0] = 33
    with pytest.raises(ValueError):
        msa_inputs(features, tape)


@pytest.mark.parametrize("failure", [False, True])
def test_pwa_operation_observer_preserves_calls_and_restores_functions(failure):
    from types import SimpleNamespace

    from bench.esmfold2_msa_prefix_probe import observe_pwa_ops

    class Tensor:
        def clone(self):
            return Tensor()

    x, y, z, probability, result = (Tensor() for _ in range(5))

    def softmax(value, *, dim):
        assert value is x and dim == -2
        return probability

    def einsum(equation, *args):
        assert equation == "bijh,bjmhd,bimhd->bimhd"
        assert args == (probability, y, z)
        return result

    torch = SimpleNamespace(softmax=softmax, einsum=einsum)

    def call():
        output = torch.einsum("bijh,bjmhd,bimhd->bimhd", torch.softmax(x, dim=-2), y, z)
        if failure:
            raise RuntimeError("native PWA failure")
        return output

    values = {}
    if failure:
        with pytest.raises(RuntimeError, match="native PWA failure"):
            observe_pwa_ops(torch, call, values)
    else:
        assert observe_pwa_ops(torch, call, values) is result
    assert len(values) == 6
    assert torch.softmax is softmax and torch.einsum is einsum


def test_stack_fusion_control_preserves_strict_baseline():
    from bench.esmfold2_msa_prefix_probe import (
        compiler_control,
        stack_compiler_control,
    )

    baseline = compiler_control("native-chunks-strict-rounding")
    assert stack_compiler_control() == baseline
    diagnostic = stack_compiler_control(True)
    assert diagnostic.pop("xla_disable_hlo_passes") == (
        baseline["xla_disable_hlo_passes"] + ",fusion"
    )
    assert diagnostic == {
        k: v for k, v in baseline.items() if k != "xla_disable_hlo_passes"
    }
    assert stack_compiler_control() == baseline


@pytest.mark.parametrize("engine,stack", [("native", True), ("jax", False)])
def test_fusion_control_rejects_wrong_capture_before_io(monkeypatch, engine, stack):
    import sys

    from bench.esmfold2_msa_prefix_probe import main

    args = ["probe", "--engine", engine, "--disable-fusion"]
    if stack:
        args.append("--stack")
    for key in (
        "embedding",
        "pair",
        "reference",
        "features",
        "weights",
        "source",
        "output",
    ):
        args.extend(["--" + key, "/nonexistent-fusion-probe"])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="fusion control requires JAX stack"):
        main()
