import numpy as np
import pytest

from bench.esmfold2_msa_prefix_probe import msa_inputs


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
def test_candidate_probe_restores_hooks_on_missing_boundary_or_failure(
    failure, full_opm
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
        linear=original, msa_encoder=encoder, outer_product_mean=original
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
            full_opm=full_opm,
        )
    assert embedders.linear is original
    assert embedders.outer_product_mean is original
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
