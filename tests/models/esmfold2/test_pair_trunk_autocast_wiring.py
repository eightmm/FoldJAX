"""Original FP32 pair-trunk weights reach native autocast, not blanket casts."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import model
from tests.models.esmfold2.test_distogram_output_flag import _cheap_features


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("cp", [False, True])
def test_predict_preserves_original_pair_affine_weights_only_in_native_arm(
    monkeypatch, dtype, cp
):
    settings = replace(
        model.ModelSettings(),
        d_pair=2,
        d_inputs=2,
        lm_encoder_n_layers=None,
        msa_n_layers=None,
        trunk_dtype=dtype,
    )
    weight = jnp.array([1.0035, -0.1037], jnp.float32)
    params = {
        "folding_trunk.blocks.0.norm.weight": weight,
        "parcae_coda.blocks.0.norm.weight": weight,
        "parcae_log_delta": weight,
        "parcae_log_a": weight,
        "parcae_b_cont": jnp.eye(2, dtype=jnp.float32) * weight,
        "parcae_input_norm.weight": weight,
        "parcae_input_norm.bias": weight,
        "msa_encoder.blocks.0.outer_product_mean.norm.weight": weight,
        "msa_encoder.blocks.0.msa_pair_weighted_averaging.norm_single.weight": weight,
    }
    native = dtype == "bfloat16" and not cp
    monkeypatch.setattr(model, "cp_mesh", lambda: object() if cp else None)
    monkeypatch.setattr(
        model, "inputs_embedding", lambda *a, **kw: jnp.ones((1, 2, 2), dtype)
    )
    for name in ("relative_position_encoding", "_token_bonds_encoding"):
        monkeypatch.setattr(
            model, name, lambda *a, **kw: jnp.zeros((1, 2, 2, 2), dtype)
        )
    monkeypatch.setattr(model, "linear", lambda x, *a, **kw: x)
    seen = []

    def recurrence(key, z, initial, lm, msa, mask, converted, **kw):
        original = kw["pair_trunk_params"]
        assert (original is not None) == native
        dynamics = kw["recurrence_params"]
        assert (dynamics is not None) == native
        norm = kw["injection_norm_params"]
        assert (norm is not None) == native
        opm = kw["msa_opm_params"]
        assert (opm is not None) == native
        if native:
            assert set(opm) == {"msa_encoder.blocks.0.outer_product_mean.norm.weight"}
            assert opm["msa_encoder.blocks.0.outer_product_mean.norm.weight"] is weight
            assert original["folding_trunk.blocks.0.norm.weight"] is weight
            assert original["parcae_coda.blocks.0.norm.weight"] is weight
            for name in ("parcae_log_delta", "parcae_log_a", "parcae_b_cont"):
                assert dynamics[name] is params[name]
            for name in ("parcae_input_norm.weight", "parcae_input_norm.bias"):
                assert norm[name] is params[name]
        assert converted["folding_trunk.blocks.0.norm.weight"].dtype == jnp.dtype(dtype)
        seen.append("recurrence")
        return z

    class EndProbeError(Exception):
        pass

    def coda(x, selected, prefix, **kw):
        assert prefix == "parcae_coda"
        assert kw["native_autocast"] == native
        expected = weight if native else weight.astype(dtype)
        np.testing.assert_array_equal(
            selected["parcae_coda.blocks.0.norm.weight"], expected
        )
        seen.append("coda")
        raise EndProbeError

    monkeypatch.setattr(model, "run_loops", recurrence)
    monkeypatch.setattr(model, "folding_trunk", coda)
    with pytest.raises(EndProbeError):
        model.predict(
            jax.random.key(0),
            _cheap_features(),
            params,
            settings=settings,
            initial_pair_state=jnp.zeros((1, 2, 2, 2), jnp.float32),
            n_chains=1,
        )
    assert seen == ["recurrence", "coda"]


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_real_recurrence_consumes_original_trunk_parameters(
    monkeypatch, compiled, native
):
    weight = jnp.array([1.0035, -0.1037], jnp.float32)
    params = {
        "parcae_log_delta": jnp.zeros(2),
        "parcae_log_a": jnp.zeros(2),
        "parcae_b_cont": jnp.eye(2),
        "parcae_input_norm.weight": jnp.ones(2),
        "parcae_input_norm.bias": jnp.zeros(2),
        "trunk.weight": weight.astype(jnp.bfloat16),
    }

    def trunk(x, selected, prefix, **kw):
        assert prefix == "folding_trunk"
        assert kw["native_autocast"] == native
        expected_dtype = jnp.float32 if native else jnp.bfloat16
        assert selected["trunk.weight"].dtype == expected_dtype
        return jnp.broadcast_to(selected["trunk.weight"].astype(jnp.float32), x.shape)

    monkeypatch.setattr(model, "folding_trunk", trunk)

    def run(value):
        z = jnp.zeros((1, 2, 2, 2), jnp.float32)
        return model.run_loops(
            jax.random.key(0),
            z,
            z,
            None,
            None,
            jnp.ones(z.shape[:-1]),
            params,
            settings=replace(model.ModelSettings(), d_pair=2),
            total_steps=2,
            pair_trunk_params={"trunk.weight": value} if native else None,
        )

    result = (jax.jit(run) if compiled else run)(weight)
    expected = weight if native else weight.astype(jnp.bfloat16).astype(jnp.float32)
    np.testing.assert_array_equal(result, jnp.broadcast_to(expected, result.shape))
