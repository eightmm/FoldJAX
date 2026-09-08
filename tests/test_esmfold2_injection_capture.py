from types import SimpleNamespace

import pytest

from bench.esmfold2_tape import observe_native_injection


class Module:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def register_forward_pre_hook(self, hook, *, with_kwargs):
        assert with_kwargs
        return self.register_forward_hook(hook)


class Tensor:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return Tensor(self.value)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("capture_inputs", [False, True])
def test_native_injection_keeps_first_values_and_removes_hooks(fail, capture_inputs):
    model = SimpleNamespace(
        **{
            k: Module()
            for k in ("msa_encoder", "lm_encoder", "parcae_input_norm", "folding_trunk")
        }
    )
    try:
        with observe_native_injection(
            model, enabled=True, capture_msa_inputs=capture_inputs
        ) as (values, counts):
            for value in (1, 2):
                for module in vars(model).values():
                    x, y = Tensor(value), Tensor(value + 10)
                    hooks = (
                        module.hooks[1:]
                        if capture_inputs and module is model.msa_encoder
                        else module.hooks
                    )
                    for hook in hooks:
                        assert hook(module, (x,), y) is None
                    x.value = y.value = -1
            if fail:
                raise RuntimeError("forward failed")
    except RuntimeError:
        assert fail
    assert counts == {k: 2 for k in ("msa", "lm", "injection", "trunk")}
    expected = {
        "msa_output": 11,
        "lm_output": 11,
        "injection_input": 1,
        "injection_output": 11,
        "trunk_input": 1,
    }
    if capture_inputs:
        expected["trunk_output"] = 11
        expected["trunk_output_last"] = 12
    assert {k: v.value for k, v in values.items()} == expected
    assert all(not module.hooks for module in vars(model).values())


def test_disabled_capture_does_not_access_model():
    with observe_native_injection(object()) as state:
        assert state == ({}, {})


@pytest.mark.parametrize("fail", [False, True])
def test_native_coda_capture_clones_and_restores(fail):
    model = SimpleNamespace(
        **{
            k: Module()
            for k in (
                "msa_encoder",
                "lm_encoder",
                "parcae_input_norm",
                "folding_trunk",
                "parcae_coda",
            )
        }
    )
    try:
        with observe_native_injection(model, enabled=True, capture_coda=True) as (
            values,
            counts,
        ):
            x, y = Tensor(3), Tensor(7)
            assert model.parcae_coda.hooks[0](model.parcae_coda, (x,), y) is None
            x.value = y.value = -1
            if fail:
                raise RuntimeError("failed after coda")
    except RuntimeError:
        assert fail
    assert counts == {"coda": 1}
    assert {k: v.value for k, v in values.items()} == {
        "coda_input": 3,
        "coda_output": 7,
    }
    assert all(not module.hooks for module in vars(model).values())


def test_native_partial_registration_is_removed_on_missing_route():
    msa = Module()
    model = SimpleNamespace(msa_encoder=msa, lm_encoder=None)
    with pytest.raises(ValueError, match="active lm_encoder"):
        with observe_native_injection(model, enabled=True):
            pytest.fail("missing route was accepted")
    assert not msa.hooks


def test_jax_capture_restores_functions_when_prediction_raises():
    import jax

    from bench.esmfold2_tape import predict_with_injection_capture

    def unused(*args, **kwargs):
        pytest.fail("unexpected model call")

    module = SimpleNamespace(
        msa_encoder=unused,
        folding_trunk=unused,
        trunk_ops=SimpleNamespace(_autocast_norm=unused),
    )
    scan = jax.lax.scan

    def fail():
        raise RuntimeError("prediction failed")

    with pytest.raises(RuntimeError, match="prediction failed"):
        predict_with_injection_capture(fail, module)
    assert module.msa_encoder is unused
    assert module.folding_trunk is unused
    assert module.trunk_ops._autocast_norm is unused
    assert jax.lax.scan is scan


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("capture_inputs", [False, True])
@pytest.mark.parametrize("capture_coda", [False, True])
def test_jax_observer_preserves_scan_and_first_boundaries(
    compiled, capture_inputs, capture_coda
):
    import jax
    import jax.numpy as jnp

    from bench.esmfold2_tape import predict_with_injection_capture

    module = SimpleNamespace(
        msa_encoder=lambda x, embedding: x + 3,
        folding_trunk=lambda x, params, prefix: x + 2,
        trunk_ops=SimpleNamespace(_autocast_norm=lambda x, params, prefix: x * 2),
    )

    def run_loops(x):
        def body(z, unused):
            lm = module.folding_trunk(z, {}, "lm_encoder")
            msa = module.msa_encoder(z, z + 10)
            injected = module.trunk_ops._autocast_norm(
                msa + lm, {}, "parcae_input_norm"
            )
            return module.folding_trunk(z + injected, {}, "folding_trunk"), None

        return jax.lax.scan(body, x, None, length=2)[0]

    def predict(x):
        result = run_loops(x)
        return {"result": module.folding_trunk(result, {}, "parcae_coda")}

    expected = predict(jnp.array(1.0))
    original = (
        module.msa_encoder,
        module.folding_trunk,
        module.trunk_ops._autocast_norm,
        jax.lax.scan,
    )

    def observed(x):
        return predict_with_injection_capture(
            predict,
            module,
            x,
            capture_msa_inputs=capture_inputs,
            capture_coda=capture_coda,
        )

    result = (jax.jit(observed) if compiled else observed)(jnp.array(1.0))
    assert result["result"] == expected["result"]
    boundaries = {
        "lm_output": 3,
        "msa_output": 4,
        "injection_input": 7,
        "injection_output": 14,
        "trunk_input": 15,
    }
    if capture_inputs:
        boundaries.update(
            msa_input_pair=1,
            msa_input_embedding=11,
            trunk_output=17,
            trunk_output_last=97,
        )
    if capture_coda:
        boundaries.update(coda_input=97, coda_output=99)
    assert result["diagnostic_injection"] == boundaries
    assert original == (
        module.msa_encoder,
        module.folding_trunk,
        module.trunk_ops._autocast_norm,
        jax.lax.scan,
    )


def test_native_msa_entry_clones_first_kwargs_and_cleans_up():
    model = SimpleNamespace(
        **{
            k: Module()
            for k in ("msa_encoder", "lm_encoder", "parcae_input_norm", "folding_trunk")
        }
    )
    with observe_native_injection(model, enabled=True, capture_msa_inputs=True) as (
        values,
        counts,
    ):
        hook = model.msa_encoder.hooks[0]
        for n in (1, 2):
            pair, embedding = Tensor(n), Tensor(n + 10)
            assert (
                hook(model.msa_encoder, (), {"x_pair": pair, "x_inputs": embedding})
                is None
            )
            pair.value = embedding.value = -1
    assert counts == {"msa_inputs": 2}
    assert {k: v.value for k, v in values.items()} == {
        "msa_input_pair": 1,
        "msa_input_embedding": 11,
    }
    assert all(not module.hooks for module in vars(model).values())
