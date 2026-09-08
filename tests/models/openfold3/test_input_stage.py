"""Input-only graphs never enter recycle or sampling stages."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3 import inference, streaming
from tests.models.openfold3.feature_fixture import minimal_features
from tests.models.openfold3.test_stable_compile import _config, _table
from tests.models.openfold3.test_streamed_msa import TinyParams


@pytest.mark.parametrize(
    "route", ["eager", "compiled", "streamed", "streamed_compiled"]
)
def test_inputs_skip_recycles_and_tail(monkeypatch, route):
    calls = []

    def initialize(batch, params, **kwargs):
        calls.append("inputs")
        x = batch["token_mask"][..., None] * params
        return x, x, x[..., None]

    def forbidden(*args, **kwargs):
        raise AssertionError("input-only executed a downstream stage")

    monkeypatch.setattr(inference, "initialize_trunk", initialize)
    monkeypatch.setattr(inference, "trunk", forbidden)
    monkeypatch.setattr(inference, "_predict_from_trunk", forbidden)
    monkeypatch.setattr(streaming, "HostMSACycles", forbidden)
    config = _config(
        stop_after_inputs=True, returned_representations=("single_inputs",),
        msa_depth=1,
    )
    batch = minimal_features(tokens=4, atoms=4, msa_rows=3)
    params = TinyParams(trunk=jnp.asarray(2.0))
    streaming._compiled_streams.clear_cache()
    if route.startswith("streamed"):
        # The streamed path must not even require a cycle index tape.
        result = streaming.compile_streamed_predict(
            config, _table(), compiled=route == "streamed_compiled"
        )(jax.random.key(0), batch, params)
    elif route == "compiled":
        result = inference.compile_predict(config, _table())(
            jax.random.key(0), batch, params
        )
    else:
        result = inference.predict(jax.random.key(0), batch, params, config, _table())
    if route == "streamed_compiled":
        assert streaming._compiled_streams._cache_size() == 1
    streaming._compiled_streams.clear_cache()
    assert calls == ["inputs"]
    assert result.coordinates is None
    assert result.single is None and result.pair is None
    np.testing.assert_array_equal(result.single_inputs, 2.0)
