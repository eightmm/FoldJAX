"""The native input stream can be requested without LM or folding execution."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.models import model as native
from tests.models.esmfold2.test_distogram_output_flag import _cheap_features


@pytest.mark.parametrize("compiled", [False, True])
def test_inputs_skip_language_model_and_pair_initialization(monkeypatch, compiled):
    expected = jnp.asarray([[[1., 2.], [3., 4.]]])

    def forbidden(*args, **kwargs):
        raise AssertionError("input-only executed a downstream stage")

    monkeypatch.setattr(native, "inputs_embedding", lambda *a, **kw: expected)
    monkeypatch.setattr(native, "linear", forbidden)
    monkeypatch.setattr(native, "run_loops", forbidden)
    monkeypatch.setattr(inference, "language_model_states", forbidden)
    model = SimpleNamespace(settings=native.ModelSettings(), parameters={})
    inference.compiled_predict.cache_clear()
    try:
        actual = inference.predict(
            jax.random.key(0), _cheap_features(), model,
            stop_after_inputs=True, return_representations=("single_inputs",),
            compile_it=compiled,
        )
        assert set(actual) == {"single_inputs"}
        np.testing.assert_array_equal(actual["single_inputs"], expected)
    finally:
        inference.compiled_predict.cache_clear()


@pytest.mark.parametrize("selector", ["all", "pair", "all,pair"])
def test_direct_backend_input_selectors(tmp_path, monkeypatch, selector):
    from foldjax.backends.esmfold2 import ESMFold2Backend
    from foldjax.schema import PredictionRequest

    job = tmp_path / "job.json"
    job.write_text('{"entities":[{"type":"protein","id":["A"],"sequence":"AG"}]}')
    seen = {}

    def predict_job(*args, **kwargs):
        seen.update(kwargs)
        return {"single_inputs": np.ones((1, 2, 3))}, {}

    modules = {
        "foldjax.models.esmfold2.inference": SimpleNamespace(
            load=lambda *a, **kw: SimpleNamespace(has_language_model=False),
            seed_key=lambda seed: seed,
            predict_job=predict_job,
        ),
        "foldjax.models.esmfold2.output": SimpleNamespace(),
    }
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )
    request = PredictionRequest(
        model="esmfold2", input=job, weights=tmp_path,
        output_dir=tmp_path / "output", stop_after="inputs",
        representations=selector,
    )
    if selector != "all":
        with pytest.raises(ValueError, match="unknown representation 'pair'"):
            ESMFold2Backend().predict(request)
        assert not seen
    else:
        result = ESMFold2Backend().predict(request)
        assert seen["return_representations"] == ("single_inputs",)
        assert result.samples == ()
