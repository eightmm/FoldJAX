from __future__ import annotations

import jax
import numpy as np
import pytest
from jax._src import compiler

from bench import opendde_profile
from foldjax.models.opendde import runner


def test_diagnostic_observes_real_executable_without_changing_outputs_or_owners(
    monkeypatch,
) -> None:
    @jax.jit
    def graph(value):
        return value @ value + 1.0

    def failing_score(*args, **kwargs):
        raise ValueError("score failed")

    monkeypatch.setattr(runner, "_predict", graph)
    monkeypatch.setattr(runner, "_score", failing_score)
    monkeypatch.setattr(opendde_profile, "_device_memory", lambda: {})
    original_compile = compiler.compile_or_get_cached
    value = np.arange(9, dtype=np.float32).reshape(3, 3)
    report = {}

    with pytest.raises(ValueError, match="score failed"):
        with opendde_profile.profile_prediction(report):
            actual = runner._predict(value)
            runner._score(actual)

    np.testing.assert_array_equal(np.asarray(actual), value @ value + 1.0)
    assert compiler.compile_or_get_cached is original_compile
    assert runner._predict is graph
    assert runner._score is failing_score
    assert [(phase["phase"], phase["status"]) for phase in report["phases"]] == [
        ("_predict", "ok"),
        ("_score", "error"),
    ]
    assert all(phase["seconds"] >= 0 for phase in report["phases"])
    executable = next(
        item for item in report["executables"] if item["module"] == "jit_graph"
    )
    assert executable["compile_or_load_seconds"] >= 0
    assert executable["memory"]["argument_size_in_bytes"] == value.nbytes
    assert executable["memory"]["output_size_in_bytes"] == value.nbytes
