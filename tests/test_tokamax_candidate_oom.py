"""A failed candidate is excluded; an all-failed operation must remain a failure."""

import pytest


def test_installed_tokamax_skips_an_oom_candidate(monkeypatch):
    from tokamax._src.autotuning import autotuner
    from tokamax._src.benchmarking import BenchmarkData

    def benchmark(factory, config, args, kwargs):
        if config == "oom":
            raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory")
        return BenchmarkData(
            compile_time_ms=1, lower_time_ms=1, evaluation_times_ms=(1,), metadata={}
        )

    monkeypatch.setattr(autotuner, "_benchmark", benchmark)
    tuner = autotuner.Autotuner(compile_executor_fn=None)
    result = tuner.autotune(lambda config: None, {"oom", "fits"})
    assert result.fastest_config == "fits"
    assert set(result.prune_errors()) == {"fits"}
    with pytest.raises(ExceptionGroup, match="All configs failed"):
        tuner.autotune(lambda config: None, {"oom"}).fastest_config
