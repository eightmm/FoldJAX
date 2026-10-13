"""Every model's Tokamax GLU must obey the selected GPU's per-block budget."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from foldjax.backends import _tokamax_autotune as policy
from foldjax.models import _glu


@pytest.fixture
def heuristic(monkeypatch):
    tile = pytest.importorskip("tokamax._src.ops.gated_linear_unit.pallas_triton")
    config = tile.Config(block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=4)

    def method(self, ba):
        return config

    monkeypatch.setattr(
        tile.PallasTritonGatedLinearUnit, "_get_heuristics_config", method
    )
    return tile.PallasTritonGatedLinearUnit, config, method


def _config(op):
    argument = SimpleNamespace(args=(SimpleNamespace(dtype=jnp.dtype("float32")),))
    return op._get_heuristics_config(None, argument)


def test_shared_model_glu_clamps_the_real_call(monkeypatch, heuristic):
    import tokamax

    op, _, original = heuristic
    device = SimpleNamespace(
        platform="gpu", compute_capability="8.6", device_kind="RTX A6000"
    )
    monkeypatch.setattr("foldjax._gpu.default_device", lambda: device)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    seen = []

    def call(**kwargs):
        seen.append(_config(op))
        return kwargs["x"]

    monkeypatch.setattr(tokamax, "gated_linear_unit", call)
    x = jnp.ones((2, 8), jnp.float32)
    _glu._fused(x, jnp.ones((8, 2, 4), jnp.float32), jax.nn.silu)
    assert seen[0].num_stages == 3
    assert seen[0].block_k == 32
    assert op._get_heuristics_config is original


def test_nested_larger_device_does_not_inherit_the_small_budget(heuristic):
    op, config, original = heuristic
    small = SimpleNamespace(compute_capability="8.9")
    large = SimpleNamespace(compute_capability="8.0")
    with policy.glu_heuristics_within_shared_memory(small):
        assert _config(op).num_stages == 3
        with policy.glu_heuristics_within_shared_memory(large):
            assert _config(op) == config
        assert _config(op).num_stages == 3
    assert op._get_heuristics_config is original


def test_another_threads_large_device_keeps_its_schedule(heuristic):
    op, config, original = heuristic
    entered, checked = Event(), Event()

    def small():
        with policy.glu_heuristics_within_shared_memory(
            SimpleNamespace(compute_capability="8.6")
        ):
            entered.set()
            assert checked.wait(5)
            assert _config(op).num_stages == 3

    def large():
        assert entered.wait(5)
        try:
            with policy.glu_heuristics_within_shared_memory(
                SimpleNamespace(compute_capability="9.0")
            ):
                assert _config(op) == config
        finally:
            checked.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks = [executor.submit(small), executor.submit(large)]
        for task in tasks:
            task.result(timeout=10)
    assert op._get_heuristics_config is original


def test_a_failed_call_restores_the_heuristic(heuristic):
    op, _, original = heuristic
    with pytest.raises(RuntimeError, match="failed"):
        with policy.glu_heuristics_within_shared_memory(
            SimpleNamespace(compute_capability="12.1")
        ):
            raise RuntimeError("failed")
    assert op._get_heuristics_config is original
