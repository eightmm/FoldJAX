"""Choose startup headroom without initializing JAX or changing caller policy."""

import os
import subprocess
import sys

import pytest

from foldjax import _cuda_memory, cli, oom


@pytest.mark.parametrize(
    ("capacities_gib", "expected"),
    [
        (None, 0.9),
        ((), 0.9),
        ((22,), 0.9),
        ((24,), 0.9),
        ((32,), 0.9),
        ((40,), 0.9),
        ((43.99,), 0.9),
        ((44,), 0.85),
        ((45, 48), 0.85),
        ((48,), 0.85),
        ((51.99,), 0.85),
        ((52,), 0.9),
        ((64,), 0.9),
        ((80,), 0.9),
        ((96,), 0.9),
        ((24, 48), 0.9),
        ((48, 96), 0.9),
    ],
)
def test_native_cli_default_depends_on_visible_capacity(
    monkeypatch, capacities_gib, expected
):
    monkeypatch.delenv(oom.CURRENT_FRACTION_ENV, raising=False)
    monkeypatch.delenv(oom.FRACTION_ENV, raising=False)
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    capacities = (
        None
        if capacities_gib is None
        else tuple(int(size * 2**30) for size in capacities_gib)
    )
    monkeypatch.setattr(_cuda_memory, "visible_device_bytes", lambda: capacities)
    cli._apply_mem_fraction(None)
    assert oom.mem_fraction() == expected
    assert os.environ[oom.CURRENT_FRACTION_ENV] == str(expected)
    assert oom.FRACTION_ENV not in os.environ


@pytest.mark.parametrize("spelling", [oom.CURRENT_FRACTION_ENV, oom.FRACTION_ENV])
@pytest.mark.parametrize("explicit", [None, 0.8])
def test_caller_policy_avoids_hardware_discovery(monkeypatch, spelling, explicit):
    monkeypatch.delenv(oom.CURRENT_FRACTION_ENV, raising=False)
    monkeypatch.delenv(oom.FRACTION_ENV, raising=False)
    monkeypatch.setenv(spelling, "0.7")
    monkeypatch.setattr(
        _cuda_memory,
        "visible_device_bytes",
        lambda: pytest.fail("explicit allocator policy must not query CUDA"),
    )
    cli._apply_mem_fraction(explicit)
    assert oom.mem_fraction() == (0.7 if explicit is None else explicit)
    other = (
        oom.FRACTION_ENV
        if spelling == oom.CURRENT_FRACTION_ENV
        else oom.CURRENT_FRACTION_ENV
    )
    assert other not in os.environ


def test_cpu_prediction_does_not_initialize_cuda(monkeypatch):
    monkeypatch.delenv(oom.CURRENT_FRACTION_ENV, raising=False)
    monkeypatch.delenv(oom.FRACTION_ENV, raising=False)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(
        _cuda_memory, "visible_device_bytes", lambda: pytest.fail("CUDA queried on CPU")
    )
    cli._apply_mem_fraction(None)
    assert oom.mem_fraction() == 0.9


def test_automatic_fraction_is_set_before_jax_is_imported():
    program = """
import os, sys
from foldjax import cli, _cuda_memory, oom
assert 'jax' not in sys.modules
_cuda_memory.visible_device_bytes = lambda: (48 * 2**30,)
cli._apply_mem_fraction(None)
assert 'jax' not in sys.modules
assert os.environ[oom.CURRENT_FRACTION_ENV] == '0.85'
"""
    environment = dict(os.environ, JAX_PLATFORMS="cuda")
    for spelling in (oom.CURRENT_FRACTION_ENV, oom.FRACTION_ENV):
        environment.pop(spelling, None)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
