"""The dual-GEMM kernel lookup is answered from the wheel's own globals.

cuEquivariance cannot load on a CPU host, so the wheel module is stood in
for; the contract under test is FoldJAX's wrapper, not the kernel.
"""

import importlib.metadata
import sys
import types

import pytest

from foldjax.models import _cueq

_NAME = "cuequivariance_jax.triangle._sigmoid_gated_dual_gemm"


@pytest.fixture
def fake_wheel(monkeypatch):
    calls = []
    gemm = types.ModuleType(_NAME)
    gemm._autotuned_forward = None
    gemm._autotuned_backward = None

    def _get_autotuned_kernel(is_forward):
        calls.append(is_forward)
        name = "_autotuned_forward" if is_forward else "_autotuned_backward"
        if getattr(gemm, name) is None:
            setattr(gemm, name, object())
        return getattr(gemm, name)

    gemm._get_autotuned_kernel = _get_autotuned_kernel
    triangle = types.ModuleType("cuequivariance_jax.triangle")
    triangle._sigmoid_gated_dual_gemm = gemm
    package = types.ModuleType("cuequivariance_jax")
    package.triangle = triangle
    monkeypatch.setitem(sys.modules, "cuequivariance_jax", package)
    monkeypatch.setitem(sys.modules, "cuequivariance_jax.triangle", triangle)
    monkeypatch.setitem(sys.modules, _NAME, gemm)
    monkeypatch.setattr(
        importlib.metadata, "version",
        lambda name: _cueq._DUAL_GEMM_MEMOIZED_VERSION,
    )
    return gemm, calls


def test_first_call_per_direction_delegates_then_the_global_answers(fake_wheel):
    gemm, calls = fake_wheel
    _cueq._memoize_dual_gemm_kernel()
    forward = gemm._get_autotuned_kernel(True)
    assert calls == [True]
    for _ in range(3):
        assert gemm._get_autotuned_kernel(True) is forward
    assert calls == [True]
    backward = gemm._get_autotuned_kernel(False)
    assert backward is not forward and calls == [True, False]
    assert gemm._get_autotuned_kernel(False) is backward
    assert calls == [True, False]


def test_memoization_wraps_once(fake_wheel):
    gemm, _ = fake_wheel
    _cueq._memoize_dual_gemm_kernel()
    wrapped = gemm._get_autotuned_kernel
    _cueq._memoize_dual_gemm_kernel()
    assert gemm._get_autotuned_kernel is wrapped


def test_other_wheel_versions_are_left_alone(fake_wheel, monkeypatch):
    gemm, _ = fake_wheel
    original = gemm._get_autotuned_kernel
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.12.0")
    _cueq._memoize_dual_gemm_kernel()
    assert gemm._get_autotuned_kernel is original


def test_a_wheel_without_the_globals_is_left_alone(fake_wheel):
    gemm, _ = fake_wheel
    original = gemm._get_autotuned_kernel
    del gemm._autotuned_backward
    _cueq._memoize_dual_gemm_kernel()
    assert gemm._get_autotuned_kernel is original
