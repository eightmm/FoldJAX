"""Capacity must follow the selected CUDA stream and ignore live env fractions."""

import ctypes
from types import SimpleNamespace

import pytest

from foldjax import _cuda_memory, _gpu, oom


@pytest.mark.parametrize("fail_at", [None, "stream", "total"])
def test_cuda_capacity_uses_the_stream_device_not_a_visible_ordinal(
    monkeypatch, fail_at
):
    device = SimpleNamespace(
        platform="gpu",
        compute_capability="8.6",
        get_stream_for_external_ready_events=lambda: 1234,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,0")

    def stream_device(stream, pointer):
        assert stream.value == 1234
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 17
        return int(fail_at == "stream")

    def total(pointer, cuda_device):
        assert cuda_device.value == 17
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_size_t))[0] = 24 * 2**30
        return int(fail_at == "total")

    monkeypatch.setattr(
        _cuda_memory,
        "_driver",
        lambda: SimpleNamespace(
            cuStreamGetDevice=stream_device, cuDeviceTotalMem_v2=total
        ),
    )
    assert _cuda_memory.total_device_bytes(device) == (
        24 * 2**30 if fail_at is None else None
    )


def test_a_changed_environment_fraction_cannot_change_the_card_capacity(monkeypatch):
    device = SimpleNamespace(memory_stats=lambda: {"bytes_limit": 18 * 2**30})
    monkeypatch.setattr(_gpu, "default_device", lambda: device)
    monkeypatch.setattr(_cuda_memory, "total_device_bytes", lambda selected: 24 * 2**30)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for fraction in ("0.75", "0.95"):
        monkeypatch.setenv(oom.CURRENT_FRACTION_ENV, fraction)
        assert oom.device_budget() == (18 * 2**30, 24 * 2**30)


def test_unavailable_capacity_is_not_inferred_from_an_environment_knob(monkeypatch):
    device = SimpleNamespace(memory_stats=lambda: {"bytes_limit": 18 * 2**30})
    monkeypatch.setattr(_gpu, "default_device", lambda: device)
    monkeypatch.setattr(_cuda_memory, "total_device_bytes", lambda selected: None)
    assert oom.device_budget() == (18 * 2**30, None)


def test_cpu_statistics_do_not_load_the_cuda_driver(monkeypatch):
    monkeypatch.setattr(
        _cuda_memory, "_driver", lambda: pytest.fail("loaded CUDA on CPU")
    )
    assert _cuda_memory.total_device_bytes(SimpleNamespace(platform="cpu")) is None
