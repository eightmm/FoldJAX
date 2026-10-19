"""Capacity must follow the selected CUDA stream and ignore live env fractions."""

import ctypes
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from foldjax import _cuda_memory, _gpu, oom


def _visible_driver(*, capacities=(48 * 2**30, 24 * 2**30), count=None, fail_at=None):
    handles = (17, 42)
    calls = []

    def init(flags):
        calls.append(("init", flags))
        return int(fail_at == "init")

    def get_count(pointer):
        calls.append(("count",))
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = (
            len(capacities) if count is None else count
        )
        return int(fail_at == "count")

    def get_device(pointer, ordinal):
        calls.append(("device", ordinal))
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = handles[ordinal]
        return int(fail_at == f"device-{ordinal}")

    def total(pointer, device):
        calls.append(("total", device.value))
        ordinal = handles.index(device.value)
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_size_t))[0] = capacities[ordinal]
        return int(fail_at == f"total-{ordinal}")

    return SimpleNamespace(
        cuInit=init,
        cuDeviceGetCount=get_count,
        cuDeviceGet=get_device,
        cuDeviceTotalMem_v2=total,
    ), calls


@pytest.mark.parametrize(
    "visibility",
    ["3,0", "GPU-bbbbbbbb,GPU-aaaaaaaa", "MIG-GPU-aaaaaaaa/2/0"],
)
def test_visible_capacity_follows_driver_devices_without_parsing_masks(
    monkeypatch, visibility
):
    driver, calls = _visible_driver()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    monkeypatch.setattr(_cuda_memory, "_visible_device_driver", lambda: driver)
    monkeypatch.setattr(
        _cuda_memory, "_driver", lambda: pytest.fail("queried a JAX stream")
    )
    assert _cuda_memory.visible_device_bytes() == (48 * 2**30, 24 * 2**30)
    assert calls == [
        ("init", 0),
        ("count",),
        ("device", 0),
        ("total", 17),
        ("device", 1),
        ("total", 42),
    ]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == visibility


def test_visible_capacity_loader_needs_no_stream_or_context_api(monkeypatch):
    library = SimpleNamespace(
        cuInit=lambda: None,
        cuDeviceGetCount=lambda: None,
        cuDeviceGet=lambda: None,
        cuDeviceTotalMem_v2=lambda: None,
    )
    loaded = []

    def load(name):
        loaded.append(name)
        return library

    monkeypatch.setattr(ctypes, "CDLL", load)
    _cuda_memory._visible_device_driver.cache_clear()
    try:
        assert _cuda_memory._visible_device_driver() is library
        assert _cuda_memory._visible_device_driver() is library
        assert loaded == ["nvcuda.dll" if os.name == "nt" else "libcuda.so.1"]
        assert library.cuInit.argtypes == [ctypes.c_uint]
        assert library.cuDeviceGetCount.argtypes == [ctypes.POINTER(ctypes.c_int)]
        assert library.cuDeviceGet.argtypes == [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        assert library.cuDeviceTotalMem_v2.argtypes == [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
        ]
        assert all(
            function.restype is ctypes.c_int for function in vars(library).values()
        )
    finally:
        _cuda_memory._visible_device_driver.cache_clear()


@pytest.mark.parametrize(
    "fail_at", ["init", "count", "device-0", "total-0", "device-1", "total-1"]
)
def test_visible_capacity_discards_incomplete_driver_inventory(monkeypatch, fail_at):
    driver, _ = _visible_driver(fail_at=fail_at)
    monkeypatch.setattr(_cuda_memory, "_visible_device_driver", lambda: driver)
    assert _cuda_memory.visible_device_bytes() is None


@pytest.mark.parametrize("count", [0, -1, 1025])
def test_visible_capacity_rejects_empty_or_implausible_counts(monkeypatch, count):
    driver, calls = _visible_driver(count=count)
    monkeypatch.setattr(_cuda_memory, "_visible_device_driver", lambda: driver)
    assert _cuda_memory.visible_device_bytes() is None
    assert calls == [("init", 0), ("count",)]


@pytest.mark.parametrize("capacity", [0, -1, 2**50 + 1])
def test_visible_capacity_rejects_invalid_native_capacity(monkeypatch, capacity):
    driver, _ = _visible_driver(capacities=(48 * 2**30, capacity))
    monkeypatch.setattr(_cuda_memory, "_visible_device_driver", lambda: driver)
    assert _cuda_memory.visible_device_bytes() is None


@pytest.mark.parametrize("error", [OSError("no driver"), AttributeError("no symbol")])
def test_visible_capacity_treats_unavailable_driver_as_unknown(monkeypatch, error):
    def unavailable():
        raise error

    monkeypatch.setattr(_cuda_memory, "_visible_device_driver", unavailable)
    assert _cuda_memory.visible_device_bytes() is None


def test_visible_capacity_import_and_query_do_not_import_jax():
    program = textwrap.dedent(
        """
        import ctypes
        import sys
        from types import SimpleNamespace

        class RejectJax:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "jax" or fullname.startswith("jax."):
                    raise AssertionError("capacity query imported JAX")

        sys.meta_path.insert(0, RejectJax())
        from foldjax import _cuda_memory

        def write_int(pointer, value):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = value
            return 0

        def total(pointer, device):
            assert device.value == 17
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_size_t))[0] = 48 * 2**30
            return 0

        _cuda_memory._visible_device_driver = lambda: SimpleNamespace(
            cuInit=lambda flags: 0,
            cuDeviceGetCount=lambda pointer: write_int(pointer, 1),
            cuDeviceGet=lambda pointer, ordinal: write_int(pointer, 17),
            cuDeviceTotalMem_v2=total,
        )
        assert _cuda_memory.visible_device_bytes() == (48 * 2**30,)
        assert "jax" not in sys.modules
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


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
