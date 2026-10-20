"""Choose startup headroom without initializing JAX or changing caller policy."""

import os
import subprocess
import sys
from types import SimpleNamespace

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
    monkeypatch.setattr(
        _cuda_memory,
        "visible_device_info",
        lambda: pytest.fail("explicit allocator policy must not query CUDA names"),
    )
    cli._apply_mem_fraction(explicit, models=("boltz2",))
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
    cli._apply_mem_fraction(None, models=("boltz2",))
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


@pytest.fixture
def boltz_startup(monkeypatch):
    for name in (oom.CURRENT_FRACTION_ENV, oom.FRACTION_ENV, "JAX_DEFAULT_DEVICE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(config=SimpleNamespace(jax_default_device=None)),
    )

    def inventory(cards):
        monkeypatch.setattr(
            _cuda_memory,
            "visible_device_bytes",
            lambda: tuple(size * 2**30 for _, size in cards) if cards else None,
        )
        monkeypatch.setattr(
            _cuda_memory,
            "visible_device_info",
            lambda: (
                tuple((name, size * 2**30) for name, size in cards) if cards else None
            ),
        )

    return inventory


@pytest.mark.parametrize("size_gib", [22, 24])
@pytest.mark.parametrize(
    "kind", ["NVIDIA GeForce RTX 4090", "GeForce RTX 4090", "rtx4090", " RTX  4090 "]
)
def test_boltz4090_uses_measured_headroom_for_ecc_and_full_capacity(
    monkeypatch, boltz_startup, kind, size_gib
):
    boltz_startup(((kind, size_gib),))
    cli._apply_mem_fraction(None, models=("boltz-2",))
    assert oom.mem_fraction() == 0.8
    assert oom.FRACTION_ENV not in os.environ


@pytest.mark.parametrize(
    ("model", "kind", "size", "expected"),
    [
        ("boltz2", "NVIDIA GeForce RTX 3090", 24, 0.9),
        ("boltz2", "NVIDIA GeForce RTX 4090 D", 24, 0.9),
        ("boltz2", "NVIDIA GeForce RTX 4090 Laptop GPU", 24, 0.9),
        ("boltz2", "NVIDIA RTX A6000", 48, 0.85),
        ("alphafold3", "NVIDIA GeForce RTX 4090", 24, 0.9),
        ("esmfold2", "NVIDIA GeForce RTX 4090", 24, 0.9),
        ("protenix", "NVIDIA GeForce RTX 4090", 24, 0.9),
        ("openfold3", "NVIDIA GeForce RTX 4090", 24, 0.9),
        ("opendde", "NVIDIA GeForce RTX 4090", 24, 0.9),
    ],
)
def test_model_card_exception_does_not_change_other_measured_policies(
    boltz_startup, model, kind, size, expected
):
    boltz_startup(((kind, size),))
    cli._apply_mem_fraction(None, models=(model,))
    assert oom.mem_fraction() == expected


def test_mixed_models_and_unknown_mixed_devices_keep_the_existing_fraction(
    monkeypatch, boltz_startup
):
    boltz_startup((("NVIDIA GeForce RTX 4090", 24),))
    cli._apply_mem_fraction(None, models=("boltz2", "protenix"))
    assert oom.mem_fraction() == 0.9
    monkeypatch.delenv(oom.CURRENT_FRACTION_ENV)
    boltz_startup((("NVIDIA GeForce RTX 4090", 24), ("NVIDIA GeForce RTX 3090", 24)))
    cli._apply_mem_fraction(None, models=("boltz2",))
    assert oom.mem_fraction() == 0.9


@pytest.mark.parametrize(
    ("selected", "cp_devices", "expected"),
    [
        (SimpleNamespace(platform="gpu", device_kind="RTX 4090", id=1), 1, 0.8),
        (SimpleNamespace(platform="gpu", device_kind="RTX 3090", id=0), 1, 0.9),
        (SimpleNamespace(platform="cpu", device_kind="cpu", id=0), 1, 0.9),
        (SimpleNamespace(platform="gpu", device_kind="RTX 4090", id=1), 2, 0.9),
    ],
)
def test_cpu_first_backend_uses_the_selected_device_kind_without_ordinal_inference(
    monkeypatch, boltz_startup, selected, cp_devices, expected
):
    boltz_startup((("RTX 3090", 24), ("RTX 4090", 24)))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu,cuda")
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(config=SimpleNamespace(jax_default_device=selected)),
    )
    cli._apply_mem_fraction(None, models=("boltz2",), cp_devices=cp_devices)
    assert oom.mem_fraction() == expected


@pytest.mark.parametrize("default", [None, "cpu", "gpu"])
def test_explicit_gpu_platform_beats_a_cpu_first_platform_list(
    monkeypatch, boltz_startup, default
):
    boltz_startup((("RTX 4090", 24),))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu,cuda")
    if default is not None:
        monkeypatch.setenv("JAX_DEFAULT_DEVICE", default)
    cli._apply_mem_fraction(None, models=("boltz2",))
    assert oom.mem_fraction() == (0.8 if default == "gpu" else 0.9)


def test_unknown_names_do_not_guess_the_card_from_its_capacity(
    monkeypatch, boltz_startup
):
    boltz_startup((("RTX 4090", 24),))
    monkeypatch.setattr(_cuda_memory, "visible_device_info", lambda: None)
    cli._apply_mem_fraction(None, models=("boltz2",))
    assert oom.mem_fraction() == 0.9


def test_boltz_policy_reaches_predict_and_warm_before_importing_jax():
    program = """
import os, sys
from foldjax import cli, _cuda_memory, oom
assert 'jax' not in sys.modules
_cuda_memory.visible_device_bytes = lambda: (24 * 2**30,)
_cuda_memory.visible_device_info = lambda: (('NVIDIA GeForce RTX 4090', 24 * 2**30),)
cli.tools_cli.prepare = lambda args: None
def dispatch(args):
    assert 'jax' not in sys.modules
    assert os.environ[oom.CURRENT_FRACTION_ENV] == '0.8'
    return 0
cli.tools_cli.dispatch = dispatch
for prefix in (['predict'], ['cache', 'warm']):
    os.environ.pop(oom.CURRENT_FRACTION_ENV, None)
    assert cli.main(prefix + ['--model', 'boltz2', '--input', 'job.yaml']) == 0
"""
    environment = dict(os.environ, JAX_PLATFORMS="cuda")
    for name in (oom.CURRENT_FRACTION_ENV, oom.FRACTION_ENV, "JAX_DEFAULT_DEVICE"):
        environment.pop(name, None)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
