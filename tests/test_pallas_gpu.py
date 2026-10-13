"""CUDA device identity comes from the device, not another card's marketing name."""

from types import SimpleNamespace

import jax
import pytest
from jax._src.pallas.triton import gpu_info

from foldjax import _pallas_gpu
from foldjax._gpu import (
    compatibility_notes,
    compute_capability,
    default_device,
    require_fused_cuda,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("8.6", (8, 6)),
        ("8.9", (8, 9)),
        ("12.1", (12, 1)),
        (86, (8, 6)),
        (120, (12, 0)),
        ((8, 6), (8, 6)),
        ("sm_75", (7, 5)),
        ("8.0", (8, 0)),
        (None, None),
        (True, None),
        ("gfx942", None),
        ("nan", None),
    ],
)
def test_compute_capability_formats(value, expected):
    assert compute_capability(SimpleNamespace(compute_capability=value)) == expected


@pytest.mark.parametrize(
    ("kind", "capability", "target"),
    [
        ("NVIDIA GeForce RTX 3090", "8.6", 86),
        ("NVIDIA RTX A5000", "8.6", 86),
        ("NVIDIA RTX A6000", "8.6", 86),
        ("NVIDIA GeForce RTX 4080", "8.9", 89),
        ("Unlisted CUDA device", "12.1", 121),
    ],
)
def test_unknown_cuda_kind_uses_its_actual_architecture(
    monkeypatch, kind, capability, target
):
    device = SimpleNamespace(
        platform="gpu", device_kind=kind, compute_capability=capability
    )
    monkeypatch.setattr(jax, "local_devices", lambda: [device])
    monkeypatch.setattr(gpu_info, "registry", {})
    assert _pallas_gpu.install() == 1
    assert _pallas_gpu.install() == 0
    info = gpu_info.registry[kind]()
    assert info.compute_capability == target
    assert info.arch_name == capability
    assert info.gpu_version is None
    monkeypatch.setattr(gpu_info, "get_device_kind", lambda: kind)
    monkeypatch.setattr(gpu_info, "get_device_platform", lambda: "gpu")
    gpu_info.get_gpu_info.cache_clear()
    try:
        assert gpu_info.get_gpu_info() == info
    finally:
        gpu_info.get_gpu_info.cache_clear()


def test_existing_names_and_custom_registry_entries_are_not_changed(monkeypatch):
    def callback():
        return "custom"

    monkeypatch.setattr(gpu_info, "registry", {"custom GPU": callback})
    devices = [
        SimpleNamespace(platform="gpu", device_kind=n, compute_capability="8.6")
        for n in ["NVIDIA A100", "custom GPU"]
    ]
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    assert _pallas_gpu.install() == 0
    assert gpu_info.registry == {"custom GPU": callback}


def test_cpu_rocm_and_unknown_architectures_are_not_registered(monkeypatch):
    devices = [
        SimpleNamespace(platform=p, device_kind=n, compute_capability=c)
        for p, n, c in [
            ("cpu", "cpu", "8.6"),
            ("rocm", "MI300X", "gfx942"),
            ("gpu", "unlisted", None),
        ]
    ]
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    monkeypatch.setattr(gpu_info, "registry", {})
    assert _pallas_gpu.install() == 0
    assert not gpu_info.registry


def test_multiple_device_kinds_keep_separate_targets(monkeypatch):
    devices = [
        SimpleNamespace(platform="gpu", device_kind=n, compute_capability=c)
        for n, c in [("unlisted Ampere", "8.6"), ("unlisted Ada", "8.9")]
    ]
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    monkeypatch.setattr(gpu_info, "registry", {})
    assert _pallas_gpu.install() == 2
    assert gpu_info.registry["unlisted Ampere"]().compute_capability == 86
    assert gpu_info.registry["unlisted Ada"]().compute_capability == 89


@pytest.mark.parametrize("explicit", [False, True])
def test_selected_cuda_device_is_registered_with_a_cpu_default(monkeypatch, explicit):
    device = SimpleNamespace(
        platform="gpu", device_kind="unlisted selected GPU", compute_capability="8.6"
    )
    monkeypatch.setattr(jax, "local_devices", lambda: [])
    monkeypatch.setattr(
        jax, "config", SimpleNamespace(jax_default_device=None if explicit else device)
    )
    monkeypatch.setattr(gpu_info, "registry", {})
    added = _pallas_gpu.install(device) if explicit else _pallas_gpu.install()
    assert added == 1
    assert gpu_info.registry[device.device_kind]().compute_capability == 86


def test_af3_registers_the_device_selected_by_its_platform_option(monkeypatch):
    from foldjax.backends import alphafold3

    device, scope = object(), object()
    seen = []
    monkeypatch.setattr(_pallas_gpu, "install", lambda chosen=None: seen.append(chosen))
    monkeypatch.setattr(jax, "default_device", lambda chosen: scope)
    assert alphafold3._model_device(device) is scope
    assert seen == [device]


def test_platform_string_resolves_a_concrete_device_for_registration(monkeypatch):
    device = SimpleNamespace(
        platform="gpu", device_kind="unlisted selected GPU", compute_capability="8.9"
    )
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device="gpu"))
    monkeypatch.setattr(
        jax, "local_devices", lambda backend=None: [device] if backend == "gpu" else []
    )
    monkeypatch.setattr(gpu_info, "registry", {})
    assert default_device() is device
    assert _pallas_gpu.install() == 1
    assert gpu_info.registry[device.device_kind]().compute_capability == 89


def test_turing_refusal_matches_the_global_gpu_support_policy(monkeypatch):
    device = SimpleNamespace(
        platform="gpu", device_kind="Quadro RTX 8000", compute_capability="7.5"
    )
    monkeypatch.setattr("foldjax._gpu.default_device", lambda: device)
    assert "sm_75" in compatibility_notes([device])[0]
    with pytest.raises(ValueError, match="FoldJAX.*sm_80") as caught:
        require_fused_cuda("Pallas pair kernel")
    assert "sm_75" in str(caught.value)
    assert "unsupported" in str(caught.value)
    assert "triangle_kernel=xla" not in str(caught.value)


@pytest.mark.parametrize("warmup", [False, True])
def test_prediction_and_warmup_install_before_runtime_validation(
    monkeypatch, tmp_path, warmup
):
    from foldjax import api
    from foldjax.schema import PredictionRequest

    input_path = tmp_path / "job.yaml"
    input_path.write_text("{}")
    seen = []
    monkeypatch.setattr(_pallas_gpu, "install", lambda: seen.append("installed"))

    def reject(request):
        assert seen == ["installed"]
        raise ValueError("validation sentinel")

    backend = SimpleNamespace(
        name="boltz2", capabilities=lambda: None, validate_request=reject
    )
    request = PredictionRequest(
        model="boltz2",
        input=input_path,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        seed=101,
    )
    with pytest.raises(ValueError, match="validation sentinel"):
        if warmup:
            from foldjax import warmup as cache_warm

            monkeypatch.setattr(cache_warm, "get_backend", lambda model: backend)
            monkeypatch.setattr(
                cache_warm, "resolve_cache_dir", lambda r, b: r.cache_dir
            )
            monkeypatch.setattr(
                cache_warm,
                "_predict_resolved",
                lambda r, **kwargs: api._predict_once(
                    r, 101, r.output_dir, backend=backend
                ),
            )
            cache_warm._warm_resolved(request, keep_output=True, output_root=None)
        else:
            api._predict_once(request, 101, request.output_dir, backend=backend)
