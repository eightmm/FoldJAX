"""The selected execution device governs admission, cache identity and AF3."""

from contextlib import nullcontext
from types import SimpleNamespace

import jax
import pytest

from foldjax import _gpu, api
from foldjax.cache import runtime_profile
from foldjax.schema import PredictionRequest


@pytest.mark.parametrize(
    "model", ["alphafold3", "boltz2", "protenix", "openfold3", "opendde", "esmfold2"]
)
def test_sm75_is_rejected_before_backend_selection(monkeypatch, tmp_path, model):
    device = SimpleNamespace(
        platform="gpu", device_kind="Quadro RTX 8000", compute_capability="7.5"
    )
    monkeypatch.setattr(_gpu, "default_device", lambda request=None: device)
    monkeypatch.setattr(api, "get_backend", lambda model: pytest.fail("backend loaded"))
    input_path = tmp_path / "input.yaml"
    input_path.write_text("{}")
    request = PredictionRequest(model=model, input=input_path, seed=101)
    with pytest.raises(ValueError, match="FoldJAX.*sm_80.*sm_75") as caught:
        api._predict_once(request, 101, tmp_path / "out")
    assert "triangle_kernel=xla" not in str(caught.value)
    assert not (tmp_path / "out").exists()


def test_cpu_default_with_selected_cuda_has_cuda_runtime_identity(monkeypatch):
    cpu = SimpleNamespace(platform="cpu", device_kind="cpu", id=0)
    devices = [
        SimpleNamespace(platform="gpu", device_kind="RTX A6000", id=i) for i in range(2)
    ]
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=devices[1]))
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(
        jax, "devices", lambda backend=None: devices if backend == "gpu" else [cpu]
    )
    monkeypatch.setattr(
        jax,
        "local_devices",
        lambda backend=None: devices if backend == "gpu" else [cpu],
    )
    monkeypatch.setattr(jax, "process_count", lambda: 1)
    monkeypatch.setattr(
        jax,
        "local_device_count",
        lambda backend=None: len(devices) if backend == "gpu" else 1,
    )
    profile = runtime_profile()
    assert profile["platform"] == "gpu"
    assert profile["device_kind"] == "RTX A6000"


def test_af3_without_device_options_respects_the_selected_gpu(monkeypatch, tmp_path):
    (tmp_path / "input.json").write_text("{}")
    cpu = SimpleNamespace(platform="cpu", device_kind="cpu", id=0)
    gpu = SimpleNamespace(platform="gpu", device_kind="RTX A6000", id=1)
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=gpu))
    monkeypatch.setattr(jax, "local_devices", lambda backend=None: [cpu])
    request = PredictionRequest(model="alphafold3", input=tmp_path / "input.json")
    assert _gpu.default_device(request) is gpu


def test_explicit_af3_device_option_overrides_the_device_context(monkeypatch, tmp_path):
    (tmp_path / "input.json").write_text("{}")
    devices = [SimpleNamespace(platform="gpu", id=i) for i in range(2)]
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=devices[1]))
    monkeypatch.setattr(jax, "local_devices", lambda backend=None: devices)
    request = PredictionRequest(
        model="alphafold3", input=tmp_path / "input.json", options={"device": 0}
    )
    assert _gpu.default_device(request) is devices[0]


def test_af3_default_attention_follows_the_selected_device(monkeypatch):
    from foldjax.backends.alphafold3 import _default_attention_backend

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(_gpu, "default_device", lambda: SimpleNamespace(platform="gpu"))
    assert _default_attention_backend(None) == "triton"


def test_pallas_defaults_follow_the_selected_device(monkeypatch):
    from foldjax.models import _pallas_pair

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(_gpu, "default_device", lambda: SimpleNamespace(platform="gpu"))
    assert _pallas_pair.gpu_process()


def test_supported_cuda_can_run_with_a_cpu_process_default(monkeypatch):
    from foldjax.models import _pallas_pair

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(
        _gpu,
        "default_device",
        lambda: SimpleNamespace(
            platform="gpu", compute_capability="8.6", device_kind="RTX A6000"
        ),
    )
    _pallas_pair._require_gpu("pair", "select an XLA backend")


def test_memory_record_uses_the_selected_gpu_and_restores_device_scope(
    monkeypatch, tmp_path
):
    from foldjax import memory_policy

    gpu = SimpleNamespace(
        platform="gpu", device_kind="RTX A6000", id=1, compute_capability="8.6"
    )
    monkeypatch.setattr(_gpu, "default_device", lambda request=None: gpu)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(jax, "default_device", lambda device: nullcontext())
    # Exercise the device scope at the real dispatch boundary without launching
    # kernels or substituting a fake GPU for a real JAX device.
    seen = {}

    def observe(*args, **kwargs):
        memory_policy.admit_unmeasured(
            model="alphafold3",
            detail="no law was fitted to this model",
            budget=memory_policy.resolve_budget(
                pool_bytes=None, card_bytes=None, override_gib=None
            ),
        )
        seen.update(memory_policy.recorded())

    monkeypatch.setattr(api, "_predict_once_on_device", observe)
    (tmp_path / "input.json").write_text("{}")
    request = PredictionRequest(model="alphafold3", input=tmp_path / "input.json")
    memory_policy.reset_warnings()
    memory_policy.clear_record()
    try:
        with pytest.warns(RuntimeWarning, match="no alphafold3 peak law"):
            api._predict_once(request, 101, tmp_path / "out")
        assert seen["selected"] is None
        assert seen["selected_device"] == {
            "platform": "gpu",
            "device_kind": "RTX A6000",
            "id": 1,
        }
        memory_policy.admit_unmeasured(
            model="alphafold3",
            detail="no law was fitted to this model",
            budget=memory_policy.resolve_budget(
                pool_bytes=None, card_bytes=None, override_gib=None
            ),
        )
        assert "selected_device" not in memory_policy.recorded()
    finally:
        memory_policy.reset_warnings()
        memory_policy.clear_record()
