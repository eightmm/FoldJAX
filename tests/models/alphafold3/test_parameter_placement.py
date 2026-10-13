"""AlphaFold 3 weights follow the adapter's selected JAX device."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from tests.models.cp_probe_env import inherited_environment


def test_model_device_context_places_lazy_arrays_on_a_nondefault_device() -> None:
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp

        from foldjax.backends.alphafold3 import _model_device

        devices = jax.devices("cpu")
        assert len(devices) == 2, devices
        with _model_device(devices[1]):
            parameter = jnp.asarray([1.0, 2.0, 3.0])
        assert parameter.device == devices[1], (parameter.device, devices[1])
        print("AF3_PARAMETER_DEVICE_OK")
        """
    )
    environment = inherited_environment()
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout
    assert "AF3_PARAMETER_DEVICE_OK" in completed.stdout


def test_api_cache_and_manifest_follow_a_real_nondefault_device() -> None:
    script = textwrap.dedent(
        """
        import json
        import tempfile
        from pathlib import Path
        import jax
        import jax.numpy as jnp
        import foldjax
        from foldjax.api import resolve_cache_dir
        from foldjax.backends.alphafold3 import _default_attention_backend
        from foldjax._gpu import default_device
        from foldjax.registry import backend_override
        from tests.test_api import DummyBackend

        class SelectedBackend(DummyBackend):
            name = "alphafold3"
            def predict(self, request):
                assert jnp.ones((2,)).device == devices[1]
                return super().predict(request)

        devices = jax.devices("cpu")
        assert len(devices) == 2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input = root / "job.yaml"
            input.write_text("native input")
            weights = root / "weights"
            weights.mkdir()
            request = foldjax.PredictionRequest(
                model="alphafold3", input=input, input_format="native",
                weights=weights, output_dir=root / "out",
                cache_dir=root / "cache", seed=101
            )
            backend = SelectedBackend()
            with backend_override("alphafold3", lambda: backend):
                ordinary = resolve_cache_dir(request, backend)
                with jax.default_device(devices[1]):
                    assert default_device(request) == devices[1]
                    assert _default_attention_backend(None) == "xla"
                    selected = resolve_cache_dir(request, backend)
                    foldjax.predict(request)
                assert selected != ordinary
            manifest = json.loads((root / "out" / "foldjax_run.json").read_text())
            recorded = json.loads(manifest["runtime"]["selected_device"])
            assert recorded["id"] == devices[1].id
            assert manifest["runtime"]["platform"] == "cpu"
        assert jax.config.jax_default_device is None
        print("SELECTED_API_CACHE_MANIFEST_OK")
        """
    )
    environment = inherited_environment()
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout
    assert "SELECTED_API_CACHE_MANIFEST_OK" in completed.stdout
