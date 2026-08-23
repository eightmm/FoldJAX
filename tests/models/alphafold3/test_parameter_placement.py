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
