"""Installation paths must not alter embedded Pallas-Triton cache inputs."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

REGEX = r".*/(site-packages|dist-packages|src)/"
ENVIRONMENT = "JAX_HLO_SOURCE_FILE_CANONICALIZATION_REGEX"
CONFIG = "jax_hlo_source_file_canonicalization_regex"


def _run(source: str, value: str | None = None) -> None:
    environment = {**os.environ, "JAX_PLATFORMS": "cpu"}
    environment.pop(ENVIRONMENT, None)
    if value is not None:
        environment[ENVIRONMENT] = value
    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("jax_first", [False, True])
def test_default_applies_before_lowering_without_importing_jax(jax_first):
    before = "import jax" if jax_first else "import sys"
    lazy = "" if jax_first else "assert 'jax' not in sys.modules"
    _run(
        f"{before}\nimport foldjax\n{lazy}\nimport jax\n"
        f"assert jax.config.{CONFIG} == {REGEX!r}"
    )


@pytest.mark.parametrize("jax_first", [False, True])
@pytest.mark.parametrize("value", ["", r"/site-policy/", REGEX])
def test_environment_override_including_empty_is_preserved(jax_first, value):
    before = "import jax" if jax_first else ""
    _run(
        f"{before}\nimport foldjax\nimport jax\n"
        f"assert jax.config.{CONFIG} == {value!r}",
        value,
    )


@pytest.mark.parametrize("value", ["", r"/explicit-config/"])
@pytest.mark.parametrize("environment", [None, r"/environment/"])
def test_config_override_wins_and_can_be_changed_after_import(value, environment):
    _run(
        "import jax\n"
        f"jax.config.update({CONFIG!r}, {value!r})\n"
        "import foldjax\n"
        f"assert jax.config.{CONFIG} == {value!r}\n"
        f"jax.config.update({CONFIG!r}, None)\n"
        f"assert jax.config.{CONFIG} is None",
        environment,
    )


def test_pallas_triton_serialization_is_independent_of_installation_path():
    import jax
    import numpy as np
    from jax._src import cache_key, config
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt
    from jax.sharding import AbstractDevice, AbstractMesh, AxisType, use_abstract_mesh

    # Cross-lower to CUDA on a CPU host; no GPU compilation or execution occurs.
    device = AbstractDevice("NVIDIA GeForce RTX 4090", 128, "gpu")
    mesh = AbstractMesh((1,), ("x",), (AxisType.Explicit,), abstract_device=device)
    paths = (
        "/checkout/src/foldjax/path_kernel.py",
        "/release/a/site-packages/foldjax/path_kernel.py",
        "/release/b/dist-packages/foldjax/path_kernel.py",
    )

    def serialized(path, increment=1):
        source = f"def kernel(x, y):\n    y[...] = x[...] + {increment}\n"
        namespace = {"__name__": "path_kernel"}
        exec(compile(source, path, "exec"), namespace)
        shape = jax.ShapeDtypeStruct((16,), np.float32)
        kernel = pl.pallas_call(
            namespace["kernel"], out_shape=shape, compiler_params=plt.CompilerParams()
        )
        with use_abstract_mesh(mesh):
            module = (
                jax.jit(kernel)
                .trace(shape)
                .lower(lowering_platforms=("cuda",))
                .compiler_ir()
            )
        assert "__gpu$xla.gpu.triton" in str(module)
        # JAX strips outer HLO debug info for its persistent cache key, but
        # source locations inside the serialized Triton custom call survive.
        return cache_key._canonicalize_ir(module, False)

    with config.hlo_source_file_canonicalization_regex(""):
        assert serialized(paths[0]) != serialized(paths[1])
    with config.hlo_source_file_canonicalization_regex(REGEX):
        results = [serialized(path) for path in paths]
        assert results[0] == results[1] == results[2]
        assert serialized(paths[0], increment=2) != results[0]
