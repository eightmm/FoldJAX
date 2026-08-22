"""The package's public surface and its process-level settings.

The allocator settings are applied on import, because they are process-local
and have to be in place before JAX initialises a device.

Matmul precision is *not*, and that is the point of the two tests below.
`jax.config.update` is process-global, so a port that pins precision on import
re-specifies the numerics of every other model sharing the process. This port
was vendored beside four others into one package and one pytest session, so it
scopes the setting to its own entry point instead.
"""

from __future__ import annotations

import subprocess
import sys

from tests.models.cp_probe_env import inherited_environment


def test_public_api_is_importable_and_complete() -> None:
    import foldjax.models.openfold3

    for name in foldjax.models.openfold3.__all__:
        assert hasattr(foldjax.models.openfold3, name), name
    assert foldjax.models.openfold3.__version__


def test_importing_does_not_change_global_matmul_precision() -> None:
    """Importing one model must not re-specify every other model's numerics.

    Checked in a fresh process, because by the time this module runs in a full
    session something else may already have set it -- in which case an
    in-process check would pass without the guarantee holding.
    """
    code = (
        "import jax; before = jax.config.jax_default_matmul_precision; "
        "import foldjax.models.openfold3; "
        "print(before, jax.config.jax_default_matmul_precision)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"JAX_PLATFORMS": "cpu", **inherited_environment()},
    )
    before, after = result.stdout.split()
    assert before == after == "None"


def test_predict_runs_under_the_precision_upstream_ships() -> None:
    """Upstream sets `set_float32_matmul_precision("high")`; so does the port.

    JAX's matmul precision is process-global and invisible on CPU, where the
    parity gate runs, so it has to be pinned rather than inherited -- but pinned
    to what upstream runs, not to something stricter. The guarantee moved from
    import time to `predict`, so this checks the decorator is actually on it
    rather than that a global was set, and it has to hold during *tracing*,
    which is where `compile_predict`'s `jax.jit` reads it.
    """
    import jax

    from foldjax.models.openfold3 import openfold3_precision, predict

    assert predict.__wrapped__ is not None, "predict is not wrapped"

    seen = []

    @openfold3_precision
    def record():
        seen.append(jax.config.jax_default_matmul_precision)

    outside_before = jax.config.jax_default_matmul_precision
    record()
    assert seen == ["high"]
    # And it is a scope, not a latch: leaving it puts back whatever the rest of
    # the session was using, which is the whole reason it is not a global.
    assert jax.config.jax_default_matmul_precision == outside_before


def test_allocator_env_is_set_on_import() -> None:
    """Checked in a fresh process so an inherited value cannot mask it."""
    code = (
        "import foldjax.models.openfold3, os; "
        "print(os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'], "
        "os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"JAX_PLATFORMS": "cpu", **inherited_environment()},
    )
    assert result.stdout.split() == ["false", "0.90"]


def test_explicit_env_settings_are_not_overridden() -> None:
    """setdefault, not assignment: an operator's choice must survive."""
    code = (
        "import foldjax.models.openfold3, os; "
        "print(os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={
            **inherited_environment(),
            "JAX_PLATFORMS": "cpu",
            # Splatted first on purpose: this test asserts an operator's
            # explicit setting survives, so nothing inherited may outrank it.
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.55",
        },
    )
    assert result.stdout.strip() == "0.55"
