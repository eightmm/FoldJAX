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

import json
import subprocess
import sys

import pytest

from tests.models.cp_probe_env import inherited_environment


def _gpu_devices():
    import jax

    try:
        return jax.devices("gpu")
    except RuntimeError:
        return []


requires_gpu = pytest.mark.skipif(
    not _gpu_devices(), reason="no GPU device visible to JAX"
)


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
    """Checked in a fresh process so an inherited value cannot mask it.

    The import sets the fraction and deliberately leaves preallocation alone.
    It used to force ``XLA_PYTHON_CLIENT_PREALLOCATE=false``; at 3,012 tokens
    with a cold cache that is the difference between the run dying on a
    54.34 GiB request at a 28.5 GiB high-water mark and completing at a
    59.8 GiB peak. Asserting the *absence* is the point -- a re-added default
    would reinstate the wall, and only this direction catches it.
    """
    code = (
        "import foldjax.models.openfold3, os; "
        "print(os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', '<unset>'), "
        "os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"JAX_PLATFORMS": "cpu", **inherited_environment()},
    )
    assert result.stdout.split() == ["<unset>", "0.90"]


@requires_gpu
def test_the_pool_is_actually_taken_up_front() -> None:
    """The absence assertion above is not enough on its own.

    Removing the `PREALLOCATE=false` default leaves JAX's own default, and
    measured on this version that is byte-identical to setting `"true"`:
    88,108 MiB held either way against 586 MiB with preallocation off. But
    "identical today" is an inherited property, not a stated one, and the
    thing it decides is whether a 3,012-token run completes at all. If JAX
    ever flips that default, the port silently regains a wall that took a
    full cold-cache A/B to attribute.

    So assert the *effect* rather than the spelling. `pool_bytes` is the
    discriminator -- 91,774,311,424 preallocated against 2,097,152 grown on
    demand -- while `bytes_limit` reads the same under both and would pass a
    test that never measured anything.
    """
    code = (
        "import foldjax.models.openfold3, jax, jax.numpy as jnp, json; "
        "jnp.zeros(1).block_until_ready(); "
        "s = jax.devices()[0].memory_stats() or {}; "
        "print(json.dumps([s.get('pool_bytes'), s.get('bytes_limit')]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=inherited_environment(),
    )
    pool, limit = json.loads(result.stdout.strip().splitlines()[-1])
    assert pool is not None and limit, result.stdout
    assert pool > 0.5 * limit, (
        f"pool_bytes {pool} is only {pool / limit:.1%} of bytes_limit {limit}: "
        "the allocator is growing on demand, which is what made OpenFold3 fail "
        "at 3,012 tokens while the card still had room"
    )


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
