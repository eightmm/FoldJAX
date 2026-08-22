"""Numerical contracts for Fold-CP atom reductions and masked attention."""

from __future__ import annotations

import subprocess
import sys
import textwrap

from tests.models.cp_probe_env import inherited_environment

_PROBE = textwrap.dedent(
    r"""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_atom import (
        pair_bias_attention_2d,
        scatter_atoms_to_tokens_mean_cp,
    )

    assert jax.device_count() == 4
    rng = np.random.default_rng(20260820)

    indices = jnp.asarray([np.arange(32) % 8], dtype=jnp.int32)
    valid = jnp.ones((1, 32), dtype=bool)
    values = jnp.asarray(rng.normal(size=(1, 32, 4)), dtype=jnp.bfloat16)
    reference = scatter_atoms_to_tokens_mean_cp(
        values, indices, valid, num_tokens=8
    )
    with context_parallel(4, layout="2d"):
        distributed = jax.jit(
            lambda x: scatter_atoms_to_tokens_mean_cp(
                x, indices, valid, num_tokens=8
            )
        )(values)
    np.testing.assert_array_equal(
        jax.device_get(reference), jax.device_get(distributed)
    )
    assert distributed.dtype == jnp.bfloat16

    q = jnp.asarray(rng.normal(size=(1, 8, 2, 3)), dtype=jnp.float32)
    k = jnp.asarray(rng.normal(size=(1, 8, 2, 3)), dtype=jnp.float32)
    v = jnp.asarray(rng.normal(size=(1, 8, 2, 3)), dtype=jnp.float32)
    bias = jnp.asarray(rng.normal(size=(1, 2, 8, 8)), dtype=jnp.float32)
    empty_mask = jnp.zeros((1, 8), dtype=jnp.float32)
    with context_parallel(4, layout="2d"):
        empty = jax.jit(
            lambda qv, kv, vv, bv, mv: pair_bias_attention_2d(
                qv, kv, vv, bv, mv, scale=3 ** -0.5
            )
        )(q, k, v, bias, empty_mask)
    got = jax.device_get(empty)
    np.testing.assert_array_equal(got, np.zeros_like(got))
    assert np.isfinite(got).all()
    print("ATOM_NUMERICS_OK")
    """
)


def test_atom_cp_reductions_and_empty_attention_are_stable() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ATOM_NUMERICS_OK" in completed.stdout
