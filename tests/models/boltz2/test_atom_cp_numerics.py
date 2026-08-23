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
    from jax.sharding import NamedSharding, PartitionSpec

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_atom import (
        pair_bias_attention_2d,
        scatter_atoms_to_tokens_mean_cp,
    )
    from foldjax.models.boltz2.models.diffusion.diffusion_transformer import (
        _attention_pair_bias_no_proj_z_forward,
    )
    from foldjax.models.boltz2.models.primitives.attention import (
        attention_pair_bias_forward,
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

    # The diffusion token attention has separate serial/1-D and 2-D
    # implementations. All three must agree that a globally empty key set
    # contributes zero, rather than the mean of V (finite mask surrogate) or
    # NaN (plain softmax over -inf).
    tokens, channels, heads = 8, 6, 2
    token_values = jnp.asarray(
        rng.normal(size=(1, tokens, channels)), dtype=jnp.float32
    )
    pair_values = jnp.asarray(
        rng.normal(size=(1, tokens, tokens, heads)), dtype=jnp.float32
    )
    empty_token_mask = jnp.zeros((1, tokens), dtype=jnp.float32)
    eye = jnp.eye(channels, dtype=jnp.float32)
    attention_params = {
        "proj_q": {"kernel": eye, "bias": jnp.zeros((channels,))},
        "proj_g": {"kernel": jnp.zeros_like(eye)},
        "proj_k": {"kernel": eye},
        "proj_v": {"kernel": eye},
        "proj_o": {"kernel": eye},
    }

    def token_attention(sv, bv, mv, *, backend="xla", chunk_size=None):
        return _attention_pair_bias_no_proj_z_forward(
            attention_params,
            s=sv,
            bias=bv,
            mask=mv,
            k_in=sv,
            multiplicity=1,
            inf=1e6,
            attention_backend=backend,
            chunk_size=chunk_size,
        )

    serial = jax.jit(lambda sv, bv, mv: token_attention(sv, bv, mv))(
        token_values,
        pair_values,
        empty_token_mask,
    )
    serial_chunked = jax.jit(
        lambda sv, bv, mv: token_attention(sv, bv, mv, chunk_size=3)
    )(token_values, pair_values, empty_token_mask)
    serial_fused = jax.jit(
        lambda sv, bv, mv: token_attention(sv, bv, mv, backend="tokamax")
    )(token_values, pair_values, empty_token_mask)
    for result in (serial, serial_chunked, serial_fused):
        got = jax.device_get(result)
        np.testing.assert_array_equal(got, np.zeros_like(got))
        assert np.isfinite(got).all()

    z_channels = 3
    trunk_params = {
        **attention_params,
        "proj_z_norm": {
            "scale": jnp.ones((z_channels,), dtype=jnp.float32),
            "bias": jnp.zeros((z_channels,), dtype=jnp.float32),
        },
        "proj_z": {
            "kernel": jnp.zeros((z_channels, heads), dtype=jnp.float32),
        },
    }
    trunk_pair = jnp.asarray(
        rng.normal(size=(1, tokens, tokens, z_channels)), dtype=jnp.float32
    )
    trunk = jax.jit(
        lambda sv, zv, mv: attention_pair_bias_forward(
            trunk_params,
            sv,
            zv,
            mv,
            chunk_size=3,
        )
    )(token_values, trunk_pair, empty_token_mask)
    got = jax.device_get(trunk)
    np.testing.assert_array_equal(got, np.zeros_like(got))
    assert np.isfinite(got).all()

    with context_parallel(4, layout="1d") as mesh:
        single = NamedSharding(mesh, PartitionSpec(None, "cp", None))
        pair_rows = NamedSharding(mesh, PartitionSpec(None, "cp", None, None))
        mask_rows = NamedSharding(mesh, PartitionSpec(None, "cp"))
        one_d = jax.jit(lambda sv, bv, mv: token_attention(sv, bv, mv))(
            jax.device_put(token_values, single),
            jax.device_put(pair_values, pair_rows),
            jax.device_put(empty_token_mask, mask_rows),
        )
    got = jax.device_get(one_d)
    np.testing.assert_array_equal(got, np.zeros_like(got))
    assert np.isfinite(got).all()

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
