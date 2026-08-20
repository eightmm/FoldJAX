"""CPU-mesh proof gates for Fold-CP atom-window communication."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run(source: str, *, devices: int) -> str:
    env = {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "PATH": "/usr/bin",
    }
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


_HALO_PROBE = textwrap.dedent(
    r"""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_atom import place_atoms, single_to_keys_cp

    assert jax.device_count() == 4
    rng = np.random.default_rng(20260820)
    x = jnp.asarray(rng.normal(size=(2, 256, 3)), dtype=jnp.float32)
    reference = jax.device_get(single_to_keys_cp(x, query_window=32, key_window=128))
    with context_parallel(4, layout="2d"):
        x_distributed = place_atoms(x, atom_axis=1)
        assert x_distributed.sharding.shard_shape(x.shape) == (2, 128, 3)
        compiled = jax.jit(
            lambda value: single_to_keys_cp(
                value,
                query_window=32,
                key_window=128,
            )
        )
        out = compiled(x_distributed)
        got = jax.device_get(out)
        hlo = compiled.lower(x_distributed).compiler_ir(
            dialect="hlo"
        ).as_hlo_text().lower()
    np.testing.assert_array_equal(reference, got)
    assert "collective-permute" in hlo or "collective_permute" in hlo, hlo
    assert "all-gather" not in hlo and "all_gather" not in hlo, hlo
    assert out.sharding.shard_shape(out.shape) != out.shape
    print("ATOM_HALO_OK")
    """
)


_ROUTING_PROBE = textwrap.dedent(
    r"""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_atom import (
        gather_token_pairs_to_atom_windows_cp,
        gather_tokens_to_atoms_cp,
        scatter_atoms_to_tokens_mean_cp,
    )

    assert jax.device_count() == 4
    rng = np.random.default_rng(7)
    token_values = jnp.asarray(rng.normal(size=(1, 8, 4)), dtype=jnp.float32)
    atom_indices = jnp.asarray([np.arange(32) % 8], dtype=jnp.int32)
    atom_valid = jnp.ones((1, 32), dtype=bool)
    atom_values = jnp.asarray(rng.normal(size=(1, 32, 4)), dtype=jnp.float32)

    gather_ref = gather_tokens_to_atoms_cp(token_values, atom_indices, atom_valid)
    scatter_ref = scatter_atoms_to_tokens_mean_cp(
        atom_values,
        atom_indices,
        atom_valid,
        num_tokens=8,
    )

    pair = jnp.asarray(rng.normal(size=(1, 8, 8, 3)), dtype=jnp.float32)
    q_idx = atom_indices.reshape(1, 8, 4)
    q_valid = atom_valid.reshape(1, 8, 4)
    k_idx = jnp.stack(
        [jnp.roll(q_idx, shift) for shift in range(4)], axis=-1
    )[:, :, :, 0]
    # Build [B,K,H] keys independent of W for a non-degenerate lookup.
    k_idx = jnp.asarray([[(i + j) % 8 for j in range(8)] for i in range(8)])
    k_idx = k_idx[None]
    k_valid = jnp.ones_like(k_idx, dtype=bool)
    pair_ref = gather_token_pairs_to_atom_windows_cp(
        pair,
        q_idx,
        q_valid,
        k_idx,
        k_valid,
    )

    with context_parallel(4, layout="2d"):
        gather_fn = jax.jit(gather_tokens_to_atoms_cp)
        scatter_fn = jax.jit(
            lambda values, indices, valid: scatter_atoms_to_tokens_mean_cp(
                values,
                indices,
                valid,
                num_tokens=8,
            )
        )
        pair_fn = jax.jit(gather_token_pairs_to_atom_windows_cp)
        gather_out = gather_fn(token_values, atom_indices, atom_valid)
        scatter_out = scatter_fn(atom_values, atom_indices, atom_valid)
        pair_out = pair_fn(pair, q_idx, q_valid, k_idx, k_valid)
        gather_hlo = gather_fn.lower(
            token_values, atom_indices, atom_valid
        ).compiler_ir(dialect="hlo").as_hlo_text().lower()
        scatter_hlo = scatter_fn.lower(
            atom_values, atom_indices, atom_valid
        ).compiler_ir(dialect="hlo").as_hlo_text().lower()
        pair_hlo = pair_fn.lower(
            pair, q_idx, q_valid, k_idx, k_valid
        ).compiler_ir(dialect="hlo").as_hlo_text().lower()

    np.testing.assert_allclose(gather_ref, jax.device_get(gather_out), atol=0, rtol=0)
    np.testing.assert_allclose(
        scatter_ref,
        jax.device_get(scatter_out),
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(pair_ref, jax.device_get(pair_out), atol=0, rtol=0)
    assert "collective-permute" in gather_hlo or "collective_permute" in gather_hlo
    assert "reduce-scatter" in scatter_hlo or "reduce_scatter" in scatter_hlo
    assert "all-gather" not in pair_hlo and "all_gather" not in pair_hlo
    print("ATOM_ROUTING_OK")
    """
)


_PAIR_BIAS_PROBE = textwrap.dedent(
    r"""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_atom import pair_bias_attention_2d

    assert jax.device_count() == 4
    rng = np.random.default_rng(11)
    q = jnp.asarray(rng.normal(size=(1, 8, 2, 3)), dtype=jnp.float32)
    k = jnp.asarray(rng.normal(size=(1, 8, 2, 3)), dtype=jnp.float32)
    v = jnp.asarray(rng.normal(size=(1, 8, 2, 3)), dtype=jnp.float32)
    bias = jnp.asarray(rng.normal(size=(1, 2, 8, 8)), dtype=jnp.float32)
    mask = jnp.asarray([[1, 1, 1, 1, 1, 0, 1, 1]], dtype=jnp.float32)
    scale = 3 ** -0.5

    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale + bias
    logits = logits + (1 - mask[:, None, None]) * -1e6
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    reference = jnp.einsum("bhqk,bkhd->bqhd", probs, v)

    with context_parallel(4, layout="2d"):
        compiled = jax.jit(
            lambda qv, kv, vv, bv, mv: pair_bias_attention_2d(
                qv, kv, vv, bv, mv, scale=scale
            )
        )
        out = compiled(q, k, v, bias, mask)
        hlo = compiled.lower(q, k, v, bias, mask).compiler_ir(
            dialect="hlo"
        ).as_hlo_text().lower()
    np.testing.assert_allclose(reference, jax.device_get(out), atol=3e-5, rtol=3e-5)
    assert "all-gather" not in hlo and "all_gather" not in hlo, hlo
    assert "all-reduce" in hlo or "all_reduce" in hlo, hlo
    print("PAIR_BIAS_CP_OK")
    """
)


@pytest.mark.parametrize("source", [_HALO_PROBE, _ROUTING_PROBE, _PAIR_BIAS_PROBE])
def test_atom_context_parallel_primitives(source: str) -> None:
    assert "_OK" in _run(source, devices=4)
