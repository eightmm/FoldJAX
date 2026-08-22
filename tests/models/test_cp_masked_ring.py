"""Numerical edge cases for the distributed online softmax."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tests.models.cp_probe_env import inherited_environment

_PROBE = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import ring_triangle_attention_2d

    devices = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    assert jax.device_count() == devices, jax.devices()
    side = int(round(devices ** 0.5))
    assert side * side == devices

    rng = np.random.default_rng(11)
    batch, outer, heads, tokens, channels = 1, 5, 2, 7, 3

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)

    q = arr(batch, outer, heads, tokens, channels)
    k = arr(batch, outer, heads, tokens, channels)
    v = arr(batch, outer, heads, tokens, channels)
    bias = arr(batch, 1, heads, tokens, tokens)

    mask = np.zeros((batch, outer, 1, 1, tokens), dtype=np.float32)
    # One query row is globally empty. Another has an empty first ring tile
    # but valid keys later, which is the -inf/-inf online-update corner.
    mask[:, 0, :, :, :] = -np.inf
    mask[:, 1, :, :, : max(1, tokens // side)] = -np.inf
    mask = jnp.asarray(mask)

    def dense(q_in, k_in, v_in, bias_in, mask_in):
        scores = (
            jnp.einsum("...hqd,...hkd->...hqk", q_in, k_in)
            + bias_in
            + mask_in
        )
        maximum = jnp.max(scores, axis=-1, keepdims=True)
        valid = jnp.isfinite(maximum)
        probabilities = jnp.exp(
            jnp.where(valid, scores - maximum, -jnp.inf)
        )
        normalizer = jnp.sum(probabilities, axis=-1, keepdims=True)
        output = jnp.einsum(
            "...hqk,...hkd->...hqd",
            probabilities,
            v_in,
        )
        return jnp.where(
            normalizer > 0,
            output / jnp.maximum(
                normalizer,
                jnp.finfo(jnp.float32).tiny,
            ),
            jnp.zeros_like(output),
        )

    reference = jax.device_get(jax.jit(dense)(q, k, v, bias, mask))
    with context_parallel(devices, layout="2d"):
        compiled = jax.jit(ring_triangle_attention_2d)
        result = compiled(q, k, v, bias, mask)
        got = jax.device_get(result)
        hlo = compiled.lower(
            q,
            k,
            v,
            bias,
            mask,
        ).compiler_ir(dialect="hlo").as_hlo_text().lower()

    np.testing.assert_allclose(reference, got, atol=4e-5, rtol=4e-5)
    assert np.isfinite(got).all()
    assert np.all(got[:, 0] == 0)
    assert "all-gather" not in hlo and "all_gather" not in hlo
    print("MASKED_RING_OK")
    """
)


@pytest.mark.parametrize("devices", [4, 9])
def test_empty_ring_tiles_remain_finite_and_exact(devices: int) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
            "FOLDJAX_CP_PROBE_DEVICES": str(devices),
            **inherited_environment(),
        },
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "MASKED_RING_OK" in completed.stdout
