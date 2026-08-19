"""Direct CP parity and HLO gates for the Protenix pair core."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


_PROBE = textwrap.dedent(
    r"""
    import os

    os.environ["PROTENIX_TRIANGLE_BACKEND"] = "xla"
    os.environ["PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND"] = "xla"

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, cp_layout
    from foldjax.models.protenix.models.primitives.attention import (
        AttentionParams,
    )
    from foldjax.models.protenix.models.primitives.primitives import (
        LayerNormParams,
        LinearParams,
    )
    from foldjax.models.protenix.models.triangle.triangle import (
        TriangleAttentionParams,
        TriangleMultiplicationParams,
        triangle_attention,
        triangle_multiplication,
    )

    devices = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    layout = os.environ["FOLDJAX_CP_PROBE_LAYOUT"]
    assert jax.device_count() == devices, jax.devices()

    channels, heads, tokens = 8, 2, 13
    rng = np.random.default_rng(20260820)

    def arr(*shape):
        return jnp.asarray(
            rng.normal(size=shape, scale=0.35),
            dtype=jnp.float32,
        )

    def linear(out_channels, in_channels, *, bias=True):
        return LinearParams(
            weight=arr(out_channels, in_channels),
            bias=arr(out_channels) if bias else None,
        )

    def norm(width):
        return LayerNormParams(
            weight=arr(width) * 0.1 + 1.0,
            bias=arr(width) * 0.1,
        )

    def multiplication():
        return TriangleMultiplicationParams(
            layer_norm_in=norm(channels),
            layer_norm_out=norm(channels),
            linear_a_p=linear(channels, channels),
            linear_a_g=linear(channels, channels),
            linear_b_p=linear(channels, channels),
            linear_b_g=linear(channels, channels),
            linear_z=linear(channels, channels),
            linear_g=linear(channels, channels),
        )

    attention_params = AttentionParams(
        linear_q=linear(channels, channels),
        linear_k=linear(channels, channels),
        linear_v=linear(channels, channels),
        linear_o=linear(channels, channels),
        linear_g=linear(channels, channels),
    )
    triangle_attention_params = TriangleAttentionParams(
        layer_norm=norm(channels),
        linear=linear(heads, channels),
        attention=attention_params,
    )
    mul_out = multiplication()
    mul_in = multiplication()

    keep = rng.random(tokens) > 0.15
    pair_mask = jnp.asarray(
        keep[:, None] & keep[None, :],
        dtype=jnp.float32,
    )
    z = arr(tokens, tokens, channels)
    traced = []

    def build():
        def run(z_in):
            traced.append(cp_layout())
            value = z_in + triangle_multiplication(
                z_in,
                pair_mask,
                mul_out,
                "outgoing",
                chunk_size=0,
            )
            value = value + triangle_multiplication(
                value,
                pair_mask,
                mul_in,
                "incoming",
                chunk_size=0,
            )
            value = value + triangle_attention(
                value,
                pair_mask,
                triangle_attention_params,
                num_heads=heads,
                starting=True,
                attention_backend="xla",
            )
            value = value + triangle_attention(
                value,
                pair_mask,
                triangle_attention_params,
                num_heads=heads,
                starting=False,
                attention_backend="xla",
            )
            return value

        return run

    reference = jax.device_get(jax.jit(build())(z))
    jax.clear_caches()
    with context_parallel(devices, layout=layout):
        compiled = jax.jit(build())
        result = compiled(z)
        got = jax.device_get(result)
        hlo = compiled.lower(z).compiler_ir(dialect="hlo").as_hlo_text().lower()

    assert traced[-2:] == [None, layout], traced
    np.testing.assert_allclose(reference, got, atol=5e-5, rtol=5e-5)
    if layout == "2d":
        assert (
            "collective-permute" in hlo
            or "collective_permute" in hlo
        ), hlo
        assert "all-gather" not in hlo and "all_gather" not in hlo, hlo
    print("PROTENIX_CP_OK")
    """
)


@pytest.mark.parametrize(
    ("devices", "layout"),
    [(4, "1d"), (4, "2d"), (9, "2d")],
)
def test_protenix_pair_core_matches_serial(
    devices: int,
    layout: str,
) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
            "FOLDJAX_CP_PROBE_DEVICES": str(devices),
            "FOLDJAX_CP_PROBE_LAYOUT": layout,
            "PATH": "/usr/bin",
        },
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PROTENIX_CP_OK" in completed.stdout
