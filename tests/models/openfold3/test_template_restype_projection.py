"""Residue-type projections stay exact when moved before pair broadcasting."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.data import (
    collapse_identical_templates,
    compact_zero_template_pair_features,
)
from foldjax.models.openfold3.models import template_module
from foldjax.models.openfold3.models.primitives import LinearParams, linear
from tests.models.cp_probe_env import inherited_environment
from tests.models.openfold3.test_template_collapse import (
    N_TOKEN,
    _embedder_params,
    _empty_templates,
    _model_batch,
)


def _pair_wide_projection(
    restype: jax.Array,
    aatype_linear_1: LinearParams,
    aatype_linear_2: LinearParams,
) -> tuple[jax.Array, jax.Array]:
    """The historical project-after-broadcast implementation."""
    n_token = restype.shape[-2]
    restype_ti = jnp.broadcast_to(
        restype[..., :, None, :], (*restype.shape[:-1], n_token, restype.shape[-1])
    )
    restype_tj = jnp.broadcast_to(
        restype[..., None, :, :], (*restype.shape[:-2], n_token, *restype.shape[-2:])
    )
    return (
        linear(restype_ti, aatype_linear_1),
        linear(restype_tj, aatype_linear_2),
    )


def _broadcast_projected_tokens(
    restype: jax.Array,
    aatype_linear_1: LinearParams,
    aatype_linear_2: LinearParams,
) -> tuple[jax.Array, jax.Array]:
    restype_ti, restype_tj = template_module._project_template_restype(
        restype, aatype_linear_1, aatype_linear_2
    )
    n_token = restype.shape[-2]
    pair_shape = (
        *restype.shape[:-2],
        n_token,
        n_token,
        restype_ti.shape[-1],
    )
    return (
        jnp.broadcast_to(restype_ti, pair_shape),
        jnp.broadcast_to(restype_tj, pair_shape),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    ("leading", "n_template", "n_token"),
    [
        ((), 1, 1),
        ((2,), 3, 5),
        ((2, 3), 2, 4),
    ],
)
def test_token_projection_is_bitwise_equal_to_the_pair_wide_oracle(
    dtype: jnp.dtype,
    leading: tuple[int, ...],
    n_template: int,
    n_token: int,
) -> None:
    keys = jax.random.split(jax.random.key(83), 3)
    restype = jax.random.normal(
        keys[0], (*leading, n_template, n_token, 32), dtype=dtype
    )
    linear_1 = LinearParams(
        weight=jax.random.normal(keys[1], (64, 32), dtype=dtype)
    )
    linear_2 = LinearParams(
        weight=jax.random.normal(keys[2], (64, 32), dtype=dtype)
    )

    expected = jax.jit(_pair_wide_projection)(restype, linear_1, linear_2)
    actual = jax.jit(_broadcast_projected_tokens)(restype, linear_1, linear_2)

    for expected_term, actual_term in zip(expected, actual, strict=True):
        assert np.asarray(actual_term).tobytes() == np.asarray(expected_term).tobytes()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
def test_signed_zero_and_nonfinite_semantics_match_the_pair_wide_oracle(
    dtype: jnp.dtype,
) -> None:
    restype = jnp.asarray([[[0.0], [-0.0], [jnp.inf], [jnp.nan]]], dtype=dtype)
    linear_1 = LinearParams(weight=jnp.asarray([[1.0], [-1.0]], dtype=dtype))
    linear_2 = LinearParams(weight=jnp.asarray([[-1.0], [1.0]], dtype=dtype))

    expected = jax.jit(_pair_wide_projection)(restype, linear_1, linear_2)
    actual = jax.jit(_broadcast_projected_tokens)(restype, linear_1, linear_2)

    expected_flat = np.concatenate(
        [np.asarray(term, dtype=np.float32).ravel() for term in expected]
    )
    actual_flat = np.concatenate(
        [np.asarray(term, dtype=np.float32).ravel() for term in actual]
    )
    for expected_term, actual_term in zip(expected, actual, strict=True):
        assert np.asarray(actual_term).tobytes() == np.asarray(expected_term).tobytes()
    np.testing.assert_array_equal(np.isnan(actual_flat), np.isnan(expected_flat))
    np.testing.assert_array_equal(np.isposinf(actual_flat), np.isposinf(expected_flat))
    np.testing.assert_array_equal(np.isneginf(actual_flat), np.isneginf(expected_flat))
    finite_zero = np.isfinite(actual_flat) & (actual_flat == 0)
    assert finite_zero.any()
    assert set(np.signbit(actual_flat[finite_zero])) == {False, True}


@pytest.mark.parametrize("compact", [False, True], ids=["dense", "compact"])
def test_full_template_embedder_matches_the_pair_wide_oracle(
    compact: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENFOLD3_TRIANGLE_BACKEND", "xla")
    params = _embedder_params()
    source = collapse_identical_templates(_empty_templates())
    if compact:
        source = compact_zero_template_pair_features(source)
    batch = _model_batch(source)
    z = jax.random.normal(
        jax.random.key(89), (1, N_TOKEN, N_TOKEN, 8), dtype=jnp.float32
    )
    pair_mask = jnp.ones((1, N_TOKEN, N_TOKEN), dtype=jnp.float32)

    def run(
        input_batch: dict[str, jax.Array],
        pair: jax.Array,
        weights: template_module.TemplateEmbedderParams,
        input_pair_mask: jax.Array,
    ) -> jax.Array:
        return template_module.template_embedder(
            input_batch,
            pair,
            weights,
            pair_mask=input_pair_mask,
            no_heads=2,
        )

    actual = np.asarray(jax.jit(run)(batch, z, params, pair_mask))
    jax.clear_caches()
    monkeypatch.setattr(
        template_module, "_project_template_restype", _pair_wide_projection
    )
    expected = np.asarray(jax.jit(run)(batch, z, params, pair_mask))

    assert actual.tobytes() == expected.tobytes()


def test_stablehlo_dots_consume_token_wide_not_pair_wide_restype() -> None:
    restype = jnp.zeros((2, 3, 5, 32), dtype=jnp.float32)
    linear_1 = LinearParams(weight=jnp.zeros((64, 32), dtype=jnp.float32))
    linear_2 = LinearParams(weight=jnp.zeros((64, 32), dtype=jnp.float32))

    new_hlo = (
        jax.jit(template_module._project_template_restype)
        .lower(restype, linear_1, linear_2)
        .as_text()
    )
    old_hlo = (
        jax.jit(_pair_wide_projection)
        .lower(restype, linear_1, linear_2)
        .as_text()
    )

    token_dot = (
        "(tensor<2x3x5x32xf32>, tensor<32x64xf32>) -> "
        "tensor<2x3x5x64xf32>"
    )
    pair_dot = (
        "(tensor<2x3x5x5x32xf32>, tensor<32x64xf32>) -> "
        "tensor<2x3x5x5x64xf32>"
    )
    assert new_hlo.count(token_dot) == 2
    assert pair_dot not in new_hlo
    assert old_hlo.count(pair_dot) == 2


_CONTEXT_PARALLEL_PROBE = textwrap.dedent(
    """
    import os
    os.environ["OPENFOLD3_TRIANGLE_BACKEND"] = "xla"

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, cp_layout
    from foldjax.models.openfold3.data import collapse_identical_templates
    from foldjax.models.openfold3.models import template_module
    from foldjax.models.openfold3.models.primitives import linear
    from tests.models.openfold3.test_template_collapse import (
        N_TOKEN, _embedder_params, _empty_templates, _model_batch,
    )

    assert jax.device_count() == 4, jax.devices()
    optimized_projection = template_module._project_template_restype

    def pair_wide_projection(restype, aatype_linear_1, aatype_linear_2):
        n_token = restype.shape[-2]
        restype_ti = jnp.broadcast_to(
            restype[..., :, None, :],
            (*restype.shape[:-1], n_token, restype.shape[-1]),
        )
        restype_tj = jnp.broadcast_to(
            restype[..., None, :, :],
            (*restype.shape[:-2], n_token, *restype.shape[-2:]),
        )
        return linear(restype_ti, aatype_linear_1), linear(
            restype_tj, aatype_linear_2
        )

    params = _embedder_params()
    params = params._replace(
        template_pair_stack=params.template_pair_stack._replace(
            blocks=params.template_pair_stack.blocks[:1]
        )
    )
    batch = _model_batch(collapse_identical_templates(_empty_templates()))
    z = jax.random.normal(
        jax.random.key(97), (1, N_TOKEN, N_TOKEN, 8), dtype=jnp.float32
    )
    pair_mask = jnp.ones((1, N_TOKEN, N_TOKEN), dtype=jnp.float32)
    traced = []

    def build(projection):
        template_module._project_template_restype = projection

        def run(input_batch, pair, weights, input_pair_mask):
            traced.append(cp_layout())
            return template_module.template_embedder(
                input_batch,
                pair,
                weights,
                pair_mask=input_pair_mask,
                no_heads=2,
            )

        return run

    arguments = (batch, z, params, pair_mask)
    reference = jax.device_get(jax.jit(build(pair_wide_projection))(*arguments))
    jax.clear_caches()
    serial = jax.device_get(jax.jit(build(optimized_projection))(*arguments))
    assert np.asarray(serial).tobytes() == np.asarray(reference).tobytes()

    for layout in ("1d", "2d"):
        jax.clear_caches()
        with context_parallel(4, layout=layout):
            old_lowered = jax.jit(build(pair_wide_projection)).lower(*arguments)
            old_compiled = old_lowered.compile()
            old = jax.device_get(old_compiled(*arguments))
            jax.clear_caches()
            new_lowered = jax.jit(build(optimized_projection)).lower(*arguments)
            new_compiled = new_lowered.compile()
            new = jax.device_get(new_compiled(*arguments))

        assert np.asarray(new).tobytes() == np.asarray(old).tobytes()
        np.testing.assert_allclose(reference, new, atol=3e-5, rtol=3e-5)
        old_hlo = old_compiled.runtime_executable().hlo_modules()[0].to_string().lower()
        hlo = new_compiled.runtime_executable().hlo_modules()[0].to_string().lower()
        collectives = ("all-gather", "all_gather", "collective-permute")
        assert any(name in hlo for name in collectives), (layout, hlo)
        new_collectives = {name: hlo.count(name) for name in collectives}
        old_collectives = {name: old_hlo.count(name) for name in collectives}
        assert all(
            new_collectives[name] <= old_collectives[name] for name in collectives
        ), (
            layout,
            old_collectives,
            new_collectives,
        )

    assert traced == [None, None, "1d", "1d", "2d", "2d"], traced
    template_module._project_template_restype = optimized_projection
    print("OPENFOLD3_TEMPLATE_RESTYPE_CP_OK")
    """
)


def test_projection_matches_pair_wide_oracle_on_four_cpu_devices() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _CONTEXT_PARALLEL_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OPENFOLD3_TEMPLATE_RESTYPE_CP_OK" in completed.stdout
