"""Experimental structural row blocks preserve the fused attention contract."""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax import execution
from foldjax.api import resolve_cache_dir
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.models.opendde import runner
from foldjax.models.opendde.cli import predict as predict_cli
from foldjax.models.protenix.models.primitives.attention import AttentionParams
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
)
from foldjax.models.protenix.models.triangle import (
    triangle,
    triangle_attention_tokamax,
    triangle_cueq,
)
from foldjax.schema import PredictionRequest
from tests._parser_capture import capture_parser


def _params(dtype):
    rng = np.random.default_rng(109)

    def projection(rows, cols):
        return LinearParams(
            jnp.asarray(rng.normal(0, 0.15, (rows, cols)), dtype),
            jnp.asarray(rng.normal(0, 0.15, (rows,)), dtype),
        )

    return triangle.TriangleAttentionParams(
        LayerNormParams(jnp.ones(4, dtype), jnp.zeros(4, dtype)),
        projection(2, 4),
        AttentionParams(
            projection(6, 4),
            projection(6, 4),
            projection(6, 4),
            projection(4, 6),
            projection(6, 4),
        ),
    )


@pytest.mark.parametrize("backend", ["cueq", "cueq_jit", "tokamax"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("lead", [(), (2,)])
def test_fused_projection_blocks_match_full_rows_and_keep_column_masks(
    monkeypatch, backend, dtype, starting, lead
):
    rng = np.random.default_rng(110)
    x = jnp.asarray(rng.normal(size=(*lead, 19, 19, 4)), dtype)
    mask = rng.integers(0, 2, size=(*lead, 19, 19)).astype(np.float32)
    mask[..., 3, :] = 0
    params = _params(dtype)
    calls, projection_rows = [], []
    observing = True

    def kernel(q, k, v, bias, mask_bias, *, scale=1.0):
        if observing:
            calls.append((q.shape, np.asarray(mask_bias), np.asarray(bias)))
        assert q.shape[-2] == k.shape[-2] == v.shape[-2] == 19
        assert bias.shape[-4:] == (1, 2, 19, 19)
        # Both arms call this same CPU reference, with each backend's scaling
        # convention. The actual fused GPU arithmetic is a separate A/B gate.
        q = q * jnp.asarray(scale, q.dtype)
        return triangle._triangle_attention_block(q, k, v, mask_bias, bias)

    monkeypatch.setattr(triangle_cueq, "cueq_attention_core", kernel)
    monkeypatch.setattr(triangle_attention_tokamax, "tokamax_attention_core", kernel)
    project_heads = triangle._project_heads

    def project(x_rows, *args):
        projection_rows.append(x_rows.shape[-3])
        return project_heads(x_rows, *args)

    monkeypatch.setattr(triangle, "_project_heads", project)
    run = partial(
        triangle.triangle_attention,
        mask=jnp.asarray(mask),
        params=params,
        num_heads=2,
        starting=starting,
        attention_backend=backend,
    )
    with jax.disable_jit():
        whole = run(x)
        whole_calls = list(calls)
        calls.clear()
        projection_rows.clear()
        blocked = run(x, fused_row_chunk_size=7)
    assert [shape[-4] for shape, _, _ in calls] == [7, 7, 5]
    assert max(projection_rows) == 7
    _, full_mask, full_bias = whole_calls[0]
    for start, size, (_, row_mask, row_bias) in zip(
        (0, 7, 14), (7, 7, 5), calls, strict=True
    ):
        np.testing.assert_array_equal(
            row_mask, full_mask[..., start : start + size, :, :, :]
        )
        np.testing.assert_array_equal(row_bias, full_bias)
    observing = False
    compiled = jax.jit(partial(run, fused_row_chunk_size=7))(x)
    assert blocked.shape == whole.shape == compiled.shape == x.shape
    assert blocked.dtype == whole.dtype == compiled.dtype == dtype
    tolerance = 2e-3 if dtype == jnp.bfloat16 else 1e-6
    for value in (blocked, compiled):
        np.testing.assert_allclose(
            np.asarray(value, dtype=np.float32),
            np.asarray(whole, dtype=np.float32),
            rtol=tolerance,
            atol=tolerance,
        )


def test_default_and_explicit_none_keep_the_shared_lowering(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the default entered experimental projection chunking")

    monkeypatch.setattr(triangle, "_triangle_attention_projection_rows", forbidden)
    x, params = jnp.ones((3, 3, 4)), _params(jnp.float32)

    def predict(x, fused_row_chunk_size=None):
        return triangle.triangle_attention(
            x,
            None,
            params,
            num_heads=2,
            attention_backend="xla_jit",
            fused_row_chunk_size=fused_row_chunk_size,
        )

    compiled = jax.jit(predict, static_argnames=("fused_row_chunk_size",))
    assert (
        compiled.lower(x).as_text()
        == compiled.lower(x, fused_row_chunk_size=None).as_text()
    )


@pytest.mark.parametrize("width", [0, -2, True, 1.5])
def test_triangle_rejects_invalid_projection_width_before_projecting(width):
    with pytest.raises(ValueError, match="positive integer"):
        triangle.triangle_attention(
            jnp.ones((3, 3, 4)),
            None,
            _params(jnp.float32),
            num_heads=2,
            attention_backend="cueq",
            fused_row_chunk_size=width,
        )


def test_projection_chunking_refuses_xla_and_context_parallelism(monkeypatch):
    kwargs = dict(num_heads=2, fused_row_chunk_size=2)
    x, params = jnp.ones((3, 3, 4)), _params(jnp.float32)
    with pytest.raises(ValueError, match="requires cueq or tokamax"):
        triangle.triangle_attention(x, None, params, attention_backend="xla", **kwargs)
    monkeypatch.setattr(triangle, "cp_mesh", lambda: object())
    with pytest.raises(ValueError, match="not validated with context parallelism"):
        triangle.triangle_attention(x, None, params, attention_backend="cueq", **kwargs)


def _request(tmp_path):
    job, weights = tmp_path / "job.json", tmp_path / "opendde.jax"
    job.write_text("[]")
    weights.write_bytes(b"fixture")
    return PredictionRequest(
        model="opendde",
        input=job,
        weights=weights,
        cache_dir=tmp_path / "compile",
        output_dir=tmp_path / "out",
        seed=0,
    )


def test_only_opted_in_requests_add_a_compile_profile_and_native_flag(tmp_path):
    request = _request(tmp_path)
    backend = OpenDDEBackend()
    baseline = backend.cache_profile(request)
    name = "structural_triangle_att_row_chunk_size"
    assert name not in baseline
    changed = dataclasses.replace(request, options={name: 32})
    candidate = backend.cache_profile(changed)
    assert candidate == {**baseline, name: 32}
    assert resolve_cache_dir(changed, backend) != resolve_cache_dir(request, backend)
    invocation = backend._native_invocation(changed)
    parsed = capture_parser(predict_cli.main).parse_args(invocation.argv)
    assert parsed.structural_triangle_att_row_chunk_size == 32
    assert invocation.config_fields[name] == 32
    assert runner.PredictionConfig(**invocation.config_fields)._asdict()[name] == 32
    with pytest.raises(ValueError, match="not validated with context parallelism"):
        backend.cache_profile(
            dataclasses.replace(changed, options={name: 32, "cp_devices": 2})
        )


def test_runner_forwards_and_records_projection_chunking_only_when_requested(
    monkeypatch,
):
    from foldjax.models.opendde.models import model

    calls = []
    monkeypatch.setattr(
        model, "opendde_infer_compiled", lambda *args, **kw: calls.append(kw)
    )
    features = {"restype": np.zeros((4, 32), np.float32)}
    settings = dict(
        seed=0,
        num_samples=1,
        num_steps=1,
        num_recycles=1,
        n_queries=2,
        n_keys=4,
        diffusion_attention_backend="xla_jit",
        trunk_single_attention_backend="xla_jit",
        structural_single_attention_backend="xla_jit",
        cycle_msa_features=(),
    )
    name = "structural_triangle_att_row_chunk_size"
    try:
        for width in (None, 2):
            execution.clear_chunking_record()
            runner._predict(features, (), **settings, **{name: width})
            chunks = execution.recorded_chunking()["profiles"][0]["chunks"]
            if width is None:
                assert name not in calls[-1] and name not in chunks
            else:
                assert calls[-1][name] == chunks[name] == width
    finally:
        execution.clear_chunking_record()
