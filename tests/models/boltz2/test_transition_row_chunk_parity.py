"""Exact-parity tests for outer-row chunking in transition_forward (rank-4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.primitives.transition import (
    _transition_rows,
    transition_forward,
)


def _random_params(
    rng: np.random.Generator,
    dim: int,
    hidden: int,
    kernel_dtype: jnp.dtype = jnp.float32,
):
    def w(*shape, dtype=jnp.float32):
        return jnp.asarray(rng.standard_normal(shape) * 0.1, dtype=dtype)

    return {
        "norm": {"scale": w(dim), "bias": w(dim)},
        "fc1": {"kernel": w(dim, hidden, dtype=kernel_dtype)},
        "fc2": {"kernel": w(dim, hidden, dtype=kernel_dtype)},
        "fc3": {"kernel": w(hidden, dim, dtype=kernel_dtype)},
    }


def _oracle_row_chunk_forward(
    params,
    x: jnp.ndarray,
    *,
    row_chunk_size: int,
    chunk_size: int | None = None,
    eps: float = 1e-5,
    native_amp_norm: bool = False,
) -> jnp.ndarray:
    """Previous Python-loop-and-concatenate outer-row chunk implementation."""
    compute_dtype = None
    if params["fc1"]["kernel"].dtype in (jnp.bfloat16, jnp.float16):
        compute_dtype = params["fc1"]["kernel"].dtype
    n = x.shape[1]
    blocks = []
    for start in range(0, n, row_chunk_size):
        stop = min(start + row_chunk_size, n)
        blocks.append(
            _transition_rows(
                params,
                x[:, start:stop],
                chunk_size=chunk_size,
                eps=eps,
                row_chunk_size=0,
                glu_backend="xla",
                compute_dtype=compute_dtype,
                native_amp_norm=native_amp_norm,
            )
        )
    return jnp.concatenate(blocks, axis=1)


def test_row_chunk_matches_single_shot() -> None:
    rng = np.random.default_rng(0)
    n, dim, hidden = 40, 8, 20
    params = _random_params(rng, dim, hidden)
    z = jnp.asarray(rng.standard_normal((1, n, n, dim)), dtype=jnp.float32)

    fwd = jax.jit(transition_forward, static_argnames=("row_chunk_size", "chunk_size"))
    single = fwd(params, z)
    chunked = fwd(params, z, row_chunk_size=11)

    diff = float(jnp.max(jnp.abs(single - chunked)))
    assert diff < 1e-6, f"max abs diff={diff}"


def test_row_and_hidden_chunk_matches_single_shot() -> None:
    rng = np.random.default_rng(2)
    n, dim, hidden = 40, 8, 32
    params = _random_params(rng, dim, hidden)
    z = jnp.asarray(rng.standard_normal((1, n, n, dim)), dtype=jnp.float32)

    fwd = jax.jit(transition_forward, static_argnames=("row_chunk_size", "chunk_size"))
    single = fwd(params, z)
    chunked = fwd(params, z, row_chunk_size=11, chunk_size=7)

    np.testing.assert_allclose(
        np.asarray(chunked),
        np.asarray(single),
        rtol=1e-5,
        atol=1e-5,
    )


def test_row_chunk_noop_when_rank3() -> None:
    rng = np.random.default_rng(1)
    dim, hidden = 8, 20
    params = _random_params(rng, dim, hidden)
    s = jnp.asarray(rng.standard_normal((1, 40, dim)), dtype=jnp.float32)

    single = transition_forward(params, s)
    with_row = transition_forward(params, s, row_chunk_size=11)
    assert float(jnp.max(jnp.abs(single - with_row))) == 0.0


@pytest.mark.parametrize(
    "param_dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"]
)
@pytest.mark.parametrize(
    "n", [24, 29, 9], ids=["no_tail", "with_tail", "one_full_and_tail"]
)
@pytest.mark.parametrize(
    "chunk_size", [None, 7], ids=["unsplit_hidden", "chunked_hidden"]
)
def test_row_chunk_bitwise_matches_python_loop_oracle(
    param_dtype: jnp.dtype,
    n: int,
    chunk_size: int | None,
) -> None:
    rng = np.random.default_rng(17)
    dim, hidden, row_chunk_size = 16, 20, 8
    params = _random_params(rng, dim, hidden, kernel_dtype=param_dtype)
    z = jnp.asarray(rng.standard_normal((1, n, 12, dim)), dtype=jnp.float32)
    native_amp_norm = param_dtype == jnp.bfloat16

    def run_new(p, x):
        return transition_forward(
            p,
            x,
            chunk_size=chunk_size,
            row_chunk_size=row_chunk_size,
            glu_backend="xla",
            native_amp_norm=native_amp_norm,
        )

    def run_oracle(p, x):
        return _oracle_row_chunk_forward(
            p,
            x,
            row_chunk_size=row_chunk_size,
            chunk_size=chunk_size,
            native_amp_norm=native_amp_norm,
        )

    actual = jax.jit(run_new)(params, z)
    expected = jax.jit(run_oracle)(params, z)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("tail", [0, 3], ids=["no_tail", "with_tail"])
def test_row_chunk_traces_each_block_shape_once(
    monkeypatch: pytest.MonkeyPatch, tail: int
) -> None:
    from foldjax.models.boltz2.models.primitives import transition as module

    dim, hidden, row_chunk_size = 16, 32, 8
    params = {
        "norm": {
            "scale": jax.ShapeDtypeStruct((dim,), jnp.float32),
            "bias": jax.ShapeDtypeStruct((dim,), jnp.float32),
        },
        "fc1": {"kernel": jax.ShapeDtypeStruct((dim, hidden), jnp.bfloat16)},
        "fc2": {"kernel": jax.ShapeDtypeStruct((dim, hidden), jnp.bfloat16)},
        "fc3": {"kernel": jax.ShapeDtypeStruct((hidden, dim), jnp.bfloat16)},
    }
    real = module._transition_rows
    block_traces: list[int] = []

    def spy(params, x, **kwargs):
        if kwargs["row_chunk_size"] == 0:
            block_traces.append(x.shape[1])
        return real(params, x, **kwargs)

    monkeypatch.setattr(module, "_transition_rows", spy)
    x = jax.ShapeDtypeStruct((1, 64 + tail, 16, dim), jnp.float32)
    jax.make_jaxpr(
        lambda p, z: transition_forward(
            p,
            z,
            row_chunk_size=row_chunk_size,
            glu_backend="xla",
            native_amp_norm=True,
        )
    )(params, x)
    # Eight full blocks and the tail: one trace per block shape.
    assert block_traces == ([8, tail] if tail else [8])


def test_row_chunk_boundary_conditions_and_invalid_inputs() -> None:
    rng = np.random.default_rng(23)
    dim, hidden, row_chunk_size = 16, 32, 8
    params = _random_params(rng, dim, hidden)

    def trace_prims(x: jnp.ndarray, rc: int) -> list[str]:
        traced = jax.make_jaxpr(
            lambda p, z: transition_forward(p, z, row_chunk_size=rc, glu_backend="xla")
        )(params, x)
        # The row blocks are the `jit` calls named for the block function.
        return [
            "block" if eqn.params.get("name") == "_transition_rows" else "other"
            for eqn in traced.eqns
        ]

    # Empty axis-1 edge: skips the blocks cleanly and returns empty output.
    empty = jnp.zeros((1, 0, 6, dim), dtype=jnp.float32)
    assert "block" not in trace_prims(empty, row_chunk_size)
    empty_out = jax.jit(
        lambda p, z: transition_forward(
            p, z, row_chunk_size=row_chunk_size, glu_backend="xla"
        )
    )(params, empty)
    assert empty_out.shape == (1, 0, 6, dim)

    # Zero row_chunk_size and n == row_chunk_size (at limit): neither blocks.
    at_limit = jnp.asarray(
        rng.standard_normal((1, row_chunk_size, 6, dim)), dtype=jnp.float32
    )
    assert "block" not in trace_prims(at_limit, 0)
    assert "block" not in trace_prims(at_limit, row_chunk_size)
    unblocked = jax.jit(
        lambda p, z: transition_forward(p, z, row_chunk_size=0, glu_backend="xla")
    )(params, at_limit)
    equal_limit = jax.jit(
        lambda p, z: transition_forward(
            p, z, row_chunk_size=row_chunk_size, glu_backend="xla"
        )
    )(params, at_limit)
    assert np.array_equal(np.asarray(equal_limit), np.asarray(unblocked))

    # One row past limit (n == row_chunk_size + 1): one full block and a tail.
    above_limit = jnp.asarray(
        rng.standard_normal((1, row_chunk_size + 1, 6, dim)), dtype=jnp.float32
    )
    assert trace_prims(above_limit, row_chunk_size).count("block") == 2

    # Failure exits on malformed chunk_size / glu_backend when row_chunk_size applies.
    for bad_chunk in (0, -4):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            transition_forward(
                params,
                above_limit,
                chunk_size=bad_chunk,
                row_chunk_size=row_chunk_size,
                glu_backend="xla",
            )
    with pytest.raises(ValueError, match="glu backend must be one of"):
        transition_forward(
            params,
            above_limit,
            row_chunk_size=row_chunk_size,
            glu_backend="invalid",
        )


def test_row_chunk_accepts_read_only_params_with_extra_entries() -> None:
    from types import MappingProxyType

    rng = np.random.default_rng(3)
    n, dim, hidden = 40, 8, 20
    params = _random_params(rng, dim, hidden)
    # A checkpoint view: read-only mappings and an entry the block never reads,
    # which is not a valid jit argument.
    view = MappingProxyType(
        {**{k: MappingProxyType(dict(v)) for k, v in params.items()}, "name": "t"}
    )
    z = jnp.asarray(rng.standard_normal((1, n, n, dim)), dtype=jnp.float32)

    expected = transition_forward(params, z, row_chunk_size=11)
    traced = jax.jit(lambda z: transition_forward(view, z, row_chunk_size=11))(z)
    eager = transition_forward(view, z, row_chunk_size=11)
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(expected))
    np.testing.assert_allclose(np.asarray(traced), np.asarray(expected), atol=1e-6)
