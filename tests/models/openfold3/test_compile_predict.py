"""``compile_predict`` must produce the same answer as eager ``predict``.

Compilation is the only practical way to run this port -- eager execution of the
released trunk is ~700x slower -- so the compiled path is the one users take, and
it needs its own gate. Two things can go wrong that the parity tests cannot see:
a value that is fine to compute eagerly but not traceable (boolean indexing on a
mask, a Python ``float()`` on an array), and a shape that only works because a
Python ``if`` looked at data.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from foldjax.models.openfold3.inference import compile_predict, predict

pytestmark = pytest.mark.torch_parity


@pytest.fixture(scope="module")
def case(openfold3_source, randomized):
    """Reuse the small end-to-end fixtures rather than build a second model."""
    from .test_inference_end_to_end import (
        _batch,
        _config,
        _params,
        _representative_atoms,
        _torch,
    )

    torch = _torch()
    return (
        _batch(torch),
        _params(torch, randomized),
        _config(),
        _representative_atoms(),
    )


def _compare(first, second) -> None:
    for name in first._fields:
        left, right = getattr(first, name), getattr(second, name)
        if left is None or right is None:
            assert left is right, name
            continue
        np.testing.assert_allclose(
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"{name} differs between the compiled and eager paths",
        )


def test_compiled_matches_eager(case) -> None:
    batch, params, config, table = case
    eager = predict(jax.random.key(0), batch, params, config, table)
    compiled = compile_predict(config, table)(jax.random.key(0), batch, params)
    _compare(eager, compiled)


def test_compiled_matches_eager_with_chains(case) -> None:
    """The ipTM path adds chain-pair reductions, which are the jit-fragile ones."""
    batch, params, config, table = case
    eager = predict(jax.random.key(0), batch, params, config, table, n_chain=2)
    compiled = compile_predict(config, table, n_chain=2)(
        jax.random.key(0), batch, params
    )
    assert eager.iptm is not None
    _compare(eager, compiled)


def test_compiled_matches_eager_with_the_pair_embedding_off(case) -> None:
    batch, params, config, table = case
    eager = predict(
        jax.random.key(0),
        batch,
        params,
        config,
        table,
        use_trunk_pair_embedding=False,
    )
    compiled = compile_predict(config, table, use_trunk_pair_embedding=False)(
        jax.random.key(0), batch, params
    )
    _compare(eager, compiled)


def test_weights_can_be_swapped_without_recompiling(case, randomized) -> None:
    """``params`` is a runtime argument, so a second checkpoint reuses the compile."""
    from .test_inference_end_to_end import _params, _torch

    batch, params, config, table = case
    compiled = compile_predict(config, table)
    first = compiled(jax.random.key(0), batch, params)

    torch = _torch()
    def differently_randomized(module, seed: int = 0, scale: float = 0.5):
        return randomized(module, 5, scale)

    other = _params(torch, differently_randomized)
    second = compiled(jax.random.key(0), batch, other)
    assert not np.allclose(
        np.asarray(first.coordinates), np.asarray(second.coordinates)
    )
