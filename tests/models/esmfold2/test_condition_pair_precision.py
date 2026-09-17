"""The pair conditioning's precision boundaries, and its row block.

`condition_pair` is run once per sampling run and was the widest unblocked
stage left outside the trunk: it holds the trunk pair, the relative position
encoding, a `2 * C` float32 concatenation of the two and that concatenation's
normalisation at once. The first two are inputs and the last two are pure
intermediates, so the block removes them; what these tests pin is that
removing them removes no arithmetic.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import diffusion, primitives


def _condition_params(channels: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), jnp.float32)

    params = {
        "c.z_input_norm.weight": arr(2 * channels) * 0.1 + 1.0,
        "c.z_input_norm.bias": arr(2 * channels) * 0.1,
        "c.z_proj.weight": arr(channels, 2 * channels),
    }
    for index in range(2):
        dot = f"c.z_transitions.{index}"
        params[f"{dot}.norm.weight"] = arr(channels) * 0.1 + 1.0
        params[f"{dot}.norm.bias"] = arr(channels) * 0.1
        params[f"{dot}.a_proj.weight"] = arr(2 * channels, channels)
        params[f"{dot}.b_proj.weight"] = arr(2 * channels, channels)
        params[f"{dot}.out_proj.weight"] = arr(channels, 2 * channels)
    return params


@pytest.mark.parametrize("tokens", [11, 12])
def test_the_blocked_pair_conditioning_matches_the_whole_one(monkeypatch, tokens):
    """Whole versus row-blocked, on a count the block does and does not divide.

    The body is row-local -- the concatenation joins channels, the
    normalisation is per-`(i, j)`, and `z_proj` and both transitions contract
    channels only -- so the block is exact arithmetic. How many times the body
    is traced is recorded here as well, because a budget that quietly failed
    to divide anything would make the comparison a program against itself.
    """
    channels = 8
    params = _condition_params(channels, 0)
    rng = np.random.default_rng(5)
    z = jnp.asarray(rng.normal(size=(1, tokens, tokens, channels), scale=0.5))
    rel = jnp.asarray(rng.normal(size=(1, tokens, tokens, channels), scale=0.5))

    traced: list[tuple[int, ...]] = []
    body = diffusion._condition_pair_body

    def counted(*args, **kwargs):
        traced.append(args[0].shape)
        return body(*args, **kwargs)

    monkeypatch.setattr(diffusion, "_condition_pair_body", counted)

    def run():
        return np.asarray(
            diffusion.condition_pair(z, rel, params, "c", trunk_dtype=jnp.bfloat16)
        )

    whole = run()
    assert len(traced) == 1 and traced[0][1] == tokens, traced
    traced.clear()

    # Two rows per block, which leaves a trailing one at eleven tokens.
    monkeypatch.setattr(
        diffusion, "_CONDITION_PAIR_BUDGET_BYTES", 2 * tokens * 4 * 4 * channels
    )
    blocked = run()
    assert len(traced) > 1, traced
    assert max(shape[1] for shape in traced) <= 2, traced
    assert sum(shape[1] for shape in traced) == tokens, traced

    assert float(np.abs(whole).max()) > 0.0
    np.testing.assert_allclose(blocked, whole, rtol=1e-5, atol=1e-6)


def test_the_pair_conditioning_block_is_off_under_the_budget():
    """A small input takes the original single-call route."""
    assert diffusion._condition_pair_rows(jnp.zeros((1, 8, 8, 8))) is None


def test_the_transition_block_costs_the_dtype_the_projections_realise():
    """A bfloat16 autocast halves the widened bytes, and the rule must see it.

    Costing the widened form at the *input's* float32 would divide rows
    nothing needs divided, which on the pair conditioning showed up as a
    second layer of blocks inside `condition_pair`'s own.
    """
    x = jnp.zeros((1, 64, 64, 8), jnp.float32)
    # `a`, `b` and their product: three times `a_proj`'s output width.
    wide = 3 * 16
    # Eight float32 rows' worth: 48 channels x 4 bytes x 64 columns x 8.
    budget = wide * 4 * 64 * 8
    assert primitives._wide_rows(x, 1, wide, budget) == 8
    assert primitives._wide_rows(x, 1, wide, budget, itemsize=2) == 16


@pytest.mark.parametrize("tokens", [23])
def test_the_blocked_transition_layer_matches_the_whole_one(tokens):
    """`transition_layer`'s own block, on an axis its width does not divide."""
    channels = 8
    rng = np.random.default_rng(6)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), jnp.float32)

    params = {
        "t.norm.weight": arr(channels) * 0.1 + 1.0,
        "t.norm.bias": arr(channels) * 0.1,
        "t.a_proj.weight": arr(2 * channels, channels),
        "t.b_proj.weight": arr(2 * channels, channels),
        "t.out_proj.weight": arr(channels, 2 * channels),
    }
    x = jnp.asarray(rng.normal(size=(1, tokens, 7, channels), scale=0.5))

    whole = np.asarray(primitives.transition_layer(x, params, "t"))
    with pytest.MonkeyPatch.context() as patch:
        # 5 rows: 3 * 16 channels * 4 bytes * 7 columns per row.
        patch.setattr(primitives, "_TRANSITION_WIDE_BUDGET_BYTES", 3 * 16 * 4 * 7 * 5)
        blocked = np.asarray(primitives.transition_layer(x, params, "t"))

    assert blocked.shape == whole.shape
    assert float(np.abs(whole).max()) > 0.0
    np.testing.assert_allclose(blocked, whole, rtol=1e-6, atol=1e-6)


def test_condition_pair_passes_unrounded_residual_and_original_params(monkeypatch):
    params = {"z_input_norm.weight": jnp.ones(4), "z_input_norm.bias": jnp.zeros(4)}
    calls = []
    monkeypatch.setattr(diffusion, "layer_norm", lambda x, *a: x)
    monkeypatch.setattr(diffusion, "linear", lambda x, *a: x)

    def transition(x, p, prefix, *, linear_dtype):
        assert p is params
        assert x.dtype == jnp.float32
        assert float(x[0, 0, 0, 0]) == float(jnp.float32(1.001))
        assert linear_dtype == jnp.bfloat16
        calls.append(prefix)
        return jnp.zeros_like(x, dtype=linear_dtype)

    monkeypatch.setattr(diffusion, "transition_layer", transition)
    x = jnp.full((1, 1, 1, 2), 1.001, dtype=jnp.float32)
    result = diffusion.condition_pair(x, x, params, trunk_dtype=jnp.bfloat16)
    assert result.dtype == jnp.float32
    assert calls == ["z_transitions.0", "z_transitions.1"]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_transition_autocast_keeps_norm_fp32(monkeypatch, dtype):
    params = {"norm.weight": jnp.ones(2), "norm.bias": jnp.zeros(2)}
    params.update(
        {f"{name}.weight": jnp.eye(2) for name in ("a_proj", "b_proj", "out_proj")}
    )
    calls = []

    def norm(x, weight, bias, **kwargs):
        assert x.dtype == weight.dtype == bias.dtype == jnp.float32
        calls.append("norm")
        return x

    def linear(x, weights, name):
        assert x.dtype == weights[f"{name}.weight"].dtype == jnp.dtype(dtype)
        calls.append(name)
        return x

    monkeypatch.setattr(primitives, "layer_norm", norm)
    monkeypatch.setattr(primitives, "linear", linear)
    result = primitives.transition_layer(jnp.ones((1, 2)), params, linear_dtype=dtype)
    assert result.dtype == jnp.dtype(dtype)
    assert calls == ["norm", "a_proj", "b_proj", "out_proj"]
    assert all(x.dtype == jnp.float32 for x in params.values())
