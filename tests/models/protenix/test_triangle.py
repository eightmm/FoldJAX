from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.bridge.torch_mapping import (
    map_triangle_attention_state_dict,
    map_triangle_multiplication_state_dict,
)
from foldjax.models.protenix.models.primitives.primitives import (
    layer_norm,
    linear,
    sigmoid,
)
from foldjax.models.protenix.models.triangle.triangle import (
    triangle_attention,
    triangle_multiplication,
)


def test_triangle_multiplication_outgoing_matches_reference_formula() -> None:
    rng = np.random.default_rng(7)
    z = rng.normal(size=(1, 3, 3, 4)).astype(np.float32)
    mask = rng.integers(0, 2, size=(1, 3, 3)).astype(np.float32)
    state = _triangle_state(rng, c_z=4, c_hidden=5)
    params = map_triangle_multiplication_state_dict(state, "tri")

    actual = np.asarray(
        triangle_multiplication(jnp.asarray(z), jnp.asarray(mask), params, "outgoing")
    )
    compiled = np.asarray(
        triangle_multiplication(
            jnp.asarray(z),
            jnp.asarray(mask),
            params,
            "outgoing",
            use_jit=True,
        )
    )

    z_norm = layer_norm(jnp.asarray(z), params.layer_norm_in)
    a = (
        jnp.asarray(mask)[..., None]
        * sigmoid(linear(z_norm, params.linear_a_g))
        * linear(z_norm, params.linear_a_p)
    )
    b = (
        jnp.asarray(mask)[..., None]
        * sigmoid(linear(z_norm, params.linear_b_g))
        * linear(z_norm, params.linear_b_p)
    )
    expected = jnp.einsum("...ikd,...jkd->...ijd", a, b)
    expected = layer_norm(expected, params.layer_norm_out)
    expected = linear(expected, params.linear_z)
    expected = expected * sigmoid(linear(z_norm, params.linear_g))

    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(compiled, actual, rtol=1e-5, atol=1e-5)


def test_triangle_multiplication_incoming_matches_reference_formula() -> None:
    rng = np.random.default_rng(8)
    z = rng.normal(size=(1, 3, 3, 4)).astype(np.float32)
    mask = rng.integers(0, 2, size=(1, 3, 3)).astype(np.float32)
    state = _triangle_state(rng, c_z=4, c_hidden=5)
    params = map_triangle_multiplication_state_dict(state, "tri")

    actual = np.asarray(
        triangle_multiplication(jnp.asarray(z), jnp.asarray(mask), params, "incoming")
    )

    z_norm = layer_norm(jnp.asarray(z), params.layer_norm_in)
    a = (
        jnp.asarray(mask)[..., None]
        * sigmoid(linear(z_norm, params.linear_a_g))
        * linear(z_norm, params.linear_a_p)
    )
    b = (
        jnp.asarray(mask)[..., None]
        * sigmoid(linear(z_norm, params.linear_b_g))
        * linear(z_norm, params.linear_b_p)
    )
    expected = jnp.einsum("...kid,...kjd->...ijd", a, b)
    expected = layer_norm(expected, params.layer_norm_out)
    expected = linear(expected, params.linear_z)
    expected = expected * sigmoid(linear(z_norm, params.linear_g))

    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_map_triangle_multiplication_state_dict_uses_protenix_keys() -> None:
    state = {
        "tri.layer_norm_in.weight": np.ones((4,), dtype=np.float32),
        "tri.layer_norm_in.bias": np.zeros((4,), dtype=np.float32),
        "tri.layer_norm_out.weight": np.ones((5,), dtype=np.float32),
        "tri.layer_norm_out.bias": np.zeros((5,), dtype=np.float32),
        "tri.linear_a_p.weight": np.ones((5, 4), dtype=np.float32),
        "tri.linear_a_g.weight": np.ones((5, 4), dtype=np.float32),
        "tri.linear_b_p.weight": np.ones((5, 4), dtype=np.float32),
        "tri.linear_b_g.weight": np.ones((5, 4), dtype=np.float32),
        "tri.linear_z.weight": np.ones((4, 5), dtype=np.float32),
        "tri.linear_g.weight": np.ones((4, 4), dtype=np.float32),
    }

    params = map_triangle_multiplication_state_dict(state, "tri")

    assert tuple(params.layer_norm_in.weight.shape) == (4,)
    assert tuple(params.layer_norm_out.weight.shape) == (5,)
    assert tuple(params.linear_a_p.weight.shape) == (5, 4)
    assert tuple(params.linear_b_g.weight.shape) == (5, 4)
    assert tuple(params.linear_z.weight.shape) == (4, 5)
    assert tuple(params.linear_g.weight.shape) == (4, 4)


def test_triangle_attention_starting_matches_reference_formula() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(1, 3, 3, 4)).astype(np.float32)
    mask = rng.integers(0, 2, size=(1, 3, 3)).astype(np.float32)
    state = _triangle_attention_state(rng, c_in=4, num_heads=2, head_dim=3)
    params = map_triangle_attention_state_dict(state, "tri_att")

    actual = np.asarray(
        triangle_attention(
            jnp.asarray(x),
            jnp.asarray(mask),
            params,
            num_heads=2,
            starting=True,
        )
    )
    expected = _triangle_attention_reference(
        jnp.asarray(x),
        jnp.asarray(mask),
        params,
        num_heads=2,
    )

    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_triangle_attention_ending_matches_transposed_starting() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(size=(1, 3, 3, 4)).astype(np.float32)
    mask = rng.integers(0, 2, size=(1, 3, 3)).astype(np.float32)
    state = _triangle_attention_state(rng, c_in=4, num_heads=2, head_dim=3)
    params = map_triangle_attention_state_dict(state, "tri_att")

    actual = np.asarray(
        triangle_attention(
            jnp.asarray(x),
            jnp.asarray(mask),
            params,
            num_heads=2,
            starting=False,
        )
    )
    expected = _triangle_attention_reference(
        jnp.swapaxes(jnp.asarray(x), -2, -3),
        jnp.swapaxes(jnp.asarray(mask), -1, -2),
        params,
        num_heads=2,
    )
    expected = jnp.swapaxes(expected, -2, -3)

    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_triangle_attention_query_chunks_match_dense() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(1, 5, 5, 4)).astype(np.float32)
    mask = rng.integers(0, 2, size=(1, 5, 5)).astype(np.float32)
    state = _triangle_attention_state(rng, c_in=4, num_heads=2, head_dim=3)
    params = map_triangle_attention_state_dict(state, "tri_att")

    dense = triangle_attention(
        jnp.asarray(x),
        jnp.asarray(mask),
        params,
        num_heads=2,
    )
    chunked = triangle_attention(
        jnp.asarray(x),
        jnp.asarray(mask),
        params,
        num_heads=2,
        q_chunk_size=2,
    )
    compiled = triangle_attention(
        jnp.asarray(x),
        jnp.asarray(mask),
        params,
        num_heads=2,
        attention_backend="xla_jit",
    )

    np.testing.assert_allclose(
        np.asarray(chunked),
        np.asarray(dense),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(compiled),
        np.asarray(dense),
        rtol=1e-5,
        atol=1e-5,
    )


def test_triangle_attention_rejects_unknown_backend(monkeypatch) -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(1, 3, 3, 4)).astype(np.float32)
    state = _triangle_attention_state(rng, c_in=4, num_heads=2, head_dim=3)
    params = map_triangle_attention_state_dict(state, "tri_att")
    monkeypatch.setenv("PROTENIX_TRIANGLE_BACKEND", "typo")

    with pytest.raises(ValueError, match="unsupported triangle attention backend"):
        triangle_attention(jnp.asarray(x), None, params, num_heads=2)


def test_map_triangle_attention_state_dict_uses_protenix_keys() -> None:
    state = {
        "tri_att.layer_norm.weight": np.ones((4,), dtype=np.float32),
        "tri_att.layer_norm.bias": np.zeros((4,), dtype=np.float32),
        "tri_att.linear.weight": np.ones((2, 4), dtype=np.float32),
        "tri_att.mha.linear_q.weight": np.ones((6, 4), dtype=np.float32),
        "tri_att.mha.linear_k.weight": np.ones((6, 4), dtype=np.float32),
        "tri_att.mha.linear_v.weight": np.ones((6, 4), dtype=np.float32),
        "tri_att.mha.linear_o.weight": np.ones((4, 6), dtype=np.float32),
        "tri_att.mha.linear_g.weight": np.ones((6, 4), dtype=np.float32),
    }

    params = map_triangle_attention_state_dict(state, "tri_att")

    assert tuple(params.layer_norm.weight.shape) == (4,)
    assert tuple(params.linear.weight.shape) == (2, 4)
    assert tuple(params.attention.linear_q.weight.shape) == (6, 4)
    assert tuple(params.attention.linear_o.weight.shape) == (4, 6)


def _triangle_state(
    rng: np.random.Generator,
    *,
    c_z: int,
    c_hidden: int,
) -> dict[str, np.ndarray]:
    return {
        "tri.layer_norm_in.weight": rng.normal(size=(c_z,)).astype(np.float32),
        "tri.layer_norm_in.bias": rng.normal(size=(c_z,)).astype(np.float32),
        "tri.layer_norm_out.weight": rng.normal(size=(c_hidden,)).astype(np.float32),
        "tri.layer_norm_out.bias": rng.normal(size=(c_hidden,)).astype(np.float32),
        "tri.linear_a_p.weight": rng.normal(size=(c_hidden, c_z)).astype(np.float32),
        "tri.linear_a_g.weight": rng.normal(size=(c_hidden, c_z)).astype(np.float32),
        "tri.linear_b_p.weight": rng.normal(size=(c_hidden, c_z)).astype(np.float32),
        "tri.linear_b_g.weight": rng.normal(size=(c_hidden, c_z)).astype(np.float32),
        "tri.linear_z.weight": rng.normal(size=(c_z, c_hidden)).astype(np.float32),
        "tri.linear_g.weight": rng.normal(size=(c_z, c_z)).astype(np.float32),
    }


def _triangle_attention_state(
    rng: np.random.Generator,
    *,
    c_in: int,
    num_heads: int,
    head_dim: int,
) -> dict[str, np.ndarray]:
    total_hidden = num_heads * head_dim
    return {
        "tri_att.layer_norm.weight": rng.normal(size=(c_in,)).astype(np.float32),
        "tri_att.layer_norm.bias": rng.normal(size=(c_in,)).astype(np.float32),
        "tri_att.linear.weight": rng.normal(size=(num_heads, c_in)).astype(
            np.float32
        ),
        "tri_att.mha.linear_q.weight": rng.normal(
            size=(total_hidden, c_in)
        ).astype(np.float32),
        "tri_att.mha.linear_k.weight": rng.normal(
            size=(total_hidden, c_in)
        ).astype(np.float32),
        "tri_att.mha.linear_v.weight": rng.normal(
            size=(total_hidden, c_in)
        ).astype(np.float32),
        "tri_att.mha.linear_o.weight": rng.normal(
            size=(c_in, total_hidden)
        ).astype(np.float32),
        "tri_att.mha.linear_g.weight": rng.normal(
            size=(total_hidden, c_in)
        ).astype(np.float32),
    }


def _triangle_attention_reference(
    x: jnp.ndarray,
    mask: jnp.ndarray,
    params,
    *,
    num_heads: int,
    inf: float = 1e9,
) -> jnp.ndarray:
    x = layer_norm(x, params.layer_norm)
    q = _project_heads(x, params.attention.linear_q, num_heads)
    k = _project_heads(x, params.attention.linear_k, num_heads)
    v = _project_heads(x, params.attention.linear_v, num_heads)
    q = q / jnp.sqrt(jnp.asarray(q.shape[-1], dtype=q.dtype))

    logits = jnp.einsum("...hid,...hjd->...hij", q, k)
    mask_bias = inf * (mask.astype(jnp.float32) - 1.0)
    mask_bias = mask_bias[..., :, None, None, :]
    tri_bias = linear(x, params.linear)
    tri_bias = jnp.moveaxis(tri_bias, -1, -3)
    tri_bias = jnp.expand_dims(tri_bias, axis=-4)
    logits = logits + mask_bias + tri_bias

    probs = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
    out = jnp.einsum("...hij,...hjd->...hid", probs.astype(v.dtype), v)
    out = jnp.swapaxes(out, -2, -3)
    gate = sigmoid(linear(x, params.attention.linear_g))
    gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
    out = out * gate
    out = out.reshape(out.shape[:-2] + (-1,))
    return linear(out, params.attention.linear_o)


def _project_heads(x, params, num_heads: int) -> jnp.ndarray:
    y = linear(x, params)
    y = y.reshape(y.shape[:-1] + (num_heads, -1))
    return jnp.swapaxes(y, -2, -3)


def test_blocked_triangle_destination_width_does_not_change_the_result(
    monkeypatch,
) -> None:
    """A narrow blocked destination is the caller's own cast, moved earlier.

    Each row block reduces over the whole of ``b`` and is written once, so the
    only rounding a ``z.dtype`` destination adds is the one
    ``triangle_multiplication`` performs on the next line. Bit equality is the
    claim; if a future edit ever made the blocks accumulate into the
    destination instead of writing it, that claim would stop holding and this
    test is what fails.
    """
    from foldjax.models.protenix.models.triangle import triangle as triangle_mod

    monkeypatch.setenv("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
    rng = np.random.default_rng(11)
    n = 7
    chunk = 2
    z = jnp.asarray(rng.normal(size=(1, n, n, 4)).astype(np.float32), jnp.bfloat16)
    mask = jnp.asarray(rng.integers(0, 2, size=(1, n, n)).astype(np.float32))
    params = map_triangle_multiplication_state_dict(
        _triangle_state(rng, c_z=4, c_hidden=5), "tri"
    )

    real = triangle_mod._triangle_contract
    seen: dict[str, object] = {}

    def spy(project_a, z_norm, mask_arg, b, direction, chunk_size, out_dtype):
        narrow = real(project_a, z_norm, mask_arg, b, direction, chunk_size, out_dtype)
        seen["chunk_size"] = chunk_size
        seen["out_dtype"] = out_dtype
        seen["narrow"] = narrow
        # The pre-change code, exactly: a float32 destination the caller casts.
        seen["wide"] = real(
            project_a, z_norm, mask_arg, b, direction, chunk_size, jnp.float32
        )
        return narrow

    monkeypatch.setattr(triangle_mod, "_triangle_contract", spy)
    for direction in ("outgoing", "incoming"):
        seen.clear()
        triangle_multiplication(z, mask, params, direction, chunk_size=chunk)

        # The blocked branch really ran: a chunk smaller than the row count,
        # carried at the trunk's width rather than float32.
        assert seen["chunk_size"] == chunk < n
        assert seen["out_dtype"] == jnp.bfloat16
        assert seen["narrow"].dtype == jnp.bfloat16
        assert seen["wide"].dtype == jnp.float32
        assert not np.all(np.asarray(seen["wide"], dtype=np.float32) == 0.0)

        np.testing.assert_array_equal(
            np.asarray(seen["narrow"].astype(jnp.float32)),
            np.asarray(seen["wide"].astype(jnp.bfloat16).astype(jnp.float32)),
        )


def test_blocked_triangle_matches_the_unblocked_contraction(monkeypatch) -> None:
    """Blocking is a schedule, not a different function."""
    monkeypatch.setenv("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
    rng = np.random.default_rng(13)
    z = jnp.asarray(rng.normal(size=(1, 7, 7, 4)).astype(np.float32))
    mask = jnp.asarray(rng.integers(0, 2, size=(1, 7, 7)).astype(np.float32))
    params = map_triangle_multiplication_state_dict(
        _triangle_state(rng, c_z=4, c_hidden=5), "tri"
    )

    for direction in ("outgoing", "incoming"):
        whole = triangle_multiplication(z, mask, params, direction)
        blocked = triangle_multiplication(z, mask, params, direction, chunk_size=2)
        np.testing.assert_allclose(
            np.asarray(blocked), np.asarray(whole), rtol=1e-6, atol=1e-6
        )


def test_inert_multiplication_chunk_size_says_so_and_says_what_it_costs(
    monkeypatch,
) -> None:
    """The cueq multiplication kernel has no chunk parameter; the knob must say so.

    The attention path already warns. This one differs in substance, not just in
    wording: cueq attention never builds the tensor the chunk size would bound,
    so nothing is lost, while cueq multiplication does build both full
    projections. The warning names their size for that reason.
    """
    from foldjax.models.protenix.models.triangle import triangle as triangle_mod

    rng = np.random.default_rng(17)
    c = 32  # c_hidden == c_z and divisible by 32, so the cueq guard is cleared.
    z = jnp.asarray(rng.normal(size=(1, 6, 6, c)).astype(np.float32))
    mask = jnp.ones((1, 6, 6), dtype=jnp.float32)
    params = map_triangle_multiplication_state_dict(
        _triangle_state(rng, c_z=c, c_hidden=c), "tri"
    )

    monkeypatch.setenv("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", "cueq")
    monkeypatch.setattr(triangle_mod, "_WARNED_UNCHUNKABLE_MUL", False)
    with pytest.warns(RuntimeWarning, match="not used by the 'cueq' backend") as got:
        triangle_multiplication(z, mask, params, "outgoing", chunk_size=2)
    text = str(got[0].message)
    # 2 * 6 * 6 * 32 * 4 B is under a MiB, so the point is the sentence, not the
    # number: it must name the buffer and offer the escape without promising it.
    assert "materialises both full projections" in text
    assert "PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND=xla" in text
    # The escape is a trade, and the text has to say so in both directions.
    # It used to say "about 24% cheaper on memory at every size tried", which
    # reads as a whole-peak claim; end to end on Protenix at 4,100 tokens that
    # is 7.5%, for 17% more wall time. Assert that both halves are present
    # rather than the sentence carrying them, which has been reworded twice.
    assert "7.5%" in text and "17%" in text
    assert "trade" in text

    # Once, not once per layer.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        triangle_multiplication(z, mask, params, "outgoing", chunk_size=2)

    # Silent where there is nothing to warn about: a chunk that covers the rows,
    # and the backend that actually honours the request.
    monkeypatch.setattr(triangle_mod, "_WARNED_UNCHUNKABLE_MUL", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        triangle_multiplication(z, mask, params, "outgoing", chunk_size=6)
        triangle_multiplication(z, mask, params, "outgoing", chunk_size=None)
        monkeypatch.setenv("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
        triangle_multiplication(z, mask, params, "outgoing", chunk_size=2)
