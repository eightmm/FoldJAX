"""CPU-sized contracts for the isolated transition-loop candidate."""

from __future__ import annotations

import numpy as np
import pytest

from bench.esmfold2_transition_loop_probe import (
    BLOCK_ROWS,
    PREFIX,
    _array_metrics,
    load_transition_weights,
    mapped_transition,
)


def original_spy(calls):
    """Tiny independent stand-in; it records the original body input shape."""

    def original(x, _params, _prefix, residual, _eps):
        calls.append(tuple(x.shape))
        update = x * 2 + 3
        return x + update if residual else update

    return original


@pytest.mark.parametrize("residual", [False, True])
def test_row_65_uses_full_block_then_exact_tail(residual):
    import jax.numpy as jnp

    calls = []
    value = jnp.arange(2 * 65 * 3, dtype=jnp.float32).reshape(2, 65, 3, 1)
    actual = mapped_transition(
        value, {}, "p", residual=residual, original=original_spy(calls)
    )
    update = value * 2 + 3
    expected = value + update if residual else update

    np.testing.assert_array_equal(actual, expected)
    assert [shape[1] for shape in calls] == [BLOCK_ROWS, 1]
    assert actual.shape == value.shape
    assert actual.dtype == value.dtype


@pytest.mark.parametrize("residual", [False, True])
def test_exact_multiple_uses_only_full_blocks_and_preserves_batch_order(residual):
    import jax.numpy as jnp

    calls = []
    value = jnp.arange(2 * 128 * 2, dtype=jnp.float32).reshape(2, 128, 2, 1)
    actual = mapped_transition(
        value, {}, "p", residual=residual, original=original_spy(calls)
    )
    update = value * 2 + 3
    expected = value + update if residual else update

    np.testing.assert_array_equal(actual, expected)
    assert [shape[1] for shape in calls] == [BLOCK_ROWS]
    np.testing.assert_array_equal(actual[1], expected[1])


def test_at_most_one_block_calls_original_directly():
    import jax.numpy as jnp

    calls = []
    value = jnp.ones((1, 64, 2, 1), jnp.bfloat16)
    actual = mapped_transition(
        value, {}, "p", residual=False, original=original_spy(calls)
    )

    assert calls == [(1, 64, 2, 1)]
    assert actual.dtype == value.dtype


def test_nonpositive_rows_are_rejected():
    import jax.numpy as jnp

    with pytest.raises(ValueError, match="positive"):
        mapped_transition(
            jnp.empty((1, 0, 2, 1)),
            {},
            "p",
            residual=False,
            original=lambda *args: args[0],
        )


def test_nonfinite_metrics_are_json_serializable():
    import json

    finite = _array_metrics(np.array([1.0]), np.array([1.0]))
    nonfinite = _array_metrics(np.array([np.nan]), np.array([np.nan]))

    assert finite["finite"] is True
    assert finite["max_abs"] == 0.0
    assert finite["rms"] == 0.0
    assert nonfinite["finite"] is False
    assert nonfinite["max_abs"] is None
    assert nonfinite["rms"] is None
    json.dumps({"finite": finite, "nonfinite": nonfinite}, allow_nan=False)


def test_load_transition_weights_preserves_optional_native_biases(tmp_path):
    from safetensors.numpy import save_file

    tensors = {
        f"{PREFIX}.norm.weight": np.ones(2, dtype=np.float32),
        f"{PREFIX}.norm.bias": np.zeros(2, dtype=np.float32),
        f"{PREFIX}.ffn.w12.weight": np.ones((4, 2), dtype=np.float32),
        f"{PREFIX}.ffn.w12.bias": np.zeros(4, dtype=np.float32),
        f"{PREFIX}.ffn.w3.weight": np.ones((2, 2), dtype=np.float32),
        f"{PREFIX}.ffn.w3.bias": np.zeros(2, dtype=np.float32),
    }
    path = tmp_path / "weights.safetensors"
    save_file(tensors, path)

    selected, record = load_transition_weights(path)

    assert selected.keys() == tensors.keys()
    assert [item["key"] for item in record["tensors"]] == sorted(tensors)
    np.testing.assert_array_equal(
        selected[f"{PREFIX}.ffn.w12.bias"], tensors[f"{PREFIX}.ffn.w12.bias"]
    )
    np.testing.assert_array_equal(
        selected[f"{PREFIX}.ffn.w3.bias"], tensors[f"{PREFIX}.ffn.w3.bias"]
    )
