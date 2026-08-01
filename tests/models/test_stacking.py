"""Pre-stacking layer weights on the host instead of inside the traced graph.

Every port runs its repeated blocks through ``lax.scan``, which needs the
per-layer parameters on one leading axis. Doing that with ``jnp.stack`` inside
``jit`` makes XLA copy the whole weight set into its temp arena -- on a
35-residue Boltz-2 job the optimized HLO carried 2,290 MiB of such
concatenates. These tests pin the properties that make hoisting it safe:
a pre-stacked container still reads like the layer list it replaced, the
stacking is a no-op the second time, and nothing that is not a layer stack is
converted.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models._stacking import (
    StackedLayers,
    prestack_layer_lists,
    stacked_or_stack,
    take_layers,
)


class Block(NamedTuple):
    kernel: Any
    bias: Any
    num_heads: int


def make_layers(count: int = 4, dim: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "kernel": np.full((dim, dim), float(index), dtype=np.float32),
            "bias": np.arange(dim, dtype=np.float32) + index,
        }
        for index in range(count)
    ]


def test_stacking_puts_layers_on_a_leading_axis():
    stacked = StackedLayers.from_layers(make_layers())
    assert stacked.stacked["kernel"].shape == (4, 3, 3)
    assert stacked.stacked["bias"].shape == (4, 3)
    assert len(stacked) == 4


def test_a_stacked_layer_reads_back_exactly():
    layers = make_layers()
    stacked = StackedLayers.from_layers(layers)
    for index, original in enumerate(layers):
        view = stacked[index]
        assert set(view) == set(original)
        for name, value in original.items():
            np.testing.assert_array_equal(np.asarray(view[name]), value)


def test_iteration_and_negative_indexing_match_the_list():
    layers = make_layers()
    stacked = StackedLayers.from_layers(layers)
    assert [float(np.asarray(view["kernel"])[0, 0]) for view in stacked] == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]
    np.testing.assert_array_equal(
        np.asarray(stacked[-1]["kernel"]), layers[-1]["kernel"]
    )
    with pytest.raises(IndexError):
        stacked[4]


def test_python_scalars_survive_the_round_trip():
    """``num_heads`` is a Python ``int`` in Boltz-2's layers.

    It has to reach ``scan`` on the layer axis like everything else, but the
    unrolled path reads it as a Python value -- a 0-d array there would break
    any caller using it as a shape.
    """
    blocks = [Block(np.ones((2, 2), np.float32), np.zeros(2, np.float32), 16)] * 3
    stacked = StackedLayers.from_layers(blocks)
    assert np.asarray(stacked.stacked.num_heads).shape == (3,)
    assert stacked[0].num_heads == 16
    assert isinstance(stacked[0].num_heads, int)


def test_scan_accepts_the_stacked_tree():
    layers = make_layers()
    stacked = stacked_or_stack(StackedLayers.from_layers(layers))

    def body(carry, layer):
        return carry + layer["kernel"].sum() + layer["bias"].sum(), None

    total, _ = jax.lax.scan(body, jnp.float32(0.0), stacked)
    expected = sum(
        float(layer["kernel"].sum() + layer["bias"].sum()) for layer in layers
    )
    assert float(total) == pytest.approx(expected)


def test_stacking_an_already_stacked_tree_is_a_no_op():
    stacked = StackedLayers.from_layers(make_layers())
    again = stacked_or_stack(stacked)
    assert again is stacked.stacked


def test_jit_sees_only_the_stacked_arrays():
    """The whole point: the traced program gets one array per leaf, not N."""
    params = {"layers": make_layers(count=6)}
    prestacked = prestack_layer_lists(params)
    leaves = jax.tree_util.tree_leaves(prestacked)
    assert len(leaves) == 2
    assert {leaf.shape for leaf in leaves} == {(6, 3, 3), (6, 3)}

    traced = []
    jax.jit(lambda tree: traced.append(jax.tree_util.tree_leaves(tree)) or 0.0)(
        prestacked
    )
    assert len(traced[0]) == 2


def test_prestacking_walks_dicts_and_named_tuples():
    class Params(NamedTuple):
        trunk: Any
        head: Any

    params = Params(
        trunk={"layers": make_layers()},
        head=tuple(
            Block(np.ones((2, 2), np.float32), np.zeros(2, np.float32), 8)
            for _ in range(3)
        ),
    )
    out = prestack_layer_lists(params)
    assert isinstance(out, Params)
    assert isinstance(out.trunk["layers"], StackedLayers)
    assert isinstance(out.head, StackedLayers)


def test_a_named_tuple_of_parameters_is_not_a_layer_stack():
    """A ``NamedTuple`` is a tuple, but it names its slots rather than repeating
    one -- converting it would replace a layer's contents with a fake stack."""
    block = Block(np.ones((2, 2), np.float32), np.ones((2, 2), np.float32), 4)
    out = prestack_layer_lists({"block": block})
    assert isinstance(out["block"], Block)


def test_loose_sequences_are_left_alone():
    params = {
        "pair": (np.ones(2, np.float32), np.zeros(2, np.float32)),
        "ragged": [
            {"a": np.ones((2, 2), np.float32), "b": np.ones(2, np.float32)},
            {"a": np.ones((3, 3), np.float32), "b": np.ones(3, np.float32)},
        ],
        "single": [{"a": np.ones((2, 2), np.float32), "b": np.ones(2, np.float32)}],
    }
    out = prestack_layer_lists(params)
    assert isinstance(out["pair"], tuple)
    assert isinstance(out["ragged"], list)
    assert isinstance(out["single"], list)


def test_truncation_keeps_the_stack_stacked():
    stacked = StackedLayers.from_layers(make_layers(count=5))
    head = take_layers(stacked, 2)
    assert isinstance(head, StackedLayers)
    assert len(head) == 2
    assert head.stacked["kernel"].shape == (2, 3, 3)
    assert take_layers(stacked, None) is stacked
    assert take_layers(stacked, 9) is stacked


def test_truncation_still_works_on_a_plain_list():
    layers = make_layers(count=5)
    assert len(take_layers(layers, 2)) == 2


def test_mixed_optional_sublayers_are_rejected():
    layers = [
        {"a": np.ones((2, 2), np.float32), "b": None},
        {"a": np.ones((2, 2), np.float32), "b": np.ones((2, 2), np.float32)},
    ]
    # The structures differ, so this is not a layer stack at all.
    assert isinstance(prestack_layer_lists({"layers": layers})["layers"], list)


def test_absent_sublayers_stack_as_absent():
    layers = [
        {"a": np.ones((2, 2), np.float32), "b": None},
        {"a": np.full((2, 2), 2.0, np.float32), "b": None},
    ]
    stacked = StackedLayers.from_layers(layers)
    assert stacked.stacked["b"] is None
    assert stacked[1]["b"] is None
    np.testing.assert_array_equal(np.asarray(stacked[1]["a"]), layers[1]["a"])


def test_an_empty_layer_list_is_an_error():
    with pytest.raises(ValueError, match="at least one layer"):
        StackedLayers.from_layers([])
    with pytest.raises(ValueError, match="empty layer list"):
        stacked_or_stack([])
