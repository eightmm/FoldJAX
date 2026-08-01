"""Layer stacks that reach the traced graph already stacked.

Every port here runs its repeated blocks through ``lax.scan``, and ``scan``
wants the per-layer parameters on one leading axis. Written the obvious way,
that stack is a ``jnp.stack`` over the layer list *inside* the traced function
-- so XLA sees the weights arrive as N separate parameters and emits a
``concatenate`` that copies all of them into a fresh buffer. The copy is real
device memory. On a 35-residue Boltz-2 job the optimized HLO contained
2,290 MiB of such concatenates (``f32[24,768,3072]`` for the token
transformer, ``f32[64,384,1536]`` for the Pairformer, ...) and XLA's temp
arena peaked at 845 MiB against 84 MiB for the same model in eager torch.

Nothing about the stack depends on the input, so it does not belong in the
graph. :func:`prestack_layer_lists` performs it once on the host at load time
and stores the result in :class:`StackedLayers`, a pytree node whose children
are the already-stacked arrays. ``jit`` then receives the stacked weights as
its arguments and the concatenates disappear; the bytes are the same, they are
just no longer duplicated.

:class:`StackedLayers` still reads like the layer list it replaced -- it has a
length, iterates layer by layer, and indexes -- so the unrolled
(``use_scan=False``) paths keep working unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _is_python_scalar(value: Any) -> bool:
    """True for a plain Python number carried alongside the arrays.

    Boltz-2 layers store ``num_heads`` as an ``int``, for instance. It is a
    leaf of the parameter tree like any other and has to end up on the scan
    axis, but it is *also* read as a Python value on the unrolled path.
    """
    return isinstance(value, (bool, int, float)) and not isinstance(value, np.generic)


def stack_leaves(*leaves: Any) -> Any:
    """Stack one leaf position across layers onto a new leading axis.

    Handles the non-array leaves these parameter trees carry: ``None`` (an
    absent optional sublayer, which must be absent in every layer) and Python
    scalars, which become a stacked array exactly as the per-port stackers
    produced them -- ``lax.scan`` requires every ``xs`` leaf to have the layer
    axis, scalars included.
    """
    first = leaves[0]
    if first is None:
        if any(leaf is not None for leaf in leaves):
            raise ValueError("cannot stack mixed None/non-None layer leaves")
        return None
    if isinstance(first, bool):
        return np.asarray(leaves, dtype=bool)
    if _is_python_scalar(first):
        return np.asarray(leaves)
    if all(isinstance(leaf, np.ndarray) for leaf in leaves):
        return np.stack(leaves, axis=0)
    return jnp.stack([jnp.asarray(leaf) for leaf in leaves], axis=0)


class StackedLayers:
    """A homogeneous layer list whose leaves carry a leading layer axis.

    Registered as a pytree node, so ``jit`` flattens it to the stacked arrays
    and never sees the individual layers. Reads like the original sequence for
    the code paths that still want one layer at a time.
    """

    __slots__ = ("stacked", "length", "scalars")

    def __init__(
        self,
        stacked: Any,
        length: int,
        scalars: tuple[tuple[int, tuple[Any, ...]], ...] = (),
    ) -> None:
        self.stacked = stacked
        self.length = length
        # Leaf positions whose per-layer values were Python scalars, kept so a
        # single layer reads back exactly as it was written -- an ``int`` that
        # came out of the stack as a 0-d array would break any caller using it
        # as a shape or a loop bound.
        self.scalars = scalars

    @classmethod
    def from_layers(cls, layers: Any) -> StackedLayers:
        layers = list(layers)
        if not layers:
            raise ValueError("StackedLayers requires at least one layer")
        stacked = jax.tree.map(
            stack_leaves, *layers, is_leaf=lambda value: value is None
        )
        per_layer = [
            jax.tree_util.tree_flatten(layer, is_leaf=_is_none)[0] for layer in layers
        ]
        scalars = tuple(
            (position, tuple(leaves[position] for leaves in per_layer))
            for position in range(len(per_layer[0]))
            if _is_python_scalar(per_layer[0][position])
        )
        return cls(stacked, len(layers), scalars)

    def __len__(self) -> int:
        return self.length

    def __bool__(self) -> bool:
        return self.length > 0

    def __iter__(self) -> Iterator[Any]:
        for index in range(self.length):
            yield self[index]

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.length))]
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(f"layer index {index} out of range")
        leaves, treedef = jax.tree_util.tree_flatten(self.stacked, is_leaf=_is_none)
        sliced = [None if leaf is None else leaf[index] for leaf in leaves]
        for position, values in self.scalars:
            sliced[position] = values[index]
        return jax.tree_util.tree_unflatten(treedef, sliced)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StackedLayers(length={self.length})"


def _is_none(value: Any) -> bool:
    return value is None


jax.tree_util.register_pytree_node(
    StackedLayers,
    lambda node: ((node.stacked,), (node.length, node.scalars)),
    lambda aux, children: StackedLayers(children[0], aux[0], aux[1]),
)


def stacked_or_stack(layers: Any) -> Any:
    """Return the stacked pytree for ``layers``, stacking only if needed.

    The per-port stack helpers call this, so a tree that went through
    :func:`prestack_layer_lists` costs nothing and one that did not behaves
    exactly as before.
    """
    if isinstance(layers, StackedLayers):
        return layers.stacked
    layers = list(layers)
    if not layers:
        raise ValueError("cannot stack an empty layer list")
    return jax.tree.map(stack_leaves, *layers, is_leaf=lambda value: value is None)


def take_layers(layers: Any, limit: int | None) -> Any:
    """The first ``limit`` layers, keeping a pre-stacked stack pre-stacked.

    Slicing a :class:`StackedLayers` with ordinary indexing would hand back a
    Python list of layer views, which the stack helper would then concatenate
    right back together -- the copy this module exists to avoid.
    """
    if limit is None or limit >= len(layers):
        return layers
    if isinstance(layers, StackedLayers):
        truncated = jax.tree.map(
            lambda leaf: None if leaf is None else leaf[:limit],
            layers.stacked,
            is_leaf=_is_none,
        )
        scalars = tuple(
            (position, values[:limit]) for position, values in layers.scalars
        )
        return StackedLayers(truncated, limit, scalars)
    return list(layers)[:limit]


def _is_layer_sequence(value: Any) -> bool:
    """True for a sequence that is a homogeneous stack of parameter blocks.

    Deliberately strict: at least two entries, every entry the same pytree
    shape with more than one leaf, and every corresponding leaf the same dtype
    and shape. A pair of loose arrays or a fixed-arity tuple of unrelated
    parameters fails at least one of those, so only real layer stacks are
    converted.

    A ``NamedTuple`` is a tuple, but it names its slots rather than repeating
    one, so it is never a layer sequence -- it is the *contents* of a layer.
    """
    if hasattr(value, "_fields"):
        return False
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        first_leaves, first_tree = jax.tree_util.tree_flatten(
            value[0], is_leaf=_is_none
        )
    except Exception:  # pragma: no cover - defensive
        return False
    if len(first_leaves) < 2:
        return False
    signature = [_leaf_signature(leaf) for leaf in first_leaves]
    if any(entry is None for entry in signature):
        return False
    for entry in value[1:]:
        leaves, tree = jax.tree_util.tree_flatten(entry, is_leaf=_is_none)
        if tree != first_tree or len(leaves) != len(first_leaves):
            return False
        if any(
            _leaf_signature(leaf) != expected
            for leaf, expected in zip(leaves, signature, strict=True)
        ):
            return False
    return True


def _leaf_signature(leaf: Any) -> Any:
    """What has to match across layers for a leaf to be stackable.

    ``None`` for anything this module cannot put on a leading axis, which
    disqualifies the whole sequence.
    """
    if leaf is None:
        return ("none",)
    if _is_python_scalar(leaf):
        return ("scalar", type(leaf))
    shape = getattr(leaf, "shape", None)
    if shape is None:
        return None
    return ("array", tuple(shape), str(leaf.dtype))


def prestack_layer_lists(params: Any) -> Any:
    """Replace every homogeneous layer sequence in ``params`` with a stack.

    Call it once on freshly loaded weights, before they reach ``jit``. When
    the leaves are still NumPy the stacking happens entirely on the host, so
    the device only ever receives the stacked form.
    """
    if isinstance(params, StackedLayers):
        return params
    if _is_layer_sequence(params):
        return StackedLayers.from_layers(
            [prestack_layer_lists(entry) for entry in params]
        )
    if isinstance(params, dict):
        return {key: prestack_layer_lists(value) for key, value in params.items()}
    if hasattr(params, "_fields") and isinstance(params, tuple):  # NamedTuple
        return type(params)(*(prestack_layer_lists(value) for value in params))
    if isinstance(params, list):
        return [prestack_layer_lists(value) for value in params]
    if isinstance(params, tuple):
        return tuple(prestack_layer_lists(value) for value in params)
    return params
