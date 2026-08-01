"""Turning an eagerly-written port into one traced program.

These ports were written the way the torch originals read: a Python function
that calls individually jitted primitives. That is easy to follow and easy to
compare against upstream, but it hands XLA the model one small program at a
time, so a single prediction dispatches hundreds or thousands of executables
and none of the fusion opportunities across module boundaries are available.

Wrapping the whole entry point in one ``jax.jit`` is the fix, and it runs into
the same three obstacles in every port. This module holds the shared answers:

* **Configuration flags live in the weights.** These checkpoints carry scalar
  ``bool`` leaves -- ``has_s``, ``cross_attention_mode`` -- that the modules
  branch on in Python. Inside ``jit`` every leaf is a tracer, so the branch
  raises. :func:`split_static_flags` moves them out of the traced arguments and
  into static ones, where they belong: they are constant for a checkpoint.
* **Features that are not model inputs.** Featurizers also emit string arrays
  for the structure writer. A string dtype cannot be traced, so
  :func:`is_traceable_feature` drops them before tracing rather than letting
  them be coerced into something meaningless.
* **Value-level validation.** Index-range checks read array *values*. They
  belong on the host, before tracing, where they still hold -- each port calls
  its own validator and then tells the traced function to skip it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp


def split_static_flags(params: Any) -> tuple[tuple, Any, tuple[tuple[int, bool], ...]]:
    """Split ``params`` into traceable arrays and static boolean flags.

    Returns ``(arrays, treedef, flags)``, where ``flags`` records the leaf
    position and value of every scalar ``bool``. Feed all three to
    :func:`merge_static_flags` inside the traced function.
    """
    leaves, treedef = jax.tree_util.tree_flatten(params)
    flags: list[tuple[int, bool]] = []
    arrays: list[Any] = []
    for index, leaf in enumerate(leaves):
        as_array = jnp.asarray(leaf)
        if as_array.dtype == jnp.bool_ and as_array.ndim == 0:
            flags.append((index, bool(leaf)))
        else:
            arrays.append(leaf)
    return tuple(arrays), treedef, tuple(flags)


def merge_static_flags(
    arrays: tuple, treedef: Any, flags: tuple[tuple[int, bool], ...]
) -> Any:
    """Rebuild the parameter tree from traced arrays and static flags."""
    flag_values = dict(flags)
    remaining = iter(arrays)
    leaves = [
        flag_values[index] if index in flag_values else next(remaining)
        for index in range(len(arrays) + len(flags))
    ]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def is_traceable_feature(value: Any) -> bool:
    """True for a feature leaf the traced graph can consume.

    Nested mappings are kept and recursed into by the caller; arrays are kept
    when their dtype is numeric or boolean. Everything else -- string arrays
    for the structure writer, Python objects -- is not a model input.
    """
    if isinstance(value, Mapping):
        return True
    dtype = getattr(value, "dtype", None)
    return dtype is not None and dtype.kind in "biufc"


def traceable_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """``features`` with the leaves the traced graph cannot consume removed."""
    return {
        name: value
        for name, value in features.items()
        if is_traceable_feature(value)
    }
