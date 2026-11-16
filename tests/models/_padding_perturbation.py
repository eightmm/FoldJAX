"""Shared assertions for the padding perturbation tests.

Each port has one such test: pad a small input to a bucket, run the default
inference forward, overwrite *only* the padded positions with large finite
garbage, run again with the same key and the same noise on the real atoms, and
require the real slices of every output to be bit-identical. A mask that
multiplies the garbage by zero leaves no trace; any difference is a padded
value reaching a real output, and the failing array names the operator family.

The comparison is bitwise on purpose. A tolerance would also accept the thing
the test is meant to separate from a leak -- the reduction-order drift a
padded shape induces -- and that drift cannot occur here, because the two runs
share one compiled program and one set of shapes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

#: Finite, far outside any value the featurizers produce.
GARBAGE = 1.0e3


def real_slices(
    outputs: Mapping[str, Any],
    crop: Callable[[str, np.ndarray], np.ndarray | None],
) -> dict[str, np.ndarray]:
    """Apply ``crop`` to every array output; ``None`` drops an output."""

    cropped: dict[str, np.ndarray] = {}
    for name, value in outputs.items():
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            continue
        array = np.asarray(value)
        if array.dtype == object:
            continue
        kept = crop(name, array)
        if kept is not None:
            cropped[name] = np.asarray(kept)
    return cropped


def assert_bitwise_identical(
    reference: Mapping[str, np.ndarray],
    perturbed: Mapping[str, np.ndarray],
    *,
    require: Iterable[str] = (),
) -> None:
    """Every shared real slice is finite and equal bit for bit.

    ``require`` names outputs that must be present -- the coordinates and the
    confidence summaries -- so a forward that silently stopped returning one
    cannot pass by omission.
    """

    missing = sorted(set(require) - set(reference))
    assert not missing, f"the forward returned no {missing}; nothing to compare"
    assert set(reference) == set(perturbed), (
        sorted(set(reference) ^ set(perturbed))
    )
    leaks = []
    for name in sorted(reference):
        expected = reference[name]
        actual = perturbed[name]
        assert expected.shape == actual.shape, (name, expected.shape, actual.shape)
        if np.issubdtype(expected.dtype, np.floating):
            assert np.isfinite(expected).all(), f"{name}: reference is not finite"
            assert np.isfinite(actual).all(), f"{name}: perturbed run is not finite"
        # Byte comparison: `array_equal` would call -0.0 and 0.0 the same.
        if np.array_equal(
            np.ascontiguousarray(expected).reshape(-1).view(np.uint8),
            np.ascontiguousarray(actual).reshape(-1).view(np.uint8),
        ):
            continue
        if np.issubdtype(expected.dtype, np.number):
            diff = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
            leaks.append(
                f"{name}: {int(np.count_nonzero(diff))}/{diff.size} elements differ, "
                f"max |diff| {float(diff.max()):.3e}"
            )
        else:
            leaks.append(f"{name}: differs")
    assert not leaks, "padded positions changed real outputs:\n  " + "\n  ".join(leaks)


def assert_some_difference(
    reference: Mapping[str, np.ndarray],
    control: Mapping[str, np.ndarray],
    names: Iterable[str],
) -> None:
    """The positive control: a *real* position changed, so outputs must move.

    Without this a model whose confidence head ignored its inputs, or a
    comparison that cropped everything away, would pass the leak test for
    nothing.
    """

    moved = [
        name
        for name in names
        if name in reference and not np.array_equal(reference[name], control[name])
    ]
    assert moved, f"perturbing a real position changed none of {sorted(names)}"


def garbage_ids(
    rng: np.random.Generator, shape: tuple[int, ...], high: int
) -> np.ndarray:
    """Random in-range category ids: wrong, but never an out-of-bounds gather."""

    return rng.integers(0, high, size=shape)
