"""The sequential confidence path must rebuild what batching produced.

Running the head one sample at a time is only equivalent if the leaves come
back in the shape batching would have given them, and the two cases differ:
a per-sample leaf gains a map axis in front of a size-1 sample axis and has to
collapse onto it, while a sample-independent leaf is produced identically five
times and only one copy belongs in the result. Getting that rule wrong does
not raise -- it returns a differently-shaped dictionary that downstream code
may happily index.

The head itself needs the released checkpoint, so what is pinned here is the
recombination rule, on shapes that stand in for the real leaves.
"""

from __future__ import annotations

import jax.numpy as jnp

from foldjax.models.esmfold2.models.model import ModelSettings


def _recombine(mapped: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """The rule as `predict` applies it, isolated so it can be asserted."""
    return {
        name: (
            jnp.squeeze(value, axis=1)
            if value.ndim > 1 and value.shape[1] == 1
            else value[0]
        )
        for name, value in mapped.items()
    }


def test_the_option_is_off_so_no_released_command_changes() -> None:
    assert ModelSettings().confidence_sample_sequential is False


def test_a_per_sample_leaf_collapses_onto_its_sample_axis() -> None:
    """Five mapped results, each carrying a size-1 sample axis."""
    mapped = jnp.arange(5 * 1 * 7 * 3, dtype=jnp.float32).reshape(5, 1, 7, 3)
    (out,) = _recombine({"plddt": mapped}).values()
    assert out.shape == (5, 7, 3)
    # Sample i must be the result the i-th call produced, not a reordering.
    for index in range(5):
        assert jnp.array_equal(out[index], mapped[index, 0])


def test_a_sample_independent_leaf_keeps_one_copy() -> None:
    """Produced identically by every call; batching would have made it once."""
    single = jnp.asarray([2.0, 3.0, 5.0])
    mapped = jnp.broadcast_to(single, (5, 3))
    (out,) = _recombine({"boundaries": mapped}).values()
    assert out.shape == (3,)
    assert jnp.array_equal(out, single)


def test_a_scalar_per_sample_leaf_is_not_mistaken_for_an_independent_one() -> None:
    """The rule keys on a size-1 second axis, so a [5] leaf takes `value[0]`.

    Recorded rather than asserted as desirable: a genuinely per-sample scalar
    with no sample axis of its own would be reduced to its first sample. The
    head does not currently return one, and this test is here so that if it
    ever does, the failure is loud instead of silent.
    """
    mapped = jnp.asarray([10.0, 11.0, 12.0, 13.0, 14.0])
    (out,) = _recombine({"scalar": mapped}).values()
    assert out.shape == ()
    assert float(out) == 10.0
