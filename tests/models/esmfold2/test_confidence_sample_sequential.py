"""The sequential confidence path must rebuild what batching produced.

Running the head one sample at a time is only equivalent if the leaves come
back in the shape batching would have given them, and the two cases differ:
a per-sample leaf gains a map axis in front of a size-1 sample axis and has to
collapse onto it, while a sample-independent leaf is produced identically once
per sample and only one copy belongs in the result. Getting that rule wrong
does not raise -- it returns a differently-shaped dictionary that downstream
code may happily index.

The head itself needs the released checkpoint, so what is pinned here is the
recombination rule, on shapes that stand in for the real leaves. It is now
imported from `model.py` rather than restated: this file used to hold its own
copy, which passes whenever the copy is right and the original is wrong.
"""

from __future__ import annotations

import jax.numpy as jnp

from foldjax.models.esmfold2.models.model import (
    ModelSettings,
)
from foldjax.models.esmfold2.models.model import (
    rebuild_batched_confidence as _recombine,
)


def test_the_option_is_on_because_the_released_command_cannot_run_without_it() -> None:
    """This default was off, and the reason it was off did not survive measurement.

    The old name here was `test_the_option_is_off_so_no_released_command_
    changes`, and that was a real argument: sequencing costs up to 1e-5 of
    pLDDT from bfloat16 reduction order, so leaving it off kept released
    behaviour identical. **It was reasoned at five samples. The release ships
    thirty-two.**

    At 32 samples and 1,003 residues the batched head asks for
    `bf16[32, L^2, 2048]` -- 122.8 GiB on a 95.6 GiB device, 27.2 GiB past the
    card in a single intermediate. It fails under preallocation on and off
    alike, at different points, because it is a capacity impossibility rather
    than a fragmentation one. There is no released behaviour to preserve: the
    released command does not complete.

    Kept and rewritten rather than deleted, so the next reader finds that this
    was considered and reversed on evidence rather than never thought about.
    """
    assert ModelSettings().confidence_sample_sequential is True


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
