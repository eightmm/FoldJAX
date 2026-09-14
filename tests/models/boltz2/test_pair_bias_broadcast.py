"""The token transformer's pair bias broadcasts over the sample batch.

``_attention_pair_bias_no_proj_z_forward`` used to materialise
``jnp.repeat(bias, multiplicity, axis=0)`` on every layer of every denoising
step. The bias is identical across the ``multiplicity`` diffusion samples of
one structure and only ever reaches the score through an elementwise add, so a
batch-1 bias broadcasts instead. These tests pin that the broadcast is not an
approximation: the reference arm passes the *already repeated* bias with
``multiplicity=1``, which is byte-for-byte the array the old code built, and
the outputs must be bitwise equal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.diffusion.diffusion_transformer import (
    _attention_pair_bias_no_proj_z_forward,
)
from tests.models.boltz2.test_attention_chunk_parity import _token_transformer_params

HEADS = 4
C_S = 16
MULTIPLICITY = 3


def _shape(var) -> tuple[int, ...]:
    """The shape of a jaxpr var's aval (``()`` for a literal without one)."""

    return tuple(getattr(var.aval, "shape", ()))


def _inputs(n: int, *, multiplicity: int = MULTIPLICITY, seed: int = 11):
    rng = np.random.default_rng(seed)
    params = _token_transformer_params(rng, C_S)
    s = jnp.asarray(
        rng.standard_normal((multiplicity, n, C_S)) * 0.5, dtype=jnp.float32
    )
    bias = jnp.asarray(rng.standard_normal((1, n, n, HEADS)), dtype=jnp.float32)
    mask = jnp.asarray((rng.random((multiplicity, n)) > 0.2).astype(np.float32))
    return params, s, bias, mask


def _run(
    params,
    s,
    bias,
    mask,
    *,
    multiplicity: int,
    chunk_size: int | None = None,
    backend: str = "xla",
) -> np.ndarray:
    run = jax.jit(
        lambda p, sv, bv, mv: _attention_pair_bias_no_proj_z_forward(
            p,
            s=sv,
            bias=bv,
            mask=mv,
            k_in=sv,
            multiplicity=multiplicity,
            inf=1e6,
            attention_backend=backend,
            chunk_size=chunk_size,
        )
    )
    return np.asarray(run(params, s, bias, mask))


def _arms(n: int, *, chunk_size: int | None, backend: str = "xla"):
    """The broadcast arm and the old materialised-repeat arm.

    ``jnp.repeat`` with ``multiplicity=1`` is the identity, so the reference
    arm runs the same function over exactly the batch-``M`` bias the removed
    ``jnp.repeat`` produced.
    """

    params, s, bias, mask = _inputs(n)
    repeated = jnp.repeat(bias, MULTIPLICITY, axis=0)
    broadcast = _run(
        params,
        s,
        bias,
        mask,
        multiplicity=MULTIPLICITY,
        chunk_size=chunk_size,
        backend=backend,
    )
    reference = _run(
        params,
        s,
        repeated,
        mask,
        multiplicity=1,
        chunk_size=chunk_size,
        backend=backend,
    )
    return broadcast, reference


@pytest.mark.parametrize("chunk_size", [None, 8])
def test_broadcast_bias_is_bit_identical(chunk_size: int | None) -> None:
    """Both XLA sub-branches agree to the word.

    ``chunk_size=8`` over 24 tokens forces three query blocks, which is the
    one place the batch-1 bias is *sliced* (``bias[:, :, start:stop]``) rather
    than only added.
    """

    broadcast, reference = _arms(24, chunk_size=chunk_size)
    assert broadcast.shape == reference.shape == (MULTIPLICITY, 24, C_S)
    assert np.array_equal(broadcast, reference)


def test_reference_arm_really_carries_the_full_batch() -> None:
    """The reference arm is the old code path, not a second broadcast.

    Without this the parity assertion above could pass by comparing the new
    behaviour with itself.
    """

    params, s, bias, mask = _inputs(24)
    assert bias.shape[0] == 1
    assert jnp.repeat(bias, MULTIPLICITY, axis=0).shape[0] == MULTIPLICITY == s.shape[0]


def test_a_shifted_bias_is_detected() -> None:
    """Tripwire: the comparison fires on a wrong bias.

    A one-column roll of the reference bias keeps its shape, dtype and every
    value, so only the *placement* changes. If the assertion above could not
    see that, it could not see a broken broadcast either.
    """

    params, s, bias, mask = _inputs(24)
    good = _run(params, s, bias, mask, multiplicity=MULTIPLICITY)
    shifted = jnp.roll(jnp.repeat(bias, MULTIPLICITY, axis=0), 1, axis=-2)
    bad = _run(params, s, shifted, mask, multiplicity=1)
    assert not np.array_equal(good, bad)


def test_the_bias_reaches_the_score() -> None:
    """Tripwire: a dropped bias would not pass as a broadcast bias.

    Zeroing the bias has to move the output, otherwise `bias` is dead in this
    path and every equality above would hold vacuously.
    """

    params, s, bias, mask = _inputs(24)
    with_bias = _run(params, s, bias, mask, multiplicity=MULTIPLICITY)
    without = _run(params, s, jnp.zeros_like(bias), mask, multiplicity=MULTIPLICITY)
    assert not np.array_equal(with_bias, without)


def test_a_per_sample_bias_still_materialises() -> None:
    """A bias that does vary per sample keeps the repeat, so it is honoured.

    ``multiplicity`` still interleaves the samples of a multi-structure bias;
    the broadcast is taken only for the batch-1 case. Feeding a batch-``M``
    bias whose slices differ must change the output per sample, which a
    silently-broadcast first slice could not do.
    """

    params, s, bias, mask = _inputs(24)
    per_sample = jnp.concatenate(
        [bias * jnp.asarray(scale, dtype=bias.dtype) for scale in (1.0, -1.0, 0.25)],
        axis=0,
    )
    out = _run(params, s, per_sample, mask, multiplicity=1)
    shared = _run(params, s, bias, mask, multiplicity=MULTIPLICITY)
    assert not np.array_equal(out[1], shared[1])
    assert np.array_equal(out[0], shared[0])


def test_the_bias_enters_the_score_still_batch_one() -> None:
    """The equality above is measured on the arm that skips the repeat.

    Byte equality alone cannot separate the arms: if the repeat came back,
    both arms would agree and every assertion here would still pass. So read
    the traced program and follow the bias forward from the argument. Under
    the broadcast it is only transposed (and query-sliced when blocked) and
    then handed straight to the score ``add`` with a leading axis of 1, while
    the score's is ``multiplicity``. The removed repeat is instead visible as
    a ``broadcast_in_dim``/``reshape`` pair that widens that axis before the
    add, which this rejects.
    """

    n = 24
    params, s, bias, mask = _inputs(n)

    def bias_consumers(chunk_size: int | None):
        closed = jax.make_jaxpr(
            lambda p, sv, bv, mv: _attention_pair_bias_no_proj_z_forward(
                p,
                s=sv,
                bias=bv,
                mask=mv,
                k_in=sv,
                multiplicity=MULTIPLICITY,
                inf=1e6,
                attention_backend="xla",
                chunk_size=chunk_size,
            )
        )(params, s, bias, mask)
        jaxpr = closed.jaxpr
        # Track by identity: jaxpr invars may also be unhashable literals.
        derived = {id(v) for v in jaxpr.invars if _shape(v) == (1, n, n, HEADS)}
        assert len(derived) == 1
        seen = []
        for eqn in jaxpr.eqns:
            operands = [v for v in eqn.invars if id(v) in derived]
            if not operands:
                continue
            seen.append((eqn.primitive.name, operands, eqn.outvars))
            for out in eqn.outvars:
                if _shape(out)[:1] == (1,):
                    derived.add(id(out))
        return seen

    for chunk_size in (None, 8):
        seen = bias_consumers(chunk_size)
        names = [name for name, _, _ in seen]
        # Anything that widens the batch axis would have to appear here.
        assert set(names) <= {"transpose", "slice", "convert_element_type", "add"}, (
            chunk_size,
            names,
        )
        adds = [(ops, outs) for name, ops, outs in seen if name == "add"]
        assert adds, (chunk_size, names)
        for operands, outvars in adds:
            assert all(_shape(v)[0] == 1 for v in operands)
            assert all(_shape(v)[0] == MULTIPLICITY for v in outvars)


def test_tokamax_public_api_accepts_a_batch_one_bias() -> None:
    """The tokamax backend takes the broadcast bias, on the CPU fallback.

    tokamax's own dispatch squeezes a size-1 batch dim and vmaps that operand
    with ``in_axes=None`` (``tokamax/_src/batching.py``), so this also
    exercises the *shape contract* the GPU kernels rely on -- not their
    lowering, which needs a GPU.
    """

    pytest.importorskip("tokamax")
    broadcast, reference = _arms(24, chunk_size=None, backend="tokamax")
    assert broadcast.shape == reference.shape == (MULTIPLICITY, 24, C_S)
    assert np.array_equal(broadcast, reference)
