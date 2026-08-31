"""The rollout optimizations must be exact, not approximate.

Two changes in :func:`~foldjax.models.openfold3.inference.predict` trade work
for identical answers, and both are only safe because of a property that is easy
to break by
editing the layer they depend on:

* pair conditioning is hoisted out of the sampler loop, which is valid exactly
  because it does not read the noise level;
* the confidence re-embedding runs one diffusion sample at a time above its
  numeric token cutoff (zero by default) or the released sample width, which is
  valid exactly because nothing in it mixes samples.

If a future change makes pair conditioning noise-dependent, or makes the
confidence path mix samples, the optimizations become silently wrong -- the code
would still run and produce plausible structures. These tests fail instead.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.inference import (
    _expand_samples,
    _per_sample_confidence,
    released_config,
)
from foldjax.models.openfold3.models.diffusion_conditioning import pair_conditioning


def test_pair_conditioning_takes_no_noise_level() -> None:
    """The hoist out of the sampler loop rests on this signature.

    Checked structurally rather than numerically: a ``t`` argument that was
    accepted and ignored would pass a value comparison and still mean the next
    person can wire the noise level in without anything complaining.
    """
    parameters = set(inspect.signature(pair_conditioning).parameters)
    assert "t" not in parameters, (
        "pair_conditioning now takes a noise level, so it is no longer constant "
        "across the rollout and predict() must stop hoisting it out of the sampler"
    )


def test_expand_samples_equals_repeat() -> None:
    """``broadcast_to`` is only an optimization if it is the same value."""
    value = jnp.asarray(np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4))
    expanded = _expand_samples(value, 5)
    repeated = jnp.repeat(value, 5, axis=0)
    assert expanded.shape == repeated.shape == (5, 2, 3, 4)
    np.testing.assert_array_equal(np.asarray(expanded), np.asarray(repeated))


def test_expand_samples_leaves_other_leading_axes_alone() -> None:
    """Scalars and already-expanded tensors have to pass through untouched."""
    scalar = jnp.asarray(3.0)
    assert _expand_samples(scalar, 5) is scalar
    already = jnp.zeros((5, 2))
    assert _expand_samples(already, 5) is already


@pytest.mark.parametrize(
    ("n_token", "num_samples", "cutoff", "expected"),
    [
        # The default serializes every multi-sample confidence pass.
        (76, 5, 0, True),
        (750, 5, 0, True),
        # Positive values retain their token boundary within the released width.
        (76, 5, 750, False),
        (750, 5, 750, False),
        (751, 5, 750, True),
        (2076, 5, 750, True),
        # Wider requests retain the existing high-sample memory guard.
        (76, 6, 750, True),
        (76, 10, 750, True),
        (76, 20, 750, True),
        # One sample has no per-sample loop to run.
        (2076, 1, 0, False),
        # ``None`` disables it however long or wide the request is.
        (2076, 5, None, False),
        (76, 20, None, False),
    ],
)
def test_per_sample_schedule_keeps_cutoff_width_guard_and_opt_out(
    n_token: int, num_samples: int, cutoff: int | None, expected: bool
) -> None:
    config = released_config(
        n_token=n_token,
        n_atom=n_token * 8,
        num_samples=num_samples,
        per_sample_token_cutoff=cutoff,
    )
    assert _per_sample_confidence(config) is expected


def test_released_default_serializes_every_multi_sample_prediction() -> None:
    """The measured bounded default is represented by the numeric cutoff zero."""
    config = released_config(n_token=100, n_atom=800)
    assert config.per_sample_token_cutoff == 0
    assert _per_sample_confidence(config)


def _with_two_confidence_blocks(params):
    """Duplicate the confidence Pairformer's block so the stack is scanned.

    The shared fixture builds one block per stack, and ``scan_stack`` falls back to
    a plain loop below two blocks. A plain loop broadcasts mismatched leading axes
    without complaint, so a rank bug in the confidence inputs is invisible with one
    block and a hard error with two: ``scan`` checks that the carry it gets back has
    the type it passed in. That is exactly how a real one slipped through -- the
    batched branch fed a batch-1 single representation alongside a 5-sample
    ``x_pred``, and only the scanned stack noticed.
    """
    stack = params.pairformer_embedding.pairformer_stack
    doubled = stack._replace(blocks=stack.blocks * 2)
    embedding = params.pairformer_embedding._replace(pairformer_stack=doubled)
    return params._replace(pairformer_embedding=embedding)


@pytest.mark.torch_parity
def test_both_confidence_branches_trace_with_a_scanned_stack(
    openfold3_source, randomized
) -> None:
    """Shapes only, in both branches, with the scan-carry check switched on."""
    import functools

    from foldjax.models.openfold3.inference import predict

    from .test_inference_end_to_end import (
        _batch,
        _config,
        _params,
        _representative_atoms,
        _torch,
    )

    torch = _torch()
    batch = _batch(torch)
    params = _with_two_confidence_blocks(_params(torch, randomized))
    table = _representative_atoms()
    base = _config()
    if base.num_samples < 2:
        pytest.skip("the per-sample path needs more than one sample")

    shapes = {}
    for cutoff, label in ((None, "batched"), (base.n_token - 1, "per_sample")):
        config = base._replace(per_sample_token_cutoff=cutoff)
        out = jax.eval_shape(
            functools.partial(predict, config=config, representative_atoms=table),
            jax.ShapeDtypeStruct((2,), jnp.uint32),
            jax.tree.map(
                lambda x: jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x)), batch
            ),
            jax.tree.map(
                lambda x: jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x)), params
            ),
        )
        shapes[label] = {
            name: getattr(out, name).shape
            for name in ("coordinates", "plddt", "pae_logits", "pde_logits")
        }
    assert shapes["batched"] == shapes["per_sample"], (
        "the two confidence branches disagree on output shapes: "
        f"{shapes['batched']} vs {shapes['per_sample']}"
    )
    assert shapes["batched"]["pae_logits"][0] == base.num_samples


@pytest.mark.torch_parity
def test_batched_and_per_sample_confidence_agree(openfold3_source, randomized) -> None:
    """The two confidence paths must give the same answer on the same input.

    Run on the small end-to-end fixture with the cutoff forced either side of the
    token count, so both branches execute on identical data. This is the test that
    would catch a per-sample loop that dropped or mis-indexed a sample -- the kind
    of bug that produces five plausible structures with four of them wrong.
    """
    from foldjax.models.openfold3.inference import predict

    from .test_inference_end_to_end import (
        _batch,
        _config,
        _params,
        _representative_atoms,
        _torch,
    )

    torch = _torch()
    batch = _batch(torch)
    params = _params(torch, randomized)
    table = _representative_atoms()
    base = _config()
    if base.num_samples < 2:
        pytest.skip("the per-sample path needs more than one sample")

    n_token = base.n_token
    batched = base._replace(per_sample_token_cutoff=None)
    per_sample = base._replace(per_sample_token_cutoff=n_token - 1)
    assert not _per_sample_confidence(batched)
    assert _per_sample_confidence(per_sample)

    key = jax.random.key(0)
    left = predict(key, batch, params, batched, table)
    right = predict(key, batch, params, per_sample, table)

    for name in ("plddt", "ptm", "pae_logits", "pde_logits", "coordinates"):
        expected, actual = getattr(left, name), getattr(right, name)
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"{name} differs between the batched and per-sample paths",
        )
