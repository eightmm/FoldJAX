"""Protenix's trunk must run at `trunk_dtype`, not merely be configured for it.

`test_cast_trunk_params_preserves_fp32_diffusion_island` asserts the dtype of
*parameters* after `cast_trunk_params`. That is a weaker claim than it looks:
ESMFold2's parameters were cast correctly too, and all forty-eight of its trunk
layers ran float32 anyway, because a single float32 activation reaching a
residual stream widens everything downstream of it. Weights say nothing about
activations.

Protenix is safe for a structural reason -- `trunk.py` derives the dtype from
its own parameters and casts `s_inputs`, `s_init`, `z_init`, the per-cycle MSA
features and the cycle carry at the boundary -- but nothing asserted it. This
does.

It watches one level below the stack, in the triangle and transition
primitives, rather than at module boundaries. Protenix's own trunk comment says
why: the state "can come back wider than it went in", so a boundary that looks
narrow can hide arithmetic that is not. And the tempting argument that a
`lax.scan` carry makes widening impossible does not apply at the default:
`use_pairformer_scan` is False in both the function signature and the CLI, so
the block stack is unrolled and no carry constrains it.
"""

from __future__ import annotations

import contextlib
import inspect

import jax
import jax.numpy as jnp
import pytest

from foldjax.models.protenix.models.model import (
    cast_trunk_params,
    protenix_infer_static,
)
from foldjax.models.protenix.models.trunk_blocks import pairformer as pairformer_module
from foldjax.models.protenix.models.trunk_blocks import trunk as trunk_module
from foldjax.models.protenix.models.trunk_blocks.pairformer import PairformerStackParams
from tests.models.protenix.test_model import _toy_features, _toy_params
from tests.models.protenix.test_trunk import _zero_pairformer_block

#: The tensor whose width sets the trunk, at the stack and one level below it.
_TARGETS = (
    (trunk_module, "recycle_embeddings", ("s", "z")),
    (trunk_module, "template_embedder", ("z",)),
    (trunk_module, "msa_module", ("z",)),
    (trunk_module, "pairformer_stack", ("s", "z")),
    (pairformer_module, "triangle_multiplication", ("z",)),
    (pairformer_module, "triangle_attention", ("x",)),
    (pairformer_module, "transition", ("x",)),
    (pairformer_module, "attention_pair_bias", ("a", "z")),
)

C_S = C_Z = 2


@contextlib.contextmanager
def _watch(module, name: str, args: tuple[str, ...]):
    """Record the dtypes `module.name` is handed and returns, then restore it.

    A copy of `foldjax-bench/dtype_spy.py`'s `watch`, deliberately inlined. The
    bench directory is not part of this repository, and a gate that loads its
    instrument from a sibling checkout is a gate that passes by skipping
    wherever that sibling is absent -- which is every clone and every CI job.
    Thirty lines duplicated is the cheaper failure.
    """
    original = getattr(module, name)
    signature = inspect.signature(original)
    seen: dict[str, set[str]] = {}

    def record(label, value):
        dtype = getattr(value, "dtype", None)
        if dtype is not None:
            seen.setdefault(label, set()).add(str(dtype))

    def spy(*call_args, **call_kwargs):
        try:
            bound = signature.bind(*call_args, **call_kwargs)
            for label in args:
                if label in bound.arguments:
                    record(label, bound.arguments[label])
        except TypeError:
            # A signature that will not bind is not worth failing the run over.
            pass
        result = original(*call_args, **call_kwargs)
        record("return", result)
        return result

    setattr(module, name, spy)
    try:
        yield seen
    finally:
        setattr(module, name, original)


def _params(trunk_dtype):
    """Splice the Pairformer block in BEFORE casting, or the fixture lies.

    `cast_trunk_params` walks `input_embedder` and `pairformer_output` only, so
    a block added afterwards stays float32 and this test would report a defect
    that is really its own setup.
    """
    params = _toy_params()
    stack = PairformerStackParams(blocks=(_zero_pairformer_block(C_S, C_Z),))
    params = params._replace(
        pairformer_output=params.pairformer_output._replace(pairformer_stack=stack)
    )
    return cast_trunk_params(params, trunk_dtype) if trunk_dtype else params


def _block_leaf_dtypes(params) -> list[str]:
    return sorted(
        {
            str(leaf.dtype)
            for leaf in jax.tree.leaves(params.pairformer_output.pairformer_stack)
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
        }
    )


def _realized(trunk_dtype, **chunk):
    params = _params(trunk_dtype)
    noise = jnp.ones((1, 3, 3), dtype=jnp.float32)
    recorded = {}
    with contextlib.ExitStack() as stack:
        for module, name, args in _TARGETS:
            if hasattr(module, name):
                recorded[name] = stack.enter_context(_watch(module, name, args))
        protenix_infer_static(
            _toy_features(),
            params,
            jnp.asarray([1.0, 0.0], dtype=jnp.float32),
            key=None,
            num_samples=1,
            init_noise=noise,
            step_noises=(jnp.zeros_like(noise),),
            num_recycles=1,
            input_atom_heads=1,
            atom_encoder_heads=1,
            token_heads=1,
            atom_decoder_heads=1,
            n_queries=2,
            n_keys=4,
            sigma_data=4.0,
            centre_each_step=False,
            run_confidence=False,
            trunk_dtype=trunk_dtype,
            **chunk,
        )
    return params, recorded


@pytest.mark.parametrize(
    ("trunk_dtype", "expected", "chunk"),
    (
        (jnp.bfloat16, "bfloat16", {}),
        (
            jnp.bfloat16,
            "bfloat16",
            {"triangle_mul_chunk_size": 1, "triangle_att_q_chunk_size": 1},
        ),
        (None, "float32", {}),
    ),
    ids=("bf16", "bf16-chunked", "fp32"),
)
def test_the_trunk_runs_at_the_dtype_it_was_configured_for(
    trunk_dtype, expected, chunk
) -> None:
    """Both dtypes on purpose: pinning one would pass for a hard-coded port."""
    params, recorded = _realized(trunk_dtype, **chunk)

    assert _block_leaf_dtypes(params) == [expected], (
        "the fixture itself is wrong -- the spliced Pairformer block was not "
        f"cast, so its leaves are {_block_leaf_dtypes(params)}"
    )
    fired = {name: seen for name, seen in recorded.items() if seen}
    assert "pairformer_stack" in fired, (
        "the Pairformer stack never ran, so this test watched nothing -- the "
        "toy fixture ships an empty stack and a block must be spliced in"
    )
    wide = {
        name: {label: sorted(dtypes) for label, dtypes in seen.items()}
        for name, seen in fired.items()
        if any(dtypes != {expected} for dtypes in seen.values())
    }
    assert not wide, f"configured {expected}, these ran wider: {wide}"


def test_the_gate_notices_when_the_trunk_stops_being_clean() -> None:
    """Widen the pair stream on purpose and require the gate above to fail.

    Without this, the file asserts that protenix is clean today and never
    demonstrates it could tell if that changed -- which is the difference
    between a gate and a decoration. The injection is the ESMFold2 defect in
    miniature: one float32 leaving `msa_module` and nothing else touched.
    """
    real = trunk_module.msa_module

    def widened(*call_args, **call_kwargs):
        return real(*call_args, **call_kwargs).astype(jnp.float32)

    trunk_module.msa_module = widened
    try:
        with pytest.raises(AssertionError, match="ran wider"):
            test_the_trunk_runs_at_the_dtype_it_was_configured_for(
                jnp.bfloat16, "bfloat16", {}
            )
    finally:
        trunk_module.msa_module = real

    # The signature is worth pinning, not just the failure: `s` stays narrow
    # and `z` widens, which is how this bug looks in every port that has had
    # it, and a gate that fired for some other reason would not show it.
    trunk_module.msa_module = widened
    try:
        _, recorded = _realized(jnp.bfloat16)
    finally:
        trunk_module.msa_module = real
    assert recorded["pairformer_stack"]["s"] == {"bfloat16"}
    assert recorded["pairformer_stack"]["z"] == {"float32"}

