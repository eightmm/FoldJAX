"""The MSA stack scans its homogeneous prefix and unrolls only the rest."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.dtype import narrow_floats
from foldjax.models.openfold3.models.msa import (
    MSAPairWeightedAveragingParams,
    OuterProductMeanParams,
)
from foldjax.models.openfold3.models.msa_module import (
    MSAModuleBlockParams,
    MSAModuleStackParams,
    msa_module_block,
    msa_module_stack,
)
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    SwiGLUParams,
    SwiGLUTransitionParams,
)
from tests.models.openfold3.test_glu_backend import HEADS, C, N, _stack_params

C_M, C_HIDDEN, N_SEQ = 4, 3, 3
SETTINGS = {"no_heads_msa": HEADS, "no_heads_pair": HEADS}


def _prefix_stack(m, z, params, **settings):
    return msa_module_stack(m, z, params, scan_prefix=True, **settings)


def _block(rng, *, skip: bool) -> MSAModuleBlockParams:
    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    def lin(out, size, *, bias=False):
        return LinearParams(weight=arr(out, size), bias=arr(out) if bias else None)

    def ln(size):
        return LayerNormParams(weight=arr(size) * 0.1 + 1.0, bias=arr(size) * 0.1)

    return MSAModuleBlockParams(
        outer_product_mean=OuterProductMeanParams(
            layer_norm=ln(C_M),
            linear_1=lin(C_HIDDEN, C_M),
            linear_2=lin(C_HIDDEN, C_M),
            linear_out=lin(C, C_HIDDEN * C_HIDDEN, bias=True),
        ),
        pair_stack=_stack_params(rng).blocks[0].pair_stack,
        msa_att_row=None
        if skip
        else MSAPairWeightedAveragingParams(
            layer_norm_m=ln(C_M),
            layer_norm_z=ln(C),
            linear_z=lin(HEADS, C),
            linear_v=lin(HEADS * C_HIDDEN, C_M),
            linear_g=lin(HEADS * C_HIDDEN, C_M),
            linear_o=lin(C_M, HEADS * C_HIDDEN),
        ),
        msa_transition=None
        if skip
        else SwiGLUTransitionParams(
            layer_norm=ln(C_M),
            swiglu=SwiGLUParams(linear_a=lin(2 * C_M, C_M), linear_b=lin(2 * C_M, C_M)),
            linear_out=lin(C_M, 2 * C_M),
        ),
    )


def _stack(layout: str) -> MSAModuleStackParams:
    """``A`` updates the MSA; ``S`` is a ``skip_msa_update`` block."""
    rng = np.random.default_rng(31)
    return MSAModuleStackParams(
        blocks=tuple(_block(rng, skip=kind == "S") for kind in layout)
    )


def _inputs():
    rng = np.random.default_rng(37)
    m = jnp.asarray(rng.normal(size=(N_SEQ, N, C_M), scale=0.5), jnp.float32)
    z = jnp.asarray(rng.normal(size=(N, N, C), scale=0.5), jnp.float32)
    msa_mask = jnp.ones((N_SEQ, N), jnp.float32).at[2, 4].set(0.0)
    token = jnp.ones((N,), jnp.float32).at[N - 1].set(0.0)
    return m, z, msa_mask, token[:, None] * token[None, :]


@pytest.mark.parametrize(
    ("layout", "scans"),
    [
        ("", []),
        ("S", []),
        ("AS", []),
        ("AAS", [2]),
        ("AAAS", [3]),
        ("AASAA", [2]),
        ("SAAA", []),
        ("AAAA", [4]),
    ],
)
def test_msa_stack_scans_the_longest_homogeneous_prefix(layout, scans) -> None:
    """Unchunked, so the stack's own scan is the only top-level one possible."""
    params = _stack(layout)
    m, z, msa_mask, pair_mask = _inputs()

    traced = jax.make_jaxpr(
        lambda m_in, z_in: _prefix_stack(
            m_in,
            z_in,
            params,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            **SETTINGS,
        )
    )(m, z)

    assert [
        equation.params["length"]
        for equation in traced.jaxpr.eqns
        if equation.primitive.name == "scan"
    ] == scans


@pytest.mark.parametrize("layout", ["AAS", "AAAS", "AASAA"])
def test_msa_stack_prefix_scan_matches_the_unrolled_blocks(layout) -> None:
    params = _stack(layout)
    m, z, msa_mask, pair_mask = _inputs()
    settings = {
        "msa_mask": msa_mask,
        "pair_mask": pair_mask,
        "chunk_size": 4,
        **SETTINGS,
    }

    def unrolled(m_in, z_in):
        for block in params.blocks:
            m_in, z_in = msa_module_block(m_in, z_in, block, **settings)
        return z_in

    expected = jax.jit(unrolled)(m, z)
    actual = jax.jit(lambda m_in, z_in: _prefix_stack(m_in, z_in, params, **settings))(
        m, z
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_msa_stack_scan_carry_keeps_bfloat16() -> None:
    """``lax.scan`` rejects a block that widens ``m`` or ``z``; a loop does not."""
    params = narrow_floats(_stack("AAAS"), jnp.bfloat16)
    m, z, msa_mask, pair_mask = (x.astype(jnp.bfloat16) for x in _inputs())

    result = jax.eval_shape(
        lambda m_in, z_in: _prefix_stack(
            m_in,
            z_in,
            params,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            chunk_size=4,
            **SETTINGS,
        ),
        m,
        z,
    )

    assert result.dtype == jnp.bfloat16


def test_msa_stack_bfloat16_prefix_scan_stays_within_bfloat16_rounding() -> None:
    """A scanned and an unrolled bfloat16 stack fuse differently, so they may
    differ, but by less than bfloat16 rounding moves either from float32."""
    params32 = _stack("AAAS")
    params = narrow_floats(params32, jnp.bfloat16)
    inputs32 = _inputs()
    m, z, msa_mask, pair_mask = (x.astype(jnp.bfloat16) for x in inputs32)
    settings = {"msa_mask": msa_mask, "pair_mask": pair_mask, **SETTINGS}
    settings32 = {"msa_mask": inputs32[2], "pair_mask": inputs32[3], **SETTINGS}

    reference = jax.jit(_unrolled(params32, settings32))(*inputs32[:2])
    unrolled = jax.jit(_unrolled(params, settings))(m, z).astype(jnp.float32)
    scanned = jax.jit(lambda m_in, z_in: _prefix_stack(m_in, z_in, params, **settings))(
        m, z
    )
    traced = jax.make_jaxpr(
        lambda m_in, z_in: _prefix_stack(m_in, z_in, params, **settings)
    )(m, z)

    assert [e for e in traced.jaxpr.eqns if e.primitive.name == "scan"]
    assert scanned.dtype == jnp.bfloat16
    scanned = scanned.astype(jnp.float32)
    assert jnp.linalg.norm(scanned - unrolled) <= jnp.linalg.norm(unrolled - reference)


def test_msa_stack_leaves_a_dtype_widening_prefix_unrolled() -> None:
    """bfloat16 activations and weights with float32 masks widen the carry."""
    params = narrow_floats(_stack("AAAS"), jnp.bfloat16)
    m, z, msa_mask, pair_mask = _inputs()
    m, z = m.astype(jnp.bfloat16), z.astype(jnp.bfloat16)
    settings = {
        "msa_mask": msa_mask,
        "pair_mask": pair_mask,
        "chunk_size": 4,
        **SETTINGS,
    }

    def unrolled(m_in, z_in):
        for block in params.blocks:
            m_in, z_in = msa_module_block(m_in, z_in, block, **settings)
        return z_in

    expected = jax.jit(unrolled)(m, z)
    actual = jax.jit(lambda m_in, z_in: _prefix_stack(m_in, z_in, params, **settings))(
        m, z
    )

    assert actual.dtype == expected.dtype == jnp.float32
    np.testing.assert_array_equal(actual, expected)


def _unrolled(params, settings):
    def run(m_in, z_in):
        for block in params.blocks:
            m_in, z_in = msa_module_block(m_in, z_in, block, **settings)
        return z_in

    return run


def test_msa_stack_leaves_a_prefix_with_a_wider_later_block_unrolled() -> None:
    """A float32 parameter in the second block widens the carry there only."""
    params = narrow_floats(_stack("AAAS"), jnp.bfloat16)
    second = params.blocks[1]
    opm = second.outer_product_mean
    wide = opm.linear_out._replace(bias=opm.linear_out.bias.astype(jnp.float32))
    second = second._replace(outer_product_mean=opm._replace(linear_out=wide))
    params = params._replace(blocks=(params.blocks[0], second, *params.blocks[2:]))
    m, z, msa_mask, pair_mask = (x.astype(jnp.bfloat16) for x in _inputs())
    settings = {
        "msa_mask": msa_mask,
        "pair_mask": pair_mask,
        "chunk_size": 4,
        **SETTINGS,
    }

    expected = jax.jit(_unrolled(params, settings))(m, z)
    actual = jax.jit(lambda m_in, z_in: _prefix_stack(m_in, z_in, params, **settings))(
        m, z
    )

    assert actual.dtype == expected.dtype == jnp.float32
    np.testing.assert_array_equal(actual, expected)


def test_msa_stack_leaves_a_prefix_that_batches_the_pair_state_unrolled() -> None:
    """An unbatched pair state meeting a batched MSA gains a leading axis."""
    params = _stack("AAAS")
    m, z, msa_mask, pair_mask = _inputs()
    settings = {
        "msa_mask": jnp.stack([msa_mask, msa_mask]),
        "pair_mask": jnp.stack([pair_mask, pair_mask]),
        "chunk_size": 4,
        **SETTINGS,
    }
    m = jnp.stack([m, m * 0.5])

    expected = jax.jit(_unrolled(params, settings))(m, z)
    actual = jax.jit(lambda m_in, z_in: _prefix_stack(m_in, z_in, params, **settings))(
        m, z
    )

    assert actual.shape == expected.shape == (2, *z.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_msa_stack_scans_no_prefix_unless_asked() -> None:
    params = _stack("AAAS")
    m, z, msa_mask, pair_mask = _inputs()

    traced = jax.make_jaxpr(
        lambda m_in, z_in: msa_module_stack(
            m_in, z_in, params, msa_mask=msa_mask, pair_mask=pair_mask, **SETTINGS
        )
    )(m, z)

    assert not [e for e in traced.jaxpr.eqns if e.primitive.name == "scan"]


def test_msa_stack_leaves_a_prefix_with_a_python_scalar_leaf_unrolled() -> None:
    """The first block's bias is a Python float, the others' 0-d arrays of the
    same shape, so only the leaf-type check can refuse the scan."""
    params = _stack("AAAS")

    def with_bias(block, bias):
        opm = block.outer_product_mean
        out = opm.linear_out._replace(bias=bias)
        return block._replace(outer_product_mean=opm._replace(linear_out=out))

    blocks = params.blocks
    params = params._replace(
        blocks=(
            with_bias(blocks[0], 0.25),
            *(with_bias(b, jnp.asarray(0.25)) for b in blocks[1:3]),
            *blocks[3:],
        )
    )
    m, z, msa_mask, pair_mask = _inputs()
    settings = {"msa_mask": msa_mask, "pair_mask": pair_mask, **SETTINGS}

    traced = jax.make_jaxpr(
        lambda m_in, z_in: _prefix_stack(m_in, z_in, params, **settings)
    )(m, z)
    expected = jax.jit(_unrolled(params, settings))(m, z)
    actual = jax.jit(lambda m_in, z_in: _prefix_stack(m_in, z_in, params, **settings))(
        m, z
    )

    assert not [e for e in traced.jaxpr.eqns if e.primitive.name == "scan"]
    np.testing.assert_array_equal(actual, expected)
