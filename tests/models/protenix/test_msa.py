from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.bridge.torch_mapping import (
    map_msa_block_state_dict,
    map_msa_module_state_dict,
    map_msa_pair_weighted_averaging_state_dict,
    map_outer_product_mean_state_dict,
)
from foldjax.models.protenix.models.primitives.primitives import transition
from foldjax.models.protenix.models.trunk_blocks import msa as msa_blocks
from foldjax.models.protenix.models.trunk_blocks.msa import (
    MSABlockParams,
    msa_block,
    msa_module,
    msa_pair_weighted_averaging,
    outer_product_mean,
    pad_msa_features_to_bucket,
    sample_msa_cycle_features,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import pairformer_block


def test_outer_product_mean_matches_reference_formula() -> None:
    rng = np.random.default_rng(21)
    state = _outer_product_mean_state(rng, "opm", c_m=3, c_hidden=2, c_z=4)
    params = map_outer_product_mean_state_dict(state, "opm")
    m = rng.normal(size=(2, 3, 3)).astype(np.float32)
    mask = np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32)

    actual = outer_product_mean(jnp.asarray(m), jnp.asarray(mask), params)

    ln = _layer_norm_np(m, state["opm.layer_norm.weight"], state["opm.layer_norm.bias"])
    a = (ln @ state["opm.linear_1.weight"].T) * mask[..., None]
    b = (ln @ state["opm.linear_2.weight"].T) * mask[..., None]
    outer = np.einsum("mic,mjd->ijcd", a, b).reshape(3, 3, 4)
    expected = outer @ state["opm.linear_out.weight"].T
    expected = expected + state["opm.linear_out.bias"]
    norm = np.einsum("mi,mj->ij", mask, mask)[..., None] + 1e-3
    expected = expected / norm

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


def test_outer_product_mean_chunking_changes_nothing() -> None:
    """Chunking the token axis is a memory transformation, not an approximation.

    Upstream passes the same dynamic ``chunk_size`` to ``outer_product_mean_msa``
    that it passes to the pair stack; this port dropped it, so the
    ``[N, N, C, C]`` outer product was built whole. The sum runs over MSA rows,
    which the blocks never split, so every output element sees the same
    reduction. Sizes that do not divide the token count exercise the short last
    block, and one larger than the count exercises the bypass.
    """
    rng = np.random.default_rng(4)
    n_token = 7
    state = _outer_product_mean_state(rng, "opm", c_m=3, c_hidden=2, c_z=4)
    params = map_outer_product_mean_state_dict(state, "opm")
    m = jnp.asarray(rng.normal(size=(2, n_token, 3)).astype(np.float32))
    mask = jnp.asarray((rng.random((2, n_token)) > 0.25).astype(np.float32))

    reference = outer_product_mean(m, mask, params)
    for chunk_size in (1, 2, 3, n_token - 1, n_token, n_token + 4, None):
        chunked = outer_product_mean(m, mask, params, chunk_size=chunk_size)
        assert chunked.shape == reference.shape
        np.testing.assert_allclose(
            np.asarray(chunked), np.asarray(reference), rtol=1e-6, atol=1e-6,
            err_msg=f"chunk_size={chunk_size}",
        )


def test_outer_product_mean_bounds_the_widened_outer_product(monkeypatch) -> None:
    """A tensor too wide to hold is blocked even when nothing asked for it.

    Upstream's chunk table asks for no chunking at or below 1,024 tokens, so the
    ``[N, N, C, C]`` outer product ran whole at exactly the sizes where it is the
    largest tensor in the trunk. Every other widening operation in this port
    already carries a byte ceiling; this one did not.

    The budget is monkeypatched down rather than the shapes up, because a shape
    that overruns the real budget is a shape no unit test can afford.
    """
    rng = np.random.default_rng(9)
    n_token, c_hidden = 7, 2
    state = _outer_product_mean_state(rng, "opm", c_m=3, c_hidden=c_hidden, c_z=4)
    params = map_outer_product_mean_state_dict(state, "opm")
    m = jnp.asarray(rng.normal(size=(2, n_token, 3)).astype(np.float32))
    mask = jnp.asarray((rng.random((2, n_token)) > 0.25).astype(np.float32))

    # ``chunk_size=0`` opts out of the ceiling as well as out of the policy.
    unblocked = outer_product_mean(m, mask, params, chunk_size=0)

    per_row = n_token * c_hidden * c_hidden * 4
    monkeypatch.setattr(msa_blocks, "_OPM_WIDE_BUDGET_BYTES", 2 * per_row)

    # Equal values alone would pass with the ceiling switched off entirely, so
    # assert the program blocked as well as that it agrees.
    emitted = jax.make_jaxpr(lambda x, k: outer_product_mean(x, k, params))(m, mask)
    assert "scan" in str(emitted), "the ceiling did not block anything"

    blocked = outer_product_mean(m, mask, params)
    np.testing.assert_allclose(
        np.asarray(blocked), np.asarray(unblocked), rtol=1e-6, atol=1e-6
    )
    assert np.asarray(outer_product_mean(m, mask, params, chunk_size=0)).shape == (
        n_token,
        n_token,
        4,
    )


def test_outer_product_mean_blocks_are_sequenced_by_a_loop() -> None:
    """Blocking has to be a loop, not a list of independent subgraphs.

    A Python list hands XLA one subgraph per block, and its scheduler may run
    every widening contraction before any of the narrowing dots -- which keeps
    every block's widened tensor live at once and bounds nothing. Compiling the
    module alone on CPU at 1,003 tokens put the unblocked form at 7.68 GiB, the
    list at 5.80 and the loop at 2.08. The difference is invisible in the
    values, so only the emitted program can assert it.
    """
    rng = np.random.default_rng(12)
    n_token = 8
    state = _outer_product_mean_state(rng, "opm", c_m=3, c_hidden=2, c_z=4)
    params = map_outer_product_mean_state_dict(state, "opm")
    m = jnp.asarray(rng.normal(size=(2, n_token, 3)).astype(np.float32))
    mask = jnp.asarray((rng.random((2, n_token)) > 0.25).astype(np.float32))

    blocked = jax.make_jaxpr(
        lambda x, k: outer_product_mean(x, k, params, chunk_size=3)
    )(m, mask)
    assert "scan" in str(blocked)
    np.testing.assert_allclose(
        np.asarray(outer_product_mean(m, mask, params, chunk_size=3)),
        np.asarray(outer_product_mean(m, mask, params, chunk_size=0)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_outer_product_mean_budget_never_widens_a_request() -> None:
    """The ceiling narrows a block size; it never hands back a larger one."""
    a = jnp.zeros((2, 1024, 32), dtype=jnp.bfloat16)
    allowed = msa_blocks._opm_block_rows(a, None)
    assert allowed is not None and 1 <= allowed < 1024
    assert msa_blocks._opm_block_rows(a, 1) == 1
    assert msa_blocks._opm_block_rows(a, 4096) == allowed
    # Small enough to fit whole: the request passes through untouched, `None`
    # included, so a short job compiles the same program it always did.
    small = jnp.zeros((2, 8, 2), dtype=jnp.float32)
    assert msa_blocks._opm_block_rows(small, None) is None
    assert msa_blocks._opm_block_rows(small, 3) == 3


def test_outer_product_mean_keeps_the_unrolled_blocks_under_a_mesh(
    monkeypatch,
) -> None:
    """A context-parallel mesh keeps the form it had, loop and ceiling included.

    The loop writes its blocks at a traced offset, which a row-sharded carry
    cannot survive, so the mesh keeps the unrolled list -- and the automatic
    ceiling stays out of the way, because under a mesh the pair tensor is split
    across devices already. Selecting the branch is what changed here; the
    arithmetic on either side of it did not.
    """
    rng = np.random.default_rng(11)
    n_token, c_hidden = 8, 2
    state = _outer_product_mean_state(rng, "opm", c_m=3, c_hidden=c_hidden, c_z=4)
    params = map_outer_product_mean_state_dict(state, "opm")
    m = jnp.asarray(rng.normal(size=(2, n_token, 3)).astype(np.float32))
    mask = jnp.asarray((rng.random((2, n_token)) > 0.25).astype(np.float32))

    dense = outer_product_mean(m, mask, params, chunk_size=0)
    monkeypatch.setattr(msa_blocks, "_OPM_WIDE_BUDGET_BYTES", 4)
    monkeypatch.setattr(msa_blocks, "cp_mesh", lambda: object())

    # The ceiling would have blocked this at one row; under a mesh it does not
    # fire at all, so nothing asks for blocks and the dense form runs.
    unasked = jax.make_jaxpr(lambda x, k: outer_product_mean(x, k, params))(m, mask)
    assert "scan" not in str(unasked), "the ceiling must not fire under a mesh"

    asked = jax.make_jaxpr(
        lambda x, k: outer_product_mean(x, k, params, chunk_size=3)
    )(m, mask)
    assert "scan" not in str(asked), "the mesh must keep the unrolled blocks"
    np.testing.assert_allclose(
        np.asarray(outer_product_mean(m, mask, params, chunk_size=3)),
        np.asarray(dense),
        rtol=1e-6,
        atol=1e-6,
    )


def test_msa_block_gives_the_outer_product_mean_its_chunk_size(monkeypatch) -> None:
    """The knob has to reach the layer, not just exist on the config.

    This is the failure that produced the bug: the chunk size was resolved and
    handed to four other consumers, and the one call that needed it most was
    written without it.
    """
    seen = {}

    def fake_outer_product_mean(m, mask, params, **kwargs):
        seen.update(kwargs)
        return jnp.zeros(m.shape[-2:-1] * 2 + (1,), dtype=m.dtype)

    monkeypatch.setattr(msa_blocks, "outer_product_mean", fake_outer_product_mean)
    monkeypatch.setattr(
        msa_blocks, "pairformer_block", lambda s, z, *a, **k: (s, z)
    )
    params = MSABlockParams(
        outer_product_mean=object(),
        msa_pair_weighted_averaging=None,
        msa_transition=None,
        pair_stack=object(),
    )
    msa_blocks.msa_block(
        jnp.zeros((2, 3, 1)),
        jnp.zeros((3, 3, 1)),
        None,
        params,
        opm_chunk_size=128,
    )
    assert seen.get("chunk_size") == 128


def test_msa_pair_weighted_averaging_matches_reference_formula() -> None:
    rng = np.random.default_rng(22)
    state = _msa_pair_weighted_state(rng, "mpwa", c_m=4, c_z=3, heads=2, c=2)
    params = map_msa_pair_weighted_averaging_state_dict(state, "mpwa")
    m = rng.normal(size=(2, 3, 4)).astype(np.float32)
    z = rng.normal(size=(3, 3, 3)).astype(np.float32)

    actual = msa_pair_weighted_averaging(jnp.asarray(m), jnp.asarray(z), params)

    m_norm = _layer_norm_np(
        m,
        state["mpwa.layernorm_m.weight"],
        state["mpwa.layernorm_m.bias"],
    )
    v = (m_norm @ state["mpwa.linear_no_bias_mv.weight"].T).reshape(2, 3, 2, 2)
    z_norm = _layer_norm_np(
        z,
        state["mpwa.layernorm_z.weight"],
        state["mpwa.layernorm_z.bias"],
    )
    b = z_norm @ state["mpwa.linear_no_bias_z.weight"].T
    w = _softmax_np(b, axis=-2)
    g = _sigmoid_np(m_norm @ state["mpwa.linear_no_bias_mg.weight"].T).reshape(
        2,
        3,
        2,
        2,
    )
    o = g * np.einsum("ijh,mjhc->mihc", w, v)
    expected = o.reshape(2, 3, 4) @ state["mpwa.linear_no_bias_out.weight"].T

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


def test_msa_block_matches_reference_composition() -> None:
    rng = np.random.default_rng(23)
    state = _msa_block_state(
        rng,
        "msa.blocks.0",
        c_m=4,
        c_z=4,
        msa_heads=2,
        msa_c=2,
        pair_heads=2,
        is_last=False,
    )
    params = map_msa_block_state_dict(state, "msa.blocks.0")
    m = rng.normal(size=(2, 3, 4)).astype(np.float32)
    z = rng.normal(size=(3, 3, 4)).astype(np.float32)
    pair_mask = np.ones((3, 3), dtype=np.float32)

    actual_m, actual_z = msa_block(
        jnp.asarray(m),
        jnp.asarray(z),
        jnp.asarray(pair_mask),
        params,
    )

    # Protenix's `MSABlock.forward`, in its order: communication, then the MSA
    # stack, then the pair stack. Writing it the other way round type-checks and
    # runs -- the shapes are identical either way -- so this ordering is the
    # whole content of the test. Getting it backwards fed the outer product mean
    # an already-updated `m` and the MSA stack a stale `z`, and the trunk's own
    # recycling then amplified the error until `z` was an order of magnitude
    # too large.
    expected_z = jnp.asarray(z) + outer_product_mean(
        jnp.asarray(m),
        None,
        params.outer_product_mean,
    )
    expected_m = jnp.asarray(m) + msa_pair_weighted_averaging(
        jnp.asarray(m),
        expected_z,
        params.msa_pair_weighted_averaging,
    )
    expected_m = expected_m + transition(expected_m, params.msa_transition)
    _, expected_z = pairformer_block(
        None,
        expected_z,
        jnp.asarray(pair_mask),
        params.pair_stack,
    )

    np.testing.assert_allclose(np.asarray(actual_m), np.asarray(expected_m), atol=1e-5)
    np.testing.assert_allclose(np.asarray(actual_z), np.asarray(expected_z), atol=1e-5)


def _recording_msa_block(monkeypatch) -> list[tuple[str, float]]:
    """Replace the block's sub-updates with recorders and return the log.

    Each stub returns a constant the others cannot produce, so the log says not
    only in what order the sub-updates ran but what each one was handed.
    """

    calls: list[tuple[str, float]] = []

    def fake_outer_product_mean(m, mask, params, **kwargs):
        calls.append(("outer_product_mean", float(jnp.sum(m))))
        return jnp.full((3, 3, 4), 7.0, dtype=jnp.float32)

    def fake_pair_weighted_averaging(m, z, params, **kwargs):
        calls.append(("msa_pair_weighted_averaging", float(jnp.sum(z))))
        return jnp.full_like(m, 30.0)

    def fake_transition(value, params, **kwargs):
        calls.append(("transition", float(jnp.sum(value))))
        return jnp.full_like(value, 500.0)

    def fake_pairformer_block(s, z, pair_mask, params, **kwargs):
        calls.append(("pair_stack", float(jnp.sum(z))))
        return None, z

    monkeypatch.setattr(msa_blocks, "outer_product_mean", fake_outer_product_mean)
    monkeypatch.setattr(
        msa_blocks, "msa_pair_weighted_averaging", fake_pair_weighted_averaging
    )
    monkeypatch.setattr(msa_blocks, "transition", fake_transition)
    monkeypatch.setattr(msa_blocks, "pairformer_block", fake_pairformer_block)
    return calls


def test_msa_block_hands_each_sub_update_the_right_tensor(monkeypatch) -> None:
    """Communication reads the incoming MSA; the MSA stack reads the updated pair.

    Swapping the two is silent: both orders type-check, both preserve every
    shape, and both produce a plausible number. Only which tensor each
    sub-update receives tells them apart, so that is what this records rather
    than recomputing the block's arithmetic -- an expectation built out of the
    same calls in the same order agrees with either arrangement.
    """

    calls = _recording_msa_block(monkeypatch)
    params = MSABlockParams(
        outer_product_mean=object(),
        msa_pair_weighted_averaging=object(),
        msa_transition=object(),
        pair_stack=object(),
    )
    m = jnp.ones((2, 3, 4), dtype=jnp.float32)
    z = jnp.zeros((3, 3, 4), dtype=jnp.float32)

    out_m, out_z = msa_block(m, z, None, params)

    assert [name for name, _ in calls] == [
        "outer_product_mean",
        "msa_pair_weighted_averaging",
        "transition",
        "pair_stack",
    ]
    # The outer product mean sees `m` exactly as the block received it, not an
    # `m` the MSA stack has already added 30 and 500 to.
    assert calls[0][1] == float(jnp.sum(m))
    # The MSA stack sees `z` after that mean has been added to it, not the zeros
    # the block started with.
    assert calls[1][1] == 7.0 * z.size
    assert float(jnp.sum(out_m)) == float(jnp.sum(m)) + (30.0 + 500.0) * m.size
    assert float(jnp.sum(out_z)) == 7.0 * z.size


def test_msa_block_reverses_the_order_for_opendde(monkeypatch) -> None:
    """`msa_stack_first=True` is OpenDDE's order, and OpenDDE shares this module.

    Protenix runs the communication first; OpenDDE runs the MSA stack first and
    says so in its own source -- "Boltz updates MSA first, then writes the
    refreshed MSA state back to z". Both orders preserve every shape, so a port
    that hands one model the other's ordering fails nothing: it just computes a
    different trunk. That is what happened when this module was corrected for
    Protenix while OpenDDE went on calling it -- OpenDDE's pair representation
    fell to a correlation of 0.47 with upstream's, and its structures moved
    31.5 A on a matched random tape, while every confidence score it reports
    stayed within 1% of upstream's.
    """

    calls = _recording_msa_block(monkeypatch)
    params = MSABlockParams(
        outer_product_mean=object(),
        msa_pair_weighted_averaging=object(),
        msa_transition=object(),
        pair_stack=object(),
    )
    m = jnp.ones((2, 3, 4), dtype=jnp.float32)
    z = jnp.zeros((3, 3, 4), dtype=jnp.float32)

    out_m, out_z = msa_block(m, z, None, params, msa_stack_first=True)

    assert [name for name, _ in calls] == [
        "msa_pair_weighted_averaging",
        "transition",
        "outer_product_mean",
        "pair_stack",
    ]
    # The MSA stack sees the `z` the block was handed, still zeros...
    assert calls[0][1] == 0.0
    # ...and the outer product mean sees the `m` that stack just produced.
    assert calls[2][1] == float(jnp.sum(m)) + (30.0 + 500.0) * m.size
    assert float(jnp.sum(out_m)) == float(jnp.sum(m)) + (30.0 + 500.0) * m.size
    assert float(jnp.sum(out_z)) == 7.0 * z.size


def test_msa_block_final_block_skips_the_msa_stack(monkeypatch) -> None:
    """Protenix drops the MSA path from the last block and returns no `m`."""

    calls = _recording_msa_block(monkeypatch)
    params = MSABlockParams(
        outer_product_mean=object(),
        msa_pair_weighted_averaging=None,
        msa_transition=None,
        pair_stack=object(),
    )
    m = jnp.ones((2, 3, 4), dtype=jnp.float32)
    z = jnp.zeros((3, 3, 4), dtype=jnp.float32)

    out_m, out_z = msa_block(m, z, None, params)

    assert [name for name, _ in calls] == ["outer_product_mean", "pair_stack"]
    assert out_m is None
    assert float(jnp.sum(out_z)) == 7.0 * z.size


def test_msa_module_builds_msa_embedding_and_discovers_blocks() -> None:
    rng = np.random.default_rng(24)
    state = {
        "msa_module.linear_no_bias_m.weight": rng.normal(size=(4, 34)).astype(
            np.float32
        ),
        "msa_module.linear_no_bias_s.weight": rng.normal(size=(4, 5)).astype(
            np.float32
        ),
    }
    state.update(
        _msa_block_state(
            rng,
            "msa_module.blocks.0",
            c_m=4,
            c_z=4,
            msa_heads=2,
            msa_c=2,
            pair_heads=2,
            is_last=False,
        )
    )
    state.update(
        _msa_block_state(
            rng,
            "msa_module.blocks.1",
            c_m=4,
            c_z=4,
            msa_heads=2,
            msa_c=2,
            pair_heads=2,
            is_last=True,
        )
    )
    params = map_msa_module_state_dict(state, "msa_module")
    features = {
        "msa": jnp.asarray([[0, 1, 2], [3, 4, 5]]),
        "has_deletion": jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
        "deletion_value": jnp.asarray([[0.0, 0.5, 0.0], [0.25, 0.0, 1.0]]),
    }
    z = rng.normal(size=(3, 3, 4)).astype(np.float32)
    s_inputs = rng.normal(size=(3, 5)).astype(np.float32)

    actual = msa_module(features, jnp.asarray(z), jnp.asarray(s_inputs), None, params)
    padded = msa_module(
        pad_msa_features_to_bucket(features, bucket_size=4),
        jnp.asarray(z),
        jnp.asarray(s_inputs),
        None,
        params,
    )
    compiled = msa_module(
        features,
        jnp.asarray(z),
        jnp.asarray(s_inputs),
        None,
        params,
        triangle_attention_backend="xla_jit",
    )

    assert len(params.blocks) == 2
    assert params.blocks[-1].msa_pair_weighted_averaging is None
    assert actual.shape == (3, 3, 4)
    np.testing.assert_allclose(
        np.asarray(padded), np.asarray(actual), rtol=1e-6, atol=2e-5
    )
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(actual), atol=1e-5)


def test_sample_msa_cycle_features_is_deterministic_and_aligned() -> None:
    features = {
        "msa": np.arange(30).reshape(5, 6),
        "has_deletion": np.arange(30).reshape(5, 6) + 100,
        "deletion_value": np.arange(30).reshape(5, 6) + 200,
    }

    first = sample_msa_cycle_features(features, n_cycle=4, seed=101, bucket_size=4)
    second = sample_msa_cycle_features(features, n_cycle=4, seed=101, bucket_size=4)

    assert len(first) == 4
    padded_depths = {cycle["msa"].shape[-2] for cycle in first}
    assert len(padded_depths) == 1
    assert next(iter(padded_depths)) <= 5
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left["msa"], right["msa"])
        np.testing.assert_array_equal(left["msa_mask"], right["msa_mask"])
        selected = left["msa_mask"].astype(bool)
        assert 1 <= selected[:, 0].sum() <= 5
        np.testing.assert_array_equal(
            left["has_deletion"][selected] - 100,
            left["msa"][selected],
        )
        np.testing.assert_array_equal(
            left["deletion_value"][selected] - 200,
            left["msa"][selected],
        )


def test_pad_msa_features_to_bucket_preserves_rows_and_adds_mask() -> None:
    features = {
        "msa": np.arange(30).reshape(5, 6),
        "has_deletion": np.arange(30).reshape(5, 6) + 100,
        "deletion_value": np.arange(30).reshape(5, 6) + 200,
        "restype": np.ones((6, 32), dtype=np.float32),
    }

    padded = pad_msa_features_to_bucket(features, bucket_size=4)

    assert padded["msa"].shape == (8, 6)
    np.testing.assert_array_equal(padded["msa"][:5], features["msa"])
    np.testing.assert_array_equal(padded["has_deletion"][:5], features["has_deletion"])
    np.testing.assert_array_equal(
        padded["deletion_value"][:5], features["deletion_value"]
    )
    np.testing.assert_array_equal(padded["msa_mask"][:5], 1.0)
    np.testing.assert_array_equal(padded["msa_mask"][5:], 0.0)
    assert padded["restype"] is features["restype"]

    skipped = pad_msa_features_to_bucket(
        features,
        bucket_size=4,
        max_padding_rows=2,
    )
    assert skipped is features


def _outer_product_mean_state(
    rng: np.random.Generator,
    prefix: str,
    *,
    c_m: int,
    c_hidden: int,
    c_z: int,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.layer_norm.weight": rng.normal(size=(c_m,)).astype(np.float32),
        f"{prefix}.layer_norm.bias": rng.normal(size=(c_m,)).astype(np.float32),
        f"{prefix}.linear_1.weight": rng.normal(size=(c_hidden, c_m)).astype(
            np.float32
        ),
        f"{prefix}.linear_2.weight": rng.normal(size=(c_hidden, c_m)).astype(
            np.float32
        ),
        f"{prefix}.linear_out.weight": rng.normal(
            size=(c_z, c_hidden * c_hidden)
        ).astype(np.float32),
        f"{prefix}.linear_out.bias": rng.normal(size=(c_z,)).astype(np.float32),
    }


def _msa_pair_weighted_state(
    rng: np.random.Generator,
    prefix: str,
    *,
    c_m: int,
    c_z: int,
    heads: int,
    c: int,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.layernorm_m.weight": rng.normal(size=(c_m,)).astype(np.float32),
        f"{prefix}.layernorm_m.bias": rng.normal(size=(c_m,)).astype(np.float32),
        f"{prefix}.linear_no_bias_mv.weight": rng.normal(size=(heads * c, c_m)).astype(
            np.float32
        ),
        f"{prefix}.layernorm_z.weight": rng.normal(size=(c_z,)).astype(np.float32),
        f"{prefix}.layernorm_z.bias": rng.normal(size=(c_z,)).astype(np.float32),
        f"{prefix}.linear_no_bias_z.weight": rng.normal(size=(heads, c_z)).astype(
            np.float32
        ),
        f"{prefix}.linear_no_bias_mg.weight": rng.normal(size=(heads * c, c_m)).astype(
            np.float32
        ),
        f"{prefix}.linear_no_bias_out.weight": rng.normal(size=(c_m, heads * c)).astype(
            np.float32
        ),
    }


def _msa_block_state(
    rng: np.random.Generator,
    prefix: str,
    *,
    c_m: int,
    c_z: int,
    msa_heads: int,
    msa_c: int,
    pair_heads: int,
    is_last: bool,
) -> dict[str, np.ndarray]:
    state = _outer_product_mean_state(
        rng,
        f"{prefix}.outer_product_mean_msa",
        c_m=c_m,
        c_hidden=2,
        c_z=c_z,
    )
    if not is_last:
        state.update(
            _msa_pair_weighted_state(
                rng,
                f"{prefix}.msa_stack.msa_pair_weighted_averaging",
                c_m=c_m,
                c_z=c_z,
                heads=msa_heads,
                c=msa_c,
            )
        )
        _add_transition_state(state, rng, f"{prefix}.msa_stack.transition_m", c_m)
    _add_pair_stack_state(state, rng, f"{prefix}.pair_stack", c_z, pair_heads)
    return state


def _add_pair_stack_state(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    c_z: int,
    heads: int,
) -> None:
    _add_triangle_multiplication_state(state, rng, f"{prefix}.tri_mul_out", c_z)
    _add_triangle_multiplication_state(state, rng, f"{prefix}.tri_mul_in", c_z)
    _add_triangle_attention_state(state, rng, f"{prefix}.tri_att_start", c_z, heads)
    _add_triangle_attention_state(state, rng, f"{prefix}.tri_att_end", c_z, heads)
    _add_transition_state(state, rng, f"{prefix}.pair_transition", c_z)


def _add_triangle_multiplication_state(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    c_z: int,
) -> None:
    state[f"{prefix}.layer_norm_in.weight"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.layer_norm_in.bias"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.layer_norm_out.weight"] = rng.normal(size=(c_z,)).astype(
        np.float32
    )
    state[f"{prefix}.layer_norm_out.bias"] = rng.normal(size=(c_z,)).astype(np.float32)
    for name in ("linear_a_p", "linear_a_g", "linear_b_p", "linear_b_g"):
        state[f"{prefix}.{name}.weight"] = rng.normal(size=(c_z, c_z)).astype(
            np.float32
        )
    state[f"{prefix}.linear_z.weight"] = rng.normal(size=(c_z, c_z)).astype(np.float32)
    state[f"{prefix}.linear_g.weight"] = rng.normal(size=(c_z, c_z)).astype(np.float32)


def _add_triangle_attention_state(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    c_z: int,
    heads: int,
) -> None:
    state[f"{prefix}.layer_norm.weight"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.layer_norm.bias"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.linear.weight"] = rng.normal(size=(heads, c_z)).astype(np.float32)
    for name in ("linear_q", "linear_k", "linear_v", "linear_o", "linear_g"):
        state[f"{prefix}.mha.{name}.weight"] = rng.normal(size=(c_z, c_z)).astype(
            np.float32
        )


def _add_transition_state(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    c_in: int,
) -> None:
    hidden = c_in * 4
    state[f"{prefix}.layernorm1.weight"] = rng.normal(size=(c_in,)).astype(np.float32)
    state[f"{prefix}.layernorm1.bias"] = rng.normal(size=(c_in,)).astype(np.float32)
    state[f"{prefix}.linear_no_bias_a.weight"] = rng.normal(size=(hidden, c_in)).astype(
        np.float32
    )
    state[f"{prefix}.linear_no_bias_b.weight"] = rng.normal(size=(hidden, c_in)).astype(
        np.float32
    )
    state[f"{prefix}.linear_no_bias.weight"] = rng.normal(size=(c_in, hidden)).astype(
        np.float32
    )


def _layer_norm_np(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_np(x: np.ndarray, axis: int) -> np.ndarray:
    y = x - np.max(x, axis=axis, keepdims=True)
    y = np.exp(y)
    return y / np.sum(y, axis=axis, keepdims=True)


def test_scanned_blocks_match_the_loop_and_actually_fire():
    """The MSA stack is scanned over its uniform prefix.

    Protenix omits the MSA path from the last block, so a whole-stack uniformity
    check would always be false and the scan would be dead code. This asserts both
    halves: that the prefix is found, and that scanning it changes nothing.
    """
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models.protenix.models.trunk_blocks.msa import (
        _uniform_prefix,
        msa_module,
    )

    rng = np.random.default_rng(11)
    state = {
        "msa_module.linear_no_bias_m.weight": rng.normal(size=(4, 34)).astype(
            np.float32
        ),
        "msa_module.linear_no_bias_s.weight": rng.normal(size=(4, 5)).astype(
            np.float32
        ),
    }
    for index in range(3):
        state.update(
            _msa_block_state(
                rng,
                f"msa_module.blocks.{index}",
                c_m=4,
                c_z=4,
                msa_heads=2,
                msa_c=2,
                pair_heads=2,
                is_last=index == 2,
            )
        )
    params = map_msa_module_state_dict(state, "msa_module")
    assert len(params.blocks) == 3
    assert params.blocks[-1].msa_pair_weighted_averaging is None
    # Two uniform blocks then a different one: the prefix is what makes this useful.
    assert _uniform_prefix(params.blocks) == 2

    features = {
        "msa": jnp.asarray([[0, 1, 2], [3, 4, 5]]),
        "has_deletion": jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
        "deletion_value": jnp.asarray([[0.0, 0.5, 0.0], [0.25, 0.0, 1.0]]),
    }
    z = jnp.asarray(rng.normal(size=(3, 3, 4)).astype(np.float32))
    s_inputs = jnp.asarray(rng.normal(size=(3, 5)).astype(np.float32))

    scanned = msa_module(features, z, s_inputs, None, params, use_scan=True)
    looped = msa_module(features, z, s_inputs, None, params, use_scan=False)
    # A scan and a loop reduce in different orders, so this is float32 noise rather
    # than exact equality: the outputs reach ~84 here, where one ulp is ~1e-5, and
    # the observed worst case is 3.1e-5.
    np.testing.assert_allclose(
        np.asarray(scanned, dtype=np.float64),
        np.asarray(looped, dtype=np.float64),
        rtol=1e-6,
        atol=1e-4,
    )


def test_a_fully_uniform_stack_is_scanned_entirely():
    from foldjax.models.protenix.models.trunk_blocks.msa import _uniform_prefix

    rng = np.random.default_rng(12)
    state = {
        "msa_module.linear_no_bias_m.weight": rng.normal(size=(4, 34)).astype(
            np.float32
        ),
        "msa_module.linear_no_bias_s.weight": rng.normal(size=(4, 5)).astype(
            np.float32
        ),
    }
    for index in range(3):
        state.update(
            _msa_block_state(
                rng,
                f"msa_module.blocks.{index}",
                c_m=4,
                c_z=4,
                msa_heads=2,
                msa_c=2,
                pair_heads=2,
                is_last=False,
            )
        )
    params = map_msa_module_state_dict(state, "msa_module")
    assert _uniform_prefix(params.blocks) == 3
