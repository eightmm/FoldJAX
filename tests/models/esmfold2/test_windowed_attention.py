"""The blocked windowed attention computes what the dense path computes.

The dense path builds `[batch, heads, atoms, atoms]` and discards everything
outside a band 129 wide. At the released 32 samples and 7,776 atoms that is
14,762 MiB, twice over with its softmax. `_windowed_attention` walks the query
rows in blocks so only one block's scores exist at a time.

The whole change is worthless unless it computes the same function, so that is
what is asserted -- against the dense formula itself rather than against a
recomputed expectation, and **in float32**. Comparing the two through
`swa_attention` instead would compare them after both were narrowed to
bfloat16, where accumulation order alone moves the answer by ~4e-3 relative
(2**-8 is bfloat16's epsilon). That comparison cannot tell a logic error from
rounding, which is the whole question here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import atom


def _dense_reference(q, k, v, valid, half_window, scale):
    """The dense branch of `swa_attention`, verbatim."""
    logits = jnp.einsum("bihd,bjhd->bhij", q, k) * scale
    allowed = atom.sliding_window_mask(valid, half_window)[:, None]
    logits = jnp.where(allowed, logits, jnp.finfo(logits.dtype).min)
    return jnp.einsum("bhij,bjhd->bihd", jax.nn.softmax(logits, axis=-1), v)


def _both(length, n_real, half_window, rows_per_block, heads=2, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    shape = (1, length, heads, dim)
    q, k, v = (jnp.asarray(rng.standard_normal(shape), jnp.float32) for _ in range(3))
    valid = jnp.asarray((np.arange(length) < n_real).astype(np.float32))[None]
    scale = dim**-0.5
    reference = _dense_reference(q, k, v, valid, half_window, scale)
    blocked = atom._windowed_attention(
        q,
        k,
        v,
        valid,
        half_window=half_window,
        rows_per_block=rows_per_block,
        scale=scale,
    )
    # Rows past the valid prefix are zeroed by the caller, so nothing is
    # claimed about them here.
    return (
        np.asarray(reference, np.float64)[:, :n_real],
        np.asarray(blocked, np.float64)[:, :n_real],
    )


@pytest.mark.parametrize(
    "name, length, n_real, half_window, rows_per_block",
    [
        # `half_window=64` is the released value for both atom stacks.
        ("length is a multiple of the block", 512, 512, 64, 256),
        ("padding at the end", 512, 400, 64, 256),
        ("length is not a multiple of the block", 500, 437, 64, 128),
        ("a small window and a small block", 64, 50, 4, 16),
        ("a different block size", 512, 512, 64, 128),
        ("exactly one slice wide", 384, 300, 64, 256),
        ("mostly padding", 512, 70, 64, 256),
        # The valid prefix ending mid-block is where an index-vs-rank confusion
        # would show, since the two stop agreeing past the prefix.
        ("the prefix ends inside a block", 512, 255, 64, 256),
        # A window narrower than a block puts every block edge in play.
        ("the window straddles every block edge", 512, 512, 1, 8),
    ],
)
def test_it_matches_the_dense_path(
    name, length, n_real, half_window, rows_per_block
) -> None:
    reference, blocked = _both(length, n_real, half_window, rows_per_block)
    assert np.abs(reference - blocked).max() < 1e-5, name


def test_the_two_paths_are_not_bit_identical_and_here_is_the_measurement() -> None:
    """Recorded so nobody re-derives the wrong conclusion from a true premise.

    The premise is right: `jnp.finfo(dtype).min` exponentiates to exactly zero,
    so masked keys contribute nothing and a windowed softmax is the dense one.
    It does not follow that the results agree bit for bit, because floating
    point addition is not associative and the two paths contract over different
    widths -- 7,776 keys against 384 -- so XLA reduces them with different
    tilings. Adding exact zeros changes no sum; regrouping the non-zero terms
    does.

    Measured on CPU in float32, fraction of entries bitwise equal:

        L=512 all valid  R=256   61.8%     max |diff| 2.7e-07
        L=512 padded     R=256   62.3%     max |diff| 2.4e-07
        L=500            R=128   11.5%     max |diff| 5.1e-07
        L=384 one slice  R=256  100.0%     max |diff| 0
        L=512 mostly pad R=256  100.0%     max |diff| 0

    So a bit-for-bit gate would fail on most shapes and pass on two by
    coincidence of tiling, which is the worst possible gate: it would go red on
    an XLA upgrade and green on a subset that proves nothing.
    """
    reference, blocked = _both(512, 512, 64, 256)
    assert not np.array_equal(reference, blocked), (
        "these agreed bitwise, which contradicts the recorded measurement -- "
        "if XLA now reduces both paths identically this test should be "
        "rewritten, not deleted"
    )
    assert np.abs(reference - blocked).max() < 1e-5


def test_blocked_stays_within_the_dense_accuracy_scale() -> None:
    """Separate harmless reduction tiling from a wrong attention window.

    A max-error ratio is an unstable comparison between two single extreme
    entries: across CPU lowerings the same seed made the blocked/dense ratio
    move from 0.976 to 1.063 even though both errors stayed near 5e-7. A
    ten-seed sweep is much steadier in RMS. The blocked/dense RMS ratio was
    0.951--1.204 across the shapes below, while an indexing or masking error
    moves both RMS and max error by orders of magnitude.

    Keep a generous but bounded RMS-ratio gate and an independent max-error
    backstop derived from float32 epsilon and the output scale. This records
    the real numerical contract without pretending that a different reduction
    tree must always beat the dense tree at its single worst element.
    """
    jax.config.update("jax_enable_x64", True)
    try:
        for length, n_real, half_window, rows_per_block in [
            (512, 512, 64, 256),
            (512, 400, 64, 256),
            (500, 437, 64, 128),
            (512, 512, 1, 8),
        ]:
            for seed in range(10):
                rng = np.random.default_rng(seed)
                shape = (1, length, 2, 32)
                wide = [
                    jnp.asarray(rng.standard_normal(shape), jnp.float64)
                    for _ in range(3)
                ]
                valid64 = jnp.asarray((np.arange(length) < n_real).astype(np.float64))[
                    None
                ]
                scale = 32**-0.5
                exact = np.asarray(
                    _dense_reference(*wide, valid64, half_window, scale)
                )[:, :n_real]

                narrow = [a.astype(jnp.float32) for a in wide]
                valid = valid64.astype(jnp.float32)
                dense_delta = (
                    np.asarray(_dense_reference(*narrow, valid, half_window, scale))[
                        :, :n_real
                    ]
                    - exact
                )
                blocked_delta = (
                    np.asarray(
                        atom._windowed_attention(
                            *narrow,
                            valid,
                            half_window=half_window,
                            rows_per_block=rows_per_block,
                            scale=scale,
                        )
                    )[:, :n_real]
                    - exact
                )
                dense_rms = np.sqrt(np.mean(np.square(dense_delta)))
                blocked_rms = np.sqrt(np.mean(np.square(blocked_delta)))
                output_scale = np.abs(exact).max()
                max_error_budget = 32 * np.finfo(np.float32).eps * output_scale

                assert blocked_rms <= dense_rms * 1.5, (
                    f"L={length} R={rows_per_block} seed={seed}: blocked RMS "
                    f"{blocked_rms:.3e} against dense {dense_rms:.3e}"
                )
                assert np.abs(blocked_delta).max() <= max_error_budget, (
                    f"L={length} R={rows_per_block} seed={seed}: blocked max "
                    f"{np.abs(blocked_delta).max():.3e} exceeds the float32 "
                    f"scale budget {max_error_budget:.3e}"
                )
    finally:
        jax.config.update("jax_enable_x64", False)


def test_the_block_size_does_not_change_the_answer() -> None:
    """A block size is a memory/time knob, so it must not be a numerical one."""
    first = _both(512, 400, 64, 128)[1]
    second = _both(512, 400, 64, 256)[1]
    assert np.abs(first - second).max() < 1e-5


def test_short_inputs_take_the_dense_path() -> None:
    """A slice wider than the array cannot be clamped into range.

    The guard is on static shapes, so exactly one of the two branches is
    traced. Asserted by watching which primitives the jaxpr contains rather
    than by reading the condition back.
    """
    rng = np.random.default_rng(0)
    width, heads = 64, 2
    params = {
        name: jnp.asarray(rng.standard_normal(shape).astype(np.float32) * 0.05)
        for name, shape in [
            ("Wqkv.weight", (3 * width, width)),
            ("gate_proj.weight", (width, width)),
            ("out_proj.weight", (width, width)),
        ]
    }

    def call(length):
        x = jnp.zeros((1, length, width), jnp.float32)
        cos, sin = atom.build_3d_rope(
            jnp.zeros((1, length, 3), jnp.float32),
            jnp.arange(length, dtype=jnp.int32)[None] // 3,
            head_dim=width // heads,
        )
        return str(
            jax.make_jaxpr(
                lambda t: atom.swa_attention(
                    t,
                    params,
                    "",
                    cos=cos,
                    sin=sin,
                    valid=jnp.ones((1, length), jnp.float32),
                    n_heads=heads,
                    half_window=64,
                    rows_per_block=256,
                )
            )(x)
        )

    # 128 < 256 + 2*64, so the dense branch; 512 >= 384, so the blocked one.
    assert "scan" not in call(128)
    assert "scan" in call(512)


def test_the_atom_mask_really_is_a_prefix() -> None:
    """The exactness argument rests on this and nothing else enforces it.

    `_windowed_attention` slices keys by raw index, which covers the window in
    packed rank only while `rank[i] == i` -- true exactly when the mask is a
    prefix. `features.py` has a single writer and fills it in atom order, so
    this holds; if a future featurizer interleaves padding, the blocked path
    would read the wrong keys and this test is what should fail first.
    """
    from foldjax.models.esmfold2.data import features

    source = features.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert text.count("atom_attention_mask[") == 1, (
        "more than one writer of atom_attention_mask; the prefix property that "
        "_windowed_attention relies on is no longer guaranteed by inspection"
    )

    # And the property itself, on a mask built the way the featurizer builds one.
    for n_real, length in [(1, 32), (17, 32), (7000, 7776)]:
        mask = np.zeros(length, dtype=bool)
        mask[:n_real] = True
        rank = np.cumsum(mask.astype(np.int32)) - 1
        index = np.arange(length)
        assert (rank[:n_real] == index[:n_real]).all()


def test_the_default_is_blocked_because_it_won_on_both_axes(monkeypatch) -> None:
    """This shipped as `dense`, and that was correct until it was measured.

    The old name here was `test_the_default_is_the_dense_path`, and its
    argument was that being exact and saving 24,633 MiB are not reasons to flip
    a default before the two paths have been timed -- the blocked form trades
    one large matmul for 31 scanned small ones, so its cost should grow where
    its benefit shrinks. **That was a reason to wait, not a finding, and the
    measurement arrived.** Shipped settings, shipped allocator, 1,003 residues,
    32 samples, three timed runs per arm, arms alternating, warmups discarded:

        dense    49.30 s  +-0.20   48.516 GiB
        blk256   46.73 s  +-0.60   24.460 GiB

    5.2% faster and 24.056 GiB smaller. Clean rather than marginal: the slowest
    blocked run beat the fastest dense one, so no pairing of runs reverses it.
    And the clock control points the conservative way -- dense held the highest
    mean SM clock, 1749 against 1711, and lost anyway, so the margin
    understates.

    Kept and renamed rather than flipped in place, so the next reader finds
    that the cautious default was chosen deliberately and retired on evidence.
    """
    monkeypatch.delenv("ESMFOLD2_ATOM_ATTENTION_BACKEND", raising=False)
    assert atom._atom_attention_backend() == "blocked"
    assert atom._resolve_rows_per_block(None) == atom.ATOM_ROWS_PER_BLOCK


def test_the_environment_can_still_reach_the_dense_path(monkeypatch) -> None:
    """`blocked` is a default, not a lock, and dense is now the escape hatch.

    While dense was the default this direction was free -- doing nothing got
    you the safe path. Now the safe path has to be asked for, so the way back
    is load-bearing in a way it was not before: a size where the scan
    misbehaves, or a backend where `lax.scan` lowers badly, needs a route out
    that does not involve editing the source.
    """
    monkeypatch.setenv("ESMFOLD2_ATOM_ATTENTION_BACKEND", "dense")
    assert atom._atom_attention_backend() == "dense"
    assert atom._resolve_rows_per_block(None) == 0


def test_the_block_size_is_not_a_knob_that_does_anything(monkeypatch) -> None:
    """Both block sizes were measured and neither wins; 256 is arbitrary.

    R=128 reads 256 keys per 128 rows against R=256's 384 per 256 -- a third
    less wasted score arithmetic -- and pays for it in twice the scan steps.
    Measured, those cancel or both sit under the noise: 46.90 s +-2.10 against
    46.73 s +-0.60, arms overlapping, and an identical peak of 24.460 GiB.

    **That is a result, not a missing measurement**, and the reason to pin it
    here is that the arithmetic argues loudly for 128 and the card disagrees.
    The override stays as an escape hatch for a size nobody has run; it is not
    a tuning knob and this test exists so nobody documents it as one.
    """
    monkeypatch.delenv("ESMFOLD2_ATOM_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("ESMFOLD2_ATOM_ROWS_PER_BLOCK", raising=False)
    assert atom.ATOM_ROWS_PER_BLOCK == 256


def test_the_environment_selects_the_blocked_path(monkeypatch) -> None:
    monkeypatch.setenv("ESMFOLD2_ATOM_ATTENTION_BACKEND", "blocked")
    assert atom._resolve_rows_per_block(None) == atom.ATOM_ROWS_PER_BLOCK
    monkeypatch.setenv("ESMFOLD2_ATOM_ROWS_PER_BLOCK", "128")
    assert atom._resolve_rows_per_block(None) == 128


def test_an_explicit_argument_beats_the_environment(monkeypatch) -> None:
    """A caller that measured its own size is not overridden by a process setting."""
    monkeypatch.setenv("ESMFOLD2_ATOM_ATTENTION_BACKEND", "blocked")
    assert atom._resolve_rows_per_block(0) == 0
    monkeypatch.delenv("ESMFOLD2_ATOM_ATTENTION_BACKEND", raising=False)
    assert atom._resolve_rows_per_block(256) == 256


def test_there_is_no_auto_mode(monkeypatch) -> None:
    """A probe-driven default puts two machines on two kernels under one name."""
    monkeypatch.setenv("ESMFOLD2_ATOM_ATTENTION_BACKEND", "auto")
    with pytest.raises(ValueError, match="unsupported atom attention backend"):
        atom._atom_attention_backend()
    assert "auto" not in atom.ATOM_ATTENTION_BACKENDS
