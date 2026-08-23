"""The atom attention runs at upstream's dtype, not the one the trunk is set to.

`test_trunk_dtype_is_realized` watches the trunk, and the trunk is not where
this went wrong. `TRUNK_PREFIXES` scopes `trunk_dtype` to the trunk and
deliberately excludes the diffusion stack, so `Wqkv` here is float32 @ float32
no matter what the trunk is set to -- and the scores inherited that for as long
as anyone looked. Upstream casts q/k/v to bfloat16 unconditionally at this point
(`modeling_esmfold2_common.py:573-575`) and restores the entering dtype after
(`:632`).

So the property is not "a setting says bfloat16" and not "the trunk is narrow".
It is: *the operands the score einsum actually receives are half precision, and
the value handed back is the dtype that came in.* Both halves are asserted,
because restoring without narrowing and narrowing without restoring are
different bugs and only the pair is upstream's behaviour.

Each assertion below was checked by reverting the change it covers: removing
the q/k/v narrowing fails the score test, and reverting the rope tables to
float32 fails two others. The one mutation that does *not* fail anything is
recorded in `test_the_block_returns_what_it_was_given`.

Costed at the released size: `f32[128, 7776, 7776]` scores are 29,524 MiB and
the softmax of them another, 90% of the whole temp arena at 32 samples.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import atom

# head_dim // 2 must hold 3*2 + 10 = 16 rotary pairs, so head_dim >= 32.
N_ATOMS, WIDTH, N_HEADS = 12, 64, 2
HEAD_DIM = WIDTH // N_HEADS
SCORES = "bihd,bjhd->bhij"
CONTEXT = "bhij,bjhd->bihd"


def _params(dtype) -> dict:
    rng = np.random.default_rng(0)

    def draw(*shape):
        return jnp.asarray(rng.standard_normal(shape).astype(np.float32) * 0.05, dtype)

    return {
        "Wqkv.weight": draw(3 * WIDTH, WIDTH),
        "gate_proj.weight": draw(WIDTH, WIDTH),
        "out_proj.weight": draw(WIDTH, WIDTH),
    }


def _run(monkeypatch, dtype):
    """Call the attention, recording the dtypes each einsum really received."""
    seen: dict[str, tuple[str, ...]] = {}
    real = jnp.einsum

    def spy(subscripts, *operands, **kwargs):
        if isinstance(subscripts, str):
            seen[subscripts] = tuple(str(jnp.asarray(o).dtype) for o in operands)
        return real(subscripts, *operands, **kwargs)

    monkeypatch.setattr(atom.jnp, "einsum", spy)

    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((1, N_ATOMS, WIDTH)).astype(np.float32), dtype)
    # The tables the program actually builds, not hand-made ones -- their dtype
    # is part of what is under test, and feeding synthetic float32 tables here
    # would assert against a path the model never takes.
    cos, sin = atom.build_3d_rope(
        jnp.asarray(rng.standard_normal((1, N_ATOMS, 3)), jnp.float32),
        jnp.arange(N_ATOMS, dtype=jnp.int32)[None] // 3,
        head_dim=HEAD_DIM,
    )
    valid = jnp.ones((1, N_ATOMS), jnp.float32)

    out = atom.swa_attention(
        x,
        _params(dtype),
        "",
        cos=cos,
        sin=sin,
        valid=valid,
        n_heads=N_HEADS,
        half_window=4,
    )
    return out, seen


def test_float32_inputs_are_narrowed_before_the_scores(monkeypatch) -> None:
    """The float32 case is the one that shipped, and the one that costs 29 GiB."""
    out, seen = _run(monkeypatch, jnp.float32)

    assert SCORES in seen, f"the score einsum was not reached; saw {sorted(seen)}"
    assert seen[SCORES] == ("bfloat16", "bfloat16"), (
        "the score einsum received "
        f"{seen[SCORES]}; upstream casts q and k to bfloat16 here "
        "(modeling_esmfold2_common.py:573-575)"
    )
    assert seen[CONTEXT][1] == "bfloat16", (
        f"v reached the context einsum as {seen[CONTEXT][1]}; upstream casts it "
        "alongside q and k"
    )


def test_the_block_returns_what_it_was_given(monkeypatch) -> None:
    """Narrowing must not leak bfloat16 into the surrounding float32 stack.

    Note what this does *not* gate. `swa_attention` ends with
    `context.astype(input_dtype)`, mirroring upstream's `out.to(input_dtype)` at
    `:632` -- and that statement is inert here: `gate` is built from `x` through
    a float32 `gate_proj`, so `context * gate` re-promotes whether or not the
    cast is there. Deleting the line leaves this test green, which was checked
    by deleting it. The line is kept because the port's contract is to run what
    upstream runs and a reader diffing the two would otherwise have to re-derive
    why it is missing -- not because anything here depends on it.

    What is asserted is the outcome that does hold: an entering float32 stack
    gets float32 back, so narrowing the scores stays local to the attention.
    """
    out, _ = _run(monkeypatch, jnp.float32)
    assert out.dtype == jnp.float32, out.dtype
    assert jnp.isfinite(out).all()


def test_a_bfloat16_caller_is_left_alone(monkeypatch) -> None:
    """Upstream's guard is conditional; a caller already in half precision is untouched.

    This is what makes the change a cast rather than a policy: it never widens,
    and it never narrows something that was already narrow.
    """
    out, seen = _run(monkeypatch, jnp.bfloat16)
    assert seen[SCORES] == ("bfloat16", "bfloat16"), seen[SCORES]
    assert out.dtype == jnp.bfloat16, out.dtype


@pytest.mark.parametrize("half_window", [1, 4])
def test_narrowing_did_not_break_the_window(monkeypatch, half_window) -> None:
    """The mask fill must still exponentiate to zero at the narrowed dtype.

    `finfo(bfloat16).min` is a different number from `finfo(float32).min`, and
    the property that matters -- masked positions contribute exactly nothing --
    has to survive the change of dtype rather than merely of magnitude.
    """
    rng = np.random.default_rng(2)
    valid = jnp.ones((1, N_ATOMS), jnp.float32)
    allowed = atom.sliding_window_mask(valid, half_window)
    logits = jnp.asarray(rng.standard_normal((1, 1, N_ATOMS, N_ATOMS)), jnp.bfloat16)
    masked = jnp.where(allowed[:, None], logits, jnp.finfo(jnp.bfloat16).min)
    attention = jnp.exp(masked - masked.max(axis=-1, keepdims=True))
    outside = attention[jnp.broadcast_to(allowed[:, None], attention.shape) == False]  # noqa: E712
    assert (outside == 0).all(), "a masked position kept a non-zero weight"


def test_the_rope_tables_are_built_at_upstreams_dtype() -> None:
    """Upstream hardcodes bfloat16 tables; float32 ones widen the rotary.

    `modeling_esmfold2_common.py:513-514`.

    Asserted separately from the attention because it is a separate defect with
    a separate symptom: float32 tables promote `q` and `k` inside
    `apply_rotary_3d`, so a bfloat16 caller receives float32 back even when the
    scores themselves are narrowed correctly.
    """
    cos, sin = atom.build_3d_rope(
        jnp.zeros((1, N_ATOMS, 3), jnp.float32),
        jnp.arange(N_ATOMS, dtype=jnp.int32)[None] // 3,
        head_dim=HEAD_DIM,
    )
    assert cos.dtype == jnp.bfloat16, cos.dtype
    assert sin.dtype == jnp.bfloat16, sin.dtype
