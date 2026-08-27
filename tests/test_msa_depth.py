"""The MSA depth cap, which is the dominant memory knob.

The trunk of every one of these models holds an ``[depth, tokens, channels]``
MSA representation. Measured on a 488-token job with a 13,254-row alignment,
that single tensor was 3,158 MiB and three of them were live at once -- 9.5 GiB
of a 12.2 GiB peak, against 2,891 MiB of activations for the same job in
AlphaFold 3. Capping the depth to 1024 halved Protenix's peak and let OpenDDE
run at all, where lowering the compute dtype and blocking the triangle
attention had each moved it by less than 0.5%.

The knob is model-neutral because the tensor exists in every trunk; each
backend translates it to its own name.
"""

from __future__ import annotations

import pytest

from foldjax.registry import capabilities, get_backend
from foldjax.schema import PredictionRequest


def _request(tmp_path, model, **kwargs):
    job = tmp_path / "job.json"
    job.write_text('{"entities": [{"type": "protein", "id": ["A"], '
                   '"sequence": "ACDEFG"}]}')
    return PredictionRequest(model=model, input=job, **kwargs)


@pytest.mark.parametrize("model", ["protenix", "opendde", "boltz2"])
def test_the_cap_is_a_neutral_knob(model: str) -> None:
    assert "max_msa_depth" in capabilities(model).sampling


@pytest.mark.parametrize(
    ("model", "native"),
    [("protenix", "max_msa_depth"), ("opendde", "max_msa_depth"),
     ("boltz2", "max_msa_depth")],
)
def test_the_cap_reaches_each_backend_under_its_own_name(
    tmp_path, model: str, native: str
) -> None:
    options = get_backend(model).apply_sampling(
        _request(tmp_path, model, max_msa_depth=1024)
    )
    assert options[native] == 1024


def test_an_unset_cap_leaves_the_backend_default_alone(tmp_path) -> None:
    """Omitting it must not pin a depth, or the port's own default is gone."""
    options = get_backend("protenix").apply_sampling(_request(tmp_path, "protenix"))
    assert "max_msa_depth" not in options


def test_the_cap_must_be_positive(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_msa_depth must be at least 1"):
        _request(tmp_path, "protenix", max_msa_depth=0)


def test_setting_both_spellings_is_an_error(tmp_path) -> None:
    """Silently preferring one would change the depth without changing the
    exit code -- the same rule the other neutral knobs follow."""
    request = _request(
        tmp_path, "protenix", max_msa_depth=1024,
        options={"max_msa_depth": 4096},
    )
    with pytest.raises(ValueError, match="both set"):
        get_backend("protenix").apply_sampling(request)


def test_the_cap_changes_the_compile_cache_namespace(tmp_path) -> None:
    """It changes the compiled program's shapes, so it cannot share an entry."""
    backend = get_backend("protenix")
    assert "max_msa_depth" in backend.compile_options


def test_a_chunk_size_the_kernel_cannot_honour_is_not_silent() -> None:
    """cuEquivariance takes the whole thing, so a chunk size never lands.

    It was the default on the assumption that it fused the score tensor away.
    It does not: at 490 tokens cuEquivariance and the unblocked XLA path peak
    identically, and the blocked XLA path peaks 28% lower. So a chunk size
    reaching this branch is a real loss, not a harmless no-op, and it says so.
    """
    import warnings

    from foldjax.models.protenix.models.triangle import triangle

    triangle._WARNED_UNCHUNKABLE = False
    with pytest.warns(RuntimeWarning, match="not used by the 'cueq' backend"):
        triangle._warn_unchunkable_backend(128)
    # Once per process: the trunk calls this per block per layer, and warning
    # every time would bury the run in identical noise.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        triangle._warn_unchunkable_backend(128)
    assert caught == []


def test_triangle_attention_blocks_rows_by_one_rule_for_every_branch() -> None:
    """One row-block rule, derived per call site from that branch's own shape.

    OpenDDE is dual-branch: the residue trunk and the structural refiner share
    a single chunk knob, but the structural branch runs on sub-residue tokens
    with three times the heads, so the same chunk size costs three times the
    score tensor. The block size therefore cannot be resolved from the token
    count alone -- it is resolved inside the attention, where both the token
    count and the head count of *that* call are known.

    The optimum is the same row count at both sizes for a given branch, which
    rules a byte budget out: a budget halves the count every time the tokens
    double, and at 1,892 tokens that gave 6 rows, measured at 43,569 MiB
    against 32,641 at 25. What is invariant is `rows * heads` -- 64*4 for
    Protenix's trunk, 25*12 for OpenDDE's structural branch.
    """
    from foldjax.models.protenix.models.triangle.triangle import (
        _MAX_ROWS_PER_BLOCK,
        _MIN_ROWS_PER_BLOCK,
        _row_block,
        _score_rows,
    )

    def score_rows(tokens: int, heads: int, requested=None):
        return _score_rows(
            rows=tokens, cols=tokens, num_heads=heads, requested=requested
        )

    # Protenix's trunk: the cap binds, not the budget.
    assert score_rows(490, 4) == 64
    assert score_rows(976, 4) == 64
    # OpenDDE's structural branch: 12 heads makes the budget bind instead, and
    # the block shrinks with the token count where the trunk's stays at the cap.
    # The head count decides, and it does not move with the token count: the
    # 12-head branch gets the same block at 948 tokens and at 1,892.
    wide = score_rows(948, 12)
    assert wide == score_rows(1892, 12) < _MAX_ROWS_PER_BLOCK
    assert wide * 12 == pytest.approx(64 * 4, rel=0.2)
    # It only ever narrows what the policy proposed.
    assert score_rows(948, 12, requested=256) == wide
    assert score_rows(948, 12, requested=16) == 16
    # The byte ceiling still takes over on a complex big enough for a fixed row
    # count to run away, and never goes below the floor.
    assert score_rows(20_000, 12) == _MIN_ROWS_PER_BLOCK
    # Small enough that blocking buys nothing: the request stands untouched.
    assert score_rows(20, 4) is None
    # The same rule sizes the triangle multiplication, off a different budget:
    # there the block bounds one projection, not a score tensor.
    from foldjax.models.protenix.models.triangle.triangle import (
        _PROJECTION_BUDGET_BYTES,
    )

    assert _row_block(
        rows=948, per_row=948 * 384 * 4, requested=256,
        budget=_PROJECTION_BUDGET_BYTES,
    ) == _MAX_ROWS_PER_BLOCK
    assert _MIN_ROWS_PER_BLOCK < _MAX_ROWS_PER_BLOCK
