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


@pytest.mark.parametrize("model", ["protenix", "opendde", "boltz2", "chai"])
def test_the_cap_is_a_neutral_knob(model: str) -> None:
    assert "max_msa_depth" in capabilities(model).sampling


@pytest.mark.parametrize(
    ("model", "native"),
    [("protenix", "max_msa_rows"), ("opendde", "max_msa_rows"),
     ("boltz2", "max_msa_depth"), ("chai", "max_msa_depth")],
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
    assert "max_msa_rows" not in options


def test_the_cap_must_be_positive(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_msa_depth must be at least 1"):
        _request(tmp_path, "protenix", max_msa_depth=0)


def test_setting_both_spellings_is_an_error(tmp_path) -> None:
    """Silently preferring one would change the depth without changing the
    exit code -- the same rule the other neutral knobs follow."""
    request = _request(
        tmp_path, "protenix", max_msa_depth=1024,
        options={"max_msa_rows": 4096},
    )
    with pytest.raises(ValueError, match="both set"):
        get_backend("protenix").apply_sampling(request)


def test_the_cap_changes_the_compile_cache_namespace(tmp_path) -> None:
    """It changes the compiled program's shapes, so it cannot share an entry."""
    backend = get_backend("protenix")
    assert "max_msa_rows" in backend.compile_options


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

    The rule is `min(64 rows, 1 GiB / row)`, and it reproduces both independent
    sweeps: 64 rows for Protenix's 4-head trunk, which the cap decides at 490
    and 976 tokens alike, and ~24 for OpenDDE's 12-head structural branch,
    where the budget decides instead.
    """
    from foldjax.models.protenix.models.triangle.triangle import (
        _MAX_ROWS_PER_BLOCK,
        _MIN_ROWS_PER_BLOCK,
        _row_block,
    )

    def score_rows(tokens: int, heads: int, requested=None):
        return _row_block(
            rows=tokens, per_row=heads * tokens * tokens * 4, requested=requested
        )

    # Protenix's trunk: the cap binds, not the budget.
    assert score_rows(490, 4) == 64
    assert score_rows(976, 4) == 64
    # OpenDDE's structural branch: 12 heads makes the budget bind instead, and
    # the block shrinks with the token count where the trunk's stays at the cap.
    wide = score_rows(948, 12)
    wider = score_rows(1892, 12)
    assert wide == 1024**3 // (12 * 948 * 948 * 4) < _MAX_ROWS_PER_BLOCK
    assert _MIN_ROWS_PER_BLOCK <= wider < wide
    # It only ever narrows what the policy proposed.
    assert score_rows(948, 12, requested=256) == wide
    assert score_rows(948, 12, requested=16) == 16
    # Never below the floor, where one more block costs more than it saves.
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


def test_chai_caps_alignment_rows_before_bucketing() -> None:
    """Chai's cap acts on the row count, which is what sizes the MSA tensor.

    Chai does not take a depth argument at the featurizer; it selects rows from
    its own mask and pads the count up to a bucket. So the cap has to act
    there, before bucketing -- capping after would round straight back up.
    Rows arrive in priority order, so the kept ones are the alignment's best.
    """
    import numpy as np

    from foldjax.models.chai.inference import _msa_row_indices_and_mask

    # 900 real rows over 4 tokens: without a cap this buckets up to 1024.
    mask = np.zeros((1, 4096, 4), dtype=bool)
    mask[:, :900, :] = True

    rows, kept = _msa_row_indices_and_mask(mask, key=None)
    assert rows.size == 1024
    assert kept.shape[1] == 1024

    # Capped to 256, which is itself a bucket, so no padding is added.
    rows, kept = _msa_row_indices_and_mask(mask, key=None, max_depth=256)
    assert rows.size == 256
    assert kept.shape[1] == 256
    # The rows kept are the first 256, not an arbitrary 256.
    assert np.array_equal(rows, np.arange(256))

    # A cap above what is there cannot invent rows.
    rows, _ = _msa_row_indices_and_mask(mask, key=None, max_depth=99_999)
    assert rows.size == 1024


def test_chai_rejects_a_non_positive_cap() -> None:
    import numpy as np

    from foldjax.models.chai.inference import _msa_row_indices_and_mask

    with pytest.raises(ValueError, match="max_depth must be positive"):
        _msa_row_indices_and_mask(
            np.ones((1, 8, 2), dtype=bool), key=None, max_depth=0
        )
