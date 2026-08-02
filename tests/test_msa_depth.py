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
    """cuEquivariance has no chunked triangle-attention kernel.

    A chunk size that reaches it bounds nothing, and the score tensor it exists
    to bound is materialised whole. That was silent, which is how a knob comes
    to look like it works.
    """
    import warnings

    from foldjax.models.protenix.models.triangle import triangle

    triangle._WARNED_UNCHUNKABLE = False
    with pytest.warns(RuntimeWarning, match="ignored by the 'cueq' backend"):
        triangle._warn_unchunkable_backend(128)
    # Once per process: the trunk calls this per block per layer, and warning
    # every time would bury the run in identical noise.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        triangle._warn_unchunkable_backend(128)
    assert caught == []


def test_the_structural_branch_is_blocked_by_bytes_not_token_count() -> None:
    """OpenDDE's structural refiner is what sizes its peak.

    Protenix's chunk policy maps a token count to a chunk size and was written
    for a four-head trunk. The structural branch runs on sub-residue tokens
    with three times the heads, so the same token count costs three times the
    score tensor: at 946 tokens the policy's 256 still materialises 10.5 GiB,
    twice over. Bounding by bytes instead took the measured peak from 32,185 to
    14,443 MiB with confidence flat to four decimals.
    """
    from foldjax.models.opendde.cli.predict import _structural_q_chunk

    class _Linear:
        weight = type("w", (), {"shape": (12, 1)})()

    class _Block:
        tri_att_start = type("t", (), {"linear": _Linear()})()

    class _Params:
        structural_refiner = type("s", (), {"blocks": [_Block()]})()

    params = _Params()
    # Big enough to bind: the policy's 256 is cut down.
    assert _structural_q_chunk(params, 946, 256) < 256
    # Small enough not to: the policy's choice stands. The threshold moves with
    # the budget, so this is well under it rather than just below.
    assert _structural_q_chunk(params, 200, 256) == 256
    # It only ever narrows.
    assert _structural_q_chunk(params, 946, 16) == 16
    # A parameter tree without the branch is left alone rather than guessed at.
    assert _structural_q_chunk(object(), 946, 256) == 256


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
