"""Admission is a comparison, so the things worth pinning are its edges.

Every test here runs on the CPU with no device: the one function in
`foldjax.memory_policy` that reads a device is :func:`device_memory_budget`,
and the budget is otherwise an argument, which is the point of the split.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import pytest

import foldjax
from foldjax import memory_policy, oom
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.backends.openfold3 import OpenFold3Backend
from foldjax.manifest import MANIFEST_NAME
from foldjax.memory_policy import (
    BOLTZ2_PEAK,
    ESMFOLD2_PEAK,
    OPENDDE_BF16_PEAK,
    OPENDDE_FP32_PEAK,
    OPENFOLD3_CHUNKED_PEAK,
    OPENFOLD3_UNCHUNKED_PEAK,
    PROTENIX_PEAK,
    resolve_budget,
    resolve_memory_policy,
)
from foldjax.models.openfold3 import inference
from foldjax.registry import backend_override
from foldjax.schema import PredictionRequest, PredictionResult, PredictionSample

_GIB = 2**30

#: The candidate set `inference.resolve_pair_chunk_size` passes: one law, for
#: the one configuration the automatic path runs.
_OPENFOLD3_CANDIDATE = (("chunked", OPENFOLD3_CHUNKED_PEAK),)


@pytest.fixture(autouse=True)
def _forget_one_time_warnings():
    memory_policy.reset_warnings()
    memory_policy.clear_record()
    yield
    memory_policy.reset_warnings()
    memory_policy.clear_record()


def _budget(
    bytes_: int | None, *, card: int | None = None
) -> memory_policy.MemoryBudget:
    return resolve_budget(pool_bytes=bytes_, card_bytes=card, override_gib=None)


def _job(tmp_path: Path) -> Path:
    path = tmp_path / "job.json"
    path.write_text(
        json.dumps({"entities": [{"type": "protein", "id": "A", "sequence": "ACD"}]})
    )
    return path


def _decide(law, n_token, budget_bytes, msa_rows=None):
    return resolve_memory_policy(
        model=law.model,
        n_token=n_token,
        msa_rows=msa_rows,
        budget_bytes=budget_bytes,
        candidates=(("released", law),),
    )


# --------------------------------------------------------------------------
# The threshold
# --------------------------------------------------------------------------


def test_a_candidate_exactly_at_the_threshold_is_admitted() -> None:
    """The threshold is the largest peak admitted, not the smallest refused.

    Worth pinning because the inequality is the whole policy: written the other
    way round, a run whose estimate lands exactly on the boundary -- which the
    caller can arrange with `--memory-budget-gib` -- flips verdict.
    """
    upper = BOLTZ2_PEAK.upper(2096)
    exact = math.ceil(upper / memory_policy.ADMISSION_FRACTION)
    # The budget that produces a threshold of exactly `upper`, allowing for the
    # int() truncation in the threshold itself.
    while int(exact * memory_policy.ADMISSION_FRACTION) > upper:
        exact -= 1
    assert int(exact * memory_policy.ADMISSION_FRACTION) == upper

    assert _decide(BOLTZ2_PEAK, 2096, exact).state == "fits"
    assert _decide(BOLTZ2_PEAK, 2096, exact).threshold == upper
    # One byte of headroom more is still admitted; one byte less is not.
    assert _decide(BOLTZ2_PEAK, 2096, exact + 1).state == "fits"
    over = _decide(BOLTZ2_PEAK, 2096, exact - 1)
    assert over.state == "over_budget"
    assert over.threshold == upper - 1
    assert over.selected is None


def test_the_allowance_is_part_of_what_is_compared() -> None:
    """An estimate that fits and an upper estimate that does not is refused."""
    estimate = BOLTZ2_PEAK.estimate(3012)
    assert BOLTZ2_PEAK.upper(3012) == estimate + BOLTZ2_PEAK.allowance_bytes
    budget = int((estimate + 1) / memory_policy.ADMISSION_FRACTION)
    assert _decide(BOLTZ2_PEAK, 3012, budget).state == "over_budget"


# --------------------------------------------------------------------------
# Everything it refuses to answer
# --------------------------------------------------------------------------


def test_no_budget_is_unknown_and_still_reports_the_estimate() -> None:
    decision = _decide(BOLTZ2_PEAK, 2096, None)
    assert decision.state == "unknown"
    assert decision.selected is None
    assert decision.threshold is None
    # The estimate is still worth recording: it is what the run expected to
    # need, and the manifest keeps it next to what the run then cost.
    assert [c.name for c in decision.estimates] == ["released"]
    assert decision.estimates[0].estimate_bytes == BOLTZ2_PEAK.estimate(2096)
    assert decision.estimates[0].fits is None


@pytest.mark.parametrize("n_token", [1002, 4889])
def test_a_token_count_outside_the_fitted_range_is_unknown(n_token: int) -> None:
    decision = _decide(BOLTZ2_PEAK, n_token, 80 * _GIB)
    assert decision.state == "unknown"
    assert decision.estimates == ()
    assert "outside the fitted range" in decision.reason
    assert BOLTZ2_PEAK.domain_tokens == (1003, 4888)
    assert BOLTZ2_PEAK.covers(1003) and BOLTZ2_PEAK.covers(4888)


def test_a_law_that_needs_msa_rows_without_them_is_unknown() -> None:
    assert PROTENIX_PEAK.needs_msa_rows
    decision = _decide(PROTENIX_PEAK, 2096, 80 * _GIB, msa_rows=None)
    assert decision.state == "unknown"
    assert "MSA row count" in decision.reason
    with pytest.raises(ValueError, match="MSA row"):
        PROTENIX_PEAK.estimate(2096)


def test_openfold3_is_estimated_over_the_blocked_arms_own_domain() -> None:
    """The blocked loop is the configuration, so its law is the whole answer.

    Its domain reaches 4,888 tokens, where the unblocked arm's measurements
    stop at 3,012 -- so a size that used to be `unknown` because one of two
    candidates had no law now gets a verdict.
    """
    assert OPENFOLD3_CHUNKED_PEAK.domain_tokens == (1003, 4888)
    assert OPENFOLD3_UNCHUNKED_PEAK.domain_tokens == (1003, 3012)
    huge = 200 * _GIB
    for n_token, state in (
        (1002, "unknown"),
        (1003, "fits"),
        (3013, "fits"),
        (4888, "fits"),
        (4889, "unknown"),
    ):
        decision = resolve_memory_policy(
            model="openfold3",
            n_token=n_token,
            msa_rows=None,
            budget_bytes=huge,
            candidates=_OPENFOLD3_CANDIDATE,
        )
        assert decision.state == state, n_token


def test_no_budget_buys_the_unblocked_loop_inside_the_domain() -> None:
    """The replacement for the candidate-order tests this file used to carry.

    There is nothing to order any more: the width is a function of the token
    count, so every budget from one that refuses the run to one far above it
    resolves to the same program. That is the property -- not the estimate --
    that the unblocked arm's 65% more peak at 2,096 tokens for 3.4% less wall
    time bought.
    """
    for gib in range(6, 200, 7):
        for n_token in (1003, 2096, 3012, 4100, 4888):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                width = inference.resolve_pair_chunk_size(
                    n_token, budget=_budget(gib * _GIB), mode="warn"
                )
            assert width == inference.RESOLVED_PAIR_CHUNK_SIZE, (gib, n_token)


def test_a_candidate_law_for_another_model_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="carries a boltz2 law"):
        resolve_memory_policy(
            model="openfold3",
            n_token=2096,
            msa_rows=None,
            budget_bytes=80 * _GIB,
            candidates=(("wrong", BOLTZ2_PEAK),),
        )


# --------------------------------------------------------------------------
# The budget side
# --------------------------------------------------------------------------


def test_an_override_wins_only_when_it_is_the_smaller_ceiling() -> None:
    pool = 90 * _GIB
    smaller = resolve_budget(pool_bytes=pool, card_bytes=96 * _GIB, override_gib=24)
    assert smaller.budget_bytes == 24 * _GIB
    assert smaller.source == "override"
    # An override above the pool would admit a job the allocator refuses.
    larger = resolve_budget(pool_bytes=pool, card_bytes=96 * _GIB, override_gib=200)
    assert larger.budget_bytes == pool
    assert larger.source == "pool"
    # With nothing readable the override is the only ceiling there is, which is
    # what makes `--memory-budget-gib` usable for planning off the machine.
    alone = resolve_budget(pool_bytes=None, card_bytes=None, override_gib=40)
    assert (alone.budget_bytes, alone.source) == (40 * _GIB, "override")
    nothing = resolve_budget(pool_bytes=None, card_bytes=None, override_gib=None)
    assert (nothing.budget_bytes, nothing.source) == (None, "none")


def test_a_nonpositive_budget_is_rejected_rather_than_refusing_everything() -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_budget(pool_bytes=None, card_bytes=None, override_gib=0)
    with pytest.raises(ValueError, match="positive"):
        memory_policy.parse_budget_gib(-1)
    with pytest.raises(ValueError, match="must be a number"):
        memory_policy.parse_budget_gib("24")
    with pytest.raises(ValueError, match="memory_check must be one of"):
        memory_policy.parse_check_mode("quiet")
    assert memory_policy.parse_check_mode(None) == "refuse"


def test_the_device_budget_is_readable_with_preallocation_off(monkeypatch) -> None:
    """The ceiling exists either way; only the reserved pool does not.

    `device_budget` used to return nothing with preallocation off, which would
    have left this whole policy inert under the bench harness and under
    `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
    """
    from foldjax import oom

    class _Device:
        def memory_stats(self):
            return {"bytes_limit": 91_779_760_128, "bytes_in_use": 1024}

    fake = type(sys)("jax")
    fake.devices = lambda: [_Device()]
    monkeypatch.setitem(sys.modules, "jax", fake)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    pool, card = oom.device_budget()
    assert pool == 91_779_760_128
    assert card == pytest.approx(91_779_760_128 / 0.9, rel=1e-9)
    # `diagnose` keeps its own rule: with nothing preallocated, `bytes_in_use`
    # no longer says how much of the ceiling the run already held.
    assert oom.diagnose(RuntimeError("RESOURCE_EXHAUSTED: allocate 1.0GiB")) is None


# --------------------------------------------------------------------------
# Enforcement
# --------------------------------------------------------------------------


def _protenix_levers() -> tuple[str, ...]:
    return (
        "--max-msa-depth lowers the estimate, but by changing the input: "
        "fewer alignment rows is a different prediction, not the same one in "
        "less memory",
    )


def test_over_budget_refuses_and_names_every_lever() -> None:
    budget = resolve_budget(
        pool_bytes=16 * _GIB, card_bytes=18 * _GIB, override_gib=None
    )
    with pytest.raises(MemoryError) as error:
        memory_policy.admit(
            model="protenix",
            n_token=3012,
            msa_rows=17542,
            candidates=(("released", PROTENIX_PEAK),),
            budget=budget,
            levers=_protenix_levers(),
        )
    message = str(error.value)
    assert "3012 tokens, 17542 processed MSA rows" in message
    # The estimate, the threshold, the pool and the card.
    assert "14.4 GiB threshold" in message
    assert "16.0 GiB of a 18.0 GiB device" in message
    assert "--mem-fraction" in message
    assert "--memory-check=warn" in message
    # Named as input-changing rather than as an equivalent way to run.
    assert "a different prediction" in message


def test_warn_mode_says_the_same_thing_without_refusing() -> None:
    budget = resolve_budget(
        pool_bytes=16 * _GIB, card_bytes=18 * _GIB, override_gib=None
    )
    with pytest.warns(RuntimeWarning) as caught:
        decision = memory_policy.admit(
            model="protenix",
            n_token=3012,
            msa_rows=17542,
            candidates=(("released", PROTENIX_PEAK),),
            budget=budget,
            mode="warn",
            levers=_protenix_levers(),
        )
    assert decision.state == "over_budget"
    message = str(caught[0].message)
    assert "14.4 GiB threshold" in message
    assert "--mem-fraction" in message
    assert "a different prediction" in message
    assert "Running anyway" in message


def test_an_off_calibration_run_keeps_its_estimate_and_loses_the_refusal() -> None:
    """A five-sample law over-predicts a one-sample run, and refusing on an
    over-prediction is a run that never starts."""
    off = memory_policy.off_profile_reason(num_samples=1)
    assert off and "1 samples" in off[0]
    assert memory_policy.off_profile_reason(num_samples=5) == ()
    budget = resolve_budget(
        pool_bytes=16 * _GIB, card_bytes=18 * _GIB, override_gib=None
    )
    with pytest.warns(RuntimeWarning, match="advisory rather than binding"):
        memory_policy.admit(
            model="boltz2",
            n_token=4888,
            msa_rows=None,
            candidates=(("released", BOLTZ2_PEAK),),
            budget=budget,
            mode="refuse",
            off_profile=off,
        )


def test_a_run_that_fits_says_nothing_at_all() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decision = memory_policy.admit(
            model="boltz2",
            n_token=1003,
            msa_rows=None,
            candidates=(("released", BOLTZ2_PEAK),),
            budget=_budget(90 * _GIB, card=96 * _GIB),
        )
    assert decision.state == "fits"


def test_an_unreadable_ceiling_warns_once_per_model() -> None:
    with pytest.warns(RuntimeWarning, match="proceeds without the check") as caught:
        for _ in range(3):
            memory_policy.admit(
                model="boltz2",
                n_token=2096,
                msa_rows=None,
                candidates=(("released", BOLTZ2_PEAK),),
                budget=_budget(None),
            )
    assert len(caught) == 1


def test_an_unknown_check_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="memory_check must be one of"):
        memory_policy.enforce(
            _decide(BOLTZ2_PEAK, 2096, None),
            model="boltz2",
            n_token=2096,
            msa_rows=None,
            budget=_budget(None),
            mode="ignore",
        )


# --------------------------------------------------------------------------
# What the policy is not allowed to do
# --------------------------------------------------------------------------


def test_nothing_offers_protenix_a_narrower_alignment_as_a_candidate() -> None:
    """The MSA cap failed this repository's accuracy-admission test, so it is
    not a memory lever the policy may reach for.

    Pinned as a property of the candidate set rather than of any one call site:
    a decision can only select a candidate it was given, and the only Protenix
    candidate is the released one. There is no depth to select.
    """
    laws = [
        value
        for name, value in vars(memory_policy).items()
        if isinstance(value, memory_policy.PeakLaw) and not name.startswith("_")
    ]
    protenix = [law for law in laws if law.model == "protenix"]
    assert protenix == [PROTENIX_PEAK]
    assert "max_msa_depth" not in str(PROTENIX_PEAK)
    decision = _decide(PROTENIX_PEAK, 3012, 16 * _GIB, msa_rows=17542)
    assert decision.state == "over_budget"
    assert decision.selected is None
    assert [c.name for c in decision.estimates] == ["released"]
    # The decision carries no knob a caller could apply, only a verdict.
    assert not hasattr(decision, "msa_rows")
    assert not hasattr(decision, "max_msa_depth")


# --------------------------------------------------------------------------
# OpenFold3's wiring
# --------------------------------------------------------------------------


def test_openfold3_blocks_inside_the_domain_whatever_the_card_reports() -> None:
    """A card with room to spare is not a reason to spend it.

    At 2,096 tokens the unblocked loop is 3.4% quicker (220.85 against
    228.5 s) and costs 65% more peak for it (22,967 against 13,990 MiB); at
    1,003 the wall time is equal (71.6 against 72.0 s) for 6,195 against
    4,317 MiB. So a 90 GiB pool resolves to the same width a 24 GiB one does.
    """
    plenty = _budget(90 * _GIB, card=96 * _GIB)
    tight = _budget(24 * _GIB, card=26 * _GIB)
    for n_token in (2096, 3012):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert (
                inference.resolve_pair_chunk_size(n_token, budget=plenty)
                == inference.RESOLVED_PAIR_CHUNK_SIZE
            )
            assert (
                inference.resolve_pair_chunk_size(n_token, budget=tight, mode="warn")
                == inference.RESOLVED_PAIR_CHUNK_SIZE
            )


def test_an_over_budget_openfold3_run_refuses_like_every_other_port() -> None:
    """With one configuration there is nothing cheaper to fall back to.

    3,012 tokens blocked estimates 23.2 GiB + 0.8 allowance, and a 24 GiB pool
    admits 21.6 -- so this is a refusal and not a second configuration.
    """
    tight = _budget(24 * _GIB, card=26 * _GIB)
    threshold = int(24 * _GIB * memory_policy.ADMISSION_FRACTION)
    assert OPENFOLD3_CHUNKED_PEAK.upper(3012) > threshold
    with pytest.raises(MemoryError, match="no openfold3 configuration fits"):
        inference.resolve_pair_chunk_size(3012, budget=tight)
    memory_policy.reset_warnings()
    with pytest.warns(RuntimeWarning, match="Running anyway") as caught:
        width = inference.resolve_pair_chunk_size(3012, budget=tight, mode="warn")
    assert width == inference.RESOLVED_PAIR_CHUNK_SIZE
    # The lever named is the one that is the same prediction.
    assert "diffusion_chunk_size" in str(caught[0].message)
    assert memory_policy.recorded()["state"] == "over_budget"
    assert memory_policy.recorded()["estimates"] == [
        {
            "name": "chunked",
            "estimate_bytes": OPENFOLD3_CHUNKED_PEAK.estimate(3012),
            "upper_bytes": OPENFOLD3_CHUNKED_PEAK.upper(3012),
            "fits": False,
        }
    ]


def test_a_run_the_law_was_not_fitted_at_is_warned_rather_than_refused() -> None:
    """`released_config` builds the off-profile list, because it is the layer
    that knows the sample count and the stages. A five-sample law over-predicts
    a one-sample run, and an over-prediction under a refusing policy is a run
    that never starts."""
    tight = _budget(24 * _GIB, card=26 * _GIB)
    with pytest.warns(RuntimeWarning, match="advisory rather than binding"):
        config = inference.released_config(
            n_token=3012, n_atom=3012 * 24, num_samples=1, memory_budget=tight
        )
    assert config.pair_chunk_size == inference.RESOLVED_PAIR_CHUNK_SIZE
    assert memory_policy.recorded()["off_profile"] == [
        "1 samples rather than the 5 the law was fitted at"
    ]


def test_above_the_blocked_domain_the_width_stands_and_the_check_says_unknown() -> (
    None
):
    """4,888 tokens is the last size measured; past it there is no law."""
    with pytest.warns(RuntimeWarning, match="proceeds without the check"):
        chunk = inference.resolve_pair_chunk_size(
            5000, budget=_budget(200 * _GIB, card=220 * _GIB)
        )
    assert chunk == inference.RESOLVED_PAIR_CHUNK_SIZE
    assert memory_policy.recorded()["state"] == "unknown"


def test_below_the_validated_domain_the_program_is_the_one_it_always_was() -> None:
    """The measured 128 rows start at 1,003 tokens. Below that the automatic
    answer stays unblocked -- the byte budget this replaced returned None for
    every such size, and those are the sizes the CPU parity captures pin.
    """
    assert inference.RESOLVED_PAIR_CHUNK_SIZE == 128
    low, _high = memory_policy.OPENFOLD3_CHUNKED_PEAK.domain_tokens
    assert low == 1003
    plenty = _budget(90 * _GIB, card=96 * _GIB)
    for n_token in (76, 128, 129, 490, low - 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert inference.resolve_pair_chunk_size(n_token, budget=None) is None
            assert inference.resolve_pair_chunk_size(n_token, budget=plenty) is None
            config = inference.released_config(
                n_token=n_token, n_atom=n_token * 24, memory_budget=plenty
            )
        assert config.pair_chunk_size is None, n_token
    # And a size the domain does cover is blocked at the measured width, so the
    # assertions above are a boundary rather than the whole function. 1,003
    # tokens blocked estimates 4.2 GiB + 0.8 allowance, which an 8 GiB card
    # admits, so the width arrives with no warning either.
    small = _budget(6 * _GIB, card=8 * _GIB)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert inference.resolve_pair_chunk_size(low, budget=small) == 128


def test_no_budget_keeps_openfold3_on_the_historical_blocked_loop() -> None:
    """`released_config`'s other callers -- checkpoint inspection,
    featurization -- must not initialize a device to resolve a chunk width."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            inference.resolve_pair_chunk_size(3012, budget=None)
            == inference.RESOLVED_PAIR_CHUNK_SIZE
        )
        config = inference.released_config(n_token=3012, n_atom=3012 * 24)
    assert config.pair_chunk_size == inference.RESOLVED_PAIR_CHUNK_SIZE


def test_the_resolved_config_carries_a_width_and_never_a_sentinel() -> None:
    for budget in (_budget(90 * _GIB), _budget(24 * _GIB), None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = inference.released_config(
                n_token=2096, n_atom=2096 * 24, memory_budget=budget
            )
        assert config.pair_chunk_size is None or isinstance(config.pair_chunk_size, int)
        # 0 and None both mean "do not block" to `map_row_chunks`, but they are
        # two different `InferenceConfig`s and so two JIT owners. 0 is the
        # request's explicit spelling and never the automatic answer: inside
        # the domain that is the width, below it `None`.
        assert config.pair_chunk_size != 0


def test_an_explicit_chunk_width_is_never_overridden_by_a_budget() -> None:
    config = inference.released_config(
        n_token=2096,
        n_atom=2096 * 24,
        pair_chunk_size=128,
        memory_budget=_budget(90 * _GIB),
    )
    assert config.pair_chunk_size == 128


def test_an_omitted_option_and_an_explicit_unblocked_run_stay_distinguishable() -> (
    None
):
    """The resolved width is `_PredictGraphIdentity`'s own key, so the three
    answers have to be three values: the width inside the domain, `None` below
    it, and the request's own `0` for the loop run whole. `map_row_chunks`
    treats `0` and `None` alike, which is exactly why they must not collapse
    here -- one means "this size has no measurement", the other "the caller
    asked for it".
    """
    plenty = _budget(90 * _GIB, card=96 * _GIB)
    inside = inference.released_config(
        n_token=2096, n_atom=2096 * 24, memory_budget=plenty
    )
    assert inside.pair_chunk_size == inference.RESOLVED_PAIR_CHUNK_SIZE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        below = inference.released_config(
            n_token=490, n_atom=490 * 24, memory_budget=plenty
        )
    assert below.pair_chunk_size is None
    explicit = inference.released_config(
        n_token=2096, n_atom=2096 * 24, pair_chunk_size=0, memory_budget=plenty
    )
    assert explicit.pair_chunk_size == 0
    assert explicit != inside


def test_two_budgets_share_one_cache_profile_and_one_program(
    tmp_path: Path,
) -> None:
    """The persistent namespace must not fork per card, and no longer could.

    The profile is a function of the request alone; the width the profile does
    *not* record is a function of the token count alone. Both halves are pinned
    here, because the budget used to decide the width and a return to that
    would break the second half silently.
    """
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"not really weights")
    request = PredictionRequest(
        model="openfold3",
        input=_job(tmp_path),
        weights=weights,
        output_dir=tmp_path / "out",
        use_compile_cache=False,
    )
    backend = OpenFold3Backend()
    baseline = backend.cache_profile(request)
    for gib in (24, 90):
        with_budget = PredictionRequest(
            model="openfold3",
            input=request.input,
            weights=request.weights,
            output_dir=request.output_dir,
            use_compile_cache=False,
            options={"memory_budget_gib": float(gib)},
        )
        assert backend.cache_profile(with_budget) == baseline
        assert "memory_budget_gib" not in backend.cache_profile(with_budget)
    assert "memory_budget_gib" not in backend.compile_options
    assert "memory_budget_gib" in backend.native_options
    # And the two budgets build the same program, not merely the same
    # namespace: a 24 GiB pool and a 90 GiB one resolve to one width.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tight = inference.released_config(
            n_token=2096, n_atom=2096 * 24, memory_budget=_budget(24 * _GIB)
        )
        roomy = inference.released_config(
            n_token=2096, n_atom=2096 * 24, memory_budget=_budget(90 * _GIB)
        )
    assert tight == roomy
    assert tight.pair_chunk_size == inference.RESOLVED_PAIR_CHUNK_SIZE


# --------------------------------------------------------------------------
# The option surface and the manifest
# --------------------------------------------------------------------------


def test_the_memory_options_are_never_compile_options() -> None:
    """Two runs that differ only in a budget must share one cache namespace."""
    from foldjax.backends.alphafold3 import AlphaFold3Backend
    from foldjax.backends.boltz2 import Boltz2Backend
    from foldjax.backends.esmfold2 import ESMFold2Backend
    from foldjax.backends.protenix import ProtenixBackend

    # Every port, including the three that gained the pair last: OpenDDE
    # derives `compile_options` from the set of options its parser takes, so
    # its two admission options have to be subtracted there by name.
    backends = (
        Boltz2Backend(),
        ProtenixBackend(),
        OpenFold3Backend(),
        OpenDDEBackend(),
        ESMFold2Backend(),
        AlphaFold3Backend(),
    )
    for backend in backends:
        for name in ("memory_check", "memory_budget_gib"):
            assert name not in backend.compile_options, backend.name
        # Accepted everywhere, so the flags mean the same thing on every port.
        # AlphaFold 3 accepts them without a law: what they buy there is one
        # `unknown` warning rather than an "unsupported options" failure.
        assert {"memory_check", "memory_budget_gib"} <= set(
            backend.native_options
        ), backend.name


class _AdmittingBackend(OpenDDEBackend):
    """A backend that takes a decision, so the manifest has one to record."""

    def predict(self, request):
        memory_policy.admit(
            model="protenix",
            n_token=2096,
            msa_rows=4096,
            candidates=(("released", PROTENIX_PEAK),),
            budget=resolve_budget(
                pool_bytes=90 * _GIB, card_bytes=96 * _GIB, override_gib=40
            ),
        )
        path = request.output_dir / f"s{request.seed}.cif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data_mock\n#\n", encoding="utf-8")
        return PredictionResult(
            model="opendde",
            samples=(
                PredictionSample(
                    seed=request.seed, structure_path=path, scores={"ptm": 0.5}
                ),
            ),
            output_dir=request.output_dir,
        )


def test_the_manifest_records_the_decision_and_survives_a_json_round_trip(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"not really weights")
    request = PredictionRequest(
        model="opendde",
        input=_job(tmp_path),
        weights=weights,
        profile="released",
        output_dir=out,
        use_compile_cache=False,
    )
    with backend_override("opendde", _AdmittingBackend):
        foldjax.predict(request)

    document = json.loads((out / MANIFEST_NAME).read_text())
    block = document["memory"]
    assert block["state"] == "fits"
    assert block["selected"] == "released"
    assert block["pool_bytes"] == 90 * _GIB
    assert block["card_bytes"] == 96 * _GIB
    assert block["budget_bytes"] == 40 * _GIB
    assert block["budget_source"] == "override"
    assert block["threshold_bytes"] == int(40 * _GIB * memory_policy.ADMISSION_FRACTION)
    assert block["calibration_id"] == PROTENIX_PEAK.calibration_id
    assert block["memory_check"] == "refuse"
    assert block["mem_fraction"] == pytest.approx(oom.mem_fraction())
    assert block["estimates"] == [
        {
            "name": "released",
            "estimate_bytes": PROTENIX_PEAK.estimate(2096, 4096),
            "upper_bytes": PROTENIX_PEAK.upper(2096, 4096),
            "fits": True,
        }
    ]
    # JSON-native throughout: `manifest.write` swallows a serialization failure
    # and discards the whole manifest, so a numpy scalar here would silently
    # cost a run its entire provenance.
    assert json.loads(json.dumps(block, sort_keys=True)) == block


def test_a_run_that_takes_no_decision_records_none_rather_than_the_last_one(
    tmp_path: Path,
) -> None:
    memory_policy.admit(
        model="boltz2",
        n_token=1003,
        msa_rows=None,
        candidates=(("released", BOLTZ2_PEAK),),
        budget=_budget(90 * _GIB, card=96 * _GIB),
    )
    assert memory_policy.recorded() is not None
    out = tmp_path / "out"
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"not really weights")
    request = PredictionRequest(
        model="opendde",
        input=_job(tmp_path),
        weights=weights,
        profile="released",
        output_dir=out,
        use_compile_cache=False,
    )

    with backend_override("opendde", _Quiet):
        foldjax.predict(request)
    assert json.loads((out / MANIFEST_NAME).read_text())["memory"] is None


class _Quiet(OpenDDEBackend):
    """Predicts without consulting the memory policy."""

    def predict(self, request):
        path = request.output_dir / f"s{request.seed}.cif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data_mock\n#\n", encoding="utf-8")
        return PredictionResult(
            model="opendde",
            samples=(
                PredictionSample(
                    seed=request.seed, structure_path=path, scores={"ptm": 0.5}
                ),
            ),
            output_dir=request.output_dir,
        )


# --------------------------------------------------------------------------
# The frozen literals
# --------------------------------------------------------------------------


def test_the_frozen_laws_are_what_the_calibration_script_fits() -> None:
    """Re-derive every literal, so a changed measurement cannot be half-applied.

    The coefficients are rounded to nine significant figures in the module, so
    they are compared at that precision; the allowances are integers and are
    compared exactly.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import calibrate_memory_policy as calibration
    finally:
        sys.path.pop(0)

    fits = calibration.fit_all()
    shipped = {
        "boltz2": BOLTZ2_PEAK,
        "protenix": PROTENIX_PEAK,
        "openfold3_chunked": OPENFOLD3_CHUNKED_PEAK,
        "openfold3_unchunked": OPENFOLD3_UNCHUNKED_PEAK,
        "opendde_bf16": OPENDDE_BF16_PEAK,
        "opendde_fp32": OPENDDE_FP32_PEAK,
        "esmfold2": ESMFOLD2_PEAK,
    }
    assert set(fits) == set(shipped)
    for name, law in shipped.items():
        fit = fits[name]
        assert law.domain_tokens == fit["domain_tokens"], name
        assert law.allowance_bytes == fit["allowance_mib"] * 2**20, name
        assert len(law.coeffs) == len(fit["phases"]), name
        for (phase, terms), (fit_phase, fit_terms, fit_coeffs) in zip(
            law.coeffs, fit["phases"]
        ):
            assert phase == fit_phase, name
            assert tuple(term for term, _c in terms) == fit_terms, name
            for (_term, shipped_value), fitted in zip(terms, fit_coeffs):
                assert shipped_value == pytest.approx(fitted, rel=1e-8), name


def test_no_measured_completed_run_is_refused() -> None:
    """The point of the in-sample allowance rule.

    Every peak in the calibration table came from a run that finished inside
    this host's pool, so a policy whose job is to say "this will not fit" must
    admit all of them. The rule used to be the leave-one-size-out error, and
    under it Boltz-2 at 4,888 tokens -- measured at 77,312 MiB, completed --
    was refused with 3.4 GiB of margin it did not need.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import calibrate_memory_policy as calibration
    finally:
        sys.path.pop(0)

    cases = (
        (BOLTZ2_PEAK, calibration.BOLTZ2_POINTS),
        (OPENFOLD3_CHUNKED_PEAK, calibration.OF3_CHUNKED_POINTS),
        (OPENFOLD3_UNCHUNKED_PEAK, calibration.OF3_UNCHUNKED_POINTS),
        (PROTENIX_PEAK, calibration.PROTENIX_PAIR + calibration.PROTENIX_MSA),
        # The validation rows are here as well as the fit points: they are
        # completed runs too, and a law fitted on one snapshot must still
        # admit the same configuration measured on another (OpenDDE) and at
        # another sample count (ESMFold2's released 32).
        (
            OPENDDE_BF16_PEAK,
            calibration.OPENDDE_BF16_POINTS + calibration.OPENDDE_BF16_VALIDATION,
        ),
        (OPENDDE_FP32_PEAK, calibration.OPENDDE_FP32_POINTS),
        (
            ESMFOLD2_PEAK,
            calibration.ESMFOLD2_POINTS + calibration.ESMFOLD2_VALIDATION,
        ),
    )
    # The ceiling this host reports at the fraction `foldjax predict` asks for.
    pool = 91_779_760_128
    for law, points in cases:
        for n_token, msa_rows, peak in points:
            upper = law.upper(n_token, msa_rows)
            # Pointwise: the allowance covers the measurement it was fitted on.
            assert upper >= peak * 2**20, (law.model, n_token)
            decision = resolve_memory_policy(
                model=law.model,
                n_token=n_token,
                msa_rows=msa_rows,
                budget_bytes=pool,
                candidates=((law.profile, law),),
            )
            assert decision.state == "fits", (law.model, n_token, decision.reason)


def test_every_law_names_the_sample_count_it_was_fitted_at() -> None:
    """Boltz-2's released default is one sample, not the five every law was
    measured at, so this is the fact that decides whether its admission binds.

    ESMFold2 is the exception, and it names the exception: its peak was
    measured at 5 samples *and* at the released 32, 0.53% apart, so its
    profile carries the whole validated range rather than the fitted point.
    """
    assert memory_policy.CALIBRATED_NUM_SAMPLES == 5
    expected = {
        BOLTZ2_PEAK: ("5 samples", "2026-09-15"),
        PROTENIX_PEAK: ("5 samples", "2026-09-15"),
        OPENFOLD3_CHUNKED_PEAK: ("5 samples", "2026-09-15"),
        OPENFOLD3_UNCHUNKED_PEAK: ("5 samples", "2026-09-15"),
        OPENDDE_BF16_PEAK: ("5 samples", "2026-09-16"),
        OPENDDE_FP32_PEAK: ("5 samples", "2026-09-16"),
        ESMFOLD2_PEAK: ("5-32 samples", "2026-09-23"),
    }
    for law, (samples, date) in expected.items():
        assert samples in law.profile, law.calibration_id
        assert law.calibration_id.endswith(date), law.calibration_id
    # Every law's own id, so a changed measurement cannot be half-applied
    # under a shared date.
    assert len({law.calibration_id for law in expected}) == len(expected)


# --------------------------------------------------------------------------
# OpenDDE and ESMFold2: the two ports admitted against a law of their own
# --------------------------------------------------------------------------


def _exact_budget(law, n_token: int) -> int:
    """The budget whose threshold is exactly this law's upper estimate."""
    upper = law.upper(n_token)
    budget = math.ceil(upper / memory_policy.ADMISSION_FRACTION)
    while int(budget * memory_policy.ADMISSION_FRACTION) > upper:
        budget -= 1
    assert int(budget * memory_policy.ADMISSION_FRACTION) == upper
    return budget


@pytest.mark.parametrize(
    ("law", "n_token"),
    [
        (OPENDDE_BF16_PEAK, 1902),
        (OPENDDE_BF16_PEAK, 4034),
        (OPENDDE_FP32_PEAK, 1902),
        (ESMFOLD2_PEAK, 2096),
        (ESMFOLD2_PEAK, 3012),
    ],
    ids=lambda value: str(value),
)
def test_each_new_law_admits_at_its_threshold_and_refuses_one_byte_below(
    law, n_token: int
) -> None:
    """The same inequality the four older laws are held to.

    Worth repeating per law rather than trusting the shared comparison: these
    two ports reach `resolve_memory_policy` through wiring of their own, and
    the boundary is what a caller can arrange with `--memory-budget-gib`.
    """
    upper = law.upper(n_token)
    assert upper == law.estimate(n_token) + law.allowance_bytes
    budget = _exact_budget(law, n_token)

    at = _decide(law, n_token, budget)
    assert at.state == "fits"
    assert at.threshold == upper
    assert _decide(law, n_token, budget + 1).state == "fits"
    below = _decide(law, n_token, budget - 1)
    assert below.state == "over_budget"
    assert below.threshold == upper - 1
    assert below.selected is None


def test_the_opendde_laws_are_keyed_on_structural_tokens_and_split_by_dtype() -> None:
    """Two laws, one per realised trunk dtype, over one domain.

    The float32 arm is not the bfloat16 arm plus a factor: it is measured, and
    at the one size both were measured at it costs twice the peak. Keeping
    them separate is what makes the dtype lever a lever rather than a guess.
    """
    assert OPENDDE_BF16_PEAK.domain_tokens == (1902, 7876)
    assert OPENDDE_FP32_PEAK.domain_tokens == (1902, 7876)
    assert "structural tokens" in OPENDDE_BF16_PEAK.profile
    assert "structural tokens" in OPENDDE_FP32_PEAK.profile
    # The two fitted rows, to the MiB, and the float32 measurement beside the
    # bfloat16 one at the same size.
    assert OPENDDE_BF16_PEAK.estimate(1902) == pytest.approx(21492 * 2**20, rel=1e-6)
    assert OPENDDE_BF16_PEAK.estimate(2978) == pytest.approx(46858.2 * 2**20, rel=1e-6)
    assert OPENDDE_FP32_PEAK.estimate(1902) == pytest.approx(42690.6 * 2**20, rel=1e-5)
    assert not OPENDDE_BF16_PEAK.needs_msa_rows
    assert not OPENDDE_FP32_PEAK.needs_msa_rows


@pytest.mark.parametrize("n_token", [1901, 7877])
def test_a_structural_token_count_outside_the_opendde_domain_is_unknown(
    n_token: int,
) -> None:
    """Below the domain is the parity sizes, and they must never be refused.

    A parity case folds a few hundred residues, which is a few hundred
    structural tokens -- far below anything measured -- and the float32 arm
    has no intercept, so its estimate down there is not a number to refuse a
    run on. `unknown` is the answer, and the run proceeds.
    """
    decision = _decide(OPENDDE_BF16_PEAK, n_token, 80 * _GIB)
    assert decision.state == "unknown"
    assert decision.estimates == ()
    assert "outside the fitted range" in decision.reason


def test_a_parity_sized_opendde_job_is_never_refused() -> None:
    """The sizes `tests/parity` runs, against a budget nothing would fit."""
    for n_structural in (257, 948):
        for law in (OPENDDE_BF16_PEAK, OPENDDE_FP32_PEAK):
            decision = _decide(law, n_structural, 1 * _GIB)
            assert decision.state == "unknown", (law.calibration_id, n_structural)


def test_an_over_budget_opendde_run_names_the_structural_tokens_and_the_lever() -> None:
    """The refusal the retired arena preflight could only warn about.

    2,096 residues is about 4,034 structural tokens: the size that runs on no
    layout but the 2x2 grid, and the one this port's ceiling is about.
    """
    from foldjax.models.opendde.runner import _BF16_TRUNK_LEVER, _FP32_TRUNK_LEVER

    budget = resolve_budget(
        pool_bytes=90 * _GIB, card_bytes=96 * _GIB, override_gib=None
    )
    with pytest.raises(MemoryError) as error:
        memory_policy.admit(
            model="opendde",
            n_token=4034,
            msa_rows=None,
            candidates=(("bf16 trunk", OPENDDE_BF16_PEAK),),
            budget=budget,
            levers=(_BF16_TRUNK_LEVER,),
            token_label="structural tokens",
        )
    message = str(error.value)
    # The unit the law is keyed on, said out loud: a message that called these
    # "tokens" would name a number the caller never typed.
    assert "4034 structural tokens" in message
    assert "80.7 GiB + 1.1 GiB allowance" in message
    assert "81.0 GiB threshold" in message
    assert "--mem-fraction" in message
    assert "--cp-devices 4" in message
    assert "--memory-check=warn" in message

    # The float32 arm names the dtype lever instead, because it has one.
    with pytest.warns(RuntimeWarning) as caught:
        decision = memory_policy.admit(
            model="opendde",
            n_token=2978,
            msa_rows=None,
            candidates=(("fp32 trunk", OPENDDE_FP32_PEAK),),
            budget=budget,
            mode="warn",
            levers=(_FP32_TRUNK_LEVER,),
            token_label="structural tokens",
        )
    assert decision.state == "over_budget"
    assert "--trunk-dtype bf16" in str(caught[0].message)
    assert "Running anyway" in str(caught[0].message)


def test_the_esmfold2_law_carries_no_sample_term_and_binds_at_32_samples() -> None:
    """The released sample count has to be inside the profile, or nothing binds.

    This port reads `num_samples` off its checkpoint, and the released value
    is 32 where every law was fitted at 5. Treating that as off profile would
    downgrade every default run's refusal to a warning -- so the measurement
    that says the sample axis is not in this peak is what the widening rests
    on, and a count *below* five is still off profile.
    """
    assert ESMFOLD2_PEAK.domain_tokens == (2096, 3012)
    assert memory_policy.off_profile_reason(num_samples=32) != ()
    assert memory_policy.off_profile_reason(num_samples=32, samples_validated=32) == ()
    assert memory_policy.off_profile_reason(num_samples=5, samples_validated=32) == ()
    below = memory_policy.off_profile_reason(num_samples=1, samples_validated=32)
    assert below and "5-32" in below[0]
    beyond = memory_policy.off_profile_reason(num_samples=64, samples_validated=32)
    assert beyond and "64 samples" in beyond[0]
    # A port may only widen the range, never narrow it.
    with pytest.raises(ValueError, match="cannot be below"):
        memory_policy.off_profile_reason(num_samples=5, samples_validated=1)


def test_an_over_budget_esmfold2_run_refuses_and_warn_runs_it_anyway() -> None:
    budget = resolve_budget(
        pool_bytes=32 * _GIB, card_bytes=40 * _GIB, override_gib=None
    )
    lever = "--cp-devices N shards the pair state"
    with pytest.raises(MemoryError) as error:
        memory_policy.admit(
            model="esmfold2",
            n_token=2096,
            msa_rows=None,
            candidates=(("released", ESMFOLD2_PEAK),),
            budget=budget,
            levers=(lever,),
        )
    assert "2096 tokens" in str(error.value)
    assert "28.8 GiB threshold" in str(error.value)
    assert lever in str(error.value)

    with pytest.warns(RuntimeWarning) as caught:
        decision = memory_policy.admit(
            model="esmfold2",
            n_token=2096,
            msa_rows=None,
            candidates=(("released", ESMFOLD2_PEAK),),
            budget=budget,
            mode="warn",
            levers=(lever,),
        )
    assert decision.state == "over_budget"
    assert "Running anyway" in str(caught[0].message)
    # 2,096 tokens fits a 40 GiB budget, so the refusal above is about the
    # budget and not about the port.
    assert _decide(ESMFOLD2_PEAK, 2096, 40 * _GIB).state == "fits"


def test_an_explicit_budget_is_what_the_new_ports_are_admitted_against() -> None:
    """`--memory-budget-gib` reaches the decision and survives into the record.

    The smaller of the two ceilings wins, and the source says which -- the
    same contract the older ports have, asserted through a decision taken
    with a law of the new ports' own so the wiring cannot report someone
    else's budget.
    """
    budget = resolve_budget(
        pool_bytes=90 * _GIB, card_bytes=96 * _GIB, override_gib=40
    )
    assert budget.source == "override"
    assert budget.budget_bytes == 40 * _GIB
    decision = memory_policy.admit(
        model="esmfold2",
        n_token=2096,
        msa_rows=None,
        candidates=(("released", ESMFOLD2_PEAK),),
        budget=budget,
        mode="warn",
    )
    assert decision.state == "fits"
    assert decision.threshold == int(40 * _GIB * memory_policy.ADMISSION_FRACTION)
    record = memory_policy.recorded()
    assert record is not None
    assert record["budget_source"] == "override"
    assert record["budget_bytes"] == 40 * _GIB
    assert record["pool_bytes"] == 90 * _GIB
    assert record["calibration_id"] == ESMFOLD2_PEAK.calibration_id
    # An override above the pool would admit a job the allocator refuses.
    wide = resolve_budget(pool_bytes=90 * _GIB, card_bytes=96 * _GIB, override_gib=200)
    assert wide.source == "pool" and wide.budget_bytes == 90 * _GIB


# --------------------------------------------------------------------------
# A port with no law at all
# --------------------------------------------------------------------------


def test_a_port_with_no_law_warns_once_by_name_and_proceeds() -> None:
    """AlphaFold 3's answer to the two flags: `unknown`, said out loud.

    Before this the flags were not in its option set, so asking for them
    ended the run with "unsupported alphafold3 options" -- which reads like a
    misspelling rather than like a missing measurement.
    """
    budget = resolve_budget(
        pool_bytes=90 * _GIB, card_bytes=96 * _GIB, override_gib=None
    )
    with pytest.warns(RuntimeWarning) as caught:
        decision = memory_policy.admit_unmeasured(
            model="alphafold3",
            budget=budget,
            mode="refuse",
            detail="no law was fitted to them",
        )
    assert decision.state == "unknown"
    assert decision.selected is None
    assert decision.estimates == ()
    message = str(caught[0].message)
    assert message.startswith("alphafold3: no alphafold3 peak law is fitted")
    assert "The run proceeds without the check." in message
    # `refuse` cannot refuse what was never estimated.
    assert len(caught) == 1

    # Once per process, like every other `unknown`: a five-seed request does
    # not say it five times.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        memory_policy.admit_unmeasured(
            model="alphafold3", budget=budget, detail="no law was fitted to them"
        )

    record = memory_policy.recorded()
    assert record is not None
    assert record["state"] == "unknown"
    assert record["model"] == "alphafold3"
    # No shape and no calibration: this port tokenizes below the adapter, and
    # there is no law to name.
    assert record["n_token"] is None
    assert record["calibration_id"] is None
    assert json.loads(json.dumps(record, sort_keys=True)) == record


def test_the_manifest_round_trips_an_opendde_decision(tmp_path: Path) -> None:
    """The structural token count, the dtype arm and the law's own id.

    The older round-trip test above records a Protenix decision; this one
    records the shape that is new -- a count in structural tokens, which is
    not the number the caller asked for, so the manifest has to carry it.
    """

    class _OpenDDEAdmitting(OpenDDEBackend):
        def predict(self, request):
            memory_policy.admit(
                model="opendde",
                n_token=1902,
                msa_rows=None,
                candidates=(("bf16 trunk", OPENDDE_BF16_PEAK),),
                budget=resolve_budget(
                    pool_bytes=90 * _GIB, card_bytes=96 * _GIB, override_gib=None
                ),
                token_label="structural tokens",
            )
            path = request.output_dir / f"s{request.seed}.cif"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("data_mock\n#\n", encoding="utf-8")
            return PredictionResult(
                model="opendde",
                samples=(
                    PredictionSample(
                        seed=request.seed, structure_path=path, scores={"ptm": 0.5}
                    ),
                ),
                output_dir=request.output_dir,
            )

    out = tmp_path / "out"
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"not really weights")
    request = PredictionRequest(
        model="opendde",
        input=_job(tmp_path),
        weights=weights,
        profile="released",
        output_dir=out,
        use_compile_cache=False,
    )
    with backend_override("opendde", _OpenDDEAdmitting):
        foldjax.predict(request)

    block = json.loads((out / MANIFEST_NAME).read_text())["memory"]
    assert block["state"] == "fits"
    assert block["selected"] == "bf16 trunk"
    assert block["n_token"] == 1902
    assert block["msa_rows"] is None
    assert block["calibration_id"] == OPENDDE_BF16_PEAK.calibration_id
    assert block["budget_source"] == "pool"
    assert block["estimates"] == [
        {
            "name": "bf16 trunk",
            "estimate_bytes": OPENDDE_BF16_PEAK.estimate(1902),
            "upper_bytes": OPENDDE_BF16_PEAK.upper(1902),
            "fits": True,
        }
    ]
    assert json.loads(json.dumps(block, sort_keys=True)) == block
