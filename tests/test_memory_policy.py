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
    OPENFOLD3_CANDIDATES,
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


def test_the_openfold3_binary_choice_is_offered_only_where_both_arms_were_fitted() -> (
    None
):
    """The narrower domain wins: 3,012 is the last size the unchunked arm has."""
    assert OPENFOLD3_UNCHUNKED_PEAK.domain_tokens == (1003, 3012)
    assert OPENFOLD3_CHUNKED_PEAK.domain_tokens == (1003, 4888)
    huge = 200 * _GIB
    inside = resolve_memory_policy(
        model="openfold3",
        n_token=3012,
        msa_rows=None,
        budget_bytes=huge,
        candidates=OPENFOLD3_CANDIDATES,
    )
    assert inside.state == "fits"
    outside = resolve_memory_policy(
        model="openfold3",
        n_token=3013,
        msa_rows=None,
        budget_bytes=huge,
        candidates=OPENFOLD3_CANDIDATES,
    )
    assert outside.state == "unknown"


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
# Candidate order
# --------------------------------------------------------------------------


def test_the_fastest_admissible_candidate_wins_and_more_memory_never_slows_it() -> None:
    """Monotone in the budget, which a "search" could easily not be."""
    previous_index = len(OPENFOLD3_CANDIDATES)
    names = [name for name, _law in OPENFOLD3_CANDIDATES]
    assert names == ["unchunked", "chunked"]
    for gib in range(10, 120, 5):
        decision = resolve_memory_policy(
            model="openfold3",
            n_token=3012,
            msa_rows=None,
            budget_bytes=gib * _GIB,
            candidates=OPENFOLD3_CANDIDATES,
        )
        index = (
            names.index(decision.selected)
            if decision.selected is not None
            else len(names)
        )
        # A larger budget never selects a candidate further down the
        # fastest-first list than a smaller one did.
        assert index <= previous_index
        previous_index = index
    assert previous_index == 0


def test_unchunked_is_selected_only_when_its_own_upper_estimate_fits() -> None:
    for n_token in (1003, 2096, 3012):
        upper = OPENFOLD3_UNCHUNKED_PEAK.upper(n_token)
        just_enough = math.ceil(upper / memory_policy.ADMISSION_FRACTION) + 1
        assert (
            resolve_memory_policy(
                model="openfold3",
                n_token=n_token,
                msa_rows=None,
                budget_bytes=just_enough,
                candidates=OPENFOLD3_CANDIDATES,
            ).selected
            == "unchunked"
        )
        short = int(upper / memory_policy.ADMISSION_FRACTION) - 1
        assert (
            resolve_memory_policy(
                model="openfold3",
                n_token=n_token,
                msa_rows=None,
                budget_bytes=short,
                candidates=OPENFOLD3_CANDIDATES,
            ).selected
            != "unchunked"
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


def test_openfold3_chunks_only_when_the_unchunked_estimate_does_not_fit() -> None:
    plenty = _budget(90 * _GIB, card=96 * _GIB)
    tight = _budget(24 * _GIB, card=26 * _GIB)
    assert inference.resolve_pair_chunk_size(3012, budget=plenty) is None
    with pytest.warns(RuntimeWarning):
        # 3,012 tokens does not fit a 24 GiB pool either way; the fallback is
        # the blocked loop, never a refusal.
        tight_chunk = inference.resolve_pair_chunk_size(3012, budget=tight)
    assert tight_chunk == inference.RESOLVED_PAIR_CHUNK_SIZE
    # 2,096 unchunked needs 22.5 GiB and the same pool admits 21.6, so this one
    # is a real choice rather than a size that fits nothing.
    assert (
        inference.resolve_pair_chunk_size(2096, budget=tight)
        == inference.RESOLVED_PAIR_CHUNK_SIZE
    )
    assert inference.resolve_pair_chunk_size(2096, budget=plenty) is None


def test_above_the_unchunked_domain_the_blocked_width_is_the_answer() -> None:
    """No measurement exists for the unblocked loop past 3,012 tokens, so no
    budget can buy it -- but the admission still runs and says so once."""
    with pytest.warns(RuntimeWarning, match="proceeds without the check"):
        chunk = inference.resolve_pair_chunk_size(
            4100, budget=_budget(200 * _GIB, card=220 * _GIB)
        )
    assert chunk == inference.RESOLVED_PAIR_CHUNK_SIZE
    assert memory_policy.recorded()["state"] == "unknown"


def test_below_the_validated_domain_the_program_is_the_one_it_always_was() -> None:
    """The measured 128 rows start at 1,003 tokens. Below that the automatic
    answer stays unblocked -- the byte budget this replaced returned None for
    every such size, and those are the sizes the CPU parity captures pin.
    """
    assert inference.RESOLVED_PAIR_CHUNK_SIZE == 128
    low, _high = memory_policy.OPENFOLD3_UNCHUNKED_PEAK.domain_tokens
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
    # tokens unblocked estimates 6.4 GiB, so it takes a small card to make the
    # blocked arm the only admissible one.
    tight = _budget(6 * _GIB, card=8 * _GIB)
    assert inference.resolve_pair_chunk_size(low, budget=tight) == 128


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
        # two different `InferenceConfig`s and so two JIT owners. The automatic
        # answer is always None.
        assert config.pair_chunk_size != 0


def test_an_explicit_chunk_width_is_never_overridden_by_a_budget() -> None:
    config = inference.released_config(
        n_token=2096,
        n_atom=2096 * 24,
        pair_chunk_size=128,
        memory_budget=_budget(90 * _GIB),
    )
    assert config.pair_chunk_size == 128


def test_two_budgets_that_compile_different_programs_share_one_cache_profile(
    tmp_path: Path,
) -> None:
    """The persistent namespace must not fork per card.

    The profile is a function of the request alone, so this holds structurally
    -- and is pinned because the chunk width it does *not* record now depends
    on the device, which is exactly the shape of change that would break it.
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
    # And the widths those two budgets resolve to really do differ, or the
    # assertion above would be vacuous.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tight = inference.released_config(
            n_token=2096, n_atom=2096 * 24, memory_budget=_budget(24 * _GIB)
        )
        roomy = inference.released_config(
            n_token=2096, n_atom=2096 * 24, memory_budget=_budget(90 * _GIB)
        )
    assert tight.pair_chunk_size != roomy.pair_chunk_size


# --------------------------------------------------------------------------
# The option surface and the manifest
# --------------------------------------------------------------------------


def test_the_memory_options_are_never_compile_options() -> None:
    """Two runs that differ only in a budget must share one cache namespace."""
    from foldjax.backends.boltz2 import Boltz2Backend
    from foldjax.backends.protenix import ProtenixBackend

    for backend in (Boltz2Backend(), ProtenixBackend(), OpenFold3Backend()):
        for name in ("memory_check", "memory_budget_gib"):
            assert name not in backend.compile_options, backend.name
    for backend in (Boltz2Backend(), ProtenixBackend()):
        assert {"memory_check", "memory_budget_gib"} <= set(backend.native_options)


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
    """
    assert memory_policy.CALIBRATED_NUM_SAMPLES == 5
    for law in (
        BOLTZ2_PEAK,
        PROTENIX_PEAK,
        OPENFOLD3_CHUNKED_PEAK,
        OPENFOLD3_UNCHUNKED_PEAK,
    ):
        assert "5 samples" in law.profile
        assert law.calibration_id.endswith("2026-09-15")
