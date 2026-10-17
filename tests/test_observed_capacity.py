"""Finite capacity reports complement fitted peak laws without inventing fits."""

from __future__ import annotations

import json
import warnings
from types import SimpleNamespace

import pytest

from foldjax import memory_policy as memory
from foldjax import slurm

_GIB = 2**30


@pytest.fixture(autouse=True)
def _clear():
    memory.clear_record()
    memory.reset_warnings()
    yield
    memory.clear_record()
    memory.reset_warnings()


@pytest.mark.parametrize(
    "model,card,quiet,risk",
    (
        ("esmfold2", 23.5, 639, 640),
        ("esmfold2", 47.5, 1024, 1025),
        ("boltz2", 23.5, 1279, 1280),
        ("boltz2", 47.5, 1280, 1281),
        ("opendde", 23.5, 896, 897),
        ("opendde", 47.5, 1280, 1281),
        ("protenix", 23.5, 2048, 2049),
        ("protenix", 47.5, 2048, 2049),
        ("openfold3", 23.5, 2048, 2049),
        ("openfold3", 47.5, 2048, 2049),
        ("alphafold3", 23.5, 1536, 1537),
        ("alphafold3", 47.5, 2048, 2049),
    ),
)
def test_finite_report_boundaries_warn_without_fitting_or_refusing(
    model, card, quiet, risk
):
    kwargs = {"card_bytes": int(card * _GIB), "budget_bytes": int(card * 0.9 * _GIB)}
    assert memory.observed_capacity(model, quiet, **kwargs)["warning"] is None
    assessment = memory.observed_capacity(model, risk, **kwargs)
    assert "advisory capacity warning" in assessment["warning"]
    assert "rather than a fitted refusal" in assessment["warning"]
    assert "A6000 success is not guaranteed" in assessment["warning"]
    assert "required peak was not measured" in assessment["warning"]
    assert "estimate_bytes" not in assessment
    assert json.loads(json.dumps(assessment)) == assessment


def test_real_card_takes_precedence_over_a_small_pool_budget():
    assessment = memory.observed_capacity(
        "esmfold2",
        768,
        20 * _GIB,
        card_bytes=48 * _GIB,
        device_kind="NVIDIA RTX A6000",
    )
    assert assessment["comparison_basis"] == "measured_card"
    assert assessment["observations"][0]["device_class"] == "A6000 48 GB"
    assert assessment["warning"] is None
    budget = memory.resolve_budget(
        pool_bytes=20 * _GIB, card_bytes=48 * _GIB, override_gib=None
    )
    with pytest.warns(RuntimeWarning, match="outside the fitted range"):
        decision = memory.admit(
            model="esmfold2",
            n_token=768,
            msa_rows=None,
            budget=budget,
            candidates=(("released", memory.ESMFOLD2_PEAK),),
        )
    assert "24 GB" not in decision.reason
    assert (
        memory.observed_capacity(
            "boltz2",
            1536,
            21 * _GIB,
            card_bytes=96 * _GIB,
        )["warning"]
        is None
    )


def test_unknown_card_is_a_conditional_report_not_an_inferred_device():
    planned = memory.observed_capacity("boltz2", 1536)
    assert planned["card_bytes"] is None
    assert planned["comparison_basis"] == "card_unknown"
    assert len(planned["observations"]) == 2
    assert "results conflict" in planned["warning"]
    assert "not established diagnoses" in planned["warning"]
    budgeted = memory.observed_capacity("boltz2", 1536, 22 * _GIB)
    assert budgeted["comparison_basis"] == "conditional_budget"
    assert budgeted["card_bytes"] is None
    nominal = memory.observed_capacity("esmfold2", 768, 24 * _GIB)
    assert nominal["observations"][0]["device_class"] == "24 GB"
    assert nominal["card_bytes"] is None


def test_preflight_records_actual_conditions_and_merges_with_the_fitted_decision():
    shape = {"actual": {"tokens": 1400}, "target": {"tokens": 1536, "msa": 2048}}
    with pytest.warns(RuntimeWarning, match="advisory capacity warning"):
        assessment = memory.capacity_preflight(
            "boltz2",
            1536,
            int(21.2 * _GIB),
            card_bytes=int(23.5 * _GIB),
            device_kind="NVIDIA RTX 4090",
            msa_rows=None,
            num_samples=5,
            shape_profile=shape,
        )
    assert assessment["shape_profile"] == shape
    assert assessment["msa_rows"] is None
    assert "MSA depth" in assessment["observations"][0]["profile_conditions"]
    budget = memory.resolve_budget(
        pool_bytes=int(21.2 * _GIB),
        card_bytes=int(23.5 * _GIB),
        override_gib=None,
    )
    decision = memory.admit(
        model="boltz2",
        n_token=1536,
        msa_rows=None,
        budget=budget,
        candidates=(("released", memory.BOLTZ2_PEAK),),
    )
    # A capacity observation does not rewrite the independently fitted law.
    assert decision.state == "fits" and decision.selected == "released"
    recorded = memory.recorded()
    assert recorded["observed_capacity"] == assessment
    assert recorded["selected"] == "released"
    assert recorded["calibration_id"] == memory.BOLTZ2_PEAK.calibration_id
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        memory.capacity_preflight("boltz2", 1536, int(21.2 * _GIB))
    memory.clear_record()
    assert memory.recorded() is None


def test_a_model_without_a_peak_law_still_records_capacity_context():
    with pytest.warns(RuntimeWarning):
        memory.capacity_preflight("alphafold3", 2048, 22 * _GIB, card_bytes=24 * _GIB)
    block = memory.recorded()
    assert block["state"] == "unknown" and block["selected"] is None
    assert block["estimates"] == []
    assert block["observed_capacity"]["n_token"] == 2048
    assert block["selected_device"] is None


def test_capacity_only_record_keeps_the_selected_device_without_claiming_a_fit():
    device = SimpleNamespace(platform="gpu", device_kind="NVIDIA RTX 4090", id=1)
    with memory.selected_device_scope(device):
        memory.capacity_preflight("alphafold3", 256, 20 * _GIB, card_bytes=22 * _GIB)
        block = memory.recorded()
    assert block["selected"] is None and block["state"] == "unknown"
    assert block["selected_device"] == {
        "platform": "gpu",
        "device_kind": "NVIDIA RTX 4090",
        "id": 1,
    }
    assert memory.recorded()["selected_device"] == block["selected_device"]
    memory.clear_record()
    assert memory.recorded() is None
    memory.capacity_preflight("alphafold3", 256, 20 * _GIB, card_bytes=22 * _GIB)
    assert memory.recorded()["selected_device"] is None


@pytest.mark.parametrize(
    "model,boundary",
    (("opendde", 896), ("alphafold3", 1536), ("protenix", 2048), ("openfold3", 2048)),
)
def test_measured_reduced_card_warns_at_a_reported_successful_boundary(model, boundary):
    assessment = memory.observed_capacity(model, boundary, card_bytes=22 * _GIB)
    assert assessment["reduced_headroom"] is True
    assert "A lower bucket may be needed" in assessment["warning"]
    assert "no new failure threshold has been measured" in assessment["warning"]
    assert assessment["observations"][0]["reported_success_tokens"][0] == boundary
    assert (
        memory.observed_capacity(model, boundary - 1, card_bytes=22 * _GIB)["warning"]
        is None
    )
    assert (
        memory.observed_capacity(model, boundary, card_bytes=int(23.5 * _GIB))[
            "warning"
        ]
        is None
    )
    conditional = memory.observed_capacity(model, boundary, budget_bytes=22 * _GIB)
    assert conditional["reduced_headroom"] is False
    assert conditional["warning"] is None


def test_structural_tokens_are_not_compared_with_public_token_observations():
    budget = memory.resolve_budget(
        pool_bytes=22 * _GIB,
        card_bytes=24 * _GIB,
        override_gib=None,
    )
    memory.capacity_preflight("opendde", 512, 22 * _GIB, card_bytes=24 * _GIB)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        memory.admit(
            model="opendde",
            n_token=1100,
            msa_rows=None,
            budget=budget,
            candidates=(("bf16 trunk", memory.OPENDDE_BF16_PEAK),),
            token_label="structural tokens",
        )
    block = memory.recorded()
    assert block["n_token"] == 1100
    assert block["observed_capacity"]["n_token"] == 512
    assert block["observed_capacity"]["warning"] is None
    assert "reported capacity" not in block["reason"]


@pytest.mark.parametrize(
    "model,tokens",
    (
        ("alphafold3", 2304),
        ("boltz2", 1536),
        ("protenix", 2304),
        ("openfold3", 2304),
        ("opendde", 1024),
        ("esmfold2", 768),
    ),
)
def test_planning_includes_reports_even_without_a_fitted_law(
    model, tokens, monkeypatch
):
    monkeypatch.setattr(
        memory,
        "device_memory_budget",
        lambda **_: pytest.fail("planning must not initialize JAX or read a GPU"),
    )
    document = {"entities": [{"type": "protein", "id": "A", "sequence": "A" * tokens}]}
    result = slurm.plan_resources(model, document)
    assert result["observed_capacity"]["n_token"] == tokens
    assert result["observed_capacity"]["card_bytes"] is None
    assert "warning" in result
    assert result["warning"] in result["reason"]


def test_padded_planning_uses_the_serving_target_and_records_the_budget(tmp_path):
    source = tmp_path / "job.json"
    source.write_text(
        json.dumps(
            {
                "entities": [{"type": "protein", "id": "A", "sequence": "A" * 512}],
            }
        )
    )
    summary = {
        "model": "esmfold2",
        "input": str(source),
        "input_format": "foldjax",
        "sampling": {"num_samples": 32},
        "padding": {},
        "padding_estimate": {"token_bucket": 768, "msa_bucket": 2048},
    }
    result = slurm.plan_slurm(summary, {"memory_budget_gib": 22, "max_msa_depth": 2048})
    assessment = result["observed_capacity"]
    assert assessment["n_token"] == 768
    assert assessment["card_bytes"] is None and assessment["budget_bytes"] == 22 * _GIB
    assert assessment["shape_profile"]["token_bucket"] == 768
    assert assessment["shape_profile"]["requested_msa_cap"] == 2048
    assert assessment["msa_rows"] is None and assessment["num_samples"] == 32
    assert "reported out-of-memory" in result["warning"]
