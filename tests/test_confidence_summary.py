"""Public scalar confidence summary contract tests."""

from __future__ import annotations

import json

import pytest

from bench.confidence_summary import build_summary


def write_record(path, *, model="model", case="case", samples=None, **metadata):
    record = {
        "model": model,
        "case": case,
        "impl": metadata.pop("impl", "foldjax"),
        "samples": [] if samples is None else samples,
        **metadata,
    }
    path.write_text(json.dumps(record))


def test_preserves_all_samples_scores_and_missing_field_counts(tmp_path):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    write_record(
        left,
        samples=[
            {"seed": 11, "scores": {"ptm": 0.2, "num_recycles": 3}},
            {"id": "second", "scores": {"ptm": 0.4}},
        ],
        schedule={"num_samples": 2},
    )
    write_record(right, samples=[{"index": 99, "scores": {"iptm": 0.7}}])

    result = build_summary(left, right)

    arm = result["arms"]["left"]
    assert arm["reported_samples"][0]["reported_scores"] == {
        "ptm": 0.2,
        "num_recycles": 3,
    }
    assert arm["reported_samples"][0]["sample_identity"] == {"seed": 11}
    assert arm["reported_samples"][0]["sample_metadata"] == {"seed": 11}
    assert arm["reported_field_summaries"]["ptm"]["values"] == [0.2, 0.4]
    assert arm["reported_field_summaries"]["num_recycles"]["missing_count"] == 1
    assert "iptm" not in arm["reported_field_summaries"]
    assert "normalized_fields" not in result
    assert result["automatic_comparability"] == "unknown"


@pytest.mark.parametrize(
    ("field", "message"), [("model", "same model"), ("case", "same case")]
)
def test_rejects_different_model_or_case(tmp_path, field, message):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    write_record(left, samples=[])
    write_record(right, samples=[], **{field: "other"})
    with pytest.raises(ValueError, match=message):
        build_summary(left, right)


def test_failed_and_empty_arms_stay_explicit(tmp_path):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    write_record(left, samples=[], failed=True, source="native")
    write_record(right, samples=[], returncode=0)

    result = build_summary(left, right)

    assert result["arms"]["left"]["status"] == "failed"
    assert result["arms"]["right"]["status"] == "no_samples"
    assert result["metadata_issues"][0]["field"] == "source"


def test_same_key_is_not_normalized_or_differenced_without_policy(tmp_path):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    write_record(left, samples=[{"scores": {"ptm": 0.1}}])
    write_record(right, samples=[{"scores": {"ptm": 0.9}}])

    result = build_summary(left, right)

    assert "normalized_fields" not in result
    assert result["arms"]["left"]["reported_field_summaries"]["ptm"]["mean"] == 0.1


def test_explicit_policy_converts_units_and_is_unpaired(tmp_path):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    write_record(
        left,
        samples=[
            {"scores": {"plddt_fraction": 0.8}},
            {"scores": {"plddt_fraction": 0.6}},
        ],
    )
    write_record(right, samples=[{"scores": {"plddt_percent": 70.0}}])
    policy = {
        "model": "model",
        "fields": [
            {
                "name": "pLDDT",
                "left_key": "plddt_fraction",
                "right_key": "plddt_percent",
                "left_scale": 100.0,
                "right_scale": 1.0,
                "unit": "percent",
                "evidence": ["writer documentation"],
            }
        ],
    }

    result = build_summary(left, right, policy)

    field = result["normalized_fields"][0]
    assert field["left"]["values"] == [80.0, 60.0]
    assert field["right"]["mean"] == 70.0
    assert field["difference_of_means"] == 0.0
    assert field["descriptive_unpaired"]
    assert result["field_policy"]["caller_supplied"] == policy


@pytest.mark.parametrize(
    "policy",
    [
        {"model": "model", "fields": []},
        {"model": "model", "fields": [{"name": "x"}]},
        {
            "model": "model",
            "fields": [
                {
                    "name": "x",
                    "left_key": "a",
                    "right_key": "a",
                    "left_scale": 0,
                    "right_scale": 1,
                    "unit": "unit",
                    "evidence": ["reference"],
                }
            ],
        },
        {
            "model": "model",
            "fields": [
                {
                    "name": "x",
                    "left_key": "a",
                    "right_key": "a",
                    "left_scale": 1,
                    "right_scale": 1,
                    "unit": "unit",
                    "evidence": [],
                },
                {
                    "name": "x",
                    "left_key": "b",
                    "right_key": "b",
                    "left_scale": 1,
                    "right_scale": 1,
                    "unit": "unit",
                    "evidence": ["reference"],
                },
            ],
        },
    ],
)
def test_invalid_policy_is_rejected(tmp_path, policy):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    write_record(left)
    write_record(right)
    with pytest.raises(ValueError):
        build_summary(left, right, policy)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), [], {"score": 1}, True])
def test_nonfinite_or_nonscalar_scores_are_rejected(tmp_path, value):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(
        '{"model":"model","case":"case","samples":[{"scores":{"score":'
        + json.dumps(value)
        + "}}]}"
    )
    write_record(right)
    with pytest.raises(ValueError, match="score"):
        build_summary(left, right)


def test_different_sample_order_and_count_are_not_paired(tmp_path):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    write_record(
        left,
        samples=[
            {"seed": 2, "scores": {"score": 2}},
            {"seed": 1, "scores": {"score": 1}},
        ],
    )
    write_record(right, samples=[{"seed": 1, "scores": {"score": 9}}])
    policy = {
        "model": "model",
        "fields": [
            {
                "name": "score",
                "left_key": "score",
                "right_key": "score",
                "left_scale": 1,
                "right_scale": 1,
                "unit": "unit",
                "evidence": ["specified"],
            }
        ],
    }

    result = build_summary(left, right, policy)

    assert result["arms"]["left"]["reported_samples"][0]["sample_identity"] == {
        "seed": 2
    }
    assert result["normalized_fields"][0]["difference_of_means"] == -7.5
    assert result["paired_analysis"] == "not_performed"


def _score_policy():
    return {
        "model": "model",
        "fields": [
            {
                "name": "score",
                "left_key": "score",
                "right_key": "score",
                "left_scale": 1,
                "right_scale": 1,
                "unit": "unit",
                "evidence": ["specified source"],
            }
        ],
    }


def test_failed_arm_with_scores_has_no_normalized_difference(tmp_path):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    write_record(left, samples=[{"scores": {"score": 1}}], returncode=1)
    write_record(right, samples=[{"scores": {"score": 2}}])

    result = build_summary(left, right, _score_policy())

    field = result["normalized_fields"][0]
    assert result["arms"]["left"]["reported_field_summaries"]["score"]["mean"] == 1
    assert field["left"] is None
    assert field["left_reason"] == "arm status is failed"
    assert field["difference_of_means"] is None
    assert field["difference_of_means_reason"]


def test_policy_json_top_level_array_is_rejected(tmp_path):
    left, right, policy_path = (
        tmp_path / "left.json",
        tmp_path / "right.json",
        tmp_path / "policy.json",
    )
    write_record(left)
    write_record(right)
    policy_path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        build_summary(left, right, policy_path)


def test_normalization_overflow_is_rejected(tmp_path):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    write_record(left, samples=[{"scores": {"score": 1e308}}])
    write_record(right, samples=[{"scores": {"score": 1}}])
    policy = _score_policy()
    policy["fields"][0]["left_scale"] = 100
    with pytest.raises(ValueError, match="nonfinite"):
        build_summary(left, right, policy)


def test_all_non_score_sample_metadata_is_preserved(tmp_path):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    sample = {
        "sample_id": "writer-id",
        "sample_index": 42,
        "producer_note": "native order retained",
        "scores": {"score": 1},
    }
    write_record(left, samples=[sample])
    write_record(right)

    result = build_summary(left, right)

    reported = result["arms"]["left"]["reported_samples"][0]
    assert reported["sample_identity"] == {}
    assert reported["sample_metadata"] == {
        "sample_id": "writer-id",
        "sample_index": 42,
        "producer_note": "native order retained",
    }
