"""Descriptive, unpaired summaries of public scalar score dictionaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
_IDENTITY_FIELDS = ("seed", "id", "index")
_METADATA_TOKENS = (
    "schedule",
    "source",
    "runtime",
    "precision",
    "option",
    "execution",
    "identity",
    "provenance",
)


def _load_json(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    raw = path.read_bytes()
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(record, dict):
        raise ValueError(f"{path}: result record must be a JSON object")
    for field in ("model", "case"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"{path}: missing nonempty {field}")
    samples = record.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"{path}: samples must be a list")
    return record, hashlib.sha256(raw).hexdigest()


def _finite_score(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: score must be a finite scalar number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{where}: score must be finite")
    return value


def _summarize(values: list[float]) -> dict[str, Any]:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("summary values must remain finite")
    try:
        mean = math.fsum(values) / len(values)
    except OverflowError as error:
        raise ValueError("summary mean is not finite") from error
    if not math.isfinite(mean):
        raise ValueError("summary mean is not finite")
    return {
        "finite_scalar_count": len(values),
        "values": values,
        "min": min(values),
        "max": max(values),
        "mean": mean,
    }


def _arm(record: dict[str, Any]) -> dict[str, Any]:
    samples = record.get("samples", [])
    reported_samples: list[dict[str, Any]] = []
    fields: dict[str, list[float]] = {}
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not isinstance(sample.get("scores"), dict):
            raise ValueError(f"samples[{sample_index}]: scores must be an object")
        scores = sample["scores"]
        numeric_scores = {
            key: _finite_score(value, f"samples[{sample_index}].scores[{key!r}]")
            for key, value in scores.items()
        }
        if any(not isinstance(key, str) for key in scores):
            raise ValueError(f"samples[{sample_index}]: score names must be strings")
        for key, value in numeric_scores.items():
            fields.setdefault(key, []).append(value)
        identity = {
            key: deepcopy(sample[key]) for key in _IDENTITY_FIELDS if key in sample
        }
        reported_samples.append(
            {
                "sample_identity": identity,
                "sample_metadata": {
                    key: deepcopy(value)
                    for key, value in sample.items()
                    if key != "scores"
                },
                "reported_scores": deepcopy(scores),
            }
        )
    failed = bool(record.get("failed")) or record.get("returncode", 0) != 0
    status = "failed" if failed else "no_samples" if not samples else "ok"
    return {
        "status": status,
        "sample_count": len(samples),
        "reported_samples": reported_samples,
        "reported_field_summaries": {
            key: _summarize(values) | {"missing_count": len(samples) - len(values)}
            for key, values in fields.items()
        },
    }


def _load_policy(field_policy: Any) -> tuple[dict[str, Any] | None, str | None]:
    if field_policy is None:
        return None, None
    if isinstance(field_policy, (str, Path)):
        raw = Path(field_policy).read_bytes()
        try:
            policy = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("field policy is invalid JSON") from error
        digest = hashlib.sha256(raw).hexdigest()
    elif isinstance(field_policy, dict):
        policy = deepcopy(field_policy)
        digest = hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    else:
        raise ValueError("field policy must be a JSON object or path")
    if not isinstance(policy, dict):
        raise ValueError("field policy must be a JSON object")
    if not isinstance(policy.get("model"), str) or not policy["model"]:
        raise ValueError("field policy requires a nonempty model")
    if not isinstance(policy.get("fields"), list) or not policy["fields"]:
        raise ValueError("field policy requires a nonempty fields list")
    names: set[str] = set()
    for item in policy["fields"]:
        if not isinstance(item, dict):
            raise ValueError("field policy entries must be objects")
        for key in ("name", "left_key", "right_key", "unit"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError(f"field policy requires nonempty {key}")
        if item["name"] in names:
            raise ValueError(f"field policy has duplicate name {item['name']!r}")
        names.add(item["name"])
        for key in ("left_scale", "right_scale"):
            value = item.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"field policy {item['name']!r} requires positive finite {key}"
                )
        evidence = item.get("evidence")
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(
                not isinstance(reference, str) or not reference.strip()
                for reference in evidence
            )
        ):
            raise ValueError(
                f"field policy {item['name']!r} requires nonempty evidence"
            )
    return policy, digest


def _metadata_issues(
    left: dict[str, Any], right: dict[str, Any]
) -> list[dict[str, Any]]:
    issues = []
    for key in sorted(set(left) | set(right)):
        if key in {"samples", "model", "case", "impl"} or not any(
            token in key.lower() for token in _METADATA_TOKENS
        ):
            continue
        if left.get(key) != right.get(key):
            issues.append(
                {
                    "field": key,
                    "left": deepcopy(left.get(key)),
                    "right": deepcopy(right.get(key)),
                    "effect": (
                        "visible metadata mismatch; fair performance is not established"
                    ),
                }
            )
    return issues


def _scaled_summary(
    values: list[float], scale: float, field: str, arm: str
) -> dict[str, Any]:
    normalized = [value * scale for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(
            f"normalization of {field!r} for {arm} produces nonfinite values"
        )
    return _summarize(normalized)


def _normalized(
    policy: dict[str, Any], left: dict[str, Any], right: dict[str, Any]
) -> list[dict[str, Any]]:
    fields = []
    for item in policy["fields"]:
        result: dict[str, Any] = {
            "name": item["name"],
            "unit": item["unit"],
            "evidence": deepcopy(item["evidence"]),
            "left_key": item["left_key"],
            "right_key": item["right_key"],
            "left_scale": item["left_scale"],
            "right_scale": item["right_scale"],
            "descriptive_unpaired": True,
        }
        for arm_name, key, scale, arm in (
            ("left", item["left_key"], item["left_scale"], left),
            ("right", item["right_key"], item["right_scale"], right),
        ):
            if arm["status"] != "ok":
                result[arm_name] = None
                result[f"{arm_name}_reason"] = f"arm status is {arm['status']}"
                continue
            summary = arm["reported_field_summaries"].get(key)
            if summary is None:
                result[arm_name] = None
                result[f"{arm_name}_reason"] = "reported field is missing"
                continue
            result[arm_name] = _scaled_summary(
                summary["values"], scale, item["name"], arm_name
            )
        if result["left"] is None or result["right"] is None:
            result["difference_of_means"] = None
            result["difference_of_means_reason"] = (
                "requires both arms to be ok with an available field"
            )
        else:
            difference = result["left"]["mean"] - result["right"]["mean"]
            if not math.isfinite(difference):
                raise ValueError(
                    f"difference of means for {item['name']!r} is not finite"
                )
            result["difference_of_means"] = difference
        fields.append(result)
    return fields


def build_summary(
    left: str | Path, right: str | Path, field_policy: Any = None
) -> dict[str, Any]:
    """Build an unpaired descriptive report; no score semantics are inferred."""
    left_record, left_hash = _load_json(left)
    right_record, right_hash = _load_json(right)
    if left_record["model"] != right_record["model"]:
        raise ValueError("result records must have the same model")
    if left_record["case"] != right_record["case"]:
        raise ValueError("result records must have the same case")
    policy, policy_hash = _load_policy(field_policy)
    if policy is not None and policy["model"] != left_record["model"]:
        raise ValueError("field policy model does not match result model")
    left_arm, right_arm = _arm(left_record), _arm(right_record)
    output = {
        "schema_version": SCHEMA_VERSION,
        "scope": "reported_public_scalar_scores",
        "inputs": {"left_sha256": left_hash, "right_sha256": right_hash},
        "model": left_record["model"],
        "case": left_record["case"],
        "arms": {"left": left_arm, "right": right_arm},
        "record_metadata": {
            "left": {
                key: deepcopy(value)
                for key, value in left_record.items()
                if key != "samples"
            },
            "right": {
                key: deepcopy(value)
                for key, value in right_record.items()
                if key != "samples"
            },
        },
        "metadata_issues": _metadata_issues(left_record, right_record),
        "automatic_comparability": "unknown",
        "raw_confidence_validation": "not_assessed",
        "raw_network_mask_validation": "not_assessed",
        "random_tape_validation": "not_assessed",
        "paired_analysis": "not_performed",
        "score_semantics": (
            "not automatically verified; pTM/ipTM are predicted confidence, "
            "not actual TM-score"
        ),
    }
    if policy is not None:
        output["field_policy"] = {
            "sha256": policy_hash,
            "caller_supplied": policy,
            "source_semantics": "not automatically verified",
        }
        output["normalized_fields"] = _normalized(policy, left_arm, right_arm)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--field-policy", type=Path)
    args = parser.parse_args()
    result = build_summary(args.left, args.right, args.field_policy)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
