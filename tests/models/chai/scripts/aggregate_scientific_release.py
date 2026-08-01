#!/usr/bin/env python3
"""Aggregate a pre-registered multi-fixture matched-tape release matrix."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
CLAIM_BOUNDARY = (
    "Matched-tape numerical port parity; not biological accuracy or "
    "native-PRNG distribution equivalence."
)
RELEASE_CONFIG = (200, 1, 1)
PROVENANCE_FIELDS = (
    "chai_jax_commit",
    "upstream_commit",
    "bundle_manifest_sha256",
    "conformer_sha256",
    "source_weight_sha256",
)


def _finite_float(report: Mapping[str, Any], name: str) -> float:
    try:
        value = float(report[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"report misses numeric {name}") from error
    if not np.isfinite(value):
        raise ValueError(f"report has non-finite {name}")
    return value


def _validate_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    if provenance.get("dirty") is not False:
        raise ValueError("release provenance requires dirty=false")
    for name in PROVENANCE_FIELDS[:-1]:
        value = provenance.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"release provenance requires nonempty {name}")
    weights = provenance.get("source_weight_sha256")
    if weights is None or weights == "" or weights == {} or weights == []:
        raise ValueError("release provenance requires nonempty source_weight_sha256")
    if isinstance(weights, Mapping):
        if not all(
            isinstance(name, str) and name and isinstance(value, str) and value
            for name, value in weights.items()
        ):
            raise ValueError(
                "release provenance source_weight_sha256 entries must be nonempty"
            )
    elif not isinstance(weights, str) or not weights:
        raise ValueError(
            "release provenance source_weight_sha256 must be a nonempty "
            "mapping or string"
        )
    return dict(provenance)


def _fixture_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for name in (
        "fasta",
        "context",
        "static_input_sha256",
        "torch_static_input_sha256",
    ):
        value = report.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"report misses nonempty fixture identity field {name}")
        identity[name] = value
    branches = report.get("branches")
    if not isinstance(branches, Mapping):
        raise ValueError("report misses fixture identity field branches")
    identity["branches"] = dict(branches)
    for name in ("model_size", "valid_atom_count"):
        try:
            value = int(report[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"report misses integer fixture identity field {name}"
            ) from error
        if value < 1:
            raise ValueError(f"fixture identity field {name} must be positive")
        identity[name] = value
    identity["optional"] = {
        name: {"present": name in report, "value": report.get(name)}
        for name in (
            "reference_augmentation_tape_sha256",
            "use_esm_embeddings",
        )
    }
    return identity


def _run_failures(
    report: Mapping[str, Any], threshold: Mapping[str, float]
) -> list[str]:
    failures = []
    raw_rmsd = _finite_float(report, "all_atom_raw_rmsd")
    maximum = _finite_float(report, "max_atom_displacement")
    correlation = _finite_float(report, "correlation")
    if raw_rmsd > float(threshold["max_all_atom_raw_rmsd"]):
        failures.append("all_atom_raw_rmsd")
    if maximum > float(threshold["max_atom_displacement"]):
        failures.append("max_atom_displacement")
    if correlation < float(threshold["min_correlation"]):
        failures.append("correlation")
    static = report.get("shared_torch_static")
    if not isinstance(static, Mapping):
        failures.append("shared_torch_static")
    else:
        static_rmsd = _finite_float(static, "all_atom_raw_rmsd")
        if static_rmsd > float(threshold["max_shared_static_raw_rmsd"]):
            failures.append("shared_torch_static.all_atom_raw_rmsd")
    static_metrics = _static_metrics(report)
    for name, threshold_name in (
        ("single_initial_valid_nrmse", "max_single_initial_valid_nrmse"),
        ("pair_initial_valid_nrmse", "max_pair_initial_valid_nrmse"),
        ("single_trunk_valid_nrmse", "max_single_trunk_valid_nrmse"),
        ("pair_trunk_valid_nrmse", "max_pair_trunk_valid_nrmse"),
    ):
        if static_metrics[name] > float(threshold[threshold_name]):
            failures.append(f"static.{name}")
    if static_metrics["identity_mismatch_count"] != 0:
        failures.append("static.identity_mismatch_count")
    return failures


def _static_metrics(report: Mapping[str, Any]) -> dict[str, float | int]:
    drift = report.get("static_input_drift")
    if not isinstance(drift, Mapping):
        raise ValueError("report misses static_input_drift")

    def valid_nrmse(name: str) -> float:
        field = drift.get(name)
        if not isinstance(field, Mapping):
            raise ValueError(f"report misses static_input_drift.{name}")
        valid = field.get("valid_region")
        if not isinstance(valid, Mapping):
            raise ValueError(f"report misses static_input_drift.{name}.valid_region")
        return _finite_float(valid, "nrmse")

    mismatch_count = 0
    for name in (
        "atom_single_mask",
        "atom_block_pair_mask",
        "token_single_mask",
        "atom_token_indices",
    ):
        field = drift.get(name)
        if not isinstance(field, Mapping):
            raise ValueError(f"report misses static_input_drift.{name}")
        try:
            mismatch_count += int(field["mismatch_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"report misses integer static_input_drift.{name}.mismatch_count"
            ) from error
    block_mask = drift["atom_block_pair_mask"]
    try:
        mismatch_count += int(block_mask["chemically_valid_mismatch_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "report misses integer chemically valid block-mask mismatch count"
        ) from error
    return {
        "single_initial_valid_nrmse": valid_nrmse("token_single_initial_repr"),
        "pair_initial_valid_nrmse": valid_nrmse("token_pair_initial_repr"),
        "single_trunk_valid_nrmse": valid_nrmse("token_single_trunk_repr"),
        "pair_trunk_valid_nrmse": valid_nrmse("token_pair_trunk_repr"),
        "identity_mismatch_count": mismatch_count,
    }


def aggregate_reports(
    reports: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    expected_fixtures: Sequence[str],
    expected_seeds: Sequence[int],
    thresholds: Mapping[str, Mapping[str, float]],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a complete fixture/seed matrix and return schema-v1 evidence."""
    validated_provenance = (
        _validate_provenance(provenance) if provenance is not None else {}
    )
    fixture_order = list(expected_fixtures)
    seed_order = list(expected_seeds)
    if not fixture_order:
        raise ValueError("expected fixtures must not be empty")
    if not seed_order:
        raise ValueError("expected seeds must not be empty")
    if len(set(fixture_order)) != len(fixture_order):
        raise ValueError("expected fixture IDs must be unique")
    if len(set(seed_order)) != len(seed_order):
        raise ValueError("expected seeds must be unique")
    if set(thresholds) != set(fixture_order):
        raise ValueError("threshold fixture IDs differ from expected fixtures")

    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for fixture_id, report in reports:
        if fixture_id not in fixture_order:
            raise ValueError(f"unexpected fixture: {fixture_id}")
        try:
            seed = int(report["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"report for {fixture_id} misses integer seed") from error
        key = (fixture_id, seed)
        if key in indexed:
            raise ValueError(f"duplicate report for {fixture_id} seed {seed}")
        indexed[key] = report

    missing = [
        f"{fixture_id}:seed={seed}"
        for fixture_id in fixture_order
        for seed in seed_order
        if (fixture_id, seed) not in indexed
    ]
    if missing:
        raise ValueError("missing reports: " + ", ".join(missing))
    unexpected_seeds = sorted(
        f"{fixture_id}:seed={seed}"
        for fixture_id, seed in indexed
        if seed not in seed_order
    )
    if unexpected_seeds:
        raise ValueError("unexpected report seeds: " + ", ".join(unexpected_seeds))

    contract_values = {str(report.get("contract")) for report in indexed.values()}
    config_values = {
        (
            int(report.get("timesteps", -1)),
            int(report.get("samples", -1)),
            int(report.get("trunk_recycles", -1)),
        )
        for report in indexed.values()
    }
    if contract_values != {"component/noise-matched"}:
        raise ValueError("reports do not share the matched-tape contract")
    if len(config_values) != 1:
        raise ValueError("reports have inconsistent sampler configuration")
    timesteps, samples, recycles = next(iter(config_values))
    if (timesteps, samples, recycles) != RELEASE_CONFIG:
        raise ValueError("release matrix requires recycles=1, timesteps=200, samples=1")

    fixture_results: dict[str, Any] = {}
    for fixture_id in fixture_order:
        identity = _fixture_identity(indexed[(fixture_id, seed_order[0])])
        run_results = []
        for seed in seed_order:
            report = indexed[(fixture_id, seed)]
            if _fixture_identity(report) != identity:
                raise ValueError(
                    f"fixture identity differs for {fixture_id} at seed {seed}"
                )
            failures = _run_failures(report, thresholds[fixture_id])
            static_metrics = _static_metrics(report)
            run_results.append(
                {
                    "seed": seed,
                    "model_size": int(report["model_size"]),
                    "coordinate_component_rmse": _finite_float(
                        report, "coordinate_component_rmse"
                    ),
                    "all_atom_raw_rmsd": _finite_float(report, "all_atom_raw_rmsd"),
                    "max_atom_displacement": _finite_float(
                        report, "max_atom_displacement"
                    ),
                    "correlation": _finite_float(report, "correlation"),
                    "shared_static_raw_rmsd": _finite_float(
                        report["shared_torch_static"], "all_atom_raw_rmsd"
                    ),
                    "static": static_metrics,
                    "pass": not failures,
                    "failure_metrics": failures,
                }
            )
        raw_values = [run["all_atom_raw_rmsd"] for run in run_results]
        correlations = [run["correlation"] for run in run_results]
        fixture_results[fixture_id] = {
            "identity": identity,
            "thresholds": dict(thresholds[fixture_id]),
            "runs": run_results,
            "aggregate": {
                "median_all_atom_raw_rmsd": float(np.median(raw_values)),
                "max_all_atom_raw_rmsd": float(np.max(raw_values)),
                "min_correlation": float(np.min(correlations)),
            },
            "pass": all(run["pass"] for run in run_results),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "provenance": validated_provenance,
        "config": {
            "contract": "component/noise-matched",
            "recycles": recycles,
            "timesteps": timesteps,
            "samples": samples,
            "seeds": seed_order,
        },
        "fixtures": fixture_results,
        "status": (
            "green"
            if all(result["pass"] for result in fixture_results.values())
            else "red"
        ),
    }


def _report_argument(value: str) -> tuple[str, Path]:
    fixture_id, separator, path = value.partition("=")
    if not separator or not fixture_id or not path:
        raise argparse.ArgumentTypeError("report must be FIXTURE_ID=PATH")
    return fixture_id, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", type=_report_argument, required=True
    )
    parser.add_argument("--expected-fixture", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = [
        (fixture_id, json.loads(path.read_text())) for fixture_id, path in args.report
    ]
    thresholds = json.loads(args.thresholds.read_text())
    provenance = json.loads(args.provenance.read_text())
    result = aggregate_reports(
        reports,
        expected_fixtures=args.expected_fixture,
        expected_seeds=args.seed,
        thresholds=thresholds,
        provenance=provenance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, sort_keys=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if result["status"] != "green":
        raise SystemExit("scientific release matrix failed")


if __name__ == "__main__":
    main()
