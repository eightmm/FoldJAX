"""Aggregate completed AF3 captures without promoting missing evidence to a pass."""

import argparse
import hashlib
import json
from pathlib import Path

from bench.af3_closure import (
    compare_arms,
    compare_confidence,
    compare_first_warm,
    compiler_policy,
)
from bench.af3_closure_capture import save, sha


def summarize_case(name, audit, performance, *, artifact_base):
    reports = {
        "audit_pair": compare_arms(audit / "native", audit / "foldjax"),
        "performance_pair": compare_arms(
            performance / "native", performance / "foldjax", require_tape=False
        ),
    }
    for arm in ("native", "foldjax"):
        reports[f"{arm}_observer_bridge"] = compare_arms(
            audit / arm, performance / arm, require_tape=False
        )
        reports[f"{arm}_first_warm"] = compare_first_warm(performance / arm)
    summary = {
        "case": name,
        "passed": all(value["passed"] for value in reports.values()),
        "gates": {key: value["passed"] for key, value in reports.items()},
        "entity_parity": reports["audit_pair"].get("entity_parity"),
        "errors": {
            key: value["error"] for key, value in reports.items() if "error" in value
        },
    }
    summary["confidence"] = {
        stage: {
            "leaves": len(comparison["leaves"]),
            "passed": comparison["passed"],
            "all_leaves_bitwise_equal": bool(comparison["leaves"])
            and all(
                value.get("bitwise_equal") is True
                for value in comparison["leaves"].values()
            ),
            "max_absolute_error": max(
                (v.get("max_absolute_error", 0) for v in comparison["leaves"].values()),
                default=None,
            ),
        }
        for stage, comparison in (
            (
                "all_raw_outputs",
                compare_confidence(audit / "native/raw.npz", audit / "foldjax/raw.npz"),
            ),
            (
                "extracted",
                reports["audit_pair"].get(
                    "confidence", {"leaves": {}, "passed": False}
                ),
            ),
        )
    }
    artifacts, provenance, timings = {}, {}, {}
    for mode, root in (("audit", audit), ("performance", performance)):
        for arm in ("native", "foldjax"):
            capture = root / arm
            key = f"{mode}/{arm}"
            provenance[key] = json.loads((capture / "provenance.json").read_text())
            source_files = provenance[key].pop("source_files")
            provenance[key]["source_tree_sha256"] = hashlib.sha256(
                json.dumps(source_files, sort_keys=True).encode()
            ).hexdigest()
            provenance[key]["compiler_policy"] = compiler_policy(
                provenance[key].pop("xla_flags")
            )
            timings[key] = json.loads((capture / "finished.json").read_text())
            artifacts[key] = {
                "path": str(capture.relative_to(artifact_base)),
                "files": {
                    p.name: sha(p)
                    for p in sorted(capture.iterdir())
                    if p.is_file() and p.suffix in (".json", ".npz", ".textproto")
                },
            }
    summary.update(provenance=provenance, timings=timings, artifacts=artifacts)
    return summary, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to replace an existing report")
    cases = json.loads(args.manifest.read_text())
    if not cases or len({value["case"] for value in cases}) != len(cases):
        parser.error("case manifest must be nonempty and names unique")
    summaries = []
    for entry in cases:
        summary, _ = summarize_case(
            entry["case"],
            args.artifact_base / entry["audit"],
            args.artifact_base / entry["performance"],
            artifact_base=args.artifact_base,
        )
        summaries.append(summary)
    report = {
        "schema": 1,
        "scope": (
            "Finite AF3 native BF16 panel on a common FoldJAX runtime, matched "
            "Tokamax/XLA decisions, separate executables; not publisher dependency "
            "lock reproduction, default-autotuning stability or universal chemistry"
        ),
        "passed": all(value["passed"] for value in summaries),
        "cases": summaries,
        "manifest_sha256": sha(args.manifest),
        "reporter_sha256": sha(Path(__file__)),
        "comparator_sha256": sha(Path(__file__).with_name("af3_closure.py")),
        "memory_definition": (
            "JAX allocator process peak_bytes_in_use, including parameter/compile "
            "work; host ru_maxrss KiB; not warm-only live peak"
        ),
        "latency_definition": (
            "Parameter load/hash excluded; first includes compilation; warm "
            "synchronized with host transfer; observer runs not speed benchmarks"
        ),
    }
    save(args.out, report)
    print(json.dumps({"passed": report["passed"], "cases": len(summaries)}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
