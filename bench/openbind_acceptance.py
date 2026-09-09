"""OpenBind cuEq-versus-cuEq acceptance over one snapshot's replay reports.

Exit 0 only when every case that is a genuine cuEq-versus-cuEq comparison
(native census: cuEq attention ran, never fell back) has every entity below
0.1 A against the native cuEq capture, and the second-process FoldJAX repeat
is bitwise equal to the first. 0.05-0.1 A is the user's deferred band, not a
failure; above 0.1 A is. Cases whose native cuEq attention fell back (at or
below 100 tokens) are reported in their own class and never counted either
way. This reads reports; it runs nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PASS, DEFER, FAIL = 0.05, 0.1, float("inf")


def classify(worst):
    if worst < PASS:
        return "pass"
    if worst < DEFER:
        return "deferred"
    return "investigate"


def census_class(capture):
    calls = json.loads((capture / "kernel-calls.json").read_text())["calls"]
    if calls.get("attention.cueq_fallback_true", 0):
        return "small-token (native cuEq attention fell back)"
    if not calls.get("attention.cueq") or not calls.get("trimul.cueq"):
        return "native did not run cuEq kernels"
    return "cueq-vs-cueq"


def cross_process(snapshot, case, backend):
    path = snapshot / f"{case}-{backend}-cross-process.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    return all(field["bitwise_equal"] for field in report["fields"].values())


def evaluate(snapshot, native, cases, backend="cueq"):
    rows = []
    for case in cases:
        report_path = snapshot / f"{case}-{backend}-vs-native-cueq-comparison.json"
        capture = native / case / "native-cueq"
        row = {"case": case, "backend": backend}
        if not report_path.exists() or not (capture / "kernel-calls.json").exists():
            row.update(status="missing")
            rows.append(row)
            continue
        report = json.loads(report_path.read_text())
        entities = report["coordinates"]["entity_max_rmsd"]
        worst = max(entities.values())
        row.update(
            entities=entities,
            worst=worst,
            plddt_max=report["public_confidence"]["plddt"]["max_absolute_error"],
            census=census_class(capture),
            structure=classify(worst),
            repeat_bitwise=cross_process(snapshot, case, backend),
        )
        row["status"] = (
            "excluded"
            if row["census"] != "cueq-vs-cueq"
            else "fail"
            if row["structure"] == "investigate" or row["repeat_bitwise"] is not True
            else row["structure"]
        )
        rows.append(row)
    counted = [r for r in rows if r["status"] not in ("excluded",)]
    accepted = bool(counted) and all(
        r["status"] in ("pass", "deferred") for r in counted
    )
    return {"accepted": accepted, "rows": rows}


def render(result):
    lines = [
        "| case | census | entity max RMSD (A) | structure | repeat bitwise | status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["rows"]:
        if r["status"] == "missing":
            lines.append(f"| {r['case']} | - | - | - | - | missing |")
            continue
        ents = "; ".join(f"{k} {v:.6f}" for k, v in r["entities"].items())
        lines.append(
            f"| {r['case']} | {r['census']} | {ents} | {r['structure']} | "
            f"{r['repeat_bitwise']} | {r['status']} |"
        )
    lines.append(f"\naccepted: {result['accepted']}")
    return "\n".join(lines)


DEFAULT_CASES = (
    "protein_ligand_5sak", "protein_protein_7st3", "protein_1ubq",
    "protein_dna_7r6r", "protein_rna_ligand_3v7e", "protein_rna_1urn",
    "rna_ligand_3gca",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--backend", default="cueq")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = evaluate(
        args.snapshot, args.native, args.cases or DEFAULT_CASES, args.backend
    )
    print(json.dumps(result, indent=2) if args.json else render(result))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
