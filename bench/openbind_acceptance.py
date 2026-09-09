"""OpenBind cuEq-versus-cuEq acceptance over one snapshot's replay reports.

Exit 0 only when every case that is a genuine cuEq-versus-cuEq comparison
(native census: cuEq attention ran, never fell back) has every entity below
0.1 A against the native cuEq capture -- or, above that, within twice the
larger of the two process floors (native cuEq versus its own repeat; the
port frozen versus unfrozen autotune, same tape) -- and
the second-process FoldJAX repeat is bitwise equal to the first. 0.05-0.1 A is
the user's deferred band, not a failure; above 0.1 A and above the native
floor is. Cases whose native cuEq attention fell back (at or
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


def native_floor(capture_root):
    """Entity max RMSD between two native cuEq processes of the same tape."""
    path = capture_root / "native-cueq-vs-repeat.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if not report.get("same_tape"):
        return None
    return max(report["coordinates"]["entity_max_rmsd"].values())


def port_floor(snapshot, case, backend):
    """Entity max RMSD between a frozen and an unfrozen-autotune port process.

    Every ``<case>-<backend>-port-floor*.json`` is one sample of that floor
    (each unfrozen process picks its own kernels); the largest sample is the
    floor. Returns ``(floor, samples)`` or ``(None, 0)``.
    """
    samples = [
        max(json.loads(path.read_text())["coordinates"]["entity_max_rmsd"].values())
        for path in sorted(snapshot.glob(f"{case}-{backend}-port-floor*.json"))
    ]
    return (max(samples), len(samples)) if samples else (None, 0)


def residual_draws(snapshot, case, backend):
    """Worst entity RMSD of every port process replayed against native cuEq."""
    draws = {}
    for path in sorted(
        snapshot.glob(f"{case}-{backend}-vs-native-cueq*-comparison.json")
    ):
        report = json.loads(path.read_text())
        draws[path.name] = max(report["coordinates"]["entity_max_rmsd"].values())
    return draws


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
        floor = native_floor(native / case)
        own, own_samples = port_floor(snapshot, case, backend)
        noise = max(x for x in (floor, own) if x is not None) if (
            floor is not None or own is not None
        ) else None
        structure = classify(worst)
        if structure == "investigate" and noise is not None and worst <= 2 * noise:
            # Each side already moves by its own floor between two processes
            # of the same tape (kernel selection); a residual within twice the
            # larger floor is that noise, not a route difference.
            structure = "at-floor"
        row.update(
            entities=entities,
            worst=worst,
            plddt_max=report["public_confidence"]["plddt"]["max_absolute_error"],
            census=census_class(capture),
            native_floor=floor,
            port_floor=own,
            port_floor_samples=own_samples,
            draws=residual_draws(snapshot, case, backend),
            structure=structure,
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
        r["status"] in ("pass", "deferred", "at-floor") for r in counted
    )
    return {"accepted": accepted, "rows": rows}


def render(result):
    lines = [
        "| case | census | entity max RMSD (A) | native floor (A) | port floor (A) "
        "| structure | repeat bitwise | status |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for r in result["rows"]:
        if r["status"] == "missing":
            lines.append(f"| {r['case']} | - | - | - | - | - | - | missing |")
            continue
        ents = "; ".join(f"{k} {v:.6f}" for k, v in r["entities"].items())
        floor = "-" if r["native_floor"] is None else f"{r['native_floor']:.4f}"
        own = (
            "-" if r["port_floor"] is None
            else f"{r['port_floor']:.4f} (n={r['port_floor_samples']})"
        )
        lines.append(
            f"| {r['case']} | {r['census']} | {ents} | {floor} | {own} | "
            f"{r['structure']} | {r['repeat_bitwise']} | {r['status']} |"
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
