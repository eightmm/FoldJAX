"""Tabulate the OpenBind comparison reports under one snapshot root.

Reads every ``*-comparison.json`` written by ``openbind_core_report.py`` and
prints entity maxima and public confidence maxima per arm. It renders what
the reports say; it admits nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(root):
    out = []
    for path in sorted(Path(root).glob("*-comparison.json")):
        report = json.loads(path.read_text())
        entities = report["coordinates"]["entity_max_rmsd"]
        public = report["public_confidence"]
        out.append({
            "arm": path.name.removesuffix("-comparison.json"),
            "native": report["native_backend"],
            "candidate": report["candidate_backend"],
            "entities": entities,
            "worst": max(entities.values()),
            "plddt": public["plddt"]["max_absolute_error"],
            "ptm": public["ptm"]["max_absolute_error"],
            "iptm": public["iptm"]["max_absolute_error"],
        })
    return out


def render(table):
    lines = [
        "| arm | native | candidate | entity max RMSD (A) "
        "| pLDDT max | pTM max | ipTM max |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in table:
        entities = "; ".join(f"{k} {v:.6f}" for k, v in row["entities"].items())
        lines.append(
            f"| {row['arm']} | {row['native']} | {row['candidate']} | {entities} | "
            f"{row['plddt']:.3f} | {row['ptm']:.6f} | {row['iptm']:.6f} |"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    table = rows(args.root)
    print(json.dumps(table, indent=2) if args.json else render(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
