"""Tabulate OpenBind warm measurements: FoldJAX arms beside native arms per case.

Reads ``<root>/<case>-<arm>/measurements.json`` written by
``openbind_warm.py`` (arms ``fj-<backend>``) and ``openbind_native_warm.py``
(arms ``native-<backend>``). Warm medians are not the same workload: native
``predict_step`` includes confidence and ranking, the FoldJAX forward does
not, so the ratio is reported as observed, never as a kernel speedup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CASES = (
    "protein_ligand_5sak", "protein_protein_7st3", "protein_1ubq",
    "protein_dna_7r6r", "protein_rna_ligand_3v7e", "protein_rna_1urn",
    "rna_ligand_3gca",
)
ARMS = ("native-triton", "native-cueq", "fj-cueq", "fj-cueq-full")


def read(root, case, arm):
    path = root / f"{case}-{arm}" / "measurements.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    peaks = [
        row["allocator"]["peak_bytes_in_use"]
        for row in data["calls"]
        if "allocator" in row and "peak_bytes_in_use" in row["allocator"]
    ]
    return {
        "warm_median_seconds": data["warm_median_seconds"],
        "first_seconds": data["calls"][0]["seconds"],
        "peak_gib": max(peaks) / 2**30 if peaks else None,
        "kernel_calls": data.get("kernel_calls"),
        "warm_bitwise_stable": all(
            all(a["equal_to_first"].values()) for a in data.get("archives", [])
        ) if data.get("archives") else None,
    }


def table(root, cases=CASES, arms=ARMS):
    rows = {case: {arm: read(root, case, arm) for arm in arms} for case in cases}
    lines = [
        "| case | " + " | ".join(f"{arm} warm s / peak GiB" for arm in arms) + " |",
        "| --- |" + " ---: |" * len(arms),
    ]
    for case, cells in rows.items():
        parts = []
        for arm in arms:
            cell = cells[arm]
            if cell is None:
                parts.append("-")
            else:
                peak = "-" if cell["peak_gib"] is None else f"{cell['peak_gib']:.2f}"
                parts.append(f"{cell['warm_median_seconds']:.3f} / {peak}")
        lines.append(f"| {case} | " + " | ".join(parts) + " |")
    return "\n".join(lines), rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    text, rows = table(args.root)
    print(json.dumps(rows, indent=2) if args.json else text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
