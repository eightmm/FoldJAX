"""Tabulate the OpenDDE master panel: native floor, port residual, port floor.

For each ``<root>/<case>`` with ``native-A``, ``native-B``, ``fj-high-A``,
``fj-high-B`` and ``fj-highest`` captures, runs ``opendde_closure_report`` on
the pairs that are missing and prints one row per case with the entity
maximum RMSD of: native A vs B (native process floor), native A vs port high
(residual), port high A vs B (port process floor), native A vs port highest.
A residual within twice the larger floor is process noise, not a route
difference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PAIRS = (
    ("native-A", "native-B"),
    ("native-A", "fj-high-A"),
    ("fj-high-A", "fj-high-B"),
    ("native-A", "fj-highest"),
)


def report(root, case, left, right, python, env):
    out = root / case / f"{left}-vs-{right}.json"
    if not out.exists():
        if not (root / case / left / "finished.json").exists() or not (
            root / case / right / "finished.json"
        ).exists():
            return None
        subprocess.run(
            [str(python), "-P", "-m", "bench.opendde_closure_report",
             str(root / case / left), str(root / case / right), "--out", str(out)],
            check=True, env=env, capture_output=True,
        )
    data = json.loads(out.read_text())
    return {k: float(v) for k, v in data["coordinates"]["entity_max_rmsd"].items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    import os

    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "PYTHONPATH": f"{args.root}/src:{args.root}"}
    rows = {}
    cases = sorted(
        p.name for p in args.root.iterdir() if (p / "native-A").exists()
    )
    for case in cases:
        rows[case] = {
            f"{left}-vs-{right}": report(
                args.root, case, left, right, args.python, env
            )
            for left, right in PAIRS
        }
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    def fmt(d):
        return "-" if d is None else "; ".join(f"{k} {v:.4f}" for k, v in d.items())

    print(
        "| case | native A vs B (floor) | native vs port high | "
        "port high A vs B (floor) | native vs port highest | verdict |"
    )
    print("| --- | --- | --- | --- | --- | --- |")
    for case, cells in rows.items():
        nf, res, pf, hi = (
            cells[f"{left}-vs-{right}"] for left, right in PAIRS
        )
        verdict = "-"
        if res is not None:
            worst = max(res.values())
            floors = [max(d.values()) for d in (nf, pf) if d is not None]
            if worst < 0.05:
                verdict = "pass"
            elif worst < 0.1:
                verdict = "deferred"
            elif floors and worst <= 2 * max(floors):
                verdict = "at-floor"
            else:
                verdict = "investigate"
        cells = " | ".join(fmt(d) for d in (nf, res, pf, hi))
        print(f"| {case} | {cells} | {verdict} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
