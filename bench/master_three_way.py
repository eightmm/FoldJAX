"""Three-way parity table for the master panels: native floor, residual, port floor.

For every ``<root>/<case>`` with ``native-A``, ``native-B``, ``port-A`` and
``port-B`` run directories, reports the entity maximum RMSD (one whole-system
Kabsch per sample, entity maxima) of native A vs B (upstream's own process
floor), native A vs port A and native B vs port A (the residual, read against
both native draws), and port A vs B (the port's own floor). A residual that is
no larger than the distance between the two native processes is upstream's
scatter, not a route difference; a residual that exceeds it on every pairing
by a margin is real. Verdicts: ``pass`` below 0.05 Å, ``deferred`` below
0.1 Å, ``at-floor`` when the worst residual is within twice the larger floor,
else ``investigate``.

Models: ``protenix`` (``prediction.npz`` + ``native-identity.npz``) and
``esmfold2`` (``upstream_coords.npz`` / ``jax_coords.npz`` + ``features.npz``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAIRS = (("native-A", "native-B"), ("native-A", "port-A"),
         ("native-B", "port-A"), ("port-A", "port-B"))


def _protenix(case_dir, left, right):
    from bench.protenix_master_diff import compare

    return compare(case_dir / left, case_dir / right, case_dir / "native-A")


def _esmfold2_coords(path):
    for name in ("upstream_coords.npz", "jax_coords.npz"):
        if (path / name).exists():
            with np.load(path / name, allow_pickle=False) as data:
                return np.asarray(data["coords"])
    raise FileNotFoundError(path)


def _esmfold2(case_dir, left, right):
    from bench.esmfold2_tape_report import compare_coordinates

    with np.load(case_dir / "native-A/features.npz", allow_pickle=False) as data:
        features = dict(data)
    report = compare_coordinates(
        _esmfold2_coords(case_dir / left), _esmfold2_coords(case_dir / right), features
    )
    return {"coordinates": report}


COMPARE = {"protenix": _protenix, "esmfold2": _esmfold2}


def finished(path):
    return path.exists() and any(
        (path / name).exists()
        for name in ("prediction.npz", "upstream_coords.npz", "jax_coords.npz")
    )


def evaluate(root, model):
    rows = {}
    for case_dir in sorted(p for p in root.iterdir() if (p / "native-A").is_dir()):
        cells = {}
        for left, right in PAIRS:
            key = f"{left}-vs-{right}"
            out = case_dir / f"{key}.json"
            if out.exists():
                report = json.loads(out.read_text())
            elif finished(case_dir / left) and finished(case_dir / right):
                report = COMPARE[model](case_dir, left, right)
                out.write_text(json.dumps(report, indent=2, sort_keys=True))
            else:
                cells[key] = None
                continue
            cells[key] = {
                "max": {k: float(v) for k, v in
                        report["coordinates"]["entity_max_rmsd"].items()},
                "per_sample": {k: [float(x) for x in v] for k, v in
                               report["coordinates"]["entity_rmsd"].items()},
            }
        rows[case_dir.name] = cells
    return rows


def verdict(cells):
    residual = cells.get("native-A-vs-port-A")
    if residual is None:
        return "-"
    worst = max(residual["max"].values())
    floors = [max(cells[k]["max"].values()) for k in
              ("native-A-vs-native-B", "port-A-vs-port-B") if cells.get(k)]
    if worst < 0.05:
        return "pass"
    if worst < 0.1:
        return "deferred"
    if floors and worst <= 2 * max(floors):
        return "at-floor"
    return "investigate"


def render(rows):
    def label(key):
        # esmfold2 keys read "entity=0|mol_type=0|asym=0"; show the asym id only.
        return key.split("asym=")[-1] if "asym=" in key else key

    def fmt(cell):
        if cell is None:
            return "-"
        return "; ".join(f"{label(k)} {v:.3f}" for k, v in cell["max"].items())

    lines = [
        "| case | native A vs B (floor) | native A vs port | native B vs port "
        "| port A vs B (floor) | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case, cells in rows.items():
        parts = " | ".join(fmt(cells.get(f"{a}-vs-{b}")) for a, b in PAIRS)
        lines.append(f"| {case} | {parts} | {verdict(cells)} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(COMPARE), required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rows = evaluate(args.root, args.model)
    print(json.dumps(rows, indent=2) if args.json else render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
