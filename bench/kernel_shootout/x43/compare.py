"""x43 rows (Pallas pair kernels) against x42 (cuEq-aligned attention, a176340).

Per row: warm wall and peak against the x42 row of the same base id, and against
this snapshot's `-ctl` row when one exists. That second comparison separates the
arm from code drift between a176340 and the snapshot. Per sample: the deposited
complex CA RMSD / TM for x43 and x42, and the same-index CA RMSD between them.
The x42 same-index floor is not in this directory; read the campaign's r1-vs-r2
rows (x42 `compare.py`) for it.

    python compare.py L3000_6ztx-boltz2-fj-1c-def-s101-pallas [...]
"""

# ruff: noqa: E501, N806 -- bench tooling: long table f-strings, A/B array names

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
X42 = HERE.parent / "x42-cueq-align-20260924"
J = HERE.parent / "jctc"
sys.path.insert(0, str(J))
import score  # noqa: E402

CODES = {"L1000_3og2": "3OG2", "L3000_6ztx": "6ZTX", "L2000_5dei": "5DEI"}


def passes(results: Path, row: str):
    out = {}
    for f in sorted((results / row).glob("*.json")):
        if "sidecar" in f.name or f.name == "native_input.json":
            continue
        out[f.stem] = json.loads(f.read_text())
    return out


def warm(p):
    return [
        d["wall_s"]
        for k, d in p.items()
        if k.startswith("warm") and not d.get("failed")
    ]


def peak(p):
    return max(
        [
            d.get("peak_mib") or 0
            for k, d in p.items()
            if k.startswith("warm") and not d.get("failed")
        ]
        or [0]
    )


def cifs(work: Path, row: str):
    return score.sample_cifs(work / row / "warm-1", "foldjax", 101)


for row in sys.argv[1:]:
    base, arm = row.rsplit("-", 1)
    case = row.split("-")[0]
    ref_path = score.REFS / f"{CODES[case]}.cif"
    ref = score.read_ca(ref_path, reference=True)[0]
    units = score.assembly_units(ref_path, ref)
    new = passes(HERE / "results", row)
    old = passes(X42 / "results", base)
    ctl_row = f"{base}-ctl"
    ctl = passes(HERE / "results", ctl_row) if arm != "ctl" else {}
    print(f"== {row}")
    wn, wo, wc = warm(new), warm(old), warm(ctl)
    if not wn:
        print("   missing warm pass", {k: d.get("reason") for k, d in new.items()})
        continue
    line = f"   warm wall  new {statistics.median(wn):9.2f} s"
    if wo:
        line += f" | x42 {statistics.median(wo):9.2f} s ratio {statistics.median(wn) / statistics.median(wo):.3f}"
    if wc:
        line += f" | ctl {statistics.median(wc):9.2f} s ratio {statistics.median(wn) / statistics.median(wc):.3f}"
    print(line)
    print(
        f"   peak MiB   new {peak(new):9.1f} | x42 {peak(old):9.1f}"
        + (f" | ctl {peak(ctl):9.1f}" if ctl else "")
    )
    a = cifs(HERE / "work", row)
    b = cifs(X42 / "work", base) if old else []
    for i, pa in enumerate(a):
        sa = score.score_structure(pa, ref, units, case, "usalign")
        text = (
            f"   sample {i}: deposited CA RMSD new {sa.get('complex_ca_rmsd', float('nan')):.3f}"
            f" TM {sa.get('complex_tm', float('nan')):.4f}"
        )
        if i < len(b):
            sb = score.score_structure(b[i], ref, units, case, "usalign")
            text += (
                f" | x42 {sb.get('complex_ca_rmsd', float('nan')):.3f}"
                f" TM {sb.get('complex_tm', float('nan')):.4f}"
                f" | same-index new-vs-x42 {score.same_index(pa, b[i]):.3f}"
            )
        print(text)
