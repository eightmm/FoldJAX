"""Per-family compiled memory of the `pallas` arm against `released` (wall_split JSON).

    python peakprobe_compare.py OUT/wall-split-1003-serial.json
"""

# ruff: noqa: E501

import json
import sys

report = json.load(open(sys.argv[1]))
arms = report["arms"]
names = list(arms)
fams = sorted({f for a in arms.values() for f in a["families"]})
mib = lambda b: b / 2**20  # noqa: E731
print(f"tokens {report['tokens']} msa_depth {report['msa_depth']} arms {names}")
print(f"{'family':32s} " + " | ".join(f"{n:>30s}" for n in names) + " | delta temp / arg+out+temp (MiB)")
for fam in fams:
    cells, totals = [], []
    for n in names:
        row = arms[n]["families"].get(fam, {})
        ma = row.get("memory_analysis")
        if not isinstance(ma, dict):
            cells.append(f"{'-':>30s}")
            totals.append(None)
            continue
        t, a, o = (mib(ma.get(k, 0)) for k in ("temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes"))
        cells.append(f"temp {t:8.1f} arg {a:7.1f} out {o:7.1f}")
        totals.append((t, t + a + o))
    delta = ""
    if len(totals) == 2 and all(totals):
        delta = f"{totals[1][0] - totals[0][0]:+8.1f} / {totals[1][1] - totals[0][1]:+8.1f}"
    print(f"{fam:32s} " + " | ".join(cells) + " | " + delta)
