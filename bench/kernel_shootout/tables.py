"""Markdown winner tables for REPORT.md from a shootout run.

python tables.py RUN_DIR
"""

# ruff: noqa: E501, N806 -- bench tooling: long table f-strings, A/B array names

import json
import sys
from pathlib import Path

import numpy as np

PAD = 3
run = Path(sys.argv[1])
PROD = {
    "af3": {
        "att": "cueq_pad_highest",
        "mul": "cueq_boltz2",
        "trans": "boltz2_tokamax_rc",
    },
    "opendde": {
        "att": "cueq_pad_high",
        "mul": "cueq_fused",
        "trans": "boltz2_xla_full",
    },
}
#: The arms each row shows beside production: the pinned candidates, then the
#: best other non-cuEq arm.
SHOW = {
    "att": ["tokamax_triton", "fj_flash_r2_128x64w8"],
    "mul": ["fj_k1k2_32w4", "tokamax_tm_xla", "tokamax_tm_triton", "xla_boltz2"],
    "trans": [
        "fj_fused_64w4_fc64",
        "fj_fused_32w4",
        "boltz2_tokamax_full",
        "boltz2_xla_rc",
    ],
}


def load(profile):
    out = run / ("af3" if profile == "af3" else "opendde")
    cells = {}
    for line in open(out / "cells.jsonl"):
        c = json.loads(line)
        cells[(c["family"], c["n"], c["arm"])] = c
    return out, cells


def err(out, fam, n, a, b):
    pa, pb = out / f"{fam}-{n}-{a}.npz", out / f"{fam}-{n}-{b}.npz"
    if not (pa.exists() and pb.exists()):
        return None
    A, B = np.load(pa), np.load(pb)
    ra, rb = list(A["rows"]), list(B["rows"])
    common = [r for r in ra if r in rb and r < n - PAD]
    x = A["out"][[ra.index(r) for r in common]]
    y = B["out"][[rb.index(r) for r in common]]
    return float(np.abs(x - y).max())


def cell_txt(out, cells, fam, n, arm, prod, ref):
    c = cells.get((fam, n, arm))
    if c is None:
        return "-"
    if not c.get("ok"):
        e = c.get("error") or c.get("status") or ""
        e = "OOM" if "RESOURCE_EXHAUSTED" in e else e[:30]
        return f"fail ({e})"
    p = cells.get((fam, n, prod), {})
    r = f"{c['ms'] / p['ms']:.2f}x" if p.get("ms") else "-"
    e = err(out, fam, n, arm, ref)
    return f"{c['ms']:.1f} ms ({r}); err {e:.3g}; peak {c.get('peak_mib', 0) / 1024:.1f} GiB"


for profile in ("af3", "opendde"):
    out, cells = load(profile)
    for fam in ("att", "mul", "trans"):
        dirs = ("_out", "_in") if fam == "mul" else ("",)
        print(f"\n#### {profile} {fam}\n")
        cols = SHOW[fam]
        print("| N | dir | production | " + " | ".join(cols) + " |")
        print("|---|---|---|" + "---|" * len(cols))
        for n in sorted({k[1] for k in cells if k[0] == fam}):
            for d in dirs:
                prod = PROD[profile][fam] + d
                ref = ("ref" + d) if fam == "mul" else "ref"
                row = [
                    str(n),
                    d.strip("_") or "-",
                    f"`{prod}` " + cell_txt(out, cells, fam, n, prod, prod, ref),
                ]
                row += [cell_txt(out, cells, fam, n, a + d, prod, ref) for a in cols]
                print("| " + " | ".join(row) + " |")
