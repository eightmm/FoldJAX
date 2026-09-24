"""Compact tables from a shootout run: per (family, N) every arm's ms, ratio to
production, temp/peak MiB, max-abs vs f32 ref (valid rows) and vs production.

    python summarize.py RUN_DIR/af3 af3 [production-attention-arm]
"""
import json
import sys
from pathlib import Path

import numpy as np

PAD = 3
out = Path(sys.argv[1])
profile = sys.argv[2]
PROD = ({"att": "cueq_pad_highest", "mul": "cueq_boltz2", "trans": "boltz2_tokamax_rc"}
        if profile == "af3" else
        {"att": "cueq_pad_high", "mul": "cueq_fused", "trans": "boltz2_xla_full"})
if len(sys.argv) > 3:
    PROD["att"] = sys.argv[3]
cells = {}
for line in open(out / "cells.jsonl"):
    c = json.loads(line)
    cells[(c["family"], c["n"], c["arm"])] = c


def rows_err(fam, n, a, b):
    pa, pb = out / f"{fam}-{n}-{a}.npz", out / f"{fam}-{n}-{b}.npz"
    if not (pa.exists() and pb.exists()):
        return float("nan")
    A, B = np.load(pa), np.load(pb)
    ra, rb = list(A["rows"]), list(B["rows"])
    common = [r for r in ra if r in rb and r < n - PAD]
    if not common:
        return float("nan")
    x = A["out"][[ra.index(r) for r in common]]
    y = B["out"][[rb.index(r) for r in common]]
    return float(np.abs(x - y).max())


for fam in ("att", "mul", "trans"):
    for n in sorted({k[1] for k in cells if k[0] == fam}):
        dirs = ("_out", "_in") if fam == "mul" else ("",)
        for d in dirs:
            prod = PROD[fam] + d
            ref = ("ref" + d) if fam == "mul" else "ref"
            p = cells.get((fam, n, prod), {})
            scale = float("nan")
            rp = out / f"{fam}-{n}-{ref}.npz"
            if rp.exists():
                scale = float(np.abs(np.load(rp)["out"]).max())
            print(f"\n## {profile} {fam}{d} N={n} (mod8={n % 8}) production={prod} "
                  f"{p.get('ms', 'FAIL')} ms  ref|max|={scale:.3g}  prod err vs ref={rows_err(fam, n, prod, ref):.3g}")
            arms = [(k[2], c) for k, c in cells.items() if k[0] == fam and k[1] == n
                    and not k[2].startswith("ref") and (not d or k[2].endswith(d))]
            arms.sort(key=lambda t: t[1].get("ms", 1e9) * (n / t[1]["rows"] if t[1].get("rows") else 1))
            for arm, c in arms:
                if c.get("ok") and "ms" in c:
                    ms = c["ms"] * (n / c["rows"] if c.get("rows") else 1)
                    ratio = ms / p["ms"] if p.get("ms") else float("nan")
                    print(f"  {arm:28s} {ms:9.2f} ms  {ratio:6.3f}x  temp {c.get('temp_mib', '-'):>8}  peak {c.get('peak_mib', '-'):>8}"
                          f"  err_ref {rows_err(fam, n, arm, ref):.3g}  err_prod {rows_err(fam, n, arm, prod):.3g}"
                          f"  padfinite {c.get('finite_pad_rows', '-')}" + ("  (16 rows, scaled)" if c.get("rows") else ""))
                else:
                    err = (c.get("error") or c.get("status") or "")[:110].replace("\n", " ")
                    print(f"  {arm:28s}      FAIL  {err}")
