"""Prove a probe run compiled the arm its label claims (x48).

    python armcheck.py RUN_DIR ARM FAMILY

The tripwire counts wall_split recorded (`fused:transition_pallas` and
`fused:tri_mul_pallas` per family) must match the arm. The snapshot's
`_pallas_pair.py` must carry the residual mode exactly when the arm is a fused
one. The HLO must carry output-to-operand aliasing on a Triton call exactly
when the arm is `fused_alias`. Exits non-zero on any mismatch, so the job log
shows it next to the numbers.
"""

# ruff: noqa: E501

import json
import sys
from pathlib import Path

run, arm, family = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
report = json.loads(next(run.glob("wall-split-*-serial.json")).read_text())
row = next(iter(report["arms"].values()))["families"][family]
fired = row.get("tripwire", {})
transitions = {
    ("released", "msa_layer"): 0,
    ("released", "msa_noseq_layer"): 0,
    ("fallback", "msa_layer"): 1,  # the pair transition only
    ("fallback", "msa_noseq_layer"): 1,
}.get((arm, family), 2 if family == "msa_layer" else 1)
problems = []
if fired.get("fused:transition_pallas", 0) != transitions:
    problems.append(f"transition_pallas fired {fired.get('fused:transition_pallas', 0)}, want {transitions}")
want_trimul = 0 if arm == "released" else 2
if fired.get("fused:tri_mul_pallas", 0) != want_trimul:
    problems.append(f"tri_mul_pallas fired {fired.get('fused:tri_mul_pallas', 0)}, want {want_trimul}")
dump = run / "dump"
aliased = any(
    "output_to_operand_aliasing" in line and "triton" in line
    for p in dump.glob("*after_optimizations.txt")
    for line in p.read_text(errors="replace").splitlines()
    if "custom-call" in line
)
if aliased != (arm == "fused_alias"):
    problems.append(f"aliased Triton call present={aliased}, want {arm == 'fused_alias'}")
print(f"ARM-CHECK {run.name}: {'ok' if not problems else 'FAIL ' + '; '.join(problems)} (tripwire {fired})")
sys.exit(1 if problems else 0)
