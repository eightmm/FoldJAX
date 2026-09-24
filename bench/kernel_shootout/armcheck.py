"""Prove a probe run compiled the arm its label claims (x48, x50).

    python armcheck.py RUN_DIR ARM FAMILY

The tripwire counts wall_split recorded (`fused:transition_pallas` and
`fused:tri_mul_pallas` per family) must match the arm. The HLO must carry
output-to-operand aliasing on a Triton call exactly when the arm is
`fused_alias`. Exits non-zero on any mismatch, so the job log shows it next to
the numbers.

Arms: `released` fires neither kernel; `trimul` fires only the multiplication;
`fallback` is `pallas` with Boltz-2's MSA transition kept on tokamax; every
other arm (`pallas`, `unfused`, `fused_*`) fires both, the transition at every
plain site of width <= 128.
"""

# ruff: noqa: E501

import json
import sys
from pathlib import Path

#: (triangle multiplications, Pallas transitions) one trace of the family makes
#: under a full `pallas` arm. The single transition (c_s 384) is out of scope.
PALLAS = {
    "msa_layer": (2, 2),  # MSA transition + pair transition
    "msa_noseq_layer": (2, 1),
    "trunk_pairformer_layer": (2, 1),
    "conf_pairformer_layer": (2, 1),
    "trunk_tri_mul_out": (1, 0),
    "trunk_tri_mul_in": (1, 0),
    "trunk_pair_transition": (0, 1),
}

run, arm, family = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if family not in PALLAS:
    sys.exit(f"armcheck: no expectation for family {family!r}; add it to PALLAS")
report = json.loads(next(run.glob("wall-split-*-serial.json")).read_text())
row = next(iter(report["arms"].values()))["families"][family]
fired = row.get("tripwire", {})
tri_mul, transitions = PALLAS[family]
if arm == "released":
    tri_mul, transitions = 0, 0
elif arm == "trimul":
    transitions = 0
elif arm == "fallback" and family == "msa_layer":
    transitions = 1  # the pair transition only
problems = []
if fired.get("fused:transition_pallas", 0) != transitions:
    problems.append(f"transition_pallas fired {fired.get('fused:transition_pallas', 0)}, want {transitions}")
if fired.get("fused:tri_mul_pallas", 0) != tri_mul:
    problems.append(f"tri_mul_pallas fired {fired.get('fused:tri_mul_pallas', 0)}, want {tri_mul}")
if fired.get("fused:tri_mul_cueq", 0) and arm != "released":
    problems.append(f"tri_mul_cueq fired {fired['fused:tri_mul_cueq']} under a Pallas arm")
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
