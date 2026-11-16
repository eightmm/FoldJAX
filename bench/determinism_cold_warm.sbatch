#!/bin/bash
#SBATCH --job-name=fj-det
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
# Same-seed repeatability across processes: each model predicts one small
# target twice in fresh processes, first against an empty compile cache (cold)
# then against the cache the first run filled (warm), and every written
# coordinate must be byte-identical. Exit 0 only if all models match; each
# run's wall time is printed alongside, since the gather loop's data-dependent
# trip count is the change's main runtime unknown on a GPU.
#
#   CODE=/path/to/foldjax sbatch bench/determinism_cold_warm.sbatch [model ...]
#
# Models default to boltz2; protenix and opendde take the same input. A residual
# difference after the atom-to-token change points at something else that
# differs between a cold compile and a cache load (e.g. GEMM autotuning), not
# at scatter-add.
set -uo pipefail
: "${SLURM_JOB_ID:?submit this script with sbatch}"
CODE=${CODE:-${SLURM_SUBMIT_DIR:?}}
PY=${PY:-python}
INPUT=${INPUT:-$CODE/examples/monomer.yaml}
SEED=${SEED:-1}
OUT=${OUT:-${TMPDIR:-/tmp}/foldjax-det-$SLURM_JOB_ID}
MODELS=("$@")
[[ ${#MODELS[@]} -gt 0 ]] || MODELS=(boltz2)

export PYTHONPATH=$CODE/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME=$OUT/xdg TRITON_CACHE_DIR=$OUT/triton
unset XLA_FLAGS JAX_PLATFORMS
mkdir -p "$OUT"
echo "job $SLURM_JOB_ID $(hostname) $(date -Is) code=$CODE out=$OUT"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
"$PY" -c "import foldjax, os; p = os.path.realpath(foldjax.__file__); assert p.startswith(os.path.realpath('$CODE')), p; print('foldjax from', p)" || exit 3

status=0
for model in "${MODELS[@]}"; do
  cache=$OUT/$model/compile
  rm -rf "${OUT:?}/$model"
  mkdir -p "$cache"
  for run in cold warm; do
    before=$(find "$cache" -type f | wc -l)
    started=$SECONDS
    "$PY" -m foldjax predict --model "$model" --input "$INPUT" --msa single \
      --seed "$SEED" --cache-dir "$cache" --output-dir "$OUT/$model/$run" \
      > "$OUT/$model/$run.log" 2>&1 || { echo "$model $run: predict failed"; tail -20 "$OUT/$model/$run.log"; status=1; continue 2; }
    echo "$model $run: $((SECONDS - started)) s, cache files $before -> $(find "$cache" -type f | wc -l)"
  done
  "$PY" - "$OUT/$model/cold" "$OUT/$model/warm" <<'EOF' || status=1
import sys
from pathlib import Path

import numpy as np

cold, warm = (Path(p) for p in sys.argv[1:3])


def coordinates(root):
    found = {}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        if path.suffix == ".cif":
            lines = path.read_text().splitlines()
            found[key] = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
        elif path.suffix == ".npz":
            with np.load(path) as data:
                found[key] = {
                    name: data[name].tobytes()
                    for name in data.files
                    if data[name].dtype.kind == "f"
                }
    return found


a, b = coordinates(cold), coordinates(warm)
if not a:
    sys.exit(f"no .cif/.npz written under {cold}")
if a.keys() != b.keys():
    sys.exit(f"different files: {sorted(a.keys() ^ b.keys())}")
bad = [key for key in a if a[key] != b[key]]
for key in bad:
    print(f"DIFFERS {key}")
print(f"{cold.parent.name}: {len(a) - len(bad)}/{len(a)} files identical")
sys.exit(1 if bad else 0)
EOF
done
echo "done $(date -Is) status=$status"
exit $status
