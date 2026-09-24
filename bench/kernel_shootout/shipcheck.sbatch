#!/bin/bash
#SBATCH --job-name=kship
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/jaemin/non-project/optimizing/foldjax-bench/kernel-shootout-20260924/out/slurm-%j.out
#SBATCH --error=/home/jaemin/non-project/optimizing/foldjax-bench/kernel-shootout-20260924/out/slurm-%j.err
# The shipped module (foldjax.models._pallas_pair) against the production arms
# and the prototypes it was pinned from, per call, at 1,003 / 2,096 / 3,012.
# Answers one question before the model-level rows are read: does the module
# the ports call time like the shootout's winning prototype?
set -uo pipefail
umask 0002
B=/home/jaemin/non-project/optimizing/foldjax-bench/kernel-shootout-20260924
CODE=${CODE:?set CODE to the x43 snapshot (whole tree)}
PY=/home/jaemin/non-project/optimizing/foldjax/.venv/bin/python
export PYTHONPATH=$CODE/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false XDG_CACHE_HOME=/tmp/kship-$SLURM_JOB_ID
export TRITON_CACHE_DIR=/tmp/kship-$SLURM_JOB_ID/triton
unset XLA_FLAGS JAX_PLATFORMS
OUT=$B/out/ship-$SLURM_JOB_ID
mkdir -p "$OUT"
echo "job $SLURM_JOB_ID $(hostname) $(id -un) $(date -Is) code=$CODE"
nvidia-smi --query-gpu=index,name,temperature.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader
cd "$CODE/bench/kernel_shootout" || exit 2
"$PY" -c "import foldjax, os; p = os.path.realpath(foldjax.__file__); assert p.startswith(os.path.realpath('$CODE')), p; print('foldjax from', p)" || exit 3
SHOOTOUT_PROFILE=af3 "$PY" shootout.py drive --out "$OUT" --family mul --sizes 1003,2096,3012 \
  --bias-dtype bfloat16 \
  --arms cueq_boltz2_out,cueq_boltz2_in,fj_k1k2_32w4_out,fj_k1k2_32w4_in,pallas_pair_out,pallas_pair_in
SHOOTOUT_PROFILE=af3 "$PY" shootout.py drive --out "$OUT" --family trans --sizes 1003,2096,3012 \
  --bias-dtype bfloat16 --arms boltz2_tokamax_rc,boltz2_xla_full,fj_fused_64w4_fc64,pallas_pair
SHOOTOUT_PROFILE=af3 "$PY" shootout.py report --out "$OUT" > "$OUT/report.txt" 2>&1
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader
echo "done $(date -Is)"
