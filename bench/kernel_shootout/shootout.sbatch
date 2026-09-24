#!/bin/bash
#SBATCH --job-name=kshoot
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=/home/jaemin/non-project/optimizing/foldjax-bench/kernel-shootout-20260924/out/slurm-%j.out
#SBATCH --error=/home/jaemin/non-project/optimizing/foldjax-bench/kernel-shootout-20260924/out/slurm-%j.err
# Pair-stack kernel shootout (availability matrix + per-primitive tables), one GPU.
# Runs from a whole-tree snapshot so edits to the worktree during the job cannot
# leak in.  Every cell is its own process (see shootout.py).  Order: availability,
# then the c_z-128 ports (Boltz-2/Protenix/OpenFold3) transition -> multiplication
# -> attention with sizes ascending, then an f32-bias attention subset, then
# OpenDDE's c_z-384 widths.  Triangle bias is bf16 in the main runs: every port
# projects it with a bf16 Linear in its bf16 trunk (traced for Boltz-2: the
# bias reaches cuex.triangle_attention as bfloat16).
set -uo pipefail
umask 0002
B=/home/jaemin/non-project/optimizing/foldjax-bench/kernel-shootout-20260924
CODE=${CODE:-$B/code-5efc64c}
PY=/home/jaemin/non-project/optimizing/foldjax/.venv/bin/python
SIZES=${SIZES:-1003,1354,2096,2100,3012,4100,4888}
DDE_SIZES=${DDE_SIZES:-1003,1354,2096}
export PYTHONPATH=$CODE/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false XDG_CACHE_HOME=/tmp/kshoot-$SLURM_JOB_ID
export TRITON_CACHE_DIR=/tmp/kshoot-$SLURM_JOB_ID/triton
unset XLA_FLAGS JAX_PLATFORMS
OUT=$B/out/run-$SLURM_JOB_ID
mkdir -p "$OUT"
echo "job $SLURM_JOB_ID $(hostname) $(id -un) $(date -Is) code=$CODE sizes=$SIZES dde=$DDE_SIZES"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,temperature.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader
cd "$CODE/bench/kernel_shootout" || exit 2
"$PY" -c "import foldjax, os; p = os.path.realpath(foldjax.__file__); assert p.startswith(os.path.realpath('$CODE')), p; print('foldjax from', p)" || exit 3
"$PY" -c "import jax, jaxlib, cuequivariance_jax, triton; from importlib.metadata import version; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__, 'tokamax', version('tokamax'), 'cueq', cuequivariance_jax.__version__, 'triton', triton.__version__, jax.devices())"
clock() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader; }

echo "== avail $(date -Is)"; clock
SHOOTOUT_PROFILE=af3 "$PY" shootout.py avail --out "$OUT/avail"

for fam in trans mul att; do
  echo "== af3 $fam $(date -Is)"; clock
  SHOOTOUT_PROFILE=af3 "$PY" shootout.py drive --out "$OUT/af3" --family $fam --sizes "$SIZES" \
    --bias-dtype bfloat16
done

echo "== af3 att f32 bias (the kernel contract _cueq documents; align.py's numbers) $(date -Is)"; clock
SHOOTOUT_PROFILE=af3 "$PY" shootout.py drive --out "$OUT/af3-f32bias" --family att --sizes 2096,3012 \
  --bias-dtype float32 \
  --arms cueq_pad_highest,cueq_pad_high,tokamax_triton,jax_cudnn_seqlen,jax_cudnn_seqlen_pad8,fj_flash_r2_64x64w4,fj_flash_r4_64x64w4,fj_flash_r4_64x32w4

for fam in trans mul att; do
  echo "== opendde $fam $(date -Is)"; clock
  SHOOTOUT_PROFILE=opendde "$PY" shootout.py drive --out "$OUT/opendde" --family $fam --sizes "$DDE_SIZES" \
    --bias-dtype bfloat16
done

for d in af3 af3-f32bias opendde; do
  SHOOTOUT_PROFILE=${d%%-*} "$PY" shootout.py report --out "$OUT/$d" > "$OUT/$d/report.txt" 2>&1
done
SHOOTOUT_PROFILE=af3 "$PY" shootout.py report --out "$OUT/avail" > "$OUT/avail/report.txt" 2>&1 || true
clock
echo "done $(date -Is)"
