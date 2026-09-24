#!/bin/bash
#SBATCH --job-name=x48-attrib
#SBATCH --gres=gpu:1
#SBATCH --mem=62G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=/home/jaemin/non-project/optimizing/foldjax-bench/x48-pallas-attrib-20260924/out/slurm-%j.out
#SBATCH --error=/home/jaemin/non-project/optimizing/foldjax-bench/x48-pallas-attrib-20260924/out/slurm-%j.err
# Which buffers make Boltz-2's MSA layer heavier under the Pallas transition?
# Compile-only on the GPU, 1,003 tokens x 8,808 alignment rows (L1000_3og2),
# msa_layer and msa_noseq_layer, one family per process. XLA dumps the buffer
# assignment, the memory-usage report and the live range at the peak step.
# Each arm is a whole-tree code snapshot, never an environment switch the
# compile cache could eat, and each run is tripwire-checked (armcheck.py):
#
#   released     code-88a41ea, arm released (cuEq, tokamax GLU)
#   unfused      code-88a41ea, arm pallas   (4e1ab01's kernel: output in the
#                                            caller's dtype, add outside)
#   fused_alias  code-8d42c6c, arm pallas   (residual fused, aliased to x)
#   fused_fresh  code-d8d28fe, arm pallas   (residual fused, fresh output)
#   fallback     code-e7aaa42, arm pallas   (MSA transition kept on the released
#                                            tokamax row-chunked GLU)
#
# attrib.py then names every buffer live at each program's peak (size, shape,
# dtype, producing instruction and its operands) and diffs each arm against
# `released`.
set -uo pipefail
umask 0002
X=/home/jaemin/non-project/optimizing/foldjax-bench/x48-pallas-attrib-20260924
PY=/home/jaemin/non-project/optimizing/foldjax/.venv/bin/python
export FOLDJAX_HOME=/home/jaemin/non-project/optimizing/foldjax/.foldjax
export FOLDJAX_BENCH_DATA=/home/jaemin/non-project/optimizing/foldjax-bench
export XLA_PYTHON_CLIENT_PREALLOCATE=false WALL_SPLIT_COMPILE_ONLY=1
unset JAX_PLATFORMS JAX_COMPILATION_CACHE_DIR
RUNS=$X/out/runs-$SLURM_JOB_ID
mkdir -p "$RUNS"
echo "job $SLURM_JOB_ID $(hostname) $(id -un) $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
status=0
for spec in released:88a41ea:released unfused:88a41ea:pallas fused_alias:8d42c6c:pallas \
            fused_fresh:d8d28fe:pallas fallback:e7aaa42:pallas; do
  IFS=: read -r arm sha kernels <<< "$spec"
  CODE=$X/code-$sha
  for family in msa_layer msa_noseq_layer; do
    run=$RUNS/$arm-$family
    mkdir -p "$run"
    echo "== $arm ($sha, $kernels) $family $(date -Is)"
    (
      export WALL_SPLIT_SRC=$CODE/src PYTHONPATH=$CODE/src
      export XDG_CACHE_HOME=/tmp/x48-$SLURM_JOB_ID-$arm TRITON_CACHE_DIR=/tmp/x48-$SLURM_JOB_ID-$arm/triton
      export XLA_FLAGS="--xla_dump_to=$run/dump --xla_dump_hlo_as_text"
      cd "$CODE/bench/kernel_shootout" &&
      "$PY" -c "import foldjax, os; p = os.path.realpath(foldjax.__file__); assert p.startswith(os.path.realpath('$CODE')), p" &&
      "$PY" wall_split_pallas.py --tokens 1003 --kernels "$kernels" --repeats 1 \
        --families "$family" --out "$run" > "$run/wall_split.log" 2>&1
    ) || { echo "   run failed: $(tail -3 "$run/wall_split.log")"; status=1; continue; }
    "$PY" "$X/armcheck.py" "$run" "$arm" "$family" || status=1
  done
done
for family in msa_layer msa_noseq_layer; do
  "$PY" "$X/attrib.py" "$RUNS" --diff "released-$family" --min-mib 32 \
    > "$X/out/attrib-$family-$SLURM_JOB_ID.txt" 2>&1
  echo "== attribution ($family): $X/out/attrib-$family-$SLURM_JOB_ID.txt"
  sed -n '/live at peak in/,$p' "$X/out/attrib-$family-$SLURM_JOB_ID.txt" | head -60
done
echo "done $(date -Is) status=$status"
exit $status
