#!/bin/bash
#SBATCH --job-name=kpeak
#SBATCH --gres=gpu:1
#SBATCH --mem=62G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/home/jaemin/non-project/optimizing/foldjax-bench/kernel-shootout-20260924/out/slurm-%j.out
#SBATCH --error=/home/jaemin/non-project/optimizing/foldjax-bench/kernel-shootout-20260924/out/slurm-%j.err
# Which Boltz-2 module grows under the Pallas pair kernels at L1000_3og2?
# x43 measured peak 11,165 MiB (-pallas) vs 8,454 (x42). Compile-only: each
# family of the released trunk, MSA and confidence stacks at 1,003 tokens and
# the case's 8,808 alignment rows, under `released` and `pallas`, with its
# memory_analysis. peakprobe_compare.py prints the per-family deltas.
set -uo pipefail
umask 0002
B=/home/jaemin/non-project/optimizing/foldjax-bench/kernel-shootout-20260924
CODE=${CODE:?set CODE to the snapshot (whole tree)}
PY=/home/jaemin/non-project/optimizing/foldjax/.venv/bin/python
export WALL_SPLIT_SRC=$CODE/src PYTHONPATH=$CODE/src
export FOLDJAX_HOME=/home/jaemin/non-project/optimizing/foldjax/.foldjax
export FOLDJAX_BENCH_DATA=/home/jaemin/non-project/optimizing/foldjax-bench
export XLA_PYTHON_CLIENT_PREALLOCATE=false XDG_CACHE_HOME=/tmp/kpeak-$SLURM_JOB_ID
export TRITON_CACHE_DIR=/tmp/kpeak-$SLURM_JOB_ID/triton
export WALL_SPLIT_COMPILE_ONLY=1
unset XLA_FLAGS JAX_PLATFORMS
OUT=$B/out/peak-$SLURM_JOB_ID
mkdir -p "$OUT"
echo "job $SLURM_JOB_ID $(hostname) $(id -un) $(date -Is) code=$CODE"
cd "$CODE/bench/kernel_shootout" || exit 2
FAMS=trunk_tri_mul_out,trunk_tri_mul_in,trunk_pair_transition,trunk_single_transition,trunk_pairformer_layer,msa_transition,msa_noseq_layer,msa_layer,conf_pairformer_layer
"$PY" wall_split_pallas.py --tokens 1003 --kernels released,pallas --repeats 1 \
  --families "$FAMS" --out "$OUT"
"$PY" peakprobe_compare.py "$OUT/wall-split-1003-serial.json" | tee "$OUT/compare.txt"
echo "done $(date -Is)"
