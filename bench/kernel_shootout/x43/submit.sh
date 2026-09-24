#!/bin/bash
# Submit the x43 GPU work as the Slurm account (see the lab-server notes): the
# shipped-module per-call check, then the model-level rows, cheapest first so an
# environment fault shows on a short row. Records job ids in submitted.txt.
#
#   bash submit.sh [ROW_GLOB]      # default: every rows/*.sbatch
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(cat "$HERE/SNAPSHOT")
CODE=$HERE/code-$SHA
SB="sudo -n -u sjm0775 sbatch --parsable"
mkdir -p "$HERE/logs"
if [ -z "${1:-}" ]; then
  id=$($SB --export=ALL,CODE="$CODE" \
    /home/jaemin/non-project/optimizing/foldjax-bench/kernel-shootout-20260924/shipcheck.sbatch)
  echo "shipcheck $id" | tee -a "$HERE/submitted.txt"
fi
for row in $(ls "$HERE"/rows/${1:-*}.sbatch | sort -t- -k1,1 -k4,4); do
  name=$(basename "$row" .sbatch)
  if [ -e "$HERE/results/$name" ]; then echo "skip $name (results exist)"; continue; fi
  id=$($SB -J "x43" -o "$HERE/logs/$name-%j.out" -e "$HERE/logs/$name-%j.err" "$row")
  echo "$name $id" | tee -a "$HERE/submitted.txt"
done
