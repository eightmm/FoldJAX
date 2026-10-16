#!/usr/bin/env bash
# Allocation values are passed by foldjax-submit; fill site defaults if needed.
PARTITIONS_GPU="${PARTITIONS_GPU:-}"
GRES="${GRES:-gpu:1}"
CPUS_PER_GPU="${CPUS_PER_GPU:-}"
PARTITION_CPU="${PARTITION_CPU:-}"
MODULE="${MODULE:-}"

set -euo pipefail
: "${SLURM_JOB_ID:?submit this script with sbatch}"
cd -- "${SLURM_SUBMIT_DIR:?Slurm did not provide the submission directory}"
if [[ -n $MODULE ]]; then module load "$MODULE"; fi
exec foldjax msa prefetch "$@"
