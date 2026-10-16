#!/usr/bin/env bash
# Pass allocation values to sbatch; these variables cannot expand #SBATCH lines.
PARTITIONS_GPU="${PARTITIONS_GPU:-}"
GRES="${GRES:-gpu:1}"
CPUS_PER_GPU="${CPUS_PER_GPU:-}"
PARTITION_CPU="${PARTITION_CPU:-}"
MODULE="${MODULE:-}"

set -euo pipefail
: "${SLURM_JOB_ID:?submit this script with sbatch; never warm on a login node}"
cd -- "${SLURM_SUBMIT_DIR:?Slurm did not provide the submission directory}"
if [[ -n $MODULE ]]; then module load "$MODULE"; fi
if [[ ${1:-} == --verify-only ]]; then
    shift
    if [[ ${1:-} == -- ]]; then shift; fi
    # Submit this mode as the ordinary Unix user, to a fresh output directory.
    exec foldjax predict "$@"
fi
if [[ ${1:-} == -- ]]; then shift; fi
foldjax cache warm --token-buckets auto --msa single "$@"
printf '%s\n' 'Warming finished. Have an ordinary Unix user submit warm.sbatch --verify-only as described in README.md; cross-user verification has not run in this job.'
