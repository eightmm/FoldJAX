#!/usr/bin/env bash
# Pass allocation values to sbatch; these variables cannot expand #SBATCH lines.
PARTITIONS_GPU="${PARTITIONS_GPU:-}"
GRES="${GRES:-gpu:1}"
CPUS_PER_GPU="${CPUS_PER_GPU:-}"
PARTITION_CPU="${PARTITION_CPU:-}"
MODULE="${MODULE:-}"
FIXED_MSA_DEPTH="${FIXED_MSA_DEPTH:-2048}"
OPENDDE_PADDING_PROFILE="${OPENDDE_PADDING_PROFILE:-}"

set -euo pipefail
: "${SLURM_JOB_ID:?submit this script with sbatch; never warm on a login node}"
cd -- "${SLURM_SUBMIT_DIR:?Slurm did not provide the submission directory}"
if [[ -n $MODULE ]]; then module load "$MODULE"; fi
die() { printf 'warm.sbatch: %s\n' "$*" >&2; exit 2; }
model_name() {
    case "$1" in
        alphafold3|af3|alphafold-3) printf alphafold3 ;;
        boltz2|boltz|boltz-2|boltz-jax) printf boltz2 ;;
        esmfold2|esm-fold2|esmfold-2) printf esmfold2 ;;
        opendde|open-dde|opendde-jax) printf opendde ;;
        openfold3|of3|openfold-3|openfold3-jax) printf openfold3 ;;
        protenix|protenix-jax) printf protenix ;;
        *) die "unknown model: $1" ;;
    esac
}
models=() common=() verify=0 exact=0 explicit_fixed=0 explicit_profile=0 has_input=0 has_weights=0
while (($#)); do
    case "$1" in
        --verify-only) verify=1; shift ;;
        --exact) exact=1; shift ;;
        --fixed-msa)
            [[ $# -ge 2 ]] || die '--fixed-msa needs a value'
            FIXED_MSA_DEPTH=$2; explicit_fixed=1; shift 2 ;;
        --fixed-msa=*) FIXED_MSA_DEPTH=${1#*=}; explicit_fixed=1; shift ;;
        --opendde-padding-profile)
            [[ $# -ge 2 ]] || die '--opendde-padding-profile needs a value'
            OPENDDE_PADDING_PROFILE=$2; explicit_profile=1; shift 2 ;;
        --opendde-padding-profile=*) OPENDDE_PADDING_PROFILE=${1#*=}; explicit_profile=1; shift ;;
        --model)
            shift
            [[ $# -gt 0 && $1 != -* ]] || die '--model needs a value'
            while (($#)) && [[ $1 != -* ]]; do
                models+=("$(model_name "$1")"); shift
            done ;;
        --model=*) models+=("$(model_name "${1#*=}")"); shift ;;
        --) shift ;;
        *)
            case "$1" in
                --input|--input=*|--sequence|--sequence=*|--dna|--dna=*|--rna|--rna=*|--ligand|--ligand=*|--ligand-smiles|--ligand-smiles=*) has_input=1 ;;
                --weights|--weights=*) has_weights=1 ;;
            esac
            common+=("$1"); shift ;;
    esac
done
((${#models[@]})) || die 'choose at least one model with --model'
((!exact || !explicit_fixed)) || die '--exact and --fixed-msa cannot be combined'
((!exact || !explicit_profile)) || die '--exact and --opendde-padding-profile cannot be combined'
if ((exact)); then FIXED_MSA_DEPTH=''; OPENDDE_PADDING_PROFILE=''; fi
[[ -z $OPENDDE_PADDING_PROFILE || $OPENDDE_PADDING_PROFILE == compact ]] || die 'OPENDDE_PADDING_PROFILE must be compact or empty'
if ((explicit_profile)) && [[ " ${models[*]} " != *' opendde '* ]]; then
    die '--opendde-padding-profile requires an OpenDDE model'
fi
[[ -z $FIXED_MSA_DEPTH || $FIXED_MSA_DEPTH =~ ^[1-9][0-9]*$ ]] || die 'FIXED_MSA_DEPTH/--fixed-msa must be a positive integer'
((${#models[@]} == 1 || !has_weights)) || die 'one explicit weights path cannot be shared by several models'
if ((!verify && exact && !has_input)); then
    die '--exact warming requires --input or --sequence; token-bucket warming always pads'
fi
seen=' '
for model in "${models[@]}"; do
    [[ $seen != *" $model "* ]] || die "duplicate model: $model"
    seen+="$model "
done
for model in "${models[@]}"; do
    preset=()
    case "$model" in
        boltz2|protenix|openfold3|opendde)
            if [[ -n $FIXED_MSA_DEPTH ]]; then preset=(--fixed-msa "$FIXED_MSA_DEPTH"); fi ;;
        alphafold3|esmfold2)
            if ((!exact)); then preset=(--padding); fi ;;
    esac
    if [[ $model == opendde && -n $OPENDDE_PADDING_PROFILE ]]; then
        preset+=(--opendde-padding-profile "$OPENDDE_PADDING_PROFILE")
    fi
    if ((verify)); then
        # Submit this mode as the ordinary Unix user, to a fresh output root.
        foldjax predict --msa single --model "$model" --batch-output "${preset[@]}" "${common[@]}"
    else
        buckets=()
        if ((!has_input)); then buckets=(--token-buckets auto); fi
        foldjax cache warm "${buckets[@]}" --msa single --model "$model" --batch-output "${preset[@]}" "${common[@]}"
    fi
done
if ((!verify)); then
    printf '%s\n' 'Warming finished. Have an ordinary Unix user submit warm.sbatch --verify-only with the same presets as described in README.md; cross-user verification has not run in this job.' >&2
fi
