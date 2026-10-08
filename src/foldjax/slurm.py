"""Batch-scheduler helpers: split a batch into shards, and size one job.

``--shard i/N`` runs every N-th unit of a batch starting at ``i`` (0-based),
where a unit is one plain input or one job of a multi-job file. Round-robin
keeps the shards balanced when a library is sorted by size, and every unit
lands in exactly the output directory the unsharded batch would give it:
plain inputs keep their path, and a multi-job file is rewritten as a smaller
multi-job file holding only this shard's jobs, with relative paths made
absolute, so each job's generated document -- and so its resume identity --
is byte-identical to the unsharded run's. ``--shard auto`` reads
``SLURM_ARRAY_TASK_ID`` (relative to ``SLURM_ARRAY_TASK_MIN``) and
``SLURM_ARRAY_TASK_COUNT``; ``auto/N`` takes only the index from Slurm.

`plan_resources` turns a model's fitted device-peak law (`foldjax.memory_policy`)
into a suggested ``--gres`` and a minimum card size. The laws describe device
memory only: no host-memory law is calibrated, so ``--mem`` is left to the
caller rather than guessed. A job outside a law's fitted composition (a
nucleic acid or ligand under OpenFold3's protein-only law), or configured to
need more than the law was fitted at (OpenFold3's float32 trunk, or its
float32 confidence head behind the bfloat16 trunk), is ``unknown`` with no
card size, as the run's own admission would call it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from foldjax import memory_policy

_GIB = 2**30


def parse_shard(text: str, environ: Mapping[str, str] | None = None) -> tuple[int, int]:
    """``"i/N"``, ``"auto/N"`` or ``"auto"`` -> (index, count), 0-based index."""
    environ = os.environ if environ is None else environ
    text = str(text).strip()
    index_text, _, count_text = text.partition("/")
    if index_text == "auto":
        if "SLURM_ARRAY_TASK_ID" not in environ:
            raise ValueError(
                "--shard auto reads SLURM_ARRAY_TASK_ID, which is not set; "
                "submit with sbatch --array or give --shard i/N"
            )
        try:
            task = int(environ["SLURM_ARRAY_TASK_ID"])
            first = int(environ.get("SLURM_ARRAY_TASK_MIN", "0"))
        except ValueError as error:
            raise ValueError(f"unreadable Slurm array variables: {error}") from error
        index = task - first
        if not count_text:
            if "SLURM_ARRAY_TASK_COUNT" not in environ:
                raise ValueError(
                    "--shard auto needs SLURM_ARRAY_TASK_COUNT; give --shard "
                    "auto/N to name the shard count"
                )
            count_text = environ["SLURM_ARRAY_TASK_COUNT"]
    else:
        if not count_text:
            raise ValueError("--shard takes i/N (0-based i), auto/N or auto")
        try:
            index = int(index_text)
        except ValueError as error:
            raise ValueError(f"--shard index must be an integer: {text!r}") from error
    try:
        count = int(count_text)
    except ValueError as error:
        raise ValueError(f"--shard count must be an integer: {text!r}") from error
    if count < 1 or not 0 <= index < count:
        raise ValueError(
            f"--shard {text!r} resolves to shard {index} of {count}; the index "
            "must be in [0, count)"
        )
    return index, count


def _shard_file(
    source: Path, jobs: Sequence[Mapping[str, Any]], *, jobs_root: Path | None = None
) -> Path:
    from foldjax import paths
    from foldjax.input import _write_text_atomic

    text = json.dumps({"jobs": list(jobs)}, indent=2)
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    root = Path(jobs_root) if jobs_root is not None else paths.runtime_dir("jobs")
    target = root / "shards" / digest / f"{source.stem}.json"
    if not (target.is_file() and target.read_text(encoding="utf-8") == text):
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(target, text)
    return target


def shard_inputs(
    inputs: Sequence[Path],
    index: int,
    count: int,
    *,
    jobs_root: Path | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    """This shard's inputs, and a summary of the split.

    ``inputs`` are already expanded from directories. Multi-job files are
    split per job; everything else is one unit.
    """
    from foldjax.input import _absolute_job_paths, is_jobs_file, read_jobs_file

    units: list[tuple[Path, int | None]] = []
    jobs_of: dict[Path, list[tuple[str, dict[str, Any]]]] = {}
    for path in inputs:
        path = Path(path)
        if not str(path).startswith("structure:") and is_jobs_file(path):
            jobs_of[path] = read_jobs_file(path)
            units.extend((path, k) for k in range(len(jobs_of[path])))
        else:
            units.append((path, None))
    chosen = units[index::count]
    selected: list[Path] = []
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for path, position in chosen:
        if position is None:
            selected.append(path)
            continue
        _name, job = jobs_of[path][position]
        if path not in grouped:
            grouped[path] = []
            selected.append(path)
        grouped[path].append(_absolute_job_paths(job, path.parent.absolute()))
    resolved = [
        _shard_file(path, grouped[path], jobs_root=jobs_root)
        if path in grouped
        else path
        for path in selected
    ]
    summary = {
        "shard": index,
        "shards": count,
        "units_total": len(units),
        "units_in_shard": len(chosen),
        "rule": "round-robin over plain inputs and the jobs of multi-job files",
    }
    return resolved, summary


# -------------------------------------------------------------------- plan


#: The law each port applies to its default configuration, and what it is
#: keyed on. Mirrors the `memory_policy.admit` call site in each port.
_DEFAULT_LAW = {
    "boltz2": memory_policy.BOLTZ2_PEAK,
    "protenix": memory_policy.PROTENIX_PEAK,
    "openfold3": memory_policy.OPENFOLD3_CHUNKED_PEAK,
    "esmfold2": memory_policy.ESMFOLD2_PEAK,
}
_NO_LAW = {
    "alphafold3": "AlphaFold 3 carries no fitted peak law",
    "opendde": (
        "OpenDDE's law is keyed on structural tokens, which only its featurizer knows"
    ),
}


def estimate_tokens(document: Mapping[str, Any]) -> tuple[int, list[str]]:
    """Tokens of a common job: one per polymer residue, one per ligand heavy atom.

    Returns the count and what it could not count (a CCD ligand's atoms are
    not known without the CCD, and modified residues tokenize per atom in
    some models), so a caller can say how exact it is.
    """
    total = 0
    notes: list[str] = []
    for entity in document.get("entities") or []:
        if not isinstance(entity, Mapping):
            continue
        ids = entity.get("id")
        copies = len(ids) if isinstance(ids, list) else 1
        kind = str(entity.get("type", "")).lower()
        if kind in {"protein", "dna", "rna"}:
            total += copies * len(str(entity.get("sequence") or ""))
            if entity.get("modifications"):
                notes.append(f"modified residues of {ids} counted as one token each")
        elif kind == "ligand":
            smiles = entity.get("smiles")
            if smiles:
                from rdkit import Chem

                molecule = Chem.MolFromSmiles(str(smiles))
                if molecule is None:
                    notes.append(f"ligand {ids}: unreadable SMILES, not counted")
                else:
                    total += copies * molecule.GetNumHeavyAtoms()
            else:
                notes.append(f"ligand {ids} (CCD {entity.get('ccd')}) not counted")
    return total, notes


def _outside_composition(model: str, document: Mapping[str, Any]) -> str | None:
    """Why ``model``'s law does not cover this job's composition, if it does not.

    The plan-time half of the check the run's admission makes on its
    features: any nucleic-acid or ligand entity, counted or not (a CCD
    ligand's atoms are unknown here, and one token is enough).
    """
    if model not in memory_policy.PROTEIN_ONLY_LAWS:
        return None
    others = [
        entity
        for entity in document.get("entities") or []
        if isinstance(entity, Mapping)
        and str(entity.get("type", "")).lower() != "protein"
    ]
    if not others:
        return None
    counted, uncounted = estimate_tokens({"entities": others})
    return memory_policy.non_protein_reason(
        f"at least {max(counted, 1)}" if uncounted else counted
    )


def _exceeding_options(model: str, options: Mapping[str, Any]) -> list[str]:
    """The options that put a run above the profile ``model``'s law was fitted at.

    The plan-time half of OpenFold3's ``exceeds_profile`` admission
    (``released_config``): a float32 trunk, or a float32 confidence head
    behind the bfloat16 one, needs more than its bfloat16 law describes. Read
    from the backend's canonical options, so `fp32`, `f32` and the
    `compute_dtype` alias resolve as the run resolves them, and an omitted
    head follows the trunk.
    """
    if model != "openfold3" or not options:
        return []
    # `_DEFAULT_DTYPE` is the adapter's JAX-free copy of `released_config`'s
    # default, which a test pins; the model package would import JAX here.
    import warnings

    from foldjax import execution, registry
    from foldjax.backends.openfold3 import _DEFAULT_DTYPE

    try:
        with warnings.catch_warnings():
            # The request's own validation has already warned about an alias.
            warnings.simplefilter("ignore", execution.Alias)
            canonical = registry.backend_class(model).canonical_options(options)
    except ValueError:
        return []
    trunk = str(canonical.get("dtype", _DEFAULT_DTYPE))
    if trunk == "float32":
        return ["a float32 trunk"]
    if str(canonical.get("confidence_dtype", trunk)) == "float32":
        return ["a float32 confidence head"]
    return []


def plan_resources(
    model: str,
    document: Mapping[str, Any] | None,
    *,
    num_samples: int | None = None,
    cp_devices: int = 1,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Suggested Slurm resources for one run, from the model's peak law.

    ``options`` are the request's backend options, read only for the ones
    that put a run above the law's profile (:func:`_exceeding_options`).
    """
    record: dict[str, Any] = {
        "gres": f"gpu:{max(1, cp_devices)}",
        "mem": None,
        "mem_reason": (
            "no host-memory law is calibrated; the fitted laws describe "
            "device memory only"
        ),
        "min_device_memory_gib": None,
        "state": "unknown",
    }
    if model in _NO_LAW:
        record["reason"] = _NO_LAW[model]
        return record
    law = _DEFAULT_LAW.get(model)
    if law is None:
        record["reason"] = f"no peak law for {model}"
        return record
    if document is None:
        record["reason"] = "the input is not a common job, so its tokens are unknown"
        return record
    tokens, notes = estimate_tokens(document)
    record["tokens_estimate"] = tokens
    if notes:
        record["tokens_not_counted"] = notes
    record["law"] = {
        "model": law.model,
        "profile": law.profile,
        "calibration": law.calibration_id,
        "domain_tokens": list(law.domain_tokens),
    }
    if law.needs_msa_rows:
        record["reason"] = (
            f"the {model} law needs the processed MSA row count, which only "
            "the featurizer knows"
        )
        return record
    if not law.covers(tokens):
        low, high = law.domain_tokens
        record["reason"] = (
            f"{tokens} tokens is outside the law's fitted range {low}-{high}; "
            "it does not extrapolate"
        )
        return record
    upper = law.upper(tokens)
    device = upper / memory_policy.ADMISSION_FRACTION
    record.update(
        {
            "state": "estimated",
            "estimate_gib": round(law.estimate(tokens) / _GIB, 2),
            "upper_gib": round(upper / _GIB, 2),
            "min_device_memory_gib": math.ceil(device / _GIB),
            "reason": (
                "upper estimate over the admission fraction "
                f"{memory_policy.ADMISSION_FRACTION}: a card with at least this "
                "much memory is admitted"
            ),
        }
    )
    exceeding = _exceeding_options(model, options or {})
    composition = _outside_composition(model, document)
    if composition is not None:
        exceeding.append(composition)
    if exceeding:
        # What the run's own admission says (`memory_policy.exceeding_profile`,
        # `memory_policy.PROTEIN_ONLY_LAWS`): the estimate stays, as a lower
        # bound, and no card is sized from it.
        record.update(
            {
                "state": "unknown",
                "min_device_memory_gib": None,
                "reason": (
                    f"{'; '.join(exceeding)}: the estimate is a lower bound "
                    "here, so no card size is suggested"
                ),
            }
        )
    if cp_devices > 1:
        record["note"] = (
            "the law is single-device; with context parallelism the per-card "
            "peak is lower but not fitted"
        )
    if num_samples is not None:
        reasons = memory_policy.off_profile_reason(num_samples=num_samples)
        if reasons:
            record["off_profile"] = list(reasons)
    return record


def plan_slurm(
    summary: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    """The ``slurm`` block `foldjax plan --json` adds to one resolved run."""
    from foldjax.input import read_job_document

    document = None
    path = Path(str(summary.get("input")))
    if summary.get("input_format") == "foldjax" or path.suffix.lower() in {
        ".json",
        ".yaml",
        ".yml",
    }:
        try:
            loaded = read_job_document(path)
            document = (
                loaded if isinstance(loaded, Mapping) and "entities" in loaded else None
            )
        except (OSError, ValueError):
            document = None
    try:
        cp = int(options.get("cp_devices", 1))
    except (TypeError, ValueError):
        cp = 1
    sampling = summary.get("sampling") or {}
    return plan_resources(
        str(summary.get("model")),
        document,
        num_samples=sampling.get("num_samples")
        if isinstance(sampling, Mapping)
        else None,
        cp_devices=cp,
        options=options,
    )
