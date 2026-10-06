"""`foldjax compare`: every structure of one input against every other.

A thin layer over `foldjax.align_structures`. For each input in a finished
output directory it aligns every structure -- all models, seeds and samples --
onto every other, and records the RMSD after the fit, the coverage, the chain
map and the residue correspondence the fit used. Proteins are fitted on CA and
nucleic acids on C4' (``selection="representative"``); ligands move with the
fit but do not drive it. Coverage is matched reference atoms over all
reference atoms, so it is directional and the matrix is written in full rather
than as a mirrored triangle. There is no TM-score.

Each structure row carries what its model never read (``ignored_msas``,
``ignored_templates``, ``ignored_constraints``) and the pocket
``constraints`` it ran with, with each ``max_distance`` and whether it came
from the job or the model's own default: a "matched-input" panel is only
matched in what the models were given, and those columns say where it was
not.
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from foldjax.alignment import (
    StructureAlignmentError,
    _parse,
    align_structures,
    residue_correspondence,
)
from foldjax.results import load_results, results_table

COMPARE_SCHEMA = "foldjax-compare-v1"

_ATOMS = (
    "CA for protein residues, C4' for nucleic-acid residues; ligands move with "
    "the fit but are not fitted"
)
_METRICS = (
    "rmsd_angstrom is measured after a rigid, reflection-free Kabsch fit of the "
    "mobile structure onto the reference; coverage is matched atoms over "
    "reference atoms and is directional. No TM-score."
)


def _signature(path: Path) -> tuple[Any, ...]:
    parsed = _parse(path)
    return tuple(
        (name, chain.kind, chain.sequence or chain.fingerprint)
        for name, chain in sorted(parsed.chains.items())
    )


def _structures(rows: Sequence[Mapping[str, Any]], root: Path, samples: str):
    """Group verified structures by input, with stable readable keys."""
    usable = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("structure_path")
        and (samples == "all" or row.get("best_within_model"))
    ]
    if samples == "best":
        # A model that ranks nothing (OpenFold3 on protein input) has no best;
        # its first sample stands in, and the row says so.
        covered = {(row["input"], row["run_dir"]) for row in usable}
        for row in rows:
            key = (row.get("input"), row.get("run_dir"))
            if (
                row.get("status") == "ok"
                and key not in covered
                and row.get("structure_path")
            ):
                usable.append({**row, "best_within_model": None})
                covered.add(key)
    runs_per_model: dict[tuple[str, str], set[str]] = {}
    jobs_per_run: dict[str, set[str]] = {}
    for row in usable:
        runs_per_model.setdefault((row["input"], row["model"]), set()).add(
            row["run_dir"]
        )
        jobs_per_run.setdefault(row["run_dir"], set()).add(str(row.get("job")))
    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for row in usable:
        several_jobs = len(jobs_per_run[row["run_dir"]]) > 1
        job = row.get("job") if several_jobs else None
        model = row["model"]
        if len(runs_per_model[(row["input"], model)]) > 1:
            try:
                relative = os.path.relpath(row["run_dir"], root)
            except ValueError:
                relative = row["run_dir"]
            model = f"{model}[{relative}]"
        job_part = f"{job}/" if job else ""
        key = f"{model}/{job_part}seed-{row['seed']}/sample-{int(row['sample']):02d}"
        groups.setdefault((row["input"], job), []).append({**row, "key": key})
    return groups


def compare_rows(
    root: str | os.PathLike[str], *, samples: str = "all"
) -> dict[str, Any]:
    """The comparison as a JSON-ready document (see this module's docstring)."""
    if samples not in {"all", "best"}:
        raise ValueError("samples must be 'all' or 'best'")
    root = Path(root)
    report = load_results(root)
    rows = results_table(report)
    groups = _structures(rows, report.root, samples)
    inputs = []
    for (input_path, job), members in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1] or "")
    ):
        paths = {member["key"]: Path(member["structure_path"]) for member in members}
        signatures = {}
        for key, path in paths.items():
            try:
                signatures[key] = _signature(path)
            except StructureAlignmentError:
                signatures[key] = None
        correspondences: dict[str, Any] = {}
        correspondence_ids: dict[Any, str] = {}
        pairs = []
        for reference_key, reference_path in paths.items():
            sources = {key: path for key, path in paths.items() if key != reference_key}
            if not sources:
                continue
            outcomes: dict[str, Any] = {}
            try:
                aligned = align_structures(sources, reference=reference_path)
                outcomes = {item.key: item for item in aligned.alignments}
            except StructureAlignmentError:
                # One unreadable or unmatched structure must not hide the rest.
                for key, path in sources.items():
                    try:
                        outcomes[key] = align_structures(
                            {key: path}, reference=reference_path
                        ).alignments[0]
                    except StructureAlignmentError as error:
                        outcomes[key] = error
            for mobile_key in sources:
                outcome = outcomes[mobile_key]
                record: dict[str, Any] = {
                    "reference": reference_key,
                    "mobile": mobile_key,
                }
                if isinstance(outcome, Exception):
                    record["error"] = str(outcome)
                    pairs.append(record)
                    continue
                chain_map = dict(outcome.chain_map)
                signature = (
                    signatures.get(mobile_key),
                    signatures.get(reference_key),
                    tuple(sorted(chain_map.items())),
                )
                if signature not in correspondence_ids:
                    identifier = f"c{len(correspondence_ids)}"
                    correspondence_ids[signature] = identifier
                    correspondences[identifier] = residue_correspondence(
                        paths[mobile_key], reference_path, chain_map
                    )
                record.update(
                    {
                        "rmsd_angstrom": outcome.rmsd,
                        "coverage": outcome.coverage,
                        "matched_atoms": outcome.matched_atoms,
                        "reference_atoms": outcome.reference_atoms,
                        "matched_residues": outcome.matched_residues,
                        "chain_map": chain_map,
                        "correspondence": correspondence_ids[signature],
                    }
                )
                pairs.append(record)
        inputs.append(
            {
                "input": input_path,
                "job": job,
                "structures": [
                    {
                        "key": member["key"],
                        "model": member["model"],
                        "configuration": member.get("configuration"),
                        "seed": member["seed"],
                        "sample": member["sample"],
                        "best_within_model": bool(member.get("best_within_model")),
                        "structure_path": member["structure_path"],
                        "structure_sha256": member.get("structure_sha256"),
                        "seed_source": member.get("seed_source"),
                        "ignored_msas": member.get("ignored_msas"),
                        "ignored_templates": member.get("ignored_templates"),
                        "ignored_constraints": member.get("ignored_constraints"),
                        "constraints": member.get("constraints"),
                    }
                    for member in members
                ],
                "pairs": pairs,
                "correspondences": correspondences,
            }
        )
    return {
        "schema": COMPARE_SCHEMA,
        "root": str(report.root),
        "samples": samples,
        "selection": "representative",
        "atoms": _ATOMS,
        "metrics": _METRICS,
        "inputs": inputs,
    }


def _csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                name: (
                    ""
                    if row.get(name) is None
                    else json.dumps(row[name], sort_keys=True, separators=(",", ":"))
                    if isinstance(row.get(name), (list, dict))
                    else row[name]
                )
                for name in columns
            }
        )
    return buffer.getvalue()


def compare_directory(
    root: str | os.PathLike[str],
    *,
    out: str | os.PathLike[str] | None = None,
    samples: str = "all",
) -> dict[str, Path]:
    """Write compare.json, compare.csv (pairs) and compare_structures.csv."""
    document = compare_rows(root, samples=samples)
    target = Path(out) if out is not None else Path(document["root"]) / "compare"
    target.mkdir(parents=True, exist_ok=True)
    pair_rows = []
    structure_rows = []
    for entry in document["inputs"]:
        models = {item["key"]: item["model"] for item in entry["structures"]}
        for pair in entry["pairs"]:
            pair_rows.append(
                {
                    "input": entry["input"],
                    "job": entry["job"],
                    "reference": pair["reference"],
                    "mobile": pair["mobile"],
                    "reference_model": models.get(pair["reference"]),
                    "mobile_model": models.get(pair["mobile"]),
                    **{
                        key: pair.get(key)
                        for key in (
                            "rmsd_angstrom",
                            "coverage",
                            "matched_atoms",
                            "reference_atoms",
                            "matched_residues",
                            "chain_map",
                            "correspondence",
                            "error",
                        )
                    },
                }
            )
        for item in entry["structures"]:
            structure_rows.append(
                {"input": entry["input"], "job": entry["job"], **item}
            )
    written = {
        "json": target / "compare.json",
        "pairs_csv": target / "compare.csv",
        "structures_csv": target / "compare_structures.csv",
    }
    written["json"].write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written["pairs_csv"].write_text(
        _csv(
            pair_rows,
            (
                "input",
                "job",
                "reference",
                "mobile",
                "reference_model",
                "mobile_model",
                "rmsd_angstrom",
                "coverage",
                "matched_atoms",
                "reference_atoms",
                "matched_residues",
                "chain_map",
                "correspondence",
                "error",
            ),
        ),
        encoding="utf-8",
    )
    written["structures_csv"].write_text(
        _csv(
            structure_rows,
            (
                "input",
                "job",
                "key",
                "model",
                "configuration",
                "seed",
                "sample",
                "best_within_model",
                "seed_source",
                "ignored_msas",
                "ignored_templates",
                "ignored_constraints",
                "constraints",
                "structure_path",
                "structure_sha256",
            ),
        ),
        encoding="utf-8",
    )
    return written
