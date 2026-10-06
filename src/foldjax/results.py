"""Read a finished output directory back as data.

`foldjax show` prints a directory for a person. A script comparing six models
over a hundred inputs needs the same information as rows -- and before this, the
only way to get them was `foldjax.api._result_from_manifest`, which is private
and answers a different question: whether a directory is exactly the request
now being made, so a resumed run can skip it. Reading results must not depend on
having that request.

So `load_results(root)` reads only the canonical files every run writes --
`foldjax_run.json`, each sample's `confidence.json` and structure, and the
batch's `foldjax_failures.json` -- and never a backend's native side files, so
two sources of truth cannot disagree in a table. A sample's `confidence.json`
takes precedence over the manifest's copy of its scores; a run written before
the common summary existed gets one computed here, from the same mapping, and is
labelled so.

`results_table` flattens that into one row per model / input / seed / sample,
with failures as rows of their own. `aggregate_table` summarizes within one
(input, model, configuration) only. Common fields standardize names and
numerical scales. They retain model-specific definitions and calibration and do
not establish comparable accuracy probabilities or authorize pooled
cross-model ranking -- so nothing here aggregates or ranks across models, and
"best" always means the top of one model's own confidence ordering within one
run.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import statistics
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from foldjax.manifest import MANIFEST_NAME
from foldjax.report import read_manifests
from foldjax.scores import EXECUTION_FIELDS
from foldjax.summary import common_summary

FAILURES_NAME = "foldjax_failures.json"

#: What "best" means wherever this module reports it.
BEST_SELECTION = "within-model confidence ranking"

#: The common fields, in column order.
SUMMARY_FIELDS = ("plddt", "ptm", "iptm")


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """One structure a run produced, with its native scores and common summary."""

    seed: int
    sample: int
    structure_path: Path | None
    structure_sha256: str | None
    #: Whether the structure exists and still hashes to the recorded digest.
    structure_verified: bool
    scores: Mapping[str, float]
    summary: Mapping[str, Mapping[str, Any]]
    #: "confidence.json" when read from the sample's file, "computed" when this
    #: loader derived it for a run written before the summary existed.
    summary_origin: str
    execution: Mapping[str, Any] = field(default_factory=dict)
    job: str | None = None
    native_rank: int | None = None
    is_best: bool = False
    #: The sample's `confidence_full.npz` record from the manifest (which
    #: arrays it holds and why others are unavailable), when one was written.
    confidence_arrays: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One finished model/input run, as its manifest describes it."""

    manifest_path: Path
    model: str
    input: str
    input_sha256: str | None
    #: A digest of what was requested (weights identity, MSA policy, sampling,
    #: options, padding, stop, FoldJAX version): runs that share it are
    #: repeats of one configuration and may be aggregated together.
    configuration: str
    seeds: tuple[int, ...]
    seed_source: str | None
    #: Inputs the common job named that the model never read; None when the
    #: input was native and FoldJAX did not inspect it.
    ignored_msas: tuple[Mapping[str, Any], ...] | None
    ignored_templates: tuple[Mapping[str, Any], ...] | None
    best: Mapping[str, Any] | None
    samples: tuple[SampleRecord, ...]
    schema_version: str | None
    manifest: Mapping[str, Any] = field(repr=False, default_factory=dict)
    #: Constraints the backend never read (OpenDDE): a native job's
    #: ``constraint`` or a common job's ``constraints``; None where the
    #: backend has no such gate or the manifest predates the field.
    ignored_constraints: tuple[Mapping[str, Any], ...] | None = None
    #: A common job's pocket and contact constraints as the model ran them,
    #: each with its ``max_distance`` and ``max_distance_source`` (``job``
    #: or ``upstream``);
    #: None for native input or a manifest that predates the field.
    constraints: tuple[Mapping[str, Any], ...] | None = None

    @property
    def directory(self) -> Path:
        return self.manifest_path.parent


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """One model/input/seed that did not finish, from `foldjax_failures.json`."""

    model: str
    input: str
    seed: int | None
    output_dir: str | None
    error_type: str
    error: str
    source: Path


@dataclass(frozen=True, slots=True)
class ResultsReport:
    """Everything `load_results` found under one directory."""

    root: Path
    runs: tuple[RunRecord, ...] = ()
    failures: tuple[FailureRecord, ...] = ()

    def samples(self) -> Iterator[tuple[RunRecord, SampleRecord]]:
        for run in self.runs:
            for sample in run.samples:
                yield run, sample


def _digest(path: Path) -> str | None:
    try:
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except OSError:
        return None


def configuration_digest(manifest: Mapping[str, Any]) -> str:
    """A short, stable name for what a run was asked to do."""
    weights = (
        manifest.get("weights") if isinstance(manifest.get("weights"), Mapping) else {}
    )
    identity = {
        "model": manifest.get("model"),
        "weights": weights.get("identity") or weights.get("label"),
        "profile": weights.get("profile"),
        "msa": manifest.get("msa"),
        "sampling": manifest.get("sampling"),
        "options": manifest.get("options"),
        "padding": manifest.get("padding"),
        "stop_after": manifest.get("stop_after"),
        "foldjax": manifest.get("foldjax"),
    }
    # Only when searched, so a run without templates keeps the digest it had
    # before the field existed.
    if manifest.get("templates", "none") != "none":
        identity["templates"] = manifest.get("templates")
        identity["template_max_date"] = manifest.get("template_max_date")
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def _structure(
    recorded: Any, manifest: Mapping[str, Any], directory: Path, root: Path
) -> Path | None:
    if not isinstance(recorded, str) or not recorded:
        return None
    path = Path(recorded)
    if manifest.get("artifact_paths") == "manifest-relative" or not path.is_absolute():
        path = directory / path
    path = Path(os.path.normpath(path))
    # Read-only, but still bounded: a manifest cannot point the loader at an
    # arbitrary file outside the directory it was asked to read.
    return path if _inside(path, root) else None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _numeric(scores: Any) -> dict[str, float]:
    if not isinstance(scores, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in scores.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _sample_record(
    entry: Mapping[str, Any],
    position: int,
    *,
    model: str,
    manifest: Mapping[str, Any],
    directory: Path,
    root: Path,
    best: Mapping[str, Any] | None,
) -> SampleRecord:
    metadata = (
        entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
    )
    seed = entry.get("seed")
    seed = int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else -1
    index = metadata.get("sample")
    index = (
        int(index)
        if isinstance(index, int) and not isinstance(index, bool)
        else position
    )
    structure = _structure(entry.get("structure_path"), manifest, directory, root)
    recorded_digest = entry.get("structure_sha256")
    recorded_digest = recorded_digest if isinstance(recorded_digest, str) else None
    verified = bool(
        structure is not None
        and structure.is_file()
        and recorded_digest
        and _digest(structure) == recorded_digest
    )

    scores = _numeric(entry.get("scores"))
    execution = {key: metadata[key] for key in EXECUTION_FIELDS if key in metadata}
    summary: Mapping[str, Any] | None = None
    origin = "computed"
    native_rank = metadata.get("native_rank")
    job = metadata.get("job")
    confidence = (
        _read_json(structure.parent / "confidence.json")
        if structure is not None
        else None
    )
    if isinstance(confidence, Mapping) and confidence.get("model") in (None, model):
        # The canonical per-sample file wins over the manifest's copy.
        scores = _numeric(confidence.get("scores")) or scores
        if isinstance(confidence.get("summary"), Mapping):
            summary = confidence["summary"]
            origin = "confidence.json"
        if isinstance(confidence.get("execution"), Mapping):
            execution = {**execution, **confidence["execution"]}
        native_rank = confidence.get("native_rank", native_rank)
        job = confidence.get("job", job)
    if summary is None:
        # Written before the common block existed: a recycle count still sits
        # among the scores there.
        for key in EXECUTION_FIELDS & set(scores):
            execution.setdefault(key, scores.pop(key))
        summary = common_summary(model, scores, structure=structure)
    is_best = bool(
        best
        and best.get("seed") == seed
        and best.get("sample") == index
        and (best.get("job") in (None, job))
    )
    return SampleRecord(
        seed=seed,
        sample=index,
        structure_path=structure,
        structure_sha256=recorded_digest,
        structure_verified=verified,
        scores=scores,
        summary=summary,
        summary_origin=origin,
        execution=execution,
        job=str(job) if job else None,
        native_rank=(
            int(native_rank)
            if isinstance(native_rank, int) and not isinstance(native_rank, bool)
            else None
        ),
        is_best=is_best,
        confidence_arrays=(
            dict(metadata["confidence_arrays"])
            if isinstance(metadata.get("confidence_arrays"), Mapping)
            else None
        ),
    )


def _ignored(value: Any) -> tuple[Mapping[str, Any], ...] | None:
    if not isinstance(value, list):
        return None
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _run_record(path: Path, manifest: Mapping[str, Any], root: Path) -> RunRecord:
    model = str(manifest.get("model") or "?")
    input_record = (
        manifest.get("input") if isinstance(manifest.get("input"), Mapping) else {}
    )
    best = manifest.get("best") if isinstance(manifest.get("best"), Mapping) else None
    entries = [
        item for item in manifest.get("samples") or [] if isinstance(item, Mapping)
    ]
    samples = tuple(
        _sample_record(
            entry,
            position,
            model=model,
            manifest=manifest,
            directory=path.parent,
            root=root,
            best=best,
        )
        for position, entry in enumerate(entries)
    )
    seeds = tuple(
        int(seed)
        for seed in manifest.get("seeds") or []
        if isinstance(seed, int) and not isinstance(seed, bool)
    )
    return RunRecord(
        manifest_path=path,
        model=model,
        input=str(input_record.get("path") or ""),
        input_sha256=input_record.get("sha256"),
        configuration=configuration_digest(manifest),
        seeds=seeds,
        seed_source=manifest.get("seed_source"),
        ignored_msas=_ignored(manifest.get("ignored_msas")),
        ignored_templates=_ignored(manifest.get("ignored_templates")),
        ignored_constraints=_ignored(manifest.get("ignored_constraints")),
        constraints=_ignored(manifest.get("constraints")),
        best=best,
        samples=samples,
        schema_version=manifest.get("schema_version"),
        manifest=manifest,
    )


def _failures(root: Path, runs: Sequence[RunRecord]) -> list[FailureRecord]:
    finished = {(run.model, run.input, seed) for run in runs for seed in run.seeds}
    candidates = (
        [root] if root.name == FAILURES_NAME else sorted(root.rglob(FAILURES_NAME))
    )
    found: list[FailureRecord] = []
    for path in candidates:
        document = _read_json(path)
        if not isinstance(document, list):
            continue
        for item in document:
            if not isinstance(item, Mapping):
                continue
            seed = item.get("seed")
            seed = (
                int(seed)
                if isinstance(seed, int) and not isinstance(seed, bool)
                else None
            )
            model = str(item.get("model") or "?")
            input_path = str(item.get("input") or "")
            # A failures file is not removed when a later invocation finishes
            # the same run, so a finished manifest for that seed supersedes it.
            if seed is not None and (model, input_path, seed) in finished:
                continue
            found.append(
                FailureRecord(
                    model=model,
                    input=input_path,
                    seed=seed,
                    output_dir=item.get("output_dir"),
                    error_type=str(item.get("error_type") or ""),
                    error=str(item.get("error") or ""),
                    source=path,
                )
            )
    return found


def load_results(root: str | os.PathLike[str]) -> ResultsReport:
    """Every finished run and recorded failure at or below ``root``.

    Reads `foldjax_run.json`, each sample's `confidence.json` and structure,
    and `foldjax_failures.json`; never a backend's native side files. Needs no
    request: unlike resume, it does not ask whether a directory matches one.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"no such output directory: {root}")
    base = root.parent if root.is_file() else root
    entries = read_manifests(root) if root.name != FAILURES_NAME else []
    runs = tuple(_run_record(path, document, base) for path, document in entries)
    failures = tuple(_failures(root, runs))
    if not runs and not failures:
        raise FileNotFoundError(
            f"no {MANIFEST_NAME} or {FAILURES_NAME} under {root}; a run writes "
            "its manifest when it finishes"
        )
    return ResultsReport(root=base, runs=runs, failures=failures)


def _field_value(summary: Mapping[str, Any], name: str) -> float | None:
    entry = summary.get(name)
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def results_table(
    report: ResultsReport | str | os.PathLike[str],
) -> list[dict[str, Any]]:
    """One row per model / input / seed / sample, and one per failure.

    Columns: identity (status, model, input, configuration, seed, sample, job,
    native rank, run directory), the common summary (``plddt`` on 0-100 with
    its source, ``ptm``, ``iptm``, the ranking key and value, and why any of
    them is empty), the native scores as ``score.<name>``, the structure path,
    its SHA-256 and whether it still matches, what the model never read
    (``ignored_msas``, ``ignored_templates``; None for a native input, which
    is not inspected, except OpenDDE's, whose dropped native templates and RNA
    alignments are listed; and ``ignored_constraints``, an OpenDDE job's native
    constraint or common constraints), the pocket and contact ``constraints``
    a common job ran with (``max_distance`` and its ``max_distance_source``),
    and for a failure its error. ``best_within_model``
    marks the top of that model's own confidence ordering within its run.
    """
    if not isinstance(report, ResultsReport):
        report = load_results(report)
    rows: list[dict[str, Any]] = []
    for run in report.runs:
        for sample in run.samples:
            summary = sample.summary
            ranking = (
                summary.get("ranking")
                if isinstance(summary.get("ranking"), Mapping)
                else {}
            )
            plddt = (
                summary.get("plddt")
                if isinstance(summary.get("plddt"), Mapping)
                else {}
            )
            reasons = {
                name: entry.get("reason")
                for name, entry in summary.items()
                if isinstance(entry, Mapping) and entry.get("value") is None
            }
            row: dict[str, Any] = {
                "status": "ok"
                if sample.structure_verified
                else "structure missing or changed",
                "model": run.model,
                "input": run.input,
                "input_name": Path(run.input).stem if run.input else None,
                "input_sha256": run.input_sha256,
                "configuration": run.configuration,
                "seed": sample.seed,
                "seed_source": run.seed_source,
                "sample": sample.sample,
                "job": sample.job,
                "native_rank": sample.native_rank,
                # Names of the arrays in this sample's confidence_full.npz;
                # None when the run wrote none.
                "confidence_arrays": (
                    list(sample.confidence_arrays.get("arrays") or [])
                    if sample.confidence_arrays is not None
                    else None
                ),
                "best_within_model": sample.is_best,
                "plddt": _field_value(summary, "plddt"),
                "plddt_source": plddt.get("source"),
                "ptm": _field_value(summary, "ptm"),
                "iptm": _field_value(summary, "iptm"),
                "ranking_key": ranking.get("key"),
                "ranking_value": _field_value(summary, "ranking"),
                "summary_missing": reasons or None,
                "summary_origin": sample.summary_origin,
                "structure_path": str(sample.structure_path)
                if sample.structure_path
                else None,
                "structure_sha256": sample.structure_sha256,
                "structure_verified": sample.structure_verified,
                "ignored_msas": (
                    [dict(item) for item in run.ignored_msas]
                    if run.ignored_msas is not None
                    else None
                ),
                "ignored_templates": (
                    [dict(item) for item in run.ignored_templates]
                    if run.ignored_templates is not None
                    else None
                ),
                "ignored_constraints": (
                    [dict(item) for item in run.ignored_constraints]
                    if run.ignored_constraints is not None
                    else None
                ),
                "constraints": (
                    [dict(item) for item in run.constraints]
                    if run.constraints is not None
                    else None
                ),
                "run_dir": str(run.directory),
                "error_type": None,
                "error": None,
            }
            for name, value in sorted(sample.scores.items()):
                row[f"score.{name}"] = value
            for name, value in sorted(sample.execution.items()):
                row[f"execution.{name}"] = value
            rows.append(row)
    for failure in report.failures:
        rows.append(
            {
                "status": "failed",
                "model": failure.model,
                "input": failure.input,
                "input_name": Path(failure.input).stem if failure.input else None,
                "seed": failure.seed,
                "run_dir": failure.output_dir,
                "error_type": failure.error_type,
                "error": failure.error,
            }
        )
    return rows


def _columns(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                columns.append(name)
    fixed = [name for name in columns if not name.startswith(("score.", "execution."))]
    extra = sorted(
        name for name in columns if name.startswith(("score.", "execution."))
    )
    # Native scores go last: their set differs by model.
    index = fixed.index("run_dir") if "run_dir" in fixed else len(fixed)
    return fixed[:index] + extra + fixed[index:]


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Rows as CSV, with nested cells as compact JSON and None as empty."""
    buffer = io.StringIO()
    columns = _columns(rows)
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _cell(row.get(name)) for name in columns})
    return buffer.getvalue()


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None, "spread": None}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def aggregate_table(
    rows: Sequence[Mapping[str, Any]] | ResultsReport | str | os.PathLike[str],
) -> list[dict[str, Any]]:
    """Count, median and spread (max - min) per (input, model, configuration).

    Never across models or configurations. ``best`` is the sample this model's
    own ranking score puts first within the group -- a within-model confidence
    selection, not an accuracy claim and not comparable to another model's.
    """
    if not isinstance(rows, list):
        rows = results_table(rows)  # type: ignore[arg-type]
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    failures: dict[tuple[str, str], int] = {}
    for row in rows:
        if row.get("status") == "failed":
            key2 = (str(row.get("input")), str(row.get("model")))
            failures[key2] = failures.get(key2, 0) + 1
            continue
        key = (
            str(row.get("input")),
            str(row.get("model")),
            str(row.get("configuration")),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (input_path, model, configuration), members in sorted(groups.items()):
        record: dict[str, Any] = {
            "input": input_path,
            "model": model,
            "configuration": configuration,
            "samples": len(members),
            "seeds": sorted(
                {row.get("seed") for row in members if row.get("seed") is not None}
            ),
            "failures": failures.get((input_path, model), 0),
        }
        for name in (*SUMMARY_FIELDS, "ranking_value"):
            values = [
                float(row[name])
                for row in members
                if isinstance(row.get(name), (int, float))
                and not isinstance(row.get(name), bool)
                and math.isfinite(float(row[name]))
            ]
            for stat, value in _stats(values).items():
                record[f"{name}_{stat}"] = value
        ranked = [
            row for row in members if isinstance(row.get("ranking_value"), (int, float))
        ]
        if ranked:
            # max keeps the first on a tie: seed order, then diffusion order.
            top = max(ranked, key=lambda row: row["ranking_value"])
            record["best_within_model"] = {
                "selection": BEST_SELECTION,
                "ranking_key": top.get("ranking_key"),
                "ranking_value": top.get("ranking_value"),
                "seed": top.get("seed"),
                "sample": top.get("sample"),
                "job": top.get("job"),
                "structure_path": top.get("structure_path"),
            }
        else:
            record["best_within_model"] = None
        out.append(record)
    for (input_path, model), count in sorted(failures.items()):
        if not any(
            item["input"] == input_path and item["model"] == model for item in out
        ):
            out.append(
                {
                    "input": input_path,
                    "model": model,
                    "configuration": None,
                    "samples": 0,
                    "seeds": [],
                    "failures": count,
                    "best_within_model": None,
                }
            )
    return out
