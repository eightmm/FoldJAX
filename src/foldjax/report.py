"""A finished run, as a person reads it.

`foldjax predict` printed one JSON object. That is the right thing for a script
and the wrong thing for the terminal it is usually run in: the number a person
wants first -- which structure came out best, and where it is -- was four levels
down inside a document long enough to scroll off the screen. `best_sample` had
computed exactly that since the output layout landed, and put it only in
`foldjax_run.json`, which nothing printed.

So this renders the manifest instead of inventing a second source of truth. It
is the same file `foldjax show` reads afterwards, which is why a run summary and
a directory summary cannot disagree, and why `show` works on a directory
produced by any earlier version.

**Scores are not comparable across models** and nothing here makes them look as
if they were. Each table is one model's own scores under that model's own names,
and the "best" column is the score that model ranks with, or nothing at all --
see `foldjax.output` for why a substitute would be a different claim wearing the
same word. The common names and scales in each `confidence.json` summary do not
change that: common fields standardize names and numerical scales. They retain
model-specific definitions and calibration and do not establish comparable
accuracy probabilities or authorize pooled cross-model ranking. Rows for
scripts come from `foldjax.results`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from foldjax.manifest import MANIFEST_NAME

#: How many score columns a table shows before it stops being scannable. The
#: complete set is always in `confidence.json` and the manifest.
_MAX_SCORE_COLUMNS = 3


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _bytes(size: Any) -> str:
    if not isinstance(size, (int, float)) or size <= 0:
        return "-"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def read_manifests(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every run manifest at or below ``root``, in directory order.

    A batch writes one per model/input pair and a multi-seed run writes one for
    the whole request plus one per seed. The per-seed manifests are dropped when
    their parent is present: the parent already lists every one of their
    samples, and showing both would count each structure twice.
    """
    root = Path(root)
    candidates = (
        [root]
        if root.name == MANIFEST_NAME
        else sorted(root.rglob(MANIFEST_NAME))
    )
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, dict):
            found.append((path, document))
    parents = {path.parent for path, _ in found}
    return [
        (path, document)
        for path, document in found
        if not any(
            parent != path.parent and path.parent.is_relative_to(parent)
            for parent in parents
        )
    ]


def sample_indices(samples: list[Any]) -> list[int]:
    """Each manifest sample's number within its own seed (and job).

    ``metadata.sample`` when recorded. A merged multi-seed manifest written
    before it was recorded lists every seed's samples in one array, so the
    fallback counts within the seed rather than across the list.
    """
    counters: dict[tuple[Any, Any], int] = {}
    indices: list[int] = []
    for sample in samples:
        metadata = sample.get("metadata") if isinstance(sample, Mapping) else None
        metadata = metadata if isinstance(metadata, Mapping) else {}
        slot = (
            sample.get("seed") if isinstance(sample, Mapping) else None,
            metadata.get("job"),
        )
        position = counters.get(slot, 0)
        counters[slot] = position + 1
        value = metadata.get("sample")
        indices.append(
            value
            if isinstance(value, int) and not isinstance(value, bool)
            else position
        )
    return indices


def is_best(best: Any, sample: Any, index: int) -> bool:
    """Whether ``sample`` (numbered ``index``) is the manifest's ``best``.

    By structure path when both record one -- a merged manifest written before
    per-seed numbering recorded ``best.sample`` as a position across seeds --
    else by seed, sample number and job.
    """
    if not isinstance(best, Mapping) or not isinstance(sample, Mapping):
        return False
    path = best.get("structure_path")
    if path and sample.get("structure_path"):
        return path == sample["structure_path"]
    metadata = (
        sample.get("metadata") if isinstance(sample.get("metadata"), Mapping) else {}
    )
    return (
        best.get("seed") == sample.get("seed")
        and best.get("sample") == index
        and best.get("job") in (None, metadata.get("job"))
    )


def _score_columns(samples: list[dict[str, Any]], ranking: str | None) -> list[str]:
    """Which score names to show, ranking score first, then the most common."""
    counts: dict[str, int] = {}
    for sample in samples:
        for name in (sample.get("scores") or {}):
            counts[name] = counts.get(name, 0) + 1
    ordered = sorted(counts, key=lambda name: (-counts[name], name))
    if ranking in ordered:
        ordered.remove(ranking)
        ordered.insert(0, ranking)
    return ordered[:_MAX_SCORE_COLUMNS]


def _relative(path: str | None, base: Path) -> str:
    if not path:
        return "-"
    candidate = Path(path)
    try:
        return str(candidate.relative_to(base))
    except ValueError:
        return str(candidate)


def _input_label(manifest: dict[str, Any]) -> str:
    """The job a run folded: a multi-job file's job by name, else its file."""
    record = manifest.get("input") if isinstance(manifest.get("input"), dict) else {}
    source = record.get("source")
    if isinstance(source, dict) and source.get("name"):
        return (
            f"{source['name']}  ({Path(str(source.get('path') or '')).name} "
            f"jobs[{source.get('index')}])"
        )
    path = record.get("path")
    return Path(str(path)).name if path else "-"


def render_failures(failures: list[Any]) -> str:
    """The footer naming each recorded failure, one line per run."""
    lines = [f"failed    {len(failures)} run(s)"]
    for failure in failures:
        seed = "" if failure.seed is None else f" seed {failure.seed}"
        label = Path(failure.input).name if failure.input else "-"
        lines.append(f"  {failure.model} · {label}{seed}: {failure.error}")
    return "\n".join(lines)


def render(manifest: dict[str, Any], *, directory: Path) -> str:
    """One run's header, best structure, and per-sample table."""
    lines: list[str] = []
    weights = manifest.get("weights") or {}
    cost = manifest.get("cost") or {}
    samples = [item for item in manifest.get("samples") or [] if isinstance(item, dict)]
    best = manifest.get("best") if isinstance(manifest.get("best"), dict) else None

    label = weights.get("label") or weights.get("profile") or "-"
    lines.append(
        f"model     {str(manifest.get('model', '?')):<16s}weights  {label}"
    )
    lines.append(f"input     {_input_label(manifest)}")
    seeds = ", ".join(str(seed) for seed in manifest.get("seeds") or [])
    lines.append(
        f"samples   {len(samples):<16d}time     "
        f"{_duration(cost.get('seconds')):<10s}peak  {_bytes(cost.get('peak_bytes'))}"
    )
    lines.append(f"seeds     {seeds or '-':<16s}msa      {manifest.get('msa', 'none')}")
    if manifest.get("templates", "none") != "none":
        kept = sum(
            len(record.get("templates") or [])
            for record in manifest.get("template_search") or []
            if isinstance(record, dict)
        )
        lines.append(f"templates {manifest['templates']:<16s}kept     {kept}")
    phases = cost.get("phases")
    if isinstance(phases, dict) and phases:
        detail = "  ".join(
            f"{name} {_duration(value)}" for name, value in phases.items()
        )
        lines.append(f"phases    {detail}")
    if best:
        best_index = next(
            (
                index
                for sample, index in zip(samples, sample_indices(samples), strict=True)
                if is_best(best, sample, index)
            ),
            best.get("sample", 0),
        )
        lines.append(
            f"best      seed {best.get('seed')} / sample "
            f"{int(best_index):02d}     "
            f"{best.get('score')} {best.get('value')}"
        )
        lines.append(f"          {_relative(best.get('structure_path'), directory)}")

    if not samples:
        return "\n".join(lines)

    columns = _score_columns(samples, best.get("score") if best else None)
    widths = {name: max(len(name), 8) for name in columns}
    heading = "  seed  sample" + "".join(
        f"  {name:>{widths[name]}s}" for name in columns
    )
    lines.append("")
    lines.append(heading + "  structure")
    for sample, index in zip(samples, sample_indices(samples), strict=True):
        scores = sample.get("scores") or {}
        row = f"  {sample.get('seed', '-'):>4}  {index:>6d}"
        for name in columns:
            value = scores.get(name)
            text = "-" if not isinstance(value, (int, float)) else f"{value:.3f}"
            row += f"  {text:>{widths[name]}s}"
        row += f"  {_relative(sample.get('structure_path'), directory)}"
        if is_best(best, sample, index):
            row += "  <- best"
        lines.append(row)
    return "\n".join(lines)


def render_all(entries: list[tuple[Path, dict[str, Any]]]) -> str:
    """Several runs, separated so a batch reads as a batch."""
    blocks = [
        render(document, directory=path.parent) for path, document in entries
    ]
    return "\n\n".join(blocks)
