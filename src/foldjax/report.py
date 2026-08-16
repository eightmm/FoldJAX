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
same word.
"""

from __future__ import annotations

import json
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
    seeds = ", ".join(str(seed) for seed in manifest.get("seeds") or [])
    lines.append(
        f"samples   {len(samples):<16d}time     "
        f"{_duration(cost.get('seconds')):<10s}peak  {_bytes(cost.get('peak_bytes'))}"
    )
    lines.append(f"seeds     {seeds or '-':<16s}msa      {manifest.get('msa', 'none')}")
    phases = cost.get("phases")
    if isinstance(phases, dict) and phases:
        detail = "  ".join(
            f"{name} {_duration(value)}" for name, value in phases.items()
        )
        lines.append(f"phases    {detail}")
    if best:
        lines.append(
            f"best      seed {best.get('seed')} / sample "
            f"{int(best.get('sample', 0)):02d}     "
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
    best_slot = (
        (best.get("seed"), int(best.get("sample", -1))) if best else (None, None)
    )
    for position, sample in enumerate(samples):
        metadata = sample.get("metadata") or {}
        index = metadata.get("sample")
        index = position if not isinstance(index, int) else index
        scores = sample.get("scores") or {}
        row = f"  {sample.get('seed', '-'):>4}  {index:>6d}"
        for name in columns:
            value = scores.get(name)
            text = "-" if not isinstance(value, (int, float)) else f"{value:.3f}"
            row += f"  {text:>{widths[name]}s}"
        row += f"  {_relative(sample.get('structure_path'), directory)}"
        if (sample.get("seed"), index) == best_slot:
            row += "  <- best"
        lines.append(row)
    return "\n".join(lines)


def render_all(entries: list[tuple[Path, dict[str, Any]]]) -> str:
    """Several runs, separated so a batch reads as a batch."""
    blocks = [
        render(document, directory=path.parent) for path, document in entries
    ]
    return "\n\n".join(blocks)
