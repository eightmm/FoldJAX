"""Turn a scale-benchmark work directory into the tables the report carries.

`drive.py` writes one `results/<case>-<model>-<label>.json` per measured run and
one `logs/<case>-<model>-<label>.slurm` per submitted job. The tables were being
assembled from those two directories by hand, which is where the errors were: a
run that never produced a result is not the same as one that died out of
memory, and only the log tells them apart.

Three parsing facts shape the module.

The filename stem is the key, not the json body. Upstream results carry
`label: null` -- the label exists only in the name `drive.py` gave the file --
so anything that buckets on the field collects every upstream run into one
nameless group.

A label may contain a hyphen (`det-notriton`, `cueq-full`) and so may a model
(`protenix-v2`, the case `bench.structures` documents). A stem is therefore split
on a known model name whenever one is available, and only falls back to "the
first two hyphens" for a model that never appears in a result.

A row exists because a FoldJAX job exists. The row set for `scale` and `mixed`
is every case carrying at least one artifact -- result or log -- at the selected
FoldJAX label. Admitting cases on the upstream label instead would pull the
`panel` cases into a scale table, since their upstream results sit in the same
directory; admitting them on results alone would drop every case that OOMed,
which is exactly the row the table exists to show.

Nothing here recomputes a structural comparison. `spread` passes
`compare-structures.json` through, and renders the columns it shares with
`bench.structures --markdown` the way that command renders them, so the two can
be diffed against each other.

Usage:

    python -m bench.scale_table --work /path/to/work --table scale --markdown
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Searched in this order, so the reported marker is the most specific one that
# the log actually contains. `Killed` is last because it is also the least
# informative: the kernel OOM killer prints it, and so does a plain `kill`.
OOM_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "CUDA out of memory",
    "OutOfMemoryError",
    "Out of memory",
    "oom-kill",
    "Killed",
)

# Score keys, in preference order. The fallback is for a *missing* key, not a
# small value: a monomer scores `iptm` 0.0 and that is its ipTM, not a reason to
# report its `ptm` instead.
IPTM_KEYS = ("iptm", "protein_iptm", "ptm")

MISSING = "-"
OOM = "OOM"
RUNNING = "running"

TABLES = ("scale", "mixed", "spread", "alternatives", "all")

_DEFAULT_LABEL = {"scale": "scale", "mixed": "mixed", "alternatives": "scale"}
_DEFAULT_PREFIX = {"scale": "", "mixed": "mixed_", "alternatives": ""}

_BAND_TOKENS = re.compile(r"^L(\d+)_")
_BAND_NAMED = re.compile(r"^mixed_(\d+)k_")

_LEGEND = (
    "cells are wall seconds / peak GiB. `OOM` is a submitted job whose log "
    "carries an allocator failure; `running` is a log with neither a result "
    "nor an OOM marker, which also covers a job that died without one; `-` is "
    "no result and no log."
)


def parse_stem(stem: str, models: object = ()) -> tuple[str, str, str] | None:
    """Split `<case>-<model>-<label>` when both case and label may be empty-ish.

    Splitting on hyphens alone is wrong in both directions: `det-notriton` is one
    label, and a model named `protenix-v2` would be read as model `protenix` with
    label `v2`. Known model names are tried longest-first so the longer of two
    names sharing a prefix wins.
    """
    for model in sorted(models, key=len, reverse=True):
        marker = f"-{model}-"
        index = stem.find(marker)
        if index > 0 and len(stem) > index + len(marker):
            return stem[:index], model, stem[index + len(marker) :]
    parts = stem.split("-", 2)
    if len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


def oom_marker(text: str) -> str | None:
    """The first OOM marker the log carries, or None."""
    for marker in OOM_MARKERS:
        if marker in text:
            return marker
    return None


def best_iptm(samples: object) -> float | None:
    """Max ipTM over a run's samples, or None when no sample reports one."""
    values: list[float] = []
    for sample in samples or []:
        scores = (sample or {}).get("scores") or {}
        for key in IPTM_KEYS:
            if key in scores:
                try:
                    values.append(float(scores[key]))
                except (TypeError, ValueError):
                    pass
                break
    return max(values) if values else None


def band(case: str) -> str | None:
    """The 1k/2k/... band the case name declares, which is not its token count.

    `L5000_8e2f` is 4,888 tokens. The band groups the ladder, the length says
    what was actually run, and both belong in the table.
    """
    match = _BAND_TOKENS.match(case)
    if match:
        return f"{int(match.group(1)) // 1000}k"
    match = _BAND_NAMED.match(case)
    if match:
        return f"{match.group(1)}k"
    return None


def _result_cell(document: dict, case: str, model: str, label: str) -> dict:
    return {
        "case": case,
        "model": model,
        "label": label,
        "impl": document.get("impl"),
        "status": "ok",
        "length": document.get("length"),
        "wall_s": document.get("wall_s"),
        "peak_mib": document.get("peak_mib"),
        "iptm": best_iptm(document.get("samples")),
        # Carried because a result is not proof of a prediction. Nine of the
        # upstream results in the 2026-09-10 scale directory record a wall time
        # and a whole-card peak with zero samples: the run died and wrote a row
        # anyway. The rendered cell still shows the numbers the file holds --
        # this table reports the directory, it does not adjudicate it -- but
        # `--json` carries the count so the reader can see which rows are empty.
        "samples": len(document.get("samples") or []),
        "marker": None,
    }


def load_work(work: Path) -> dict[tuple[str, str, str], dict]:
    """Every artifact under `work`, keyed by (case, model, label).

    A result wins over a log for the same key: the log of a finished job says
    nothing the result does not, and its text may still contain the word
    `Killed` from an unrelated line.
    """
    results = Path(work) / "results"
    logs = Path(work) / "logs"

    documents: dict[str, dict] = {}
    models: set[str] = set()
    for path in sorted(results.glob("*.json")):
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        documents[path.stem] = document
        name = document.get("model")
        if isinstance(name, str) and name:
            models.add(name)

    artifacts: dict[tuple[str, str, str], dict] = {}
    for stem, document in documents.items():
        parsed = parse_stem(stem, models)
        if parsed is None:
            continue
        case, model, label = parsed
        artifacts[parsed] = _result_cell(document, case, model, label)

    if logs.is_dir():
        for path in sorted(logs.glob("*.slurm")):
            parsed = parse_stem(path.stem, models)
            if parsed is None or parsed in artifacts:
                continue
            case, model, label = parsed
            try:
                text = path.read_text(errors="replace")
            except OSError:
                text = ""
            marker = oom_marker(text)
            artifacts[parsed] = {
                "case": case,
                "model": model,
                "label": label,
                "impl": None,
                "status": "oom" if marker else "running",
                "length": None,
                "wall_s": None,
                "peak_mib": None,
                "iptm": None,
                "samples": None,
                "marker": marker,
            }
    return artifacts


def case_lengths(artifacts: dict[tuple[str, str, str], dict]) -> dict[str, int]:
    """Token count per case, taken from whichever run recorded one.

    A case whose every FoldJAX run OOMed still has a length, because its
    upstream runs wrote one.
    """
    lengths: dict[str, int] = {}
    for (case, _model, _label), cell in sorted(artifacts.items()):
        length = cell.get("length")
        if length is not None and case not in lengths:
            lengths[case] = int(length)
    return lengths


def measurement_rows(
    artifacts: dict[tuple[str, str, str], dict],
    *,
    label: str,
    upstream_label: str,
    case_prefix: str = "",
) -> list[dict]:
    """One row per case, cases admitted by the FoldJAX label, sorted by length."""
    lengths = case_lengths(artifacts)
    cases = sorted(
        {
            case
            for (case, _model, artifact_label) in artifacts
            if artifact_label == label and case.startswith(case_prefix)
        }
    )
    models = sorted(
        {
            model
            for (case, model, artifact_label) in artifacts
            if case in cases and artifact_label in (label, upstream_label)
        }
    )
    rows = []
    for case in cases:
        cells = {
            model: {
                "foldjax": artifacts.get((case, model, label)),
                "upstream": artifacts.get((case, model, upstream_label)),
            }
            for model in models
        }
        rows.append(
            {
                "case": case,
                "band": band(case),
                "length": lengths.get(case),
                "models": models,
                "cells": cells,
            }
        )
    rows.sort(key=lambda row: (row["length"] is None, row["length"] or 0, row["case"]))
    return rows


def alternative_rows(
    artifacts: dict[tuple[str, str, str], dict],
    *,
    label: str,
    upstream_label: str,
    case_prefix: str = "",
) -> list[dict]:
    """Every (case, model) measured under more than one FoldJAX label.

    Ratios are against `label`; a group whose baseline never produced a result
    -- OpenFold3 at 3,012 tokens has `cueq` and `cueq-full` and no `scale` --
    keeps its rows and reports no ratio.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for (case, model, artifact_label), cell in artifacts.items():
        if artifact_label == upstream_label or cell.get("impl") == "upstream":
            continue
        if not case.startswith(case_prefix):
            continue
        groups.setdefault((case, model), []).append(cell)

    rows = []
    for (case, model), cells in sorted(groups.items()):
        if len({cell["label"] for cell in cells}) < 2:
            continue
        baseline = next(
            (
                cell
                for cell in cells
                if cell["label"] == label and cell["status"] == "ok"
            ),
            None,
        )
        ordered = sorted(
            cells, key=lambda cell: (cell["label"] != label, cell["label"])
        )
        for cell in ordered:
            rows.append(
                {
                    "case": case,
                    "model": model,
                    "label": cell["label"],
                    "status": cell["status"],
                    "wall_s": cell["wall_s"],
                    "peak_mib": cell["peak_mib"],
                    "baseline": label,
                    "wall_ratio": _ratio(cell, baseline, "wall_s"),
                    "peak_ratio": _ratio(cell, baseline, "peak_mib"),
                }
            )
    return rows


def _ratio(cell: dict, baseline: dict | None, field: str) -> float | None:
    if baseline is None or cell["status"] != "ok":
        return None
    value, reference = cell.get(field), baseline.get(field)
    if value is None or not reference:
        return None
    return float(value) / float(reference)


def measurement(cell: dict | None) -> str:
    """`wall / peak`, or why there is no measurement."""
    if cell is None:
        return MISSING
    if cell["status"] == "oom":
        return OOM
    if cell["status"] != "ok":
        return RUNNING
    wall = "?" if cell.get("wall_s") is None else f"{float(cell['wall_s']):.0f}"
    peak = (
        "?" if cell.get("peak_mib") is None else f"{float(cell['peak_mib']) / 1024:.1f}"
    )
    return f"{wall} / {peak}"


def _iptm(cell: dict | None) -> str:
    if cell is None or cell.get("iptm") is None:
        return ""
    return f"{float(cell['iptm']):.3f}"


def _table(header: list[str], body: list[list[str]], legend: str) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return "\n".join(lines) + "\n\n" + legend


def render_measurements(rows: list[dict], *, iptm: bool = False) -> str:
    models = rows[0]["models"] if rows else []
    header = ["band", "case", "tokens"]
    for model in models:
        header += [f"{model} FoldJAX", f"{model} upstream"]
        if iptm:
            header += [f"{model} FJ ipTM", f"{model} up ipTM"]
    body = []
    for row in rows:
        length = row["length"]
        cells = [
            row["band"] or MISSING,
            row["case"],
            MISSING if length is None else str(length),
        ]
        for model in models:
            pair = row["cells"].get(model) or {}
            cells += [
                measurement(pair.get("foldjax")),
                measurement(pair.get("upstream")),
            ]
            if iptm:
                cells += [_iptm(pair.get("foldjax")), _iptm(pair.get("upstream"))]
        body.append(cells)
    legend = _LEGEND
    if iptm:
        legend += (
            " ipTM is the best of the run's samples, reported from `iptm` where "
            "the model writes one and from `protein_iptm` or `ptm` where it does "
            "not; blank means no sample carried any of the three."
        )
    return _table(header, body, legend)


def render_alternatives(rows: list[dict]) -> str:
    header = ["case", "model", "label", "wall s / peak GiB", "vs baseline wall / peak"]
    body = []
    for row in rows:
        ratio = MISSING
        if row["wall_ratio"] is not None or row["peak_ratio"] is not None:
            wall = MISSING if row["wall_ratio"] is None else f"{row['wall_ratio']:.2f}x"
            peak = MISSING if row["peak_ratio"] is None else f"{row['peak_ratio']:.2f}x"
            ratio = f"{wall} / {peak}"
        body.append(
            [
                row["case"],
                row["model"],
                row["label"],
                measurement(row),
                ratio,
            ]
        )
    legend = (
        _LEGEND
        + " Ratios are against the `"
        + (rows[0]["baseline"] if rows else "scale")
        + "` label of the same case and model; a group whose baseline produced "
        "no result reports none."
    )
    return _table(header, body, legend)


def _spread_cell(block: object, field: str = "tm") -> str:
    """The cell `bench.structures --markdown` prints, rendered identically."""
    if not block or not block.get(field):
        return MISSING
    values = block[field]
    return f"{values['median']:.3f} ({values['min']:.3f}-{values['max']:.3f})"


def _spread_permutations(row: dict) -> str:
    parts = [
        f"{name} {block['permuted']}/{block['tm']['n']}"
        for name, block in (
            ("cross", row.get("cross")),
            ("fj", row.get("within_foldjax")),
            ("up", row.get("within_upstream")),
        )
        if block and block.get("permuted")
    ]
    return "; ".join(parts) if parts else MISSING


def render_spread(rows: list[dict]) -> str:
    """`bench.structures --markdown`'s columns, plus the two RMSD it drops.

    The shared columns are byte-identical to that command's output on the same
    rows, which is the only way to check that this reads the comparison rather
    than reinventing it.
    """
    header = [
        "model",
        "case",
        "cross TM",
        "within FoldJAX TM",
        "within upstream TM",
        "cross RMSD A",
        "within FoldJAX RMSD A",
        "within upstream RMSD A",
        "chain perm",
    ]
    body = [
        [
            row.get("model", MISSING),
            row.get("case", MISSING),
            _spread_cell(row.get("cross")),
            _spread_cell(row.get("within_foldjax")),
            _spread_cell(row.get("within_upstream")),
            _spread_cell(row.get("cross"), "rmsd"),
            _spread_cell(row.get("within_foldjax"), "rmsd"),
            _spread_cell(row.get("within_upstream"), "rmsd"),
            _spread_permutations(row),
        ]
        for row in rows
    ]
    legend = (
        "Median over all scored pairs, min-max in brackets, passed through from "
        "`compare-structures.json` unchanged. Read `cross` against the two "
        "`within` columns: torch and JAX do not share a diffusion random tape "
        "even from one seed, so `within` -- how well an implementation agrees "
        "with itself across samples -- is the closest any correct port can come. "
        "`chain perm` counts the scored pairs whose interchangeable chains had "
        "to be reassigned before scoring, which is a labelling difference and "
        "not a structural one."
    )
    return _table(header, body, legend)


def load_comparison(path: Path) -> list[dict]:
    try:
        rows = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return rows if isinstance(rows, list) else []


def _resolve(table: str, label: str | None, prefix: str | None) -> tuple[str, str]:
    return (
        label if label is not None else _DEFAULT_LABEL[table],
        prefix if prefix is not None else _DEFAULT_PREFIX[table],
    )


def build(
    work: Path,
    table: str,
    *,
    label: str | None = None,
    upstream_label: str = "upstream",
    case_prefix: str | None = None,
    comparison: Path | None = None,
) -> dict[str, list[dict]]:
    """The parsed rows for one table, or for all four."""
    every = ("scale", "mixed", "spread", "alternatives")
    wanted = every if table == "all" else (table,)
    artifacts = load_work(work) if set(wanted) - {"spread"} else {}
    built: dict[str, list[dict]] = {}
    for name in wanted:
        if name == "spread":
            path = comparison or Path(work) / "compare-structures.json"
            built[name] = load_comparison(path)
            continue
        chosen, prefix = _resolve(name, label, case_prefix)
        builder = alternative_rows if name == "alternatives" else measurement_rows
        built[name] = builder(
            artifacts,
            label=chosen,
            upstream_label=upstream_label,
            case_prefix=prefix,
        )
    return built


def render(name: str, rows: list[dict]) -> str:
    if name == "spread":
        return render_spread(rows)
    if name == "alternatives":
        return render_alternatives(rows)
    return render_measurements(rows, iptm=name == "mixed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--table", choices=TABLES, default="scale")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--label",
        help="FoldJAX label to read; defaults per table (scale, mixed)",
    )
    parser.add_argument("--upstream-label", default="upstream")
    parser.add_argument(
        "--case-prefix",
        help="restrict to cases whose name starts with this; defaults per table",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="comparison json for the spread table "
        "(default: <work>/compare-structures.json)",
    )
    args = parser.parse_args(argv)

    built = build(
        args.work,
        args.table,
        label=args.label,
        upstream_label=args.upstream_label,
        case_prefix=args.case_prefix,
        comparison=args.compare,
    )

    if args.json:
        payload = built if args.table == "all" else built[args.table]
        print(json.dumps(payload, indent=2))
        return 0

    for index, (name, rows) in enumerate(built.items()):
        if args.table == "all":
            print(("\n" if index else "") + f"## {name}\n")
        print(render(name, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
