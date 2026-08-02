"""Render the collected rows as the comparison table.

Reads whatever `drive.py` has written so far, so an interrupted matrix still
reports every row that finished. A row that failed is printed as a failure
rather than omitted -- a model that could not run at a size is a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(results: Path) -> list[dict]:
    rows = []
    for path in sorted(results.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return rows


def _confidence(row: dict) -> str:
    """One comparable number per row, named, or '-' if the run produced none.

    The models do not report the same confidence fields, and the ones they
    share are not interchangeable across models -- only across the two
    implementations of the same model, which is the comparison this column is
    for.
    """
    samples = row.get("samples") or []
    for key in ("ranking_score", "aggregate_score", "confidence_score", "ptm"):
        values = [
            sample["scores"][key]
            for sample in samples
            if isinstance(sample.get("scores"), dict) and key in sample["scores"]
        ]
        if values:
            return f"{key} {max(values):.4f}"
    return "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    rows = load(args.results)
    if not rows:
        print("no results yet")
        return 0

    schedule = next(
        (row["schedule"] for row in rows if isinstance(row.get("schedule"), dict)), {}
    )
    if schedule:
        print(
            f"schedule: {schedule['num_samples']} samples, "
            f"{schedule['num_steps']} steps, {schedule['num_recycles']} recycles, "
            "seed 101, one alignment shared by both sides\n"
        )

    by_key: dict[tuple, dict] = {
        (row["model"], row["case"], row["impl"]): row for row in rows
    }
    cases = sorted(
        {(row["length"], row["case"]) for row in rows}, key=lambda item: item[0]
    )
    models = sorted({row["model"] for row in rows})

    header = (
        "| tokens | model | FoldJAX s | upstream s | FoldJAX MiB | upstream MiB "
        "| speed | memory | FoldJAX confidence | upstream confidence |"
    )
    print(header)
    print("|" + "---|" * 10)
    for length, case in cases:
        for model in models:
            fj = by_key.get((model, case, "foldjax"))
            up = by_key.get((model, case, "upstream"))
            if fj is None and up is None:
                continue

            def cell(row, field, suffix=""):
                if row is None:
                    return "-"
                if row.get("failed") or row.get("returncode", 0) != 0:
                    return "failed"
                return f"{row[field]:,.0f}{suffix}"

            ratio_time = ratio_mem = "-"
            usable = (
                fj
                and up
                and not fj.get("failed")
                and not up.get("failed")
                and up.get("returncode", 0) == 0
            )
            if usable and up["wall_s"]:
                ratio_time = f"{up['wall_s'] / fj['wall_s']:.2f}x"
            if usable and fj["peak_mib"]:
                ratio_mem = f"{up['peak_mib'] / fj['peak_mib']:.2f}x"

            print(
                f"| {length} | {model} | {cell(fj, 'wall_s')} | {cell(up, 'wall_s')} "
                f"| {cell(fj, 'peak_mib')} | {cell(up, 'peak_mib')} "
                f"| {ratio_time} | {ratio_mem} "
                f"| {_confidence(fj) if fj else '-'} "
                f"| {_confidence(up) if up else '-'} |"
            )
    print(
        "\nspeed/memory are upstream divided by FoldJAX: above 1.00x means "
        "FoldJAX is faster / uses less."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
