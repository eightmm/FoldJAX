"""Render the benchmark figure from the results directory.

One horizontal pair of bars per model and size — FoldJAX against the repository
it reimplements — wall time on the left, peak memory on the right. The figure is
generated from the same JSON record format and source that `bench/report.py`
reads. The Markdown table is maintained separately, so regenerate or copy the
report and figure together and review their values for drift.

Two series, and only one of them is the subject, so this is the emphasis form:
FoldJAX in the accent, upstream in the de-emphasis gray. The ratio rides each
row as a direct label, because it is the number the reader would otherwise
compute by eye, and it is the only label on the row — a value beside every bar
would be thirty-six numbers nobody reads.

Bars are square-ended. A 4px pixel-space round on the data end is the house
mark spec, and it is omitted here rather than approximated: one panel is on a
log scale, where a data-space radius makes every bar a different shape and the
rounding starts reading as an encoding.

A size that produced no structure is a label at the axis edge, not a bar. A
failure has no magnitude; drawing it as a full-width hatch claims one, and the
reasons differ anyway — upstream Protenix refuses protenix-v2 above 2,560 tokens
before it allocates anything, which is not running out of memory.

    uv run --with matplotlib python docs/make_benchmark_figure.py \
        --results ../foldjax-bench/results-merged-20260826

That directory is the maintainer's external historical measurement archive; it
is not shipped in this repository, so this is not a clean-clone reproduction
recipe.

Writes docs/benchmark-light.png and docs/benchmark-dark.png, the pair the README
picks between by colour scheme.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_style import THEMES, apply_theme, did_not_run  # noqa: E402

MODELS = ("boltz2", "protenix", "protenix-v2", "openfold3", "opendde", "alphafold3")
SIZES = (499, 1003, 1354, 2096, 3012, 4926)
# `nvidia-smi` reported 97,887 MiB. Keep the plotted unit binary, matching the
# MiB-to-GiB conversion used for every result bar.
CARD_GIB = 97_887 / 1024


def load(results: Path) -> dict[tuple[str, str, int], dict]:
    records: dict[tuple[str, str, int], dict] = {}
    for path in results.glob("*.json"):
        body = json.loads(path.read_text())
        if "model" not in body:
            continue
        records[(body["model"], body["impl"], body["length"])] = body
    return records


def _value(body: dict | None, metric: str) -> float | None:
    if body is None or body.get("failed"):
        return None
    value = body.get(metric)
    return float(value) if value else None


def _absent_label(body: dict | None) -> str | None:
    """Why this cell has no bar, in the fewest words that stay true.

    Three outcomes, not two. `out of memory` used to be the default for
    anything that was not a refusal, which put a cause on cells that have no
    evidence for one: upstream Boltz-2 at 4,926 tokens reaches 92.2 GiB,
    produces nothing, exits 0 and writes no error anywhere. A peak that is
    close to the card is not proof of an allocation that failed, and labelling
    it as one is the figure asserting something the run never said.

    The refusal is read from the recorded text rather than from prose someone
    remembered to write: at 3,012 tokens the row carries a hand-written reason,
    while at 4,926 the same refusal only reaches `stderr_tail`.
    """
    if body is None or not body.get("failed"):
        return None
    evidence = f"{body.get('reason') or ''}\n{body.get('stderr_tail') or ''}"
    if "refuses" in evidence or "does not support n_token" in evidence:
        return "refused"
    if "out of memory" in evidence.lower() or "MemoryError" in evidence:
        return "out of memory"
    return "no structure"


def render(records: dict, out: Path, theme) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[tuple[str, str | None, int]] = []
    for size in SIZES:
        rows.append(("header", None, size))
        for model in MODELS:
            rows.append(("model", model, size))

    height = 0.34 * len(rows) + 1.6
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.5, height), sharey=True, dpi=200
    )
    apply_theme(figure, (left, right), theme)

    positions = {row: len(rows) - index for index, row in enumerate(rows)}
    offset = 0.23

    panels = ((left, "wall_s", "seconds", True), (right, "peak_mib", "GiB", False))
    for axis, metric, unit, log in panels:
        seen = [
            _value(records.get((m, i, s)), metric)
            for kind, m, s in rows
            if kind == "model"
            for i in ("foldjax", "upstream")
        ]
        top = max(v for v in seen if v)
        if metric == "peak_mib":
            top = max(top / 1024.0, CARD_GIB) * 1.06
            axis.set_xlim(0, top)
        else:
            axis.set_xscale("log")
            axis.set_xlim(20, top * 1.6)
        axis.grid(axis="x", color=theme.grid, linewidth=1.0, linestyle="-")
        axis.set_xlabel(unit, color=theme.muted, fontsize=8.5, labelpad=6)

    left.set_yticks([positions[row] for row in rows])
    left.set_yticklabels(
        [
            f"{size:,} tokens" if kind == "header" else f"   {model}"
            for kind, model, size in rows
        ]
    )
    for label, (kind, _m, _s) in zip(left.get_yticklabels(), rows):
        label.set_color(theme.ink if kind == "header" else theme.ink_secondary)
        label.set_fontweight("bold" if kind == "header" else "normal")
        label.set_fontsize(9.5 if kind == "header" else 9)
    left.set_ylim(0.3, len(rows) + 0.9)

    right.axvline(CARD_GIB, color=theme.limit, linewidth=1.0, zorder=1)
    right.text(
        CARD_GIB,
        len(rows) + 0.7,
        f"{CARD_GIB:.1f} GiB card ",
        color=theme.muted,
        fontsize=7.5,
        va="top",
        ha="right",
    )

    left.set_title(
        "wall time", color=theme.ink, fontsize=11, fontweight="bold", loc="left", pad=10
    )
    right.set_title(
        "peak GPU memory",
        color=theme.ink,
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    figure.suptitle(
        "FoldJAX against the repository it reimplements   ·   5 samples · 200 "
        "steps · 10 recycles · requested seed 101†",
        color=theme.ink_secondary,
        fontsize=9.5,
        x=0.008,
        ha="left",
        y=0.985,
    )

    handles = [
        plt.Line2D([], [], color=theme.accent, linewidth=7, solid_capstyle="round"),
        plt.Line2D([], [], color=theme.context, linewidth=7, solid_capstyle="round"),
    ]
    legend = figure.legend(
        handles,
        ["FoldJAX", "upstream"],
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.995, 1.0),
    )
    for text in legend.get_texts():
        text.set_color(theme.ink_secondary)

    figure.text(
        0.008,
        0.012,
        "bold figure on each row is upstream \u00f7 FoldJAX \u2014 above 1.00x "
        "means FoldJAX is faster, or uses less\n"
        "† historical OpenFold3 upstream rows used generated seed 2746317213; "
        "time/memory remain valid, confidence is not seed-matched",
        color=theme.muted,
        fontsize=7.5,
        ha="left",
    )
    figure.subplots_adjust(
        left=0.115, right=0.995, top=0.925, bottom=0.085, wspace=0.06
    )

    for axis, metric, _unit, _log in panels:
        floor = axis.get_xlim()[0]
        for kind, model, size in rows:
            if kind == "header":
                continue
            y = positions[(kind, model, size)]
            pair: list[float | None] = []
            for impl, colour, sign in (
                ("foldjax", theme.accent, +1),
                ("upstream", theme.context, -1),
            ):
                body = records.get((model, impl, size))
                value = _value(body, metric)
                if value is not None and metric == "peak_mib":
                    value = value / 1024.0
                pair.append(value)
                if value is None:
                    label = _absent_label(body)
                    if label:
                        did_not_run(axis, y + sign * offset, label, theme)
                    continue
                axis.barh(
                    y + sign * offset,
                    value - floor,
                    left=floor,
                    height=0.30,
                    color=colour,
                    edgecolor="none",
                    zorder=3,
                )
            fj, up = pair
            if fj and up:
                axis.text(
                    0.995,
                    y,
                    f"{up / fj:.2f}x",
                    transform=axis.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=8,
                    color=theme.ink_secondary,
                    fontweight="bold",
                    zorder=5,
                )

    figure.savefig(out, facecolor=theme.surface)
    plt.close(figure)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    records = load(args.results)
    here = Path(__file__).resolve().parent
    for theme in THEMES:
        print(render(records, here / f"benchmark-{theme.name}.png", theme))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
