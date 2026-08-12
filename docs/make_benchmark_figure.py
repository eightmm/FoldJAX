"""Render the benchmark figure from the results directory.

One horizontal pair of bars per model and size — FoldJAX against the
repository it reimplements — wall time on the left, peak memory on the right.
The figure is generated from the same JSON records `bench/report.py` reads, so
it cannot drift from the table: re-run the bench, re-run this, and both tell
the same story. Failures are drawn as a hatched bar to the axis edge and
labelled OOM, because "did not run" is a measurement here, not missing data.

    uv run --with matplotlib python docs/make_benchmark_figure.py \
        --results ../foldjax-bench/results-lengths

Writes docs/benchmark-light.png and docs/benchmark-dark.png, the same pair the
banner keeps, picked by the reader's colour scheme.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ("boltz2", "protenix", "openfold3", "opendde", "alphafold3")
SIZES = (499, 1003, 3012)

#: JAX-tile palette from the banner: FoldJAX bars in the gradient's blue,
#: upstream in a neutral grey that reads in both themes.
FOLDJAX_COLOR = "#4a8fe7"
UPSTREAM_COLOR = "#9a9a9a"


def load(results: Path) -> dict[tuple[str, str, int], dict]:
    records: dict[tuple[str, str, int], dict] = {}
    for path in results.glob("*.json"):
        body = json.loads(path.read_text())
        records[(body["model"], body["impl"], body["length"])] = body
    return records


def render(records: dict, out: Path, *, dark: bool) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fg = "#e6e6e6" if dark else "#222222"
    bg = "#0d1117" if dark else "#ffffff"
    grid = "#30363d" if dark else "#dddddd"

    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 6.4), dpi=200, facecolor=bg, sharey=True
    )
    labels: list[str] = []
    positions: list[float] = []
    y = 0.0
    rows: list[tuple[float, str, int]] = []
    for size in SIZES:
        for model in MODELS:
            rows.append((y, model, size))
            labels.append(model)
            positions.append(y)
            y += 1.0
        y += 0.9  # gap between size groups

    for axis, metric, title, unit in (
        (axes[0], "wall_s", "wall time (5 samples · 200 steps · 10 recycles)", "s"),
        (axes[1], "peak_mib", "peak GPU memory", "GiB"),
    ):
        axis.set_facecolor(bg)
        limit = 0.0
        for row_y, model, size in rows:
            for impl, color, offset in (
                ("foldjax", FOLDJAX_COLOR, -0.19),
                ("upstream", UPSTREAM_COLOR, 0.19),
            ):
                body = records.get((model, impl, size))
                if body is None:
                    continue  # alphafold3 has no upstream column by design
                value = body.get(metric)
                failed = body.get("failed") or not value
                if not failed:
                    shown = value / 1024.0 if metric == "peak_mib" else value
                    limit = max(limit, shown)
                    axis.barh(
                        row_y + offset, shown, height=0.36, color=color, zorder=3
                    )
                else:
                    axis.barh(
                        row_y + offset,
                        1e9,  # clipped to the axis; the hatch says why
                        height=0.36,
                        color="none",
                        edgecolor=color,
                        hatch="///",
                        linewidth=0.8,
                        zorder=2,
                    )
                    axis.text(
                        0.99,
                        (row_y + offset),
                        "OOM",
                        transform=axis.get_yaxis_transform(),
                        ha="right",
                        va="center",
                        fontsize=8,
                        color=color,
                        zorder=4,
                    )
        if metric == "peak_mib":
            # The card the whole table ran on; the hatched bars died against
            # this line, and FoldJAX's tallest bar clears it with room.
            axis.axvline(97.9, color=fg, linewidth=0.9, linestyle="--", zorder=1)
            axis.text(
                97.9, -1.15, "97.9 GiB card", ha="center", fontsize=8, color=fg
            )
            axis.set_xlim(0, 104)
        else:
            axis.set_xlim(0, limit * 1.12)
        axis.set_title(title, color=fg, fontsize=11)
        axis.set_xlabel(unit, color=fg, fontsize=9)
        axis.tick_params(colors=fg, labelsize=9)
        axis.grid(axis="x", color=grid, linewidth=0.6, zorder=0)
        for spine in axis.spines.values():
            spine.set_color(grid)

    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    for row_y, model, size in rows:
        if model == MODELS[0]:
            axes[0].text(
                -0.16,
                row_y - 0.72,
                f"{size:,} tokens",
                transform=axes[0].get_yaxis_transform(),
                fontsize=10,
                fontweight="bold",
                color=fg,
                va="center",
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FOLDJAX_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=UPSTREAM_COLOR),
    ]
    fig.legend(
        handles,
        ["FoldJAX", "upstream"],
        loc="upper right",
        frameon=False,
        labelcolor=fg,
        fontsize=10,
        bbox_to_anchor=(0.99, 0.98),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    records = load(args.results)
    here = Path(__file__).parent
    for dark in (False, True):
        path = render(
            records,
            here / f"benchmark-{'dark' if dark else 'light'}.png",
            dark=dark,
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
