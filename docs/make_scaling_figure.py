"""How wall time and peak memory scale with token count.

Two figures, because they answer two questions.

`scaling-*.png` is one panel per model, FoldJAX against upstream, from the
length sweep alone. `upstream-scaling-*.png` puts the five reference
implementations on one pair of axes, where they can be read against each other,
and there it is safe to pool the earlier sweep's sizes as well: reference-side
code and configuration were held fixed between those sweeps, so their points
belong on the same curve. The FoldJAX column has no such licence -- its own code
changed between the two sweeps -- which is why the faceted figure does not pool
them.


The benchmark table answers "what did this cost" at six sizes. The question
this figure answers is a different one -- *how fast does the cost grow* -- and
the answer is a slope, not a number. Both axes are logarithmic, so a power law
is a straight line and its exponent is the slope: time proportional to n^2 rises
twice as steeply as n^1, and a model whose curve steepens between sizes is one
whose next size is worse than an extrapolation would predict.

The exponent printed in each panel is a least-squares fit through that series'
completed runs. It describes those completed points, not a law -- a fit over
two surviving points is labelled as such, because a line through two points has
no residual and can look more confident than the evidence warrants.

Runs that did not complete are drawn at the top of the panel as an open marker,
not omitted: a size a model cannot reach is the most important thing a scaling
figure can say about it, and dropping the point would bend the curve into a
claim that it kept scaling.

    uv run --with matplotlib python docs/make_scaling_figure.py \
        --results ../foldjax-bench/results-merged-20260826

That directory is the maintainer's external historical measurement archive; it
is not shipped in this repository, so this is not a clean-clone reproduction
recipe. The current checked-in upstream figure also adds `bench/results` via
`--upstream-extra`, as documented in `docs/benchmark.md`.

Writes docs/scaling-light.png and docs/scaling-dark.png.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

#: Cheapest first, matching the benchmark figure's row order.
MODELS = (
    "boltz2",
    "protenix",
    "protenix-v2",
    "openfold3",
    "opendde",
    "alphafold3",
)

#: Two series per panel is what lets this figure keep the JAX-tile blue against
#: a neutral upstream: five model hues on one pair of axes cannot clear the
#: all-pairs colour-vision floors (deuteranopia ΔE 1.6 at worst), so the models
#: are faceted instead of coloured -- the substitution the palette rules ask for
#: past three series. Both greys are stepped per theme to clear 3:1 on their own
#: surface.
COLORS = {
    "light": {"foldjax": "#4a8fe7", "upstream": "#7d7d7d"},
    "dark": {"foldjax": "#4a8fe7", "upstream": "#949494"},
}
THEMES = {
    "light": {"fg": "#222222", "bg": "#ffffff", "grid": "#dddddd"},
    "dark": {"fg": "#e6e6e6", "bg": "#0d1117", "grid": "#30363d"},
}

METRICS = (
    ("wall_s", "wall time (s)", 1.0),
    ("peak_mib", "peak GPU memory (GiB)", 1 / 1024),
)

# `nvidia-smi` reported 97,887 MiB. The memory series is converted from MiB to
# GiB, so the capacity line must use the same binary unit.
CARD_GIB = 97_887 / 1024

#: One hue per reference implementation, assigned in the palette's fixed order
#: and never cycled. Five series on one pair of axes clear the colour-vision
#: gates on the adjacent pairlist, which is the one that governs line charts --
#: but lines cross, so identity is carried three times over: hue, marker shape,
#: and a direct label at the end of every line. That also supplies the relief
#: the light surface needs, where aqua sits below 3:1 contrast.
UPSTREAM_COLORS = {
    "light": {
        "boltz2": "#2a78d6",
        "protenix": "#eb6834",
        "openfold3": "#1baf7a",
        "opendde": "#4a3aa7",
        "alphafold3": "#e34948",
    },
    "dark": {
        "boltz2": "#3987e5",
        "protenix": "#d95926",
        "openfold3": "#199e70",
        "opendde": "#9085e9",
        "alphafold3": "#e66767",
    },
}
MARKERS = {
    "boltz2": "o",
    "protenix": "s",
    "openfold3": "^",
    "opendde": "D",
    "alphafold3": "v",
}

#: Which column holds each model's own reference implementation.
#:
#: AlphaFold 3's is the `foldjax` one, and that is not a shortcut: FoldJAX
#: drives the official installation rather than reimplementing it, so that row
#: *is* upstream AlphaFold 3 running its own code. It is the same fact that
#: leaves AlphaFold 3's upstream column blank in the table -- both columns would
#: be the same program -- read from the other side.
REFERENCE_IMPL = {
    "boltz2": "upstream",
    "protenix": "upstream",
    "openfold3": "upstream",
    "opendde": "upstream",
    "alphafold3": "foldjax",
}

#: Series that could not reach their own released fast path on this card, drawn
#: dashed -- and only dashed. The caveat is a paragraph in `docs/benchmark.md`,
#: not a caption: a figure that has to explain itself on the canvas is one that
#: will be read without its explanation anyway. OpenFold3's upstream is the
#: only one: DS4Sci's evoformer attention
#: refuses to build for sm_120 and 0.3.1's experimental cuEquivariance flag
#: crashes at more than one diffusion sample, so it runs plain torch attention
#: chunked four rows at a time at every size. That is where its 6,722 s lives.
#: See `bench/upstream-environments.md`.
HANDICAPPED = {"openfold3"}


def load(results: list[Path]) -> dict:
    """{model: {impl: {"ok": [(tokens, wall, peak)], "failed": [tokens]}}}."""
    grouped: dict = {}
    for rank, directory in enumerate(results):
        for path in sorted(directory.glob("*.json")):
            body = json.loads(path.read_text())
            model, impl = body.get("model"), body.get("impl")
            length = body.get("length")
            if model is None or impl is None or length is None:
                continue
            side = grouped.setdefault(model, {}).setdefault(
                impl, {"ok": [], "failed": []}
            )
            if body.get("failed") or not body.get("wall_s") or not body.get("peak_mib"):
                side["failed"].append(length)
            else:
                side["ok"].append((length, body["wall_s"], body["peak_mib"], rank))
    for entry in grouped.values():
        for side in entry.values():
            # A size measured in both sweeps keeps the later run, and a size
            # that failed once and completed later is not a failure.
            side["ok"] = _one_run_per_size(side["ok"])
            completed = {value[0] for value in side["ok"]}
            side["failed"] = sorted(set(side["failed"]) - completed)
    return grouped


def _one_run_per_size(points: list[tuple], tolerance: float = 0.06) -> list[tuple]:
    """One point per size, and every point a single real run.

    The two sweeps used different proteins, and two of their sizes land within
    a few percent of each other -- 490 against 499, and 970 against 1,003. Drawn
    as separate points they overlap into one smeared marker; averaged they
    become a statistic this benchmark never measured, since nothing here is run
    more than once and a bar between two different proteins is not a standard
    deviation.

    So neither. Where the sweeps overlap in size the length sweep wins, and the
    earlier sweep contributes only the sizes the length sweep does not have --
    132 and 1,531 tokens. Every plotted point stays one run of one sequence.
    """
    kept: list[tuple] = []
    for point in sorted(points, key=lambda value: (value[0], value[3])):
        if kept and point[0] <= kept[-1][0] * (1 + tolerance):
            if point[3] < kept[-1][3]:
                kept[-1] = point
            continue
        kept.append(point)
    return kept


def growth(fitted: float) -> str:
    """The exponent as what it costs to double the input.

    `n^1.83` is the same fact as "2x the tokens costs 3.6x", and only one of
    them can be read without stopping to compute it.
    """
    return f"{2**fitted:.1f}x"


def exponent(points: list[tuple[float, float]]) -> float | None:
    """The least-squares slope of log(y) against log(x): the scaling exponent."""
    if len(points) < 2:
        return None
    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def render(grouped: dict, out: Path, *, theme: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = THEMES[theme]
    fg, bg, grid = style["fg"], style["bg"], style["grid"]
    models = [name for name in MODELS if name in grouped]

    figure, axes = plt.subplots(
        len(METRICS),
        len(models),
        figsize=(3.35 * len(models), 7.4),
        dpi=200,
        facecolor=bg,
        sharex=True,
        squeeze=False,
    )

    for row, (metric, ylabel, scale) in enumerate(METRICS):
        # Shared y within a row, independent between rows: the comparison a
        # reader makes is across models at one metric, never seconds against
        # gibibytes.
        top = max(
            (
                value[1 if metric == "wall_s" else 2] * scale
                for entry in grouped.values()
                for side in entry.values()
                for value in side["ok"]
            ),
            default=1.0,
        )
        bottom = min(
            (
                value[1 if metric == "wall_s" else 2] * scale
                for entry in grouped.values()
                for side in entry.values()
                for value in side["ok"]
            ),
            default=1.0,
        )
        for column, model in enumerate(models):
            axis = axes[row][column]
            axis.set_facecolor(bg)
            for impl in ("upstream", "foldjax"):
                side = grouped[model].get(impl)
                if side is None:
                    continue
                color = COLORS[theme][impl]
                points = [
                    (value[0], value[1 if metric == "wall_s" else 2] * scale)
                    for value in side["ok"]
                ]
                if points:
                    axis.plot(
                        [x for x, _ in points],
                        [y for _, y in points],
                        color=color,
                        linewidth=1.6,
                        marker="o",
                        markersize=4.5,
                        zorder=3,
                    )
                for tokens in side["failed"]:
                    # At the ceiling, hollow: the run reached this size and did
                    # not finish it. Placing it on the axis top says "off this
                    # chart" without inventing a value for it.
                    axis.plot(
                        [tokens],
                        [top * 1.7],
                        marker="o",
                        markersize=6,
                        markerfacecolor="none",
                        markeredgecolor=color,
                        markeredgewidth=1.3,
                        zorder=4,
                        clip_on=False,
                    )

            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(420, 5400)
            axis.set_ylim(bottom * 0.55, top * 2.4)
            axis.set_xticks([500, 1000, 2000, 3000, 5000])
            axis.set_xticklabels(["500", "1k", "2k", "3k", "5k"])
            # A log axis labels its minor ticks too, which puts "6x10^2" on top
            # of "500" at this width.
            axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            axis.grid(color=grid, linewidth=0.6, which="major", zorder=0)
            axis.tick_params(colors=fg, labelsize=9)
            for spine in axis.spines.values():
                spine.set_color(grid)
            if row == 0:
                axis.set_title(model, color=fg, fontsize=11.5, loc="left", pad=16)
            if row == len(METRICS) - 1:
                axis.set_xlabel("tokens", color=fg, fontsize=9)
            if column == 0:
                axis.set_ylabel(ylabel, color=fg, fontsize=9.5)

            # The exponent, one line per implementation, inside the panel. This
            # is the figure's actual answer -- the curve only shows that it is
            # a straight line -- so it is direct-labelled rather than legended.
            offset = 0.0
            for impl in ("foldjax", "upstream"):
                side = grouped[model].get(impl)
                if side is None or len(side["ok"]) < 2:
                    continue
                series = [
                    (value[0], value[1 if metric == "wall_s" else 2])
                    for value in side["ok"]
                ]
                fitted = exponent(series)
                if fitted is None:
                    continue
                caveat = "*" if len(series) == 2 else ""
                text = f"{growth(fitted)}{caveat}"
                # A fit over the whole range is a bad summary of a curve that
                # bends, and these bend for a real reason: a per-run cost that
                # does not grow with n -- weight loading, compilation -- is a
                # constant added to a power law, and it flattens the fitted
                # slope at the small end. AlphaFold 3's 302 s at 499 tokens is
                # mostly that, and a lone "n^0.68" beside a curve that triples
                # its slope by 3,012 would be a wrong claim printed on top of
                # right data. So when the local slope at the large end departs
                # from the fit, both are shown and the arrow says which is
                # which.
                local = exponent(series[-2:])
                if local is not None and abs(local - fitted) > 0.25:
                    text += f" → {growth(local)}"
                axis.text(
                    0.04,
                    0.955 - offset,
                    text,
                    transform=axis.transAxes,
                    color=COLORS[theme][impl],
                    fontsize=9.5,
                    va="top",
                    ha="left",
                    zorder=5,
                    fontweight="bold" if impl == "foldjax" else "normal",
                )
                offset += 0.085

    handles = [
        plt.Line2D(
            [], [], color=COLORS[theme][impl], linewidth=2, marker="o", markersize=4.5
        )
        for impl in ("foldjax", "upstream")
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            color=THEMES[theme]["fg"],
            linewidth=0,
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgewidth=1.3,
        )
    )
    figure.legend(
        handles,
        ["FoldJAX", "upstream", "did not run"],
        loc="upper right",
        frameon=False,
        labelcolor=fg,
        fontsize=9.5,
        ncol=3,
        bbox_to_anchor=(0.995, 1.0),
    )
    figure.suptitle(
        "cost against sequence length · 5 samples · 200 steps · 10 recycles · "
        f"one RTX PRO 6000 ({CARD_GIB:.1f} GiB)\n"
        "printed: what doubling the token count multiplies the cost by, over "
        "every completed size → at the top end where it differs; "
        "* marks a fit over two surviving points",
        color=fg,
        fontsize=10.5,
        x=0.006,
        ha="left",
        va="top",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.945))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=bg, bbox_inches="tight")
    plt.close(figure)
    return out


def render_upstream(grouped: dict, out: Path, *, theme: str) -> Path:
    """The five reference implementations against each other, on one axes.

    No prose on the canvas: the figure is meant to be read beside the text that
    explains it, not to carry that text. Names go in a legend, each with what
    doubling the token count multiplies that panel's cost by.

    OpenFold3's memory falling from 2,096 to 3,012 tokens is not a bad point.
    Upstream tunes its attention chunk size at runtime by binary-searching the
    largest chunk that does not raise (`chunk_utils.py:_determine_favorable_
    chunk_size`), so at 3,012 the large chunks it used at 2,096 no longer fit,
    it drops to a smaller one, and the peak comes down while the wall time goes
    from 2,410 s to 6,722 s. Memory traded for time, visible only if both
    panels are read together -- which is why interpolating that point to
    somewhere more plausible would delete the finding.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = THEMES[theme]
    fg, bg, grid = style["fg"], style["bg"], style["grid"]
    palette = UPSTREAM_COLORS[theme]
    models = [name for name in MODELS if name in palette and name in grouped]

    def reference(model):
        return grouped[model].get(REFERENCE_IMPL[model])

    figure, axes = plt.subplots(
        1, len(METRICS), figsize=(12.5, 5.4), dpi=200, facecolor=bg, squeeze=False
    )

    for column, (metric, ylabel, scale) in enumerate(METRICS):
        axis = axes[0][column]
        axis.set_facecolor(bg)
        index = 1 if metric == "wall_s" else 2
        top = max(
            (
                value[index] * scale
                for model in models
                for value in (reference(model) or {}).get("ok", [])
            ),
            default=1.0,
        )
        del column
        # The failures live in a band of their own above the data, so a reader
        # never has to decide whether a marker up there is a measurement.
        # Not all of them are out-of-memory: upstream Protenix refuses
        # protenix-v2 above 2,560 tokens by assertion, before allocating,
        # so the band says only that the size produced no structure.
        floor, ceiling = top * 1.06, top * 1.25
        axis.axhspan(floor, ceiling, color=grid, alpha=0.55, zorder=0, lw=0)
        axis.axhline(floor, color=grid, linewidth=1.0, zorder=1)
        axis.text(
            60,
            (floor + ceiling) / 2,
            "did not run",
            color=fg,
            fontsize=9,
            fontweight="bold",
            va="center",
            ha="left",
            zorder=5,
        )

        handles, labels = [], []
        for order, model in enumerate(models):
            side = reference(model)
            if side is None:
                continue
            color = palette[model]
            # One point per length, not per run: two sequences of the same
            # length are replicates of it, and drawing them as separate points
            # a couple of pixels apart made the curve look broken.
            points = [(value[0], value[index]) for value in side["ok"]]
            if points:
                (line,) = axis.plot(
                    [size for size, _ in points],
                    [value * scale for _, value in points],
                    color=color,
                    linewidth=1.8,
                    marker=MARKERS[model],
                    markersize=5.5,
                    zorder=3,
                    linestyle=(0, (5, 2)) if model in HANDICAPPED else "-",
                )
                fitted = exponent(points)
                handles.append(line)
                labels.append(
                    f"{model}" + (f"   {growth(fitted)} / 2x tokens" if fitted else "")
                )
            for slot, tokens in enumerate(side["failed"]):
                # Staggered by model: two implementations failing at the same
                # size would otherwise draw one marker on top of another, and
                # the figure would report one failure where there were two.
                axis.plot(
                    [tokens],
                    [floor + (ceiling - floor) * (order + 0.5) / len(models)],
                    marker=MARKERS[model],
                    markersize=6.5,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=1.4,
                    zorder=4,
                )
                del slot

        # Linear, not log-log. Log axes are for reading an exponent off a
        # slope; they turn "the cost explodes" into five tidy straight lines,
        # which is the opposite of what a growth plot is for. On linear axes
        # OpenFold3's upstream visibly runs away from the other four, which is
        # the finding.
        axis.set_xlim(0, 5250)
        axis.set_ylim(0, ceiling)
        axis.set_xticks([500, 1000, 2000, 3000, 4000, 5000])
        axis.set_xticklabels(["500", "1,000", "2,000", "3,000", "4,000", "5,000"])
        axis.set_xlabel("tokens", color=fg, fontsize=10)
        axis.set_ylabel(ylabel, color=fg, fontsize=10)
        axis.grid(color=grid, linewidth=0.6, zorder=0)
        axis.tick_params(colors=fg, labelsize=9.5)
        for spine in axis.spines.values():
            spine.set_color(grid)
        if metric == "peak_mib":
            axis.axhline(CARD_GIB, color=fg, linewidth=0.9, linestyle="--", zorder=1)
            axis.text(
                60,
                CARD_GIB - 1.5,
                f"{CARD_GIB:.1f} GiB card",
                color=fg,
                fontsize=9,
                va="top",
                ha="left",
            )
        # Upper left, tucked under the failure band: on linear axes every
        # curve leaves that corner empty, and the lower right -- free when
        # these were log-log -- is now where four of the five lines run.
        legend = axis.legend(
            handles,
            labels,
            loc="upper left",
            frameon=False,
            fontsize=9.5,
            labelcolor="linecolor",
            handlelength=1.6,
            borderpad=0.2,
            labelspacing=0.35,
            bbox_to_anchor=(0.015, 0.83),
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")

    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=bg, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="the length sweep; both figures read it",
    )
    parser.add_argument(
        "--upstream-extra",
        type=Path,
        nargs="*",
        default=[],
        help="further result directories, pooled into the upstream-only figure "
        "only. Safe there and not in the faceted one: reference-side code and "
        "configuration were held fixed between sweeps, while FoldJAX's own "
        "code changed.",
    )
    args = parser.parse_args()
    here = Path(__file__).parent
    faceted = load(args.results)
    pooled = load([*args.results, *args.upstream_extra])
    for theme in THEMES:
        print(render(faceted, here / f"scaling-{theme}.png", theme=theme))
        print(
            render_upstream(pooled, here / f"upstream-scaling-{theme}.png", theme=theme)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
