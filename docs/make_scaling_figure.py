"""How wall time and peak memory scale with token count, per model.

The benchmark table answers "what did this cost" at three sizes. The question
this figure answers is a different one -- *how fast does the cost grow* -- and
the answer is a slope, not a number. Both axes are logarithmic, so a power law
is a straight line and its exponent is the slope: time proportional to n^2 rises
twice as steeply as n^1, and a model whose curve steepens between sizes is one
whose next size is worse than an extrapolation would predict.

The exponent printed in each panel is a least-squares fit through that series'
completed runs. It is a description of four points, not a law -- a fit over two
surviving points is labelled as such, because a line through two points has no
residual and looks exactly as confident as one through four.

Runs that did not complete are drawn at the top of the panel as an open marker,
not omitted: a size a model cannot reach is the most important thing a scaling
figure can say about it, and dropping the point would bend the curve into a
claim that it kept scaling.

    uv run --with matplotlib python docs/make_scaling_figure.py \
        --results ../foldjax-bench/results-lengths

Writes docs/scaling-light.png and docs/scaling-dark.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Cheapest first, matching the benchmark figure's row order.
MODELS = ("boltz2", "protenix", "openfold3", "opendde", "alphafold3")

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


def load(results: Path) -> dict:
    """{model: {impl: {"ok": [(tokens, wall, peak)], "failed": [tokens]}}}."""
    grouped: dict = {}
    for path in sorted(results.glob("*.json")):
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
            side["ok"].append((length, body["wall_s"], body["peak_mib"]))
    for entry in grouped.values():
        for side in entry.values():
            side["ok"].sort()
            side["failed"].sort()
    return grouped


def exponent(points: list[tuple[float, float]]) -> float | None:
    """The least-squares slope of log(y) against log(x): the scaling exponent."""
    if len(points) < 2:
        return None
    import math

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
        len(METRICS), len(models), figsize=(3.35 * len(models), 7.4), dpi=200,
        facecolor=bg, sharex=True, squeeze=False,
    )

    for row, (metric, ylabel, scale) in enumerate(METRICS):
        # Shared y within a row, independent between rows: the comparison a
        # reader makes is across models at one metric, never seconds against
        # gibibytes.
        top = max(
            (value[1 if metric == "wall_s" else 2] * scale
             for entry in grouped.values() for side in entry.values()
             for value in side["ok"]),
            default=1.0,
        )
        bottom = min(
            (value[1 if metric == "wall_s" else 2] * scale
             for entry in grouped.values() for side in entry.values()
             for value in side["ok"]),
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
                        [x for x, _ in points], [y for _, y in points],
                        color=color, linewidth=1.6, marker="o", markersize=4.5,
                        zorder=3,
                    )
                for tokens in side["failed"]:
                    # At the ceiling, hollow: the run reached this size and did
                    # not finish it. Placing it on the axis top says "off this
                    # chart" without inventing a value for it.
                    axis.plot(
                        [tokens], [top * 1.7], marker="o", markersize=6,
                        markerfacecolor="none", markeredgecolor=color,
                        markeredgewidth=1.3, zorder=4, clip_on=False,
                    )

            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(420, 3600)
            axis.set_ylim(bottom * 0.55, top * 2.4)
            axis.set_xticks([500, 1000, 2000, 3000])
            axis.set_xticklabels(["500", "1k", "2k", "3k"])
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
                text = f"n^{fitted:.2f}{caveat}"
                # A fit over the whole range is a bad summary of a curve that
                # bends, and these bend for a real reason: a per-run cost that
                # does not grow with n -- weight loading, compilation -- is a
                # constant added to a power law, and it flattens the fitted
                # slope at the small end. AlphaFold 3's 285 s at 499 tokens is
                # mostly that, and a lone "n^0.68" beside a curve that triples
                # its slope by 3,012 would be a wrong claim printed on top of
                # right data. So when the local slope at the large end departs
                # from the fit, both are shown and the arrow says which is
                # which.
                local = exponent(series[-2:])
                if local is not None and abs(local - fitted) > 0.25:
                    text += f" → {local:.2f}"
                axis.text(
                    0.04, 0.955 - offset, text,
                    transform=axis.transAxes, color=COLORS[theme][impl],
                    fontsize=9.5, va="top", ha="left", zorder=5,
                    fontweight="bold" if impl == "foldjax" else "normal",
                )
                offset += 0.085

    handles = [
        plt.Line2D([], [], color=COLORS[theme][impl], linewidth=2, marker="o",
                   markersize=4.5)
        for impl in ("foldjax", "upstream")
    ]
    handles.append(
        plt.Line2D([], [], color=THEMES[theme]["fg"], linewidth=0, marker="o",
                   markersize=6, markerfacecolor="none", markeredgewidth=1.3)
    )
    figure.legend(
        handles, ["FoldJAX", "upstream", "did not complete"],
        loc="upper right", frameon=False, labelcolor=fg, fontsize=9.5,
        ncol=3, bbox_to_anchor=(0.995, 1.0),
    )
    figure.suptitle(
        "cost against sequence length · 5 samples · 200 steps · 10 recycles · "
        "one RTX PRO 6000 (97.9 GiB)\n"
        "log-log, so the printed exponent is the slope: fitted over every "
        "completed size, → the local slope at the top end where it differs; "
        "* marks a fit over two surviving points",
        color=fg, fontsize=10.5, x=0.006, ha="left", va="top", y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.945))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=bg, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    grouped = load(args.results)
    here = Path(__file__).parent
    for theme in THEMES:
        print(render(grouped, here / f"scaling-{theme}.png", theme=theme))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
