"""Follow one prediction's device memory over time, and plot the traces.

The tables in `docs/benchmark.md` report a peak, which is one number off a curve
that has a shape: weights loading, a trunk that climbs and holds, a sampler that
spikes once per diffusion step, a confidence head that scales with the sample
count. Two models with the same peak can spend very different fractions of the
run near it, and that is what decides whether a job fits on a card.

The measurement is the one `bench/README.md` already defends -- live bytes,
sampled inside the measured process by `peakhook/benchtrace.py`, never
`nvidia-smi`'s reserved pool. So a curve's top is the same quantity as its
model's row in the table, and the trace needs nothing switched off to be read:
it is taken under the same preallocation the shipped command uses.

    python -m bench.trace run --model protenix --impl foldjax --case L1000_3og2 \
        --out traces/protenix-foldjax.jsonl -- <the measurement command>
    python -m bench.trace plot --out docs/memory-trace.png traces/*.jsonl

`run` is a thin wrapper: it puts the hook on PYTHONPATH, names a directory for
the samples, and runs the command it was given. It deliberately does not know
how to run a prediction -- `bench.run_foldjax` and `bench.run_upstream` already
do, and a second way to launch one is a second thing to keep true.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOOK = HERE / "peakhook"

#: The JAX-tile blue FoldJAX uses for itself, and a neutral grey for the
#: repository it reimplements -- upstream is the baseline being measured
#: against, not a peer category, which is why it is deliberately unsaturated.
#: Both greys are stepped per theme so each clears 3:1 on its own surface;
#: separation from the blue is ΔE 15+ for normal vision and 13+ under
#: protanopia/deuteranopia (`dataviz/scripts/validate_palette.js`).
COLORS = {
    "light": {"foldjax": "#4a8fe7", "upstream": "#7d7d7d"},
    "dark": {"foldjax": "#4a8fe7", "upstream": "#949494"},
}
THEMES = {
    "light": {"fg": "#222222", "bg": "#ffffff", "grid": "#dddddd"},
    "dark": {"fg": "#e6e6e6", "bg": "#0d1117", "grid": "#30363d"},
}

#: The order panels appear in, cheapest first, matching the benchmark figure.
MODELS = ("boltz2", "protenix", "openfold3", "opendde", "alphafold3")


def run(argv: dict, command: list[str], out: Path, interval: float) -> int:
    """Run `command` with the sampler installed, and keep what it wrote."""
    out.parent.mkdir(parents=True, exist_ok=True)
    shards = out.parent / f".{out.stem}-shards"
    shutil.rmtree(shards, ignore_errors=True)
    shards.mkdir(parents=True)

    environment = dict(os.environ)
    environment["BENCH_TRACE_DIR"] = str(shards)
    environment["BENCH_TRACE_INTERVAL"] = str(interval)
    # The upstream runners already prepend this directory for the peak hook;
    # prepending it here too covers the FoldJAX side and the case where this
    # wrapper drives something else. A directory holding only the hook cannot
    # shadow anything the command imports.
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{HOOK}:{existing}" if existing else str(HOOK)

    started = time.perf_counter()
    completed = subprocess.run(command, env=environment)
    wall = round(time.perf_counter() - started, 2)

    written = []
    for path in sorted(shards.glob("trace-*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        samples = [row for row in rows if "t" in row]
        if samples:
            written.append((rows, samples))
    if not written:
        # The command's own exit status, not a failure of our own: a trace is
        # an observer, and a prediction that succeeded must not be recorded as
        # a failed row because the instrument found nothing to watch. The
        # missing file is the report.
        print(f"[trace] no samples: {command[0]} never initialized a device")
        shutil.rmtree(shards, ignore_errors=True)
        return completed.returncode
    if len(written) > 1:
        # One measurement per process is this directory's invariant; two
        # sampling processes means the wrapper was pointed at a driver that
        # spawns them, and the larger one is a guess rather than a measurement.
        print(f"[trace] {len(written)} processes sampled; keeping the longest")
        written.sort(key=lambda pair: len(pair[1]), reverse=True)

    rows, samples = written[0]
    head = rows[0]
    footer = next((row for row in rows if "peak_bytes" in row), {})
    header = {
        **argv,
        "command": command,
        "returncode": completed.returncode,
        "wall_s": wall,
        "backend": head.get("backend"),
        "interval_s": head.get("interval_s"),
        "peak_bytes": footer.get("peak_bytes"),
    }
    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"header": header}) + "\n")
        for row in samples:
            handle.write(json.dumps(row) + "\n")
    shutil.rmtree(shards, ignore_errors=True)
    print(f"[trace] {out} · {len(samples)} samples · exit {completed.returncode}")
    return completed.returncode


def load(paths: list[Path]) -> dict[str, dict[str, dict]]:
    """Traces grouped by model, then by implementation."""
    grouped: dict[str, dict[str, dict]] = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        header = rows[0]["header"]
        points = [(row["t"], row["bytes"] / 2**30) for row in rows[1:] if "t" in row]
        if not points:
            continue
        grouped.setdefault(header["model"], {})[header["impl"]] = {
            "points": points,
            "peak_gib": (header.get("peak_bytes") or 0) / 2**30,
            "label": header.get("label"),
        }
    return grouped


def plot(grouped: dict, out: Path, case: str) -> None:
    """One panel per model, FoldJAX against upstream, on shared axes.

    Shared axes rather than per-panel ones: the question a reader brings here is
    how much of the card a model holds and for how long, which is a comparison
    between panels. A model whose curve is a flat line near the floor is the
    finding, not a panel to be rescaled until it looks busy.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [name for name in MODELS if name in grouped]
    models += [name for name in sorted(grouped) if name not in MODELS]
    span = max(
        (point[0] for entry in grouped.values() for side in entry.values()
         for point in side["points"]),
        default=1.0,
    )
    ceiling = max(
        (max(side["peak_gib"], max(p[1] for p in side["points"]))
         for entry in grouped.values() for side in entry.values()),
        default=1.0,
    )

    for theme, style in THEMES.items():
        fg, bg, grid = style["fg"], style["bg"], style["grid"]
        columns = 3
        rows = (len(models) + columns - 1) // columns
        figure, axes = plt.subplots(
            rows, columns, figsize=(13, 4.1 * rows), dpi=200,
            facecolor=bg, sharex=True, sharey=True, squeeze=False,
        )
        flat = [axis for row in axes for axis in row]

        for axis, model in zip(flat, models):
            axis.set_facecolor(bg)
            axis.set_title(model, color=fg, fontsize=11, loc="left")
            for impl in ("upstream", "foldjax"):  # FoldJAX drawn on top
                side = grouped[model].get(impl)
                if side is None:
                    continue
                color = COLORS[theme][impl]
                axis.plot(
                    [t for t, _ in side["points"]],
                    [g for _, g in side["points"]],
                    linewidth=1.4, color=color, zorder=3,
                )
                # The sampler is a 4 Hz observer, so its largest sample is a
                # lower bound on the peak. The allocator's high-water mark --
                # the number in the table -- is drawn as its own rule, so the
                # figure cannot understate what the run actually held.
                if side["peak_gib"]:
                    axis.axhline(
                        side["peak_gib"], color=color, linewidth=0.8,
                        linestyle=(0, (4, 3)), zorder=2, alpha=0.85,
                    )
                    # Left-aligned: every curve here starts at the floor and
                    # climbs, so the upper left is the one region of the panel
                    # a peak rule's label cannot land on top of the data.
                    # Above its own rule, and nudged clear of the other
                    # series' label when the two peaks are close enough to
                    # collide -- which they are wherever the port is near its
                    # upstream, i.e. exactly the panels worth reading.
                    other = grouped[model].get(
                        "upstream" if impl == "foldjax" else "foldjax", {}
                    ).get("peak_gib", 0)
                    clash = other and abs(other - side["peak_gib"]) < ceiling * 0.07
                    axis.text(
                        0.012,
                        side["peak_gib"] + ceiling * (0.055 if clash and
                                                      side["peak_gib"] > other
                                                      else 0.015),
                        f"{side['peak_gib']:.1f} GiB peak",
                        transform=axis.get_yaxis_transform(),
                        ha="left", va="bottom", fontsize=8, color=fg, zorder=4,
                    )
            axis.grid(color=grid, linewidth=0.6, zorder=0)
            axis.tick_params(colors=fg, labelsize=9)
            for spine in axis.spines.values():
                spine.set_color(grid)
        for axis in flat[len(models):]:
            axis.set_visible(False)

        for index, axis in enumerate(flat[:len(models)]):
            if index % columns == 0:
                axis.set_ylabel("device memory in use (GiB)", color=fg, fontsize=9)
            # `sharex` hides the tick labels of every panel that has one below
            # it -- including the ones whose lower neighbour is an empty cell,
            # which would leave a column with no time axis at all.
            if index + columns >= len(models):
                axis.set_xlabel("seconds", color=fg, fontsize=9)
                axis.tick_params(labelbottom=True)
        flat[0].set_xlim(0, span * 1.02)
        flat[0].set_ylim(0, ceiling * 1.12)

        handles = [
            plt.Line2D([], [], color=COLORS[theme][impl], linewidth=2)
            for impl in ("foldjax", "upstream")
        ]
        figure.legend(
            handles, ["FoldJAX", "upstream"], loc="upper right", frameon=False,
            labelcolor=fg, fontsize=10, bbox_to_anchor=(0.995, 0.995),
        )
        figure.suptitle(
            f"live device memory through one prediction · {case} · "
            "5 samples · 200 steps · 10 recycles",
            color=fg, fontsize=11.5, x=0.008, ha="left",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.955))
        target = out.with_name(f"{out.stem}-{theme}{out.suffix}")
        figure.savefig(target, facecolor=bg, bbox_inches="tight")
        plt.close(figure)
        print(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    runner = sub.add_parser("run")
    runner.add_argument("--model", required=True)
    runner.add_argument("--impl", required=True, choices=("foldjax", "upstream"))
    runner.add_argument("--case", required=True)
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument("--interval", type=float, default=0.25)
    runner.add_argument("command", nargs=argparse.REMAINDER)

    plotter = sub.add_parser("plot")
    plotter.add_argument("--out", type=Path, required=True)
    plotter.add_argument("--case", default="")
    plotter.add_argument("traces", nargs="+", type=Path)

    args = parser.parse_args()
    if args.mode == "run":
        command = [item for item in args.command if item != "--"]
        if not command:
            parser.error("nothing to run: put the command after --")
        return run(
            {
                "label": f"{args.model}-{args.impl}",
                "model": args.model,
                "impl": args.impl,
                "case": args.case,
            },
            command,
            args.out,
            args.interval,
        )
    plot(load(args.traces), args.out, args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
