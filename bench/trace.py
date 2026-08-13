"""Follow one prediction's GPU memory over time, and plot the traces together.

The benchmark tables report a peak, which is one number off a curve that has a
shape: weights loading, a trunk that climbs and holds, a sampler that spikes
once per step, a confidence head that scales with the sample count. Two models
with the same peak can spend very different amounts of the run near it, and
that is what decides whether a job fits on a card.

Two parts:

* `trace` runs a command and samples `nvidia-smi` alongside it. Sampling the
  *driver's* used-bytes rather than the framework's allocator is deliberate --
  it is the only figure both a JAX and a torch process report the same way, and
  the only one that includes what the framework does not know about. The cost
  is that a preallocating pool shows as a step to its full size, so traces are
  taken with `XLA_PYTHON_CLIENT_PREALLOCATE=false`, which the header records.
* `plot` draws every trace on one pair of axes.

    python -m bench.trace run --label protenix-upstream --out t.jsonl -- <cmd>
    python -m bench.trace plot --out docs/trace.png t1.jsonl t2.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

INTERVAL = 0.25


def trace(label: str, command: list[str], out: Path, interval: float) -> int:
    """Run `command`, sampling device memory until it exits."""
    out.parent.mkdir(parents=True, exist_ok=True)
    env_note = {"label": label, "interval_s": interval, "command": command}
    samples: list[tuple[float, int]] = []

    process = subprocess.Popen(command)
    start = time.perf_counter()
    while process.poll() is None:
        try:
            used = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split("\n")[0]
            samples.append((round(time.perf_counter() - start, 3), int(used)))
        except (subprocess.CalledProcessError, ValueError):
            pass  # a transient nvidia-smi failure must not kill the trace
        time.sleep(interval)

    with out.open("w") as handle:
        handle.write(json.dumps({"header": env_note}) + "\n")
        for seconds, mib in samples:
            handle.write(json.dumps({"t": seconds, "mib": mib}) + "\n")
    return process.returncode


def plot(paths: list[Path], out: Path) -> None:
    """Every trace on one pair of axes, light and dark."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = []
    for path in paths:
        lines = [json.loads(line) for line in path.read_text().splitlines() if line]
        label = lines[0]["header"]["label"]
        points = [(row["t"], row["mib"] / 1024) for row in lines[1:] if "t" in row]
        if points:
            series.append((label, points))

    for theme, face, text in (("light", "white", "#222"), ("dark", "#111", "#eee")):
        figure, axes = plt.subplots(figsize=(9, 5), facecolor=face)
        axes.set_facecolor(face)
        for label, points in series:
            axes.plot(
                [t for t, _ in points], [g for _, g in points], linewidth=1.6,
                label=label,
            )
        axes.set_xlabel("seconds", color=text)
        axes.set_ylabel("device memory in use (GiB)", color=text)
        axes.tick_params(colors=text)
        for spine in axes.spines.values():
            spine.set_color(text)
        axes.legend(frameon=False, labelcolor=text)
        axes.grid(alpha=0.2)
        target = out.with_name(f"{out.stem}-{theme}{out.suffix}")
        figure.tight_layout()
        figure.savefig(target, dpi=160, facecolor=face)
        plt.close(figure)
        print(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    runner = sub.add_parser("run")
    runner.add_argument("--label", required=True)
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument("--interval", type=float, default=INTERVAL)
    runner.add_argument("command", nargs=argparse.REMAINDER)

    plotter = sub.add_parser("plot")
    plotter.add_argument("--out", type=Path, required=True)
    plotter.add_argument("traces", nargs="+", type=Path)

    args = parser.parse_args()
    if args.mode == "run":
        command = [item for item in args.command if item != "--"]
        return trace(args.label, command, args.out, args.interval)
    plot(args.traces, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
