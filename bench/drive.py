"""Run the comparison matrix, one measurement per process, results as they land.

Each row is written to its own JSON file the moment it completes, so a matrix
that is interrupted -- or a model that fails on the largest case -- still leaves
every row that did finish. Nothing is held in memory across runs.

Between runs the card is allowed to go idle. A process exiting does not hand
its device memory back synchronously, so a job started immediately after a
heavy one can fail an allocation that would otherwise fit, and report it as the
model being too large.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def gpu_idle(threshold_mib: int = 2048, timeout_s: int = 300, settled: int = 3) -> None:
    """Wait for the previous run's memory to be back, and to stay back.

    One reading below the threshold is not enough. A process that has exited
    releases its memory to the driver over some interval, and `nvidia-smi` can
    report a small figure while the release is still in flight; the next run
    then asks for its pool -- 0.9 of the card, under the shipped setting -- and
    cannot get it. Requiring several consecutive quiet readings costs seconds
    and removes a failure that is indistinguishable from the model being too
    large for the card.

    Giving up is reported rather than silent: a run started on a busy card is
    still a measurement, but not of the thing the table claims.
    """
    waited = 0
    quiet = 0
    while waited < timeout_s:
        try:
            query = [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]
            used = subprocess.run(
                query, capture_output=True, text=True, timeout=30
            ).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            return
        if not used:
            return
        if int(used[0]) <= threshold_mib:
            quiet += 1
            if quiet >= settled:
                return
        else:
            quiet = 0
        time.sleep(5)
        waited += 5
    print(
        f"[warn] card still above {threshold_mib} MiB after {timeout_s}s; "
        "starting anyway",
        flush=True,
    )


def native_input(model: str, case, destination: Path) -> Path:
    """The job file this upstream reads, produced by FoldJAX's own translator.

    Both sides therefore run the same job naming the same alignment, rather
    than two inputs that are believed to say the same thing.
    """
    from foldjax.input import materialize_native_input
    from foldjax.registry import capabilities

    # Bench labels are (weights, schedule) pairs; FoldJAX's registry is keyed by
    # backend. Protenix's two supported checkpoints are two rows and one
    # backend, and they read the same input dialect, so the label resolves to
    # the backend before the translator is asked what it accepts.
    backend = "protenix" if model == "protenix-v2" else model
    return materialize_native_input(
        case.job, capabilities(backend), destination, seed=101
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--impls", nargs="+", default=["foldjax", "upstream"])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--skip-existing", action="store_true", help="leave finished rows alone"
    )
    parser.add_argument(
        "--traces",
        type=Path,
        help="also record device memory over time, one file per measurement. "
        "The warm-up run is never traced: it is discarded work, and tracing it "
        "would put a compile-bound curve next to the ones being compared.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    from bench.spec import cases as all_cases

    args.results.mkdir(parents=True, exist_ok=True)
    lookup = {case.name: case for case in all_cases()}

    for case_name in args.cases:
        case = lookup[case_name]
        for model in args.models:
            for impl in args.impls:
                row = args.results / f"{model}-{impl}-{case_name}.json"
                if args.skip_existing and row.is_file():
                    print(f"[skip] {row.name}", flush=True)
                    continue
                work = args.work / f"{model}-{impl}-{case_name}"
                subprocess.run(["rm", "-rf", str(work)], check=False)
                work.mkdir(parents=True, exist_ok=True)
                gpu_idle()
                print(f"[run ] {model} {impl} {case_name}", flush=True)

                if impl == "foldjax":
                    # Fill the compile cache first and throw the run away. A
                    # cold JAX run is dominated by XLA compilation, which the
                    # torch side has no equivalent of, so timing one against
                    # the other measures the compiler rather than the model.
                    # Compilation is paid once per shape and replayed from disk
                    # after that, which is what a second prediction costs.
                    warm = args.work / f"{model}-{impl}-{case_name}-warmup"
                    subprocess.run(["rm", "-rf", str(warm)], check=False)
                    warm.mkdir(parents=True, exist_ok=True)
                    print(f"[warm] {model} {case_name}", flush=True)
                    filled = subprocess.run(
                        [
                            str(REPO / ".venv/bin/python"), "-m", "bench.run_foldjax",
                            "--model", model, "--case", case_name,
                            "--output-dir", str(warm), "--warmup",
                        ],
                        cwd=REPO, capture_output=True, text=True,
                        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
                        timeout=args.timeout + 600,
                    )
                    # A warm-up that died is not a neutral event: the run it was
                    # supposed to prepare now pays compilation inside the
                    # measured time, and whatever killed it is likely to kill
                    # that run too. Say so, and keep the reason.
                    if filled.returncode != 0:
                        log = args.results / f"{model}-{impl}-{case_name}.warmup.txt"
                        log.write_text(filled.stderr or "")
                        print(
                            f"[warn] warm-up exited {filled.returncode}; "
                            f"reason in {log.name}",
                            flush=True,
                        )
                    subprocess.run(["rm", "-rf", str(warm)], check=False)
                    gpu_idle()
                    argv = [
                        str(REPO / ".venv/bin/python"), "-m", "bench.run_foldjax",
                        "--model", model, "--case", case_name,
                        "--output-dir", str(work), "--json-out", str(row),
                    ]
                else:
                    job = native_input(model, case, work / "input")
                    argv = [
                        str(REPO / ".venv/bin/python"), "-m", "bench.run_upstream",
                        "--model", model, "--case", case_name,
                        "--job", str(job), "--output-dir", str(work / "out"),
                        "--json-out", str(row), "--timeout", str(args.timeout),
                    ]

                if args.traces:
                    # A wrapper, not a different runner: the measurement below
                    # is byte-for-byte the one an untraced matrix runs, so a
                    # trace and its table row can never be of different things.
                    argv = [
                        str(REPO / ".venv/bin/python"), "-m", "bench.trace", "run",
                        "--model", model, "--impl", impl, "--case", case_name,
                        "--out",
                        str(args.traces / f"{model}-{impl}-{case_name}.jsonl"),
                        "--", *argv,
                    ]

                started = time.perf_counter()
                completed = subprocess.run(
                    argv, cwd=REPO, capture_output=True, text=True,
                    env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
                    timeout=args.timeout + 600,
                )
                if completed.returncode != 0 or not row.is_file():
                    # Keep all of it beside the row, not just the tail. An XLA
                    # allocation failure ends in a stack dump long enough to
                    # fill any tail on its own, so the line that says how many
                    # bytes were asked for -- the only part that identifies the
                    # failure -- is exactly what a truncated tail loses.
                    log = row.with_suffix(".stderr.txt")
                    log.write_text(completed.stderr or "")
                    row.write_text(
                        json.dumps(
                            {
                                "impl": impl, "model": model, "case": case_name,
                                "length": case.length, "failed": True,
                                "wall_s": round(time.perf_counter() - started, 2),
                                "stderr_bytes": len(completed.stderr or ""),
                                "stderr_path": str(log),
                                "stderr_tail": completed.stderr[-3000:],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    print(f"[FAIL] {model} {impl} {case_name}", flush=True)
                else:
                    print(f"[ ok ] {row.read_text().strip()[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
