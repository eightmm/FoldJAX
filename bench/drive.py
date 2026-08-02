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


def gpu_idle(threshold_mib: int = 2048, timeout_s: int = 300) -> None:
    waited = 0
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
            return
        time.sleep(5)
        waited += 5


def native_input(model: str, case, destination: Path) -> Path:
    """The job file this upstream reads, produced by FoldJAX's own translator.

    Both sides therefore run the same job naming the same alignment, rather
    than two inputs that are believed to say the same thing.
    """
    from foldjax.input import materialize_native_input
    from foldjax.registry import capabilities

    return materialize_native_input(
        case.job, capabilities(model), destination, seed=101
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

                started = time.perf_counter()
                completed = subprocess.run(
                    argv, cwd=REPO, capture_output=True, text=True,
                    env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
                    timeout=args.timeout + 600,
                )
                if completed.returncode != 0 or not row.is_file():
                    row.write_text(
                        json.dumps(
                            {
                                "impl": impl, "model": model, "case": case_name,
                                "length": case.length, "failed": True,
                                "wall_s": round(time.perf_counter() - started, 2),
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
