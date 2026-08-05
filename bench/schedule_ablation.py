"""Split a run's wall time across trunk, diffusion and confidence.

Protenix and OpenDDE trace the whole prediction as one XLA program, so wrapping
Python functions says nothing about where the time goes -- there is only one
call. What still separates the stages is the schedule: recycles drive the trunk
alone, steps drive the diffusion alone, and samples drive the diffusion and the
confidence head together. Varying one at a time and reading the slope gives the
split without a profiler.

    time(recycles, steps, samples)
      ~ fixed + recycles*trunk + samples*steps*step + samples*confidence

Four runs determine the four terms. They are run one per process, warm, on an
idle card, for the same reason every other number here is.

    python -m bench.schedule_ablation --model protenix --case t1531
"""

from __future__ import annotations

import argparse
import json
import shutil
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
            used = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            return
        if not used or int(used[0]) <= threshold_mib:
            return
        time.sleep(5)
        waited += 5


def run_one(
    model: str,
    case: str,
    schedule: dict[str, int],
    work: Path,
    options: list[str],
    timeout_s: int,
) -> dict | None:
    """One measurement, in its own process, after a warmup of the same shape."""
    script = (
        "import os, json, time, sys\n"
        "os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')\n"
        "import foldjax\n"
        "from bench.spec import SEED, cases\n"
        "from foldjax.schema import PredictionRequest\n"
        "model, case_name, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "warm = sys.argv[4] == '1'\n"
        "schedule = json.loads(sys.argv[5])\n"
        "options = dict(entry.split('=', 1) for entry in sys.argv[6:])\n"
        "case = next(c for c in cases() if c.name == case_name)\n"
        "request = PredictionRequest(model=model, input=case.job, output_dir=out_dir,\n"
        "                            seed=SEED, options=options, **schedule)\n"
        "start = time.perf_counter()\n"
        "foldjax.predict(request)\n"
        "elapsed = time.perf_counter() - start\n"
        "import jax\n"
        "stats = jax.local_devices()[0].memory_stats() or {}\n"
        "if not warm:\n"
        "    print('MEASURE ' + json.dumps({'wall_s': round(elapsed, 2),\n"
        "          'peak_mib': round(stats.get('peak_bytes_in_use', 0) / 2**20, 1)}))\n"
    )
    payload = json.dumps(schedule)
    for warm in ("1", "0"):
        destination = work / ("warm" if warm == "1" else "measure")
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        gpu_idle()
        completed = subprocess.run(
            [
                sys.executable, "-c", script, model, case,
                str(destination), warm, payload, *options,
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if completed.returncode != 0:
            print(completed.stdout[-2000:])
            print(completed.stderr[-2000:], file=sys.stderr)
            return None
        for line in completed.stdout.splitlines():
            if line.startswith("MEASURE "):
                return json.loads(line[len("MEASURE "):])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", default="t1531")
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=int, default=5400)
    parser.add_argument(
        "--work", type=Path, default=Path("/tmp/claude-1000/schedule-ablation")
    )
    args = parser.parse_args()

    # One knob moves per row, and every row keeps the other two at the bench
    # schedule, so each difference names one term and nothing else.
    plan = [
        ("baseline", {"num_samples": 5, "num_steps": 200, "num_recycles": 10}),
        ("half_recycles", {"num_samples": 5, "num_steps": 200, "num_recycles": 5}),
        ("half_steps", {"num_samples": 5, "num_steps": 100, "num_recycles": 10}),
        ("one_sample", {"num_samples": 1, "num_steps": 200, "num_recycles": 10}),
    ]

    rows: dict[str, dict] = {}
    for name, schedule in plan:
        print(f"=== {args.model} {args.case} {name}: {schedule}", flush=True)
        result = run_one(
            args.model, args.case, schedule,
            args.work / name, args.option, args.timeout,
        )
        if result is None:
            print(f"  FAILED: {name}", flush=True)
            continue
        rows[name] = {**result, "schedule": schedule}
        print(f"  {result}", flush=True)

    report: dict[str, object] = {
        "model": args.model,
        "case": args.case,
        "options": args.option,
        "rows": rows,
    }

    base = rows.get("baseline")
    if base:
        seconds = base["wall_s"]
        terms: dict[str, float] = {}
        if "half_recycles" in rows:
            # 5 fewer recycles removed this much, so 10 of them cost twice it.
            terms["trunk_s"] = 2 * (seconds - rows["half_recycles"]["wall_s"])
        if "half_steps" in rows:
            terms["diffusion_s"] = 2 * (seconds - rows["half_steps"]["wall_s"])
        if "one_sample" in rows:
            # Dropping 4 of 5 samples removes 4/5 of everything per-sample --
            # the diffusion steps and the confidence head both.
            per_sample = (seconds - rows["one_sample"]["wall_s"]) / 4
            terms["per_sample_s"] = per_sample
            if "diffusion_s" in terms:
                terms["confidence_s"] = max(
                    0.0, 5 * per_sample - terms["diffusion_s"]
                )
        terms["unattributed_s"] = seconds - sum(
            value for key, value in terms.items()
            if key in {"trunk_s", "diffusion_s", "confidence_s"}
        )
        report["split"] = {key: round(value, 1) for key, value in terms.items()}
        print("\n=== split of the baseline wall time ===")
        print(f"  total          {seconds:>8.1f} s")
        for key, value in report["split"].items():
            share = 100 * value / seconds if seconds else 0
            print(f"  {key:<15}{value:>8.1f} s  ({share:.0f}%)")

    print()
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
