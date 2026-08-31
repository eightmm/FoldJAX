"""Warm and measure one targeted optimization A/B arm in fresh processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_ARMS = {
    "af3-samples": (
        REPOSITORY_ROOT
        / "tests/models/alphafold3/scripts/run_sample_shard_end_to_end_arm.py",
        "--sample-arm",
        "alphafold3",
        frozenset(("batched", "sharded")),
    ),
    "opendde-relp": (
        REPOSITORY_ROOT
        / "tests/models/opendde/scripts/run_structural_relp_end_to_end_arm.py",
        "--relp-arm",
        "opendde",
        frozenset(("dense", "direct")),
    ),
    "openfold3-confidence": (
        REPOSITORY_ROOT
        / "tests/models/openfold3/scripts/run_confidence_schedule_end_to_end_arm.py",
        "--confidence-arm",
        "openfold3",
        frozenset(("batched", "serial")),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(_ARMS), required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()

    wrapper, arm_flag, expected_model, arms = _ARMS[args.kind]
    if args.model != expected_model:
        parser.error(
            f"{args.kind} only applies to model {expected_model!r}, not {args.model!r}"
        )
    if args.arm not in arms:
        parser.error(
            f"{args.kind} arm must be one of {', '.join(sorted(arms))}; "
            f"got {args.arm!r}"
        )
    run_name = f"{args.kind}-{args.arm}-{args.case}-n{args.num_samples}"
    run_root = args.work_root / run_name
    warm = run_root / "warm"
    measured = run_root / "measured"
    cache = run_root / "cache"
    result = run_root / "result.json"
    warm.mkdir(parents=True, exist_ok=True)
    measured.mkdir(parents=True, exist_ok=True)

    base = [
        sys.executable,
        str(wrapper),
        arm_flag,
        args.arm,
        "--model",
        args.model,
        "--case",
        args.case,
        "--num-samples",
        str(args.num_samples),
        "--cache-dir",
        str(cache),
        "--timing-state",
        "warm-after-successful-prefill",
    ]
    subprocess.run([*base, "--output-dir", str(warm), "--warmup"], check=True)
    subprocess.run(
        [
            *base,
            "--output-dir",
            str(measured),
            "--json-out",
            str(result),
            "--label",
            f"{args.kind}-{args.arm}",
        ],
        check=True,
    )
    print(result.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
