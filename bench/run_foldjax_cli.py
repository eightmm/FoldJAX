"""Measure the public FoldJAX CLI as one child process.

This is a separate ``cli_subprocess`` timing boundary.  It deliberately does
not redefine the historical in-process ``bench.run_foldjax`` wall time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from bench.provenance import (
    CURRENT_RESULT_SCHEMA,
    ArtifactFingerprintError,
    artifact_identity,
    benchmark_identity,
    device_identity,
    execution_identity,
    foldjax_checkpoint_paths,
    foldjax_implicit_asset_paths,
    portable_options,
    require_unchanged,
    runtime_identity,
    source_identity,
)
from bench.run_upstream import produced_structures


def cli_argv(args, case, schedule: dict[str, int], seed: int) -> list[str]:
    """Build the public ``foldjax predict`` invocation without importing JAX."""
    model = "protenix" if args.model == "protenix-v2" else args.model
    argv = [
        sys.executable,
        "-m",
        "foldjax.cli",
        "predict",
        "--model",
        model,
        "--input",
        str(case.job),
        "--weights",
        str(args.weights),
        "--output-dir",
        str(args.output_dir),
        "--seed",
        str(seed),
        "--num-samples",
        str(schedule["num_samples"]),
        "--num-steps",
        str(schedule["num_steps"]),
        "--num-recycles",
        str(schedule["num_recycles"]),
        "--json",
        "--quiet",
    ]
    if args.model == "protenix-v2":
        argv.extend(("--profile", "v2"))
    if args.cache_dir is not None:
        argv.extend(("--cache-dir", str(args.cache_dir)))
    for option in args.option:
        argv.extend(("--option", option))
    return argv


def cli_environment(peak_file: Path) -> dict[str, str]:
    """Set the JAX peak observer only in the measured child environment."""
    environment = dict(os.environ)
    environment["BENCH_PEAK_FILE"] = str(peak_file)
    environment["BENCH_PEAK_ENGINE"] = "jax"
    hook = str(Path(__file__).parent / "peakhook")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{hook}:{existing}" if existing else hook
    return environment


def _read_peak(path: Path) -> float | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return value / 2**20 if value >= 0 else None


def _stdout_samples(stdout: str) -> list[dict]:
    try:
        payload = json.loads(
            stdout,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError):
        return []
    rows = payload if isinstance(payload, list) else [payload]
    samples = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("samples"), list):
            continue
        for sample in row["samples"]:
            if isinstance(sample, dict) and isinstance(sample.get("scores"), dict):
                if any(
                    isinstance(value, float) and not math.isfinite(value)
                    for value in sample["scores"].values()
                ):
                    return []
                samples.append(sample)
    return samples


def _same_file(left: Path, right: Path) -> bool:
    """Return whether two existing paths refer to the same filesystem entry."""
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def run_cli_child(argv: list[str], environment: dict[str, str], timeout: int):
    """Run exactly one public CLI child and return its boundary evidence."""
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            cwd=Path(__file__).resolve().parent.parent,
            env=environment,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        return (
            completed.returncode,
            completed.stdout,
            completed.stderr,
            time.perf_counter() - started,
            None,
        )
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else error.stdout or ""
        )
        stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else error.stderr or ""
        )
        return (
            124,
            stdout,
            stderr,
            time.perf_counter() - started,
            f"FoldJAX CLI exceeded {timeout}s timeout",
        )
    except OSError as error:
        return (
            127,
            "",
            str(error),
            time.perf_counter() - started,
            f"cannot launch FoldJAX CLI: {error}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--num-recycles", type=int)
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--timing-state",
        choices=("cold-or-unspecified", "warm-after-successful-prefill"),
        default="cold-or-unspecified",
    )
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")

    from bench.spec import SCHEDULE, SEED, cases
    from foldjax.cli import _options

    case = next(item for item in cases() if item.name == args.case)
    seed = SEED if args.seed is None else args.seed
    schedule = dict(SCHEDULE)
    for name in ("num_samples", "num_steps", "num_recycles"):
        value = getattr(args, name)
        if value is not None:
            if value < 1:
                parser.error(f"--{name.replace('_', '-')} must be positive")
            schedule[name] = value
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be fresh and empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.json_out is not None:
        if args.json_out.exists():
            parser.error("--json-out must name a new file")
        if (
            _same_file(args.json_out, case.job)
            or _same_file(args.json_out, args.weights)
            or _is_within(args.json_out, args.output_dir)
            or (args.weights.is_dir() and _is_within(args.json_out, args.weights))
        ):
            parser.error("--json-out must be outside inputs, weights, and output-dir")
    peak_file = args.output_dir / "peak_bytes.txt"
    if peak_file.exists():
        peak_file.unlink()
    options = _options(args.option)
    weights = args.weights
    model = "protenix" if args.model == "protenix-v2" else args.model
    environment = cli_environment(peak_file)
    try:
        checkpoints = foldjax_checkpoint_paths(model, weights)
        assets = foldjax_implicit_asset_paths(model, weights, options=options)
        artifacts = artifact_identity(
            job=case.job, checkpoints=checkpoints, implicit_assets=assets
        )
        source = source_identity(Path(__file__).resolve().parent.parent)
        runtime = runtime_identity()
        device = device_identity(environment)
        execution = execution_identity(
            environment, timing_state=args.timing_state, traced=False
        )
    except ArtifactFingerprintError as error:
        parser.error(str(error))
    identity = benchmark_identity(
        impl="foldjax-cli",
        model=args.model,
        case=case.name,
        length=case.length,
        schedule=schedule,
        seed=seed,
        options=portable_options(options),
        artifacts=artifacts,
        source=source,
        runtime=runtime,
        # device_identity uses nvidia-smi and does not initialize a CUDA context.
        device=device,
        execution=execution,
    )
    argv = cli_argv(args, case, schedule, seed)
    returncode, stdout, stderr, elapsed, launch_error = run_cli_child(
        argv, environment, args.timeout
    )
    (args.output_dir / "cli_stdout.txt").write_text(stdout, encoding="utf-8")
    (args.output_dir / "cli_stderr.txt").write_text(stderr, encoding="utf-8")
    postflight_error = None
    try:
        require_unchanged(
            artifacts,
            artifact_identity(
                job=case.job,
                checkpoints=checkpoints,
                implicit_assets=foldjax_implicit_asset_paths(
                    model, weights, options=options
                ),
            ),
        )
        require_unchanged(
            source, source_identity(Path(__file__).resolve().parent.parent)
        )
        require_unchanged(runtime, runtime_identity())
        require_unchanged(device, device_identity(environment))
        require_unchanged(
            execution,
            execution_identity(
                environment, timing_state=args.timing_state, traced=False
            ),
        )
    except (ArtifactFingerprintError, RuntimeError) as error:
        postflight_error = str(error)
    peak = _read_peak(peak_file)
    structures = produced_structures(args.output_dir)
    samples = _stdout_samples(stdout)
    record = {
        "schema": CURRENT_RESULT_SCHEMA,
        "identity": identity,
        "impl": "foldjax-cli",
        "timing_scope": "cli_subprocess",
        "model": args.model,
        "case": case.name,
        "length": case.length,
        "schedule": schedule,
        "seed": seed,
        "options": portable_options(options),
        "artifacts": artifacts,
        "source": source,
        "runtime": runtime,
        "device": device,
        "execution": execution,
        "argv": argv,
        "wall_s": round(elapsed, 2),
        "peak_mib": None if peak is None else round(peak, 1),
        "returncode": returncode,
        "samples": samples,
        "public_manifest": str(args.output_dir / "foldjax_run.json")
        if (args.output_dir / "foldjax_run.json").is_file()
        else None,
    }
    if postflight_error:
        record.update(
            failed=True,
            reason="benchmark provenance changed during prediction",
            provenance_error=postflight_error,
        )
    elif returncode != 0:
        record.update(
            failed=True,
            reason=launch_error or f"FoldJAX CLI exited {returncode}",
            stderr_tail=stderr[-2000:],
        )
    elif peak is None:
        record.update(
            failed=True, reason="FoldJAX CLI produced no peak observer result"
        )
    elif not structures:
        record.update(
            failed=True,
            reason="exited 0 but produced no structures",
            stderr_tail=stderr[-2000:],
        )
    elif (
        len(structures) != schedule["num_samples"]
        or len(samples) != schedule["num_samples"]
    ):
        record.update(
            failed=True, reason="CLI outputs do not match requested sample count"
        )
    elif any(
        not sample["scores"]
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in sample["scores"].values()
        )
        for sample in samples
    ):
        record.update(
            failed=True, reason="CLI produced non-finite or empty sample scores"
        )
    text = json.dumps(record, allow_nan=False, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 1 if record.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
