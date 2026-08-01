"""Run full Chai-JAX smoke inference for every public token bucket.

Each bucket runs in a fresh process so an OOM or compiler failure is isolated.
The child benchmark performs two identical executions: the first records cold
compile/execution time and the second records warm executable-cache latency.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

PUBLIC_BUCKETS = (256, 384, 512, 768, 1024, 1536, 2048)
MINIMAL_TOKENS = dict(
    zip(PUBLIC_BUCKETS, (1, 257, 385, 513, 769, 1025, 1537), strict=True)
)
_MEASURED_256_PEAK_BYTES = 4_592_000_000


def bucket_for_tokens(tokens: int) -> int:
    """Return the public Chai bucket selected for an actual token count."""
    return next(bucket for bucket in PUBLIC_BUCKETS if tokens <= bucket)


def _representative_tensor_bytes(bucket: int) -> int:
    """Size representative unfused feature/state tensors for risk planning."""
    depth = 16_384
    atoms = 23 * bucket
    blocks = atoms // 32
    return sum(
        (
            depth * bucket * 42 * 4,  # MSA feature concat, fp32
            depth * bucket * 64 * 2,  # projected MSA, bf16
            bucket * bucket * 163 * 4,  # token-pair feature concat, fp32
            bucket * bucket * 512 * 2,  # projected token-pair, bf16
            4 * bucket * bucket * 76 * 4,  # template feature concat, fp32
            4 * bucket * bucket * 64 * 2,  # projected templates, bf16
            bucket * bucket * 256 * 2,  # one trunk pair state, bf16
            atoms * 395 * 4,  # atom feature concat, fp32
            blocks * 32 * 128 * 14 * 4,  # blocked atom-pair concat, fp32
        )
    )


def memory_forecast(bucket: int) -> dict[str, int | str]:
    """Return a conservative heuristic, not a fit guarantee."""
    representative = _representative_tensor_bytes(bucket)
    baseline = _representative_tensor_bytes(256)
    return {
        "representative_tensor_bytes": representative,
        "heuristic_peak_bytes": _MEASURED_256_PEAK_BYTES
        + max(0, representative - baseline),
        "basis": "256-bucket measured peak plus representative tensor growth",
    }


def _tail(value: str | bytes | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def run_bucket_smokes(
    *,
    buckets: Sequence[int],
    bundle: Path,
    conformers: Path,
    python: Path,
    timeout_seconds: int,
    cuda_visible_devices: str,
    process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[dict[str, Any]]:
    """Run bucket workers sequentially and retain every success or failure."""
    repo = Path(__file__).resolve().parents[1]
    benchmark = repo / "scripts" / "benchmark_inference.py"
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="chai-jax-buckets-") as temporary:
        workspace = Path(temporary)
        for bucket in buckets:
            tokens = MINIMAL_TOKENS[bucket]
            fasta = workspace / f"{bucket}.fasta"
            fasta.write_text(
                f">protein|name=bucket_{bucket}\n{'A' * tokens}\n",
                encoding="utf-8",
            )
            command = [
                str(python),
                str(benchmark),
                "--fasta",
                str(fasta),
                "--bundle",
                str(bundle),
                "--conformers",
                str(conformers),
                "--recycles",
                "1",
                "--timesteps",
                "2",
                "--samples",
                "1",
                "--iterations",
                "2",
                "--compile-cache",
                str(workspace / f"compile_cache_{bucket}"),
                "--no-use-esm-embeddings",
            ]
            base: dict[str, Any] = {
                "bucket": bucket,
                "actual_tokens": tokens,
                "padded_atoms": 23 * bucket,
                "memory_forecast": memory_forecast(bucket),
            }
            try:
                completed = process_runner(
                    command,
                    cwd=repo,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                records.append(
                    {
                        **base,
                        "status": "timeout",
                        "timeout_seconds": timeout_seconds,
                        "stdout_tail": _tail(error.stdout or ""),
                        "stderr_tail": _tail(error.stderr or ""),
                    }
                )
                continue
            if completed.returncode != 0:
                records.append(
                    {
                        **base,
                        "status": "failed",
                        "returncode": completed.returncode,
                        "stdout_tail": _tail(completed.stdout),
                        "stderr_tail": _tail(completed.stderr),
                    }
                )
                continue
            try:
                payload = json.loads(completed.stdout)
                iterations = payload["iterations"]
                if len(iterations) != 2 or payload["model_size"] != bucket:
                    raise ValueError(
                        "worker returned the wrong bucket or iteration count"
                    )
                peak = payload.get("memory", {}).get("peak_bytes_in_use")
                if peak is None:
                    raise ValueError("worker did not report peak_bytes_in_use")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                records.append(
                    {
                        **base,
                        "status": "invalid_output",
                        "error": str(error),
                        "stdout_tail": _tail(completed.stdout),
                        "stderr_tail": _tail(completed.stderr),
                    }
                )
                continue
            records.append(
                {
                    **base,
                    "status": "success",
                    "device": payload.get("device"),
                    "cold_seconds": iterations[0]["seconds"],
                    "warm_seconds": iterations[1]["seconds"],
                    "peak_bytes_in_use": peak,
                    "coordinate_fingerprints": payload.get("coordinate_fingerprints"),
                    "stderr_tail": _tail(completed.stderr),
                }
            )
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--conformers", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--buckets", nargs="+", type=int, choices=PUBLIC_BUCKETS, default=PUBLIC_BUCKETS
    )
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    if args.plan_only:
        records = [
            {
                "bucket": bucket,
                "actual_tokens": MINIMAL_TOKENS[bucket],
                "padded_atoms": 23 * bucket,
                "memory_forecast": memory_forecast(bucket),
            }
            for bucket in args.buckets
        ]
    else:
        records = run_bucket_smokes(
            buckets=args.buckets,
            bundle=args.bundle,
            conformers=args.conformers,
            python=args.python,
            timeout_seconds=args.timeout_seconds,
            cuda_visible_devices=args.cuda_visible_devices,
        )
    payload = json.dumps({"buckets": records}, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return int(any(record.get("status") not in {None, "success"} for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
