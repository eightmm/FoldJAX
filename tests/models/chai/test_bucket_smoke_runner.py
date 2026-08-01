from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parent / "scripts" / "benchmark_buckets.py"
    spec = importlib.util.spec_from_file_location("benchmark_buckets", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_minimal_token_counts_force_every_public_bucket(runner) -> None:
    assert runner.MINIMAL_TOKENS == {
        256: 1,
        384: 257,
        512: 385,
        768: 513,
        1024: 769,
        1536: 1025,
        2048: 1537,
    }
    for bucket, tokens in runner.MINIMAL_TOKENS.items():
        assert runner.bucket_for_tokens(tokens) == bucket


def test_bucket_processes_continue_after_failure(tmp_path: Path, runner) -> None:
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        bucket = int(Path(command[command.index("--fasta") + 1]).stem)
        if bucket == 384:
            return subprocess.CompletedProcess(command, 1, "", "out of memory")
        payload = {
            "device": "cuda:0",
            "model_size": bucket,
            "iterations": [{"seconds": 2.0}, {"seconds": 0.5}],
            "memory": {"peak_bytes_in_use": bucket},
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    records = runner.run_bucket_smokes(
        buckets=(256, 384, 512),
        bundle=tmp_path / "bundle",
        conformers=tmp_path / "conformers.npz",
        python=Path("python"),
        timeout_seconds=10,
        cuda_visible_devices="0",
        process_runner=run,
    )

    assert len(calls) == 3
    assert [record["status"] for record in records] == ["success", "failed", "success"]
    assert records[0]["cold_seconds"] == 2.0
    assert records[0]["warm_seconds"] == 0.5
    assert records[0]["peak_bytes_in_use"] == 256
    assert "out of memory" in records[1]["stderr_tail"]
