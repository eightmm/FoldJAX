"""Run the native prediction CLI and persist wall/RSS/JAX allocator metrics."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import jax

from foldjax.models.opendde.cli.predict import main as predict_main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    args, predict_args = parser.parse_known_args()
    if not predict_args:
        parser.error("prediction arguments are required after --metrics PATH")

    started = time.perf_counter()
    predict_main(predict_args)
    jax.effects_barrier()
    elapsed_seconds = time.perf_counter() - started

    device = jax.devices()[0]
    memory_stats = device.memory_stats() or {}
    metrics = {
        "elapsed_seconds": elapsed_seconds,
        "device_kind": device.device_kind,
        "platform": device.platform,
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_peak_bytes_in_use": memory_stats.get("peak_bytes_in_use"),
        "gpu_peak_pool_bytes": memory_stats.get("peak_pool_bytes"),
        "gpu_largest_allocation_bytes": memory_stats.get("largest_alloc_size"),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("JAX_BENCHMARK " + json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
