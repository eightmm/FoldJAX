"""Measure one FoldJAX prediction, in a process that runs nothing else.

The process boundary is the measurement. `peak_bytes_in_use` is a
process-lifetime high-water mark, so two runs in one process report the larger
of the two and the second one's number is unknowable. Every row in the report
is therefore its own `python -m bench.run_foldjax`.

Wall time is measured warm: the caller runs the same case once to fill the
compile cache and discards it, then measures. A cold number is dominated by XLA
compilation, which is real but is paid once per shape, and reporting it as the
cost of a prediction would be wrong in both directions -- too slow for a served
model, too fast for a one-off.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Must precede the first JAX import: with preallocation on, the allocator grabs
# a fixed fraction of the card and `peak_bytes_in_use` still reports live bytes,
# but an OOM would be reported against the pool rather than the need.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="run for the compile cache only; report nothing",
    )
    args = parser.parse_args()

    import foldjax
    from bench.spec import SCHEDULE, SEED, cases
    from foldjax.schema import PredictionRequest

    case = next(item for item in cases() if item.name == args.case)
    request = PredictionRequest(
        model=args.model,
        input=case.job,
        output_dir=args.output_dir,
        seed=SEED,
        **SCHEDULE,
    )

    start = time.perf_counter()
    result = foldjax.predict(request)
    elapsed = time.perf_counter() - start

    import jax

    stats = jax.local_devices()[0].memory_stats() or {}
    record = {
        "impl": "foldjax",
        "model": args.model,
        "case": case.name,
        "length": case.length,
        "schedule": dict(SCHEDULE),
        "seed": SEED,
        "wall_s": round(elapsed, 2),
        "peak_mib": round(stats.get("peak_bytes_in_use", 0) / 2**20, 1),
        "samples": [
            {"seed": sample.seed, "scores": sample.scores} for sample in result.samples
        ],
    }
    if args.warmup:
        return 0
    text = json.dumps(record, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
