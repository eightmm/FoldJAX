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
import sys
import time
from pathlib import Path

# Preallocation is left at JAX's default -- on -- because that is what `foldjax
# predict` runs, and a benchmark that turns it off is not measuring the product.
#
# This used to force it off, for a better OOM message: with the pool
# preallocated, a failure is reported against the pool rather than against what
# the program needed. That reason was real and the cost was not worth it. Off,
# the allocator grows on demand and a single large contiguous request can fail
# against a carved-up address space even when the total is there: Boltz-2 at
# 3,012 tokens dies asking for its 41.67 GiB arena, and completes with
# preallocation on at a 73.5 GiB peak. The benchmark was reporting a size the
# product runs as one it cannot.
#
# The peak is unaffected, which is what made the old choice look free. Measured
# at 1,003 tokens: 10,214 MiB off against 10,024 MiB on, both `peak_bytes_in_use`
# and neither anywhere near the 85.5 GiB pool -- so this reports live bytes
# either way.

# And the same pool fraction `foldjax predict` gives itself. This harness calls
# the Python API, and `PREDICT_MEM_FRACTION` is applied by `foldjax/cli.py`
# alone -- deliberately, since a library must not resize a host application's
# pool on import. The consequence for a benchmark is that it was measuring the
# port at JAX's 0.75 default while the shipped command runs at 0.9: a quarter of
# the card reserved and unused. At 3,012 tokens that is the difference between
# `jit_run_model`'s 73.5 GiB fitting and missing a 71.2 GiB pool by 0.6, and it
# was reported as OOM in a table next to an upstream that had the whole card.
# `oom.py` had already recorded this exact case; the benchmark had not.
os.environ.setdefault(
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    str(__import__("foldjax.oom", fromlist=["oom"]).PREDICT_MEM_FRACTION),
)


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
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="backend option, e.g. compute_dtype=bfloat16. Used to match a "
        "precision the upstream runs by default and FoldJAX does not",
    )
    parser.add_argument("--label", help="tag for this row, defaults to the model")
    args = parser.parse_args()

    import foldjax
    from bench.spec import SCHEDULE, SEED, cases
    from foldjax.schema import PredictionRequest

    options = dict(entry.split("=", 1) for entry in args.option)
    case = next(item for item in cases() if item.name == args.case)
    request = PredictionRequest(
        model=args.model,
        input=case.job,
        output_dir=args.output_dir,
        seed=SEED,
        options=options,
        **SCHEDULE,
    )

    import jax

    device = jax.local_devices()[0]
    if not args.warmup:
        # The same live-bytes quantity the peak below reports, sampled as the
        # run goes. `bytes_in_use` is unaffected by preallocation -- the pool is
        # reserved address space, not an allocation -- so the trace is taken
        # under exactly the configuration the table rows ran under.
        sys.path.insert(0, str(Path(__file__).parent / "peakhook"))
        import benchtrace

        benchtrace.start(
            lambda: (device.memory_stats() or {}).get("bytes_in_use", 0),
            lambda: (device.memory_stats() or {}).get("peak_bytes_in_use", 0),
            backend="jax",
        )

    start = time.perf_counter()
    result = foldjax.predict(request)
    elapsed = time.perf_counter() - start

    stats = device.memory_stats() or {}
    record = {
        "impl": "foldjax",
        "model": args.model,
        "label": args.label or args.model,
        "options": options,
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
