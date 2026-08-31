"""Measure one FoldJAX prediction, in a process that runs nothing else.

The process boundary is the measurement. `peak_bytes_in_use` is a
process-lifetime high-water mark, so two runs in one process report the larger
of the two and the second one's number is unknowable. Every row in the report
is therefore its own `python -m bench.run_foldjax`.

Matrix wall time is measured after a successful prefill. An eligible readable
persistent-cache entry can avoid compilation, but cache rejection or
deserialization failure can still recompile. Direct invocations are explicitly
recorded as cold-or-unspecified.
"""

from __future__ import annotations

import argparse
import json
import os
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
    foldjax_effective_environment,
    foldjax_implicit_asset_paths,
    portable_options,
    require_unchanged,
    reusable_result,
    runtime_identity,
    source_identity,
)

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
_oom = __import__("foldjax.oom", fromlist=["oom"])
# Never as a second spelling beside one the caller chose: jaxlib rejects the
# pair and falls back to the CPU, and its message blames a missing CUDA jaxlib.
_oom.set_mem_fraction(_oom.PREDICT_MEM_FRACTION)


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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the pinned bench seed. The table is measured at one "
        "seed so its rows are comparable; this exists to measure the spread "
        "that pinning hides -- how far two seeds move the same job.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="override the pinned diffusion sample count. The table is "
        "measured at one schedule so its rows are comparable; this exists to "
        "measure how a peak grows with the sample axis, which is a different "
        "question and gets its own runs.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="override diffusion steps for a targeted gate; ordinary benchmark "
        "rows keep the pinned schedule",
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="override recycles for a targeted gate; ordinary benchmark rows "
        "keep the pinned schedule",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="use an isolated persistent-cache root for a cache-behaviour gate",
    )
    parser.add_argument(
        "--timing-state",
        choices=("cold-or-unspecified", "warm-after-successful-prefill"),
        default="cold-or-unspecified",
    )
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    import foldjax
    from bench.spec import SCHEDULE, SEED, cases

    # Resolved once, then both used and reported. The report used to record the
    # pinned constants rather than these, so a `--seed` run wrote "seed": 101
    # whatever it had actually run -- a label that certifies the control it did
    # not check. Anything overridable has to be reported from the value that
    # reached the request.
    seed = SEED if args.seed is None else args.seed
    schedule = dict(SCHEDULE)
    if args.num_samples is not None:
        schedule["num_samples"] = args.num_samples
    if args.num_steps is not None:
        schedule["num_steps"] = args.num_steps
    if args.num_recycles is not None:
        schedule["num_recycles"] = args.num_recycles
    from foldjax.schema import PredictionRequest

    options = dict(entry.split("=", 1) for entry in args.option)
    case = next(item for item in cases() if item.name == args.case)
    # A benchmark row is a (weights, schedule) pair, so Protenix's two
    # supported checkpoints are two rows. FoldJAX spells the second as a
    # profile of one model rather than as a second model, and the bench label
    # has to survive that: `protenix-v2` names the row, `--profile v2` names
    # the weights.
    model, profile = args.model, None
    if model == "protenix-v2":
        model, profile = "protenix", "v2"

    request = foldjax.resolve_request(
        PredictionRequest(
            model=model,
            input=case.job,
            output_dir=args.output_dir,
            seed=seed,
            options=options,
            profile=profile,
            cache_dir=args.cache_dir,
            **schedule,
        )
    )
    assert request.weights is not None
    try:
        checkpoint_paths = foldjax_checkpoint_paths(model, Path(request.weights))
        implicit_asset_paths = foldjax_implicit_asset_paths(
            model,
            Path(request.weights),
            options=request.options,
        )
        artifacts = artifact_identity(
            job=Path(request.input),
            checkpoints=checkpoint_paths,
            implicit_assets=implicit_asset_paths,
        )
        source = source_identity(Path(__file__).resolve().parent.parent)
        runtime = runtime_identity()
        environment = foldjax_effective_environment(
            dict(os.environ),
            model=model,
            options=request.options,
        )
        device_identity_record = device_identity(environment)
        execution = execution_identity(
            environment,
            timing_state=args.timing_state,
            traced=args.traced,
        )
    except ArtifactFingerprintError as error:
        parser.error(str(error))
    portable = portable_options(options)
    identity = benchmark_identity(
        impl="foldjax",
        model=args.model,
        case=case.name,
        length=case.length,
        schedule=schedule,
        seed=seed,
        options=portable,
        artifacts=artifacts,
        source=source,
        runtime=runtime,
        device=device_identity_record,
        execution=execution,
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
    postflight_environment = foldjax_effective_environment(
        dict(os.environ),
        model=model,
        options=request.options,
    )
    require_unchanged(
        artifacts,
        artifact_identity(
            job=Path(request.input),
            checkpoints=checkpoint_paths,
            implicit_assets=foldjax_implicit_asset_paths(
                model,
                Path(request.weights),
                options=request.options,
            ),
        ),
    )
    require_unchanged(
        source,
        source_identity(Path(__file__).resolve().parent.parent),
    )
    require_unchanged(runtime, runtime_identity())
    require_unchanged(
        device_identity_record,
        device_identity(postflight_environment),
    )
    require_unchanged(
        execution,
        execution_identity(
            postflight_environment,
            timing_state=args.timing_state,
            traced=args.traced,
        ),
    )

    stats = device.memory_stats() or {}
    record = {
        "schema": CURRENT_RESULT_SCHEMA,
        "identity": identity,
        "impl": "foldjax",
        "model": args.model,
        "label": args.label or args.model,
        "options": portable,
        "artifacts": artifacts,
        "source": source,
        "runtime": runtime,
        "device": device_identity_record,
        "execution": execution,
        "case": case.name,
        "length": case.length,
        "schedule": schedule,
        "seed": seed,
        "wall_s": round(elapsed, 2),
        "peak_mib": round(stats.get("peak_bytes_in_use", 0) / 2**20, 1),
        "samples": [
            {"seed": sample.seed, "scores": sample.scores} for sample in result.samples
        ],
    }
    if not reusable_result(record, expected_identity=identity):
        record["failed"] = True
        record["reason"] = (
            "measurement record has missing or non-finite timing, peak, or "
            "sample scores"
        )
    if args.warmup:
        return 0
    text = json.dumps(record, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 1 if record.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
