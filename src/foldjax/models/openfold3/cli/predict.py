"""Run a prediction and write structures and scores.

Completes the split the other CLIs already have: ``openfold3-jax-featurize`` needs
upstream's data stack, this needs only JAX and a checkpoint. Passing features as
``.npz`` is what makes that possible -- the two steps can run on different machines.

    openfold3-jax-predict ubq.npz --checkpoint of3_ft3_v1.pt -o out/
    openfold3-jax-predict ubq.npz --checkpoint w.pt -o out/ --samples 5 --steps 200
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openfold3-jax-predict",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("features", type=Path, help="feature .npz")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--name", default=None, help="defaults to the features' stem")
    parser.add_argument("--samples", type=int, default=None, help="released: 5")
    parser.add_argument("--steps", type=int, default=None, help="released: 200")
    parser.add_argument("--cycles", type=int, default=None, help="released: 4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prefix", default=None, help="checkpoint model root (default: auto-detect)"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="run eagerly; far slower, but skips a multi-minute compile",
    )
    parser.add_argument(
        "--cp-devices",
        type=int,
        default=1,
        help="shard the pair representations across this many JAX devices "
        "(context parallelism, the JAX form of OpenDDE's Fold-CP); needs "
        "that many visible devices and runs the XLA triangle kernels",
    )
    parser.add_argument(
        "--cp-layout",
        choices=("auto", "1d", "2d"),
        default="auto",
        help="how those devices are arranged: '1d' splits pair rows only, "
        "'2d' is Fold-CP's square grid and splits columns too (needs a square "
        "device count); 'auto' is '1d' until the grid has its own GPU numbers",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="persistent XLA cache directory (default: FoldJAX cache)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable the persistent XLA compilation cache",
    )
    parser.add_argument(
        "--all-arrays",
        action="store_true",
        help="write the per-bin PAE/PDE/distogram logits whatever their size. They "
        "are [samples, tokens, tokens, bins], so at 2000 tokens they are over "
        "50 GiB; by default they are omitted once the file would exceed 4 GiB",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="run the prediction this many times and report the steady-state "
        "median separately from the first call. The first call includes "
        "compilation, which at long sequences is minutes and is not what a "
        "throughput comparison against another implementation should measure",
    )
    return parser


def _report_peak_memory(jax: Any, config: Any) -> None:
    """Print peak device memory, and the chunk size that bounded it.

    Whether a long target fits is the question this CLI most often has to answer,
    and timing alone does not answer it. The peak is read from the allocator rather
    than from ``nvidia-smi``, which reports the pool: with the default
    ``XLA_PYTHON_CLIENT_PREALLOCATE`` the pool is reserved up front and does not move
    however much the program actually uses, which is how an earlier round of work
    here concluded that chunking changed nothing.

    Note what the number includes: compilation autotunes kernels by allocating real
    buffers, and the pool's high-water mark keeps that, so a first-call peak is not a
    steady-state peak. Run with ``--repeats`` and treat the figure as an upper bound.
    """
    device = jax.devices()[0]
    stats = getattr(device, "memory_stats", lambda: None)()
    if not stats or "peak_bytes_in_use" not in stats:
        return
    gib = stats["peak_bytes_in_use"] / 2**30
    chunk = (
        "no chunking"
        if config.pair_chunk_size is None
        else f"pair rows chunked at {config.pair_chunk_size}"
    )
    print(f"peak device memory {gib:.2f} GiB ({chunk})")
    if "bytes_limit" in stats and stats["bytes_limit"]:
        share = stats["peak_bytes_in_use"] / stats["bytes_limit"]
        if share > 0.9:
            print(
                f"  that is {share:.0%} of the device. A longer target will not fit "
                "at this setting; lower PAIR_SCORE_BUDGET_BYTES or pass a smaller "
                "pair chunk size."
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.no_cache and args.cache_dir is not None:
        parser.error("--cache-dir and --no-cache are mutually exclusive")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models.openfold3.bridge.checkpoint import load_checkpoint
    from foldjax.models.openfold3.bridge.torch_mapping import map_inference_params
    from foldjax.models.openfold3.data import (
        load_feature_archive,
        normalize_asym_ids,
        subsample_msa_rows,
    )
    from foldjax.models.openfold3.inference import (
        compile_predict,
        predict,
        released_config,
    )
    from foldjax.models.openfold3.output import (
        DEFAULT_ARRAY_BUDGET_BYTES,
        write_prediction_outputs,
    )

    try:
        raw, table, output_metadata = load_feature_archive(args.features)
    except (OSError, ValueError) as error:
        print(f"cannot load features: {error}")
        return 1

    n_token = raw["token_mask"].shape[-1]
    n_atom = raw["atom_mask"].shape[-1]
    overrides = {
        name: value
        for name, value in (
            ("num_samples", args.samples),
            ("no_rollout_steps", args.steps),
            ("num_cycles", args.cycles),
        )
        if value is not None
    }
    if args.cp_devices < 1:
        print("cp-devices must be positive")
        return 1
    if args.cp_devices > 1:
        if args.no_compile:
            print("context parallelism requires the compiled graph; drop "
                  "--no-compile or --cp-devices")
            return 1
        overrides["cp_shards"] = args.cp_devices
        overrides["cp_layout"] = args.cp_layout
    config = released_config(n_token=n_token, n_atom=n_atom, **overrides)
    # Keep a full alignment on the host only. At long sequences it can be
    # several GiB, so converting before this cut defeats the memory saving.
    raw = subsample_msa_rows(raw, config.msa_depth)
    raw, n_chain = normalize_asym_ids(raw)
    features = {name: jnp.asarray(value) for name, value in raw.items()}
    print(
        f"{n_token} tokens, {n_atom} atoms | {config.num_samples} samples x "
        f"{config.no_rollout_steps} steps, {config.num_cycles} cycles"
    )

    state = load_checkpoint(args.checkpoint)
    params = map_inference_params(state, args.prefix)
    print(
        f"mapped {len(params.trunk.pairformer_stack.blocks)} Pairformer / "
        f"{len(params.denoiser.diffusion_transformer.blocks)} diffusion blocks"
    )

    key = jax.random.key(args.seed)
    if args.no_compile:
        started = time.perf_counter()
        prediction = predict(
            key, features, params, config, table, n_chain=n_chain
        )
        jax.block_until_ready(prediction.coordinates)
        print(f"predicted in {time.perf_counter() - started:.1f}s (eager)")
    else:
        if not args.no_cache:
            from foldjax.models.openfold3.compilation import (
                enable_compilation_cache,
            )

            cache = enable_compilation_cache(args.cache_dir)
            print(f"persistent compilation cache {cache}")
        print("compiling (one-time, minutes at long sequences) ...")
        compiled = compile_predict(config, table, n_chain=n_chain)
        elapsed = []
        for _ in range(max(1, args.repeats)):
            started = time.perf_counter()
            prediction = compiled(key, features, params)
            jax.block_until_ready(prediction.coordinates)
            elapsed.append(time.perf_counter() - started)
        print(f"first call {elapsed[0]:.1f}s (includes compilation)")
        if len(elapsed) > 1:
            warm = sorted(elapsed[1:])
            median = warm[len(warm) // 2]
            print(
                f"warm {median:.2f}s median of {len(warm)} "
                f"(min {warm[0]:.2f}s, max {warm[-1]:.2f}s)"
            )

    _report_peak_memory(jax, config)

    coordinates = np.asarray(prediction.coordinates)
    if not np.isfinite(coordinates).all():
        print("non-finite coordinates; refusing to write a structure")
        return 1

    written = write_prediction_outputs(
        prediction,
        features,
        args.output,
        name=args.name or args.features.stem,
        max_array_bytes=None if args.all_arrays else DEFAULT_ARRAY_BUDGET_BYTES,
        output_metadata=output_metadata,
    )
    for path in written["structures"]:
        print(f"wrote {path}")
    print(f"wrote {written['scores']}")
    print(f"wrote {written['arrays']}")
    if written.get("omitted_arrays"):
        names = ", ".join(written["omitted_arrays"])
        print(
            f"  left out {names}: per-bin pair distributions are quadratic in token "
            "count, and at this size they would dominate the file. Pass "
            "--all-arrays to write them anyway."
        )
    return 0


def entrypoint() -> None:
    raise SystemExit(main())

if __name__ == "__main__":  # ``python -m foldjax.models.openfold3.cli.<name>``
    entrypoint()
