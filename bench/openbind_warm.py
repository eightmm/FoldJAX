"""Ordinary-RNG, model-only warm calls; no tape or internal observers."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench.boltz_historical_replay import digest, save_new, source_hashes
from bench.openbind_core_replay import (
    compiler_environment,
    model_feature_batch,
    positive_count,
)
from bench.openbind_tape_adapter import prepare_core_features


def measure_calls(
    call, synchronize, observe, *, warm_repeats, clock=time.perf_counter, prepare=None
):
    """Exclude observation/serialization; release each result before the next call."""
    if (
        isinstance(warm_repeats, bool)
        or not isinstance(warm_repeats, int)
        or warm_repeats < 1
    ):
        raise ValueError("warm_repeats must be a positive integer")
    rows = []
    for index in range(warm_repeats + 1):
        if prepare is not None:
            prepare()
        start = clock()
        result = call()
        synchronize(result)
        elapsed = clock() - start
        observation = observe(result)
        del result
        rows.append(
            {
                "phase": "first" if index == 0 else "warm",
                "seconds": elapsed,
                **observation,
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("cueq", "cueq-full", "xla", "native-private"),
        required=True,
    )
    parser.add_argument("--warm-repeats", type=positive_count, default=3)
    args = parser.parse_args(argv)

    import jax

    from foldjax.models.openfold3.bridge.checkpoint import load_checkpoint
    from foldjax.models.openfold3.bridge.chemistry import representative_atom_table
    from foldjax.models.openfold3.bridge.torch_mapping import (
        map_inference_params,
        prune_sample_diffusion_aliases,
        resolve_model_prefix,
    )
    from foldjax.models.openfold3.data.featurize import prepare_msa_cycle_features
    from foldjax.models.openfold3.inference import compile_predict, released_config

    input_path = args.capture / "input.npz"
    effective_path = args.capture / "effective-model.json"
    effective = json.loads(effective_path.read_text())
    with np.load(input_path, allow_pickle=False) as data:
        batch = model_feature_batch(
            prepare_core_features(dict(data), max_atoms_per_token=23)
        )
    config = released_config(
        n_token=batch["token_mask"].shape[-1],
        n_atom=batch["atom_mask"].shape[-1],
        per_sample_token_cutoff=effective["settings"]["memory"]["eval"][
            "per_sample_token_cutoff"
        ],
    )
    # The CLI plans native MSA row selection per recycle from an ordinary
    # NumPy stream; direct predict refuses an over-depth MSA without it.
    batch = prepare_msa_cycle_features(
        batch,
        config.msa_depth,
        num_recycles=config.num_recycles,
        rng=np.random.default_rng(101),
    )
    identity = {
        "source": source_hashes(args.source_root),
        "checkpoint": digest(args.checkpoint),
        "input": digest(input_path),
        "effective": digest(effective_path),
        "harness": digest(Path(__file__)),
        "compiler_environment": compiler_environment(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    save_new(
        args.out_dir / "preflight.json",
        {
            **identity,
            "backend": args.backend,
            "config": config._asdict(),
            "rng": (
                "ordinary JAX key 101 reset per call; MSA cycle rows from "
                "numpy default_rng(101) once per process; no tape replay"
            ),
            "scope": (
                "resident-input model forward; excludes preprocessing, loading, "
                "transfers and output writing"
            ),
            "memory_scope": (
                "allocator process-lifetime peak including setup/compile; "
                "not reset warm-only peak"
            ),
        },
    )
    state = load_checkpoint(args.checkpoint)
    prefix = resolve_model_prefix(state, None)
    prune_sample_diffusion_aliases(state, prefix=prefix)
    params = jax.device_put(map_inference_params(state, prefix))
    del state
    batch = jax.device_put(batch)
    key = jax.random.key(101)
    jax.block_until_ready((params, batch, key))
    run = compile_predict(
        config, representative_atom_table(), triangle_kernel=args.backend
    )
    device = jax.local_devices()[0]
    public = []

    def observe(result):
        arrays = jax.device_get(
            {
                name: getattr(result, name)
                for name in ("coordinates", "plddt", "ptm", "iptm")
            }
        )
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("nonfinite warm output")
        public.append(arrays)
        return {"allocator": device.memory_stats()}

    rows = measure_calls(
        lambda: run(key, batch, params),
        jax.block_until_ready,
        observe,
        warm_repeats=args.warm_repeats,
    )
    archives = []
    for index, arrays in enumerate(public):
        path = args.out_dir / f"public-{index}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        archives.append(
            {
                "file": path.name,
                "sha256": digest(path),
                "equal_to_first": {
                    k: bool(np.array_equal(v, public[0][k])) for k, v in arrays.items()
                },
            }
        )
    if (
        source_hashes(args.source_root) != identity["source"]
        or digest(args.checkpoint) != identity["checkpoint"]
        or digest(input_path) != identity["input"]
        or digest(effective_path) != identity["effective"]
        or digest(Path(__file__)) != identity["harness"]
        or compiler_environment() != identity["compiler_environment"]
    ):
        raise RuntimeError("warm source/input/environment changed")
    save_new(
        args.out_dir / "finished.json",
        {
            "calls": rows,
            "archives": archives,
            "warm_median_seconds": float(np.median([r["seconds"] for r in rows[1:]])),
            "upstream_comparison": "not measured by this runner",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
