"""Benchmark upstream Boltz-2 warm inference on cached feature tensors."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
BOLTZ_SRC = REPO.parent / "boltz" / "src"
sys.path.insert(0, str(BOLTZ_SRC))

from boltz.main import (  # noqa: E402
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz2 import Boltz2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO.parent / "boltz" / ".cache/boltz/boltz2_conf.ckpt",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--recycling", type=int, default=3)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    device = torch.device("cuda")

    model = Boltz2.load_from_checkpoint(
        args.checkpoint,
        strict=True,
        predict_args={},
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        use_kernels=True,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(
            MSAModuleArgs(
                subsample_msa=True,
                num_subsampled_msa=1024,
                use_paired_feature=True,
            )
        ),
        steering_args=asdict(BoltzSteeringParams()),
    ).eval().to(device)
    obj = torch.load(args.features, map_location="cpu", weights_only=False)
    batch = {
        key: value.to(device)
        for key, value in obj.items()
        if not key.startswith("_") and torch.is_tensor(value)
    }

    stage_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
    if args.profile_stages:
        for name in (
            "input_embedder",
            "msa_module",
            "pairformer_module",
            "diffusion_conditioning",
            "distogram_module",
            "bfactor_module",
            "confidence_module",
        ):
            module = getattr(model, name, None)
            if module is None:
                continue

            def pre_hook(_module, _inputs, *, stage=name):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                stage_events.setdefault(stage, []).append((start, end))

            def post_hook(_module, _inputs, output, *, stage=name):
                stage_events[stage][-1][1].record()

            module.register_forward_pre_hook(pre_hook)
            module.register_forward_hook(post_hook)

        original_sample = model.structure_module.sample

        def timed_sample(*sample_args, **sample_kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            stage_events.setdefault("structure_sample", []).append((start, end))
            result = original_sample(*sample_args, **sample_kwargs)
            end.record()
            return result

        model.structure_module.sample = timed_sample

    @torch.inference_mode()
    def call():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(
                batch,
                recycling_steps=args.recycling,
                num_sampling_steps=args.steps,
                diffusion_samples=args.multiplicity,
                max_parallel_samples=5,
                run_confidence_sequentially=True,
            )

    start = time.perf_counter()
    out = call()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - start) * 1000
    for _ in range(1, args.warmup):
        out = call()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    stage_events.clear()
    times_ms = []
    for _ in range(args.iters):
        start = time.perf_counter()
        out = call()
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)

    coords = out["sample_atom_coords"].float().cpu().numpy()
    payload = {
        "features": str(args.features),
        "dtype": "bfloat16-mixed",
        "steps": args.steps,
        "recycling": args.recycling,
        "multiplicity": args.multiplicity,
        "n_tokens": int(batch["token_pad_mask"].shape[1]),
        "n_atoms": int(batch["atom_pad_mask"].shape[1]),
        "cold_ms": cold_ms,
        "warm_median_ms": statistics.median(times_ms),
        "warm_mean_ms": statistics.mean(times_ms),
        "warm_times_ms": times_ms,
        "memory_mib": {
            "allocated": torch.cuda.memory_allocated() / 1024**2,
            "peak_allocated": torch.cuda.max_memory_allocated() / 1024**2,
            "reserved": torch.cuda.memory_reserved() / 1024**2,
            "peak_reserved": torch.cuda.max_memory_reserved() / 1024**2,
        },
        "coords_finite": bool(np.isfinite(coords).all()),
    }
    if args.profile_stages:
        payload["stage_cuda_ms_total"] = {
            name: sum(start.elapsed_time(end) for start, end in events)
            for name, events in stage_events.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(args.output.with_suffix(".coords.npy"), coords)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
