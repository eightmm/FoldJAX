#!/usr/bin/env python3
"""Capture upstream Torch Boltz-2 trunk tensors with deterministic MSA rows."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent / "boltz" / "src"))

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
    parser.add_argument("--recycling", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
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
    ).eval().cuda()
    obj = torch.load(args.features, map_location="cpu", weights_only=False)
    batch = {
        key: value.cuda()
        for key, value in obj.items()
        if not key.startswith("_") and torch.is_tensor(value)
    }
    captured = {}

    def capture(_module, _inputs, output):
        captured["s"] = output[0].detach()
        captured["z"] = output[1].detach()

    handle = model.pairformer_module.register_forward_hook(capture)
    original_randperm = torch.randperm

    def ordered_indices(n: int, *args, **kwargs):
        return torch.arange(n, device=kwargs.get("device"), dtype=torch.int64)

    torch.randperm = ordered_indices
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(
                batch,
                recycling_steps=args.recycling,
                num_sampling_steps=2,
                num_samples=1,
                max_parallel_samples=1,
                run_confidence_sequentially=True,
            )
        torch.cuda.synchronize()
    finally:
        torch.randperm = original_randperm
        handle.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        s=captured["s"].float().cpu().numpy(),
        z=captured["z"].float().cpu().numpy(),
    )
    print({key: tuple(value.shape) for key, value in captured.items()})


if __name__ == "__main__":
    main()
