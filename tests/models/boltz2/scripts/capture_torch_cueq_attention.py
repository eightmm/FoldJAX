#!/usr/bin/env python3
"""Capture upstream Torch cuEquivariance triangle-attention output."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent / "boltz" / "src"))

from boltz.model.layers.cue_runtime import preload_nvidia_cu13_libs  # noqa: E402


def _bf16(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.copy()).view(torch.bfloat16).cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()

    preload_nvidia_cu13_libs()
    from cuequivariance_torch.primitives.triangle import triangle_attention

    fixture = np.load(args.fixture)
    q = _bf16(fixture["q_bits"])
    k = _bf16(fixture["k_bits"])
    v = _bf16(fixture["v_bits"])
    bias = _bf16(fixture["bias_bits"]).float()
    mask = torch.from_numpy(fixture["mask"]).cuda()
    scale = float(fixture["scale"])

    def call() -> torch.Tensor:
        return triangle_attention(q, k, v, bias, mask, scale)

    with torch.inference_mode():
        output = call()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            output = call()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output.cpu().view(torch.uint16).numpy())
    payload = {
        "warm_median_ms": statistics.median(times),
        "warm_times_ms": times,
        "finite": bool(torch.isfinite(output).all().item()),
    }
    args.metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
