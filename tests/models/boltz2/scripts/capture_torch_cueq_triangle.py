#!/usr/bin/env python3
"""Capture the upstream Torch cuEquivariance triangle output for parity."""

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

from boltz.model.layers.triangular_mult import kernel_triangular_mult  # noqa: E402


def _bf16(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.copy()).view(torch.bfloat16).cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()

    fixture = np.load(args.fixture)
    x = _bf16(fixture["x_bits"])
    mask = _bf16(fixture["mask_bits"])

    def value(module: str, name: str) -> torch.Tensor:
        return _bf16(fixture[f"{module}_{name}_bits"])

    weights = {
        "norm_in_weight": value("norm_in", "scale"),
        "norm_in_bias": value("norm_in", "bias"),
        "p_in_weight": value("p_in", "kernel").T.contiguous(),
        "g_in_weight": value("g_in", "kernel").T.contiguous(),
        "norm_out_weight": value("norm_out", "scale"),
        "norm_out_bias": value("norm_out", "bias"),
        "p_out_weight": value("p_out", "kernel").T.contiguous(),
        "g_out_weight": value("g_out", "kernel").T.contiguous(),
    }

    def call() -> torch.Tensor:
        return kernel_triangular_mult(
            x,
            direction="outgoing",
            mask=mask,
            **weights,
            eps=1e-5,
        )

    with torch.inference_mode():
        output = call()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            output = call()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e3)

    output_bits = output.cpu().view(torch.uint16).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output_bits)
    payload = {
        "warm_median_ms": statistics.median(times),
        "warm_times_ms": times,
        "finite": bool(torch.isfinite(output).all().item()),
    }
    args.metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
