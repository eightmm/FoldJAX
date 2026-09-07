"""Replay saved native Linear operands in an existing Torch runtime.

This is an operator control, not a model or performance benchmark. Archives
preserve values, not original CUDA strides; old-runtime replay is necessary
before attributing a difference to the new runtime.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    records = json.loads((args.native / "linear-policy.json").read_text())
    if not records:
        raise ValueError("no captured operators")
    versions = {}
    for name in ("torch", "numpy", "nvidia-cublas", "nvidia-cublas-cu12"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    with torch.inference_mode():
        for record in records:
            path = args.native / record["file"]
            with np.load(path, allow_pickle=False) as archive:
                values = dict(archive)
            operands = [values["x"], values["weight"]]
            if "bias" in values:
                operands.append(values["bias"])
            if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in operands):
                raise ValueError("expected finite FP32 operands")
            x, weight = (torch.from_numpy(v.copy()).cuda() for v in operands[:2])
            bias = (
                torch.from_numpy(operands[2].copy()).cuda()
                if len(operands) == 3
                else None
            )
            arrays, comparisons = {}, {}
            for enabled in (False, True):
                torch.backends.cuda.matmul.allow_tf32 = enabled
                result = torch.nn.functional.linear(x, weight, bias).cpu().numpy()
                key = "tf32" if enabled else "fp32"
                arrays[key] = result
                comparisons[key] = {}
                for reference in ("y", "fp32_control"):
                    error = result.astype(np.float64) - values[reference]
                    comparisons[key][reference] = {
                        "max_abs": float(np.max(abs(error))),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "unequal": int(np.count_nonzero(error)),
                    }
            np.savez_compressed(args.out / record["file"], **arrays)
            record.update(capture_sha256=sha(path), comparisons=comparisons)
            print(json.dumps(record), flush=True)
    save(
        args.out / "report.json",
        {
            "scope": __doc__,
            "versions": versions,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "capture_provenance_sha256": sha(args.native / "provenance.json"),
            "wrapper_sha256": sha(Path(__file__)),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
