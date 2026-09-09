"""Real-weight native input-encoder dtype trace; diagnostic, not admission."""

import argparse
import json
from pathlib import Path


def main():
    import numpy as np
    import torch
    from protenix.model.modules.embedders import InputFeatureEmbedder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("native CUDA autocast observation requires GPU")
    state = torch.load(args.weights, map_location="cpu", mmap=True, weights_only=True)
    prefix = "module.input_embedder."
    model = InputFeatureEmbedder()
    model.load_state_dict(
        {
            k.removeprefix(prefix): v
            for k, v in state["model"].items()
            if k.startswith(prefix)
        },
        strict=True,
    )
    model = model.eval().cuda()
    features = {}
    for name in ("native-input.npz", "native-derived.npz"):
        with np.load(args.reference / name, allow_pickle=False) as archive:
            for key in archive.files:
                parts = key.split(".")
                target = features
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                value = archive[key]
                target[parts[-1]] = (
                    value.item() if value.ndim == 0 else torch.from_numpy(value).cuda()
                )

    def hook(name):
        def observe(module, inputs, output):
            if isinstance(output, torch.Tensor):
                print(
                    json.dumps(
                        {
                            "module": name,
                            "input": [
                                str(x.dtype)
                                for x in inputs
                                if isinstance(x, torch.Tensor)
                            ],
                            "output": str(output.dtype),
                            "shape": list(output.shape),
                        }
                    ),
                    flush=True,
                )

        return observe

    handles = [
        module.register_forward_hook(hook(name))
        for name, module in model.named_modules()
        if name and not list(module.children())
    ]
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(features, inplace_safe=False)
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "complete": True,
                    "output_dtype": str(result.dtype),
                    "shape": list(result.shape),
                }
            )
        )
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
