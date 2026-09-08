"""Native learned InputsEmbedder capture and isolated reference-rounding control."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from bench.esmfold2_input_boundary import INPUT_NAMES
from bench.esmfold2_tape import _npz, _save_npz, _sha256


def validate_boundary(directory: Path):
    report = json.loads((directory / "report.json").read_text())
    archive = directory / "inputs.npz"
    if report["engine"] != "native" or _sha256(archive) != report["archive_sha256"]:
        raise ValueError("requires intact native input boundary")
    arrays = _npz(archive)
    if set(arrays) != set(INPUT_NAMES) or set(report["fields"]) != set(INPUT_NAMES):
        raise ValueError("native input fields differ")
    for name, value in arrays.items():
        field = report["fields"][name]
        if (
            list(value.shape) != field["shape"]
            or str(value.dtype) != field["dtype"]
            or str(value.dtype) != field["storage_dtype"]
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"native input schema differs: {name}")
    return report, arrays


def difference(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("comparison schema differs")
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {
        "shape": list(a.shape),
        "unequal": int(np.count_nonzero(a != b)),
        "max_abs": float(np.max(np.abs(delta), initial=0)),
        "rmse": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("boundary", "reference", "weights", "native-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    boundary, arrays = validate_boundary(args.boundary)
    reference = json.loads((args.reference / "metadata.json").read_text())
    config_path = args.weights / "config.json"
    weight_path = args.weights / "model.safetensors"
    config = json.loads(config_path.read_text())
    for path in (config_path, weight_path):
        if _sha256(path) != reference["binding"]["checkpoint"][path.name]:
            raise ValueError("native checkpoint differs")
    if config != reference["config"] or boundary["bindings"].get(
        str((args.reference / "metadata.json").resolve())
    ) != _sha256(args.reference / "metadata.json"):
        raise ValueError("boundary reference differs")
    sys.path.insert(0, str(args.native_root / "src"))
    import torch
    from safetensors import safe_open
    from transformers.models.esmfold2 import modeling_esmfold2_common as common
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config

    source = Path(inspect.getfile(common)).resolve()
    expected = (
        args.native_root
        / "src/transformers/models/esmfold2/modeling_esmfold2_common.py"
    )
    if (
        source != expected.resolve()
        or _sha256(source)
        != reference["binding"]["source"]["native"][
            "transformers/models/esmfold2/modeling_esmfold2_common.py"
        ]
    ):
        raise ValueError("native source differs")
    if (
        torch.cuda.device_count() != 1
        or str(torch.__version__) != boundary["runtime"]["torch"]
        or torch.version.git_version != boundary["runtime"]["torch_git"]
    ):
        raise ValueError("native CUDA runtime differs")
    paths = [
        Path(__file__),
        Path(inspect.getfile(_sha256)),
        Path(__file__).with_name("esmfold2_input_boundary.py"),
        args.boundary / "inputs.npz",
        args.boundary / "report.json",
        args.reference / "metadata.json",
        config_path,
        weight_path,
        *source.parent.rglob("*.py"),
    ]
    bindings = {str(path.resolve()): _sha256(path) for path in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    model = common.InputsEmbedder(ESMFold2Config(**config)).eval()
    with safe_open(weight_path, framework="pt") as handle:
        state = {
            key.removeprefix("inputs_embedder."): handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith("inputs_embedder.")
        }
    model.load_state_dict(state, strict=True)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("expected original FP32 parameters")
    model.cuda()
    inputs = {
        key: torch.as_tensor(value, device="cuda") for key, value in arrays.items()
    }
    stored, dtypes, hooks = {}, {}, []

    def save(key, value):
        if key in stored:
            raise ValueError(f"repeated module capture: {key}")
        dtypes[key] = str(value.dtype).removeprefix("torch.")
        stored[key] = value.detach().float().cpu().numpy().copy()

    def observe(name):
        def hook(module, positional, output):
            del module
            if positional and isinstance(positional[0], torch.Tensor):
                save(name + ".input", positional[0])
            if isinstance(output, torch.Tensor):
                save(name + ".output", output)
            elif name == "atom_attention_encoder":
                for label, value in zip(
                    ("tokens", "queries", "conditioning"), output[:3], strict=True
                ):
                    save(name + "." + label, value)
                save(name + ".rope_cos", output[3][0])
                save(name + ".rope_sin", output[3][1])

        return hook

    def forward(values):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model(**values).float().cpu().numpy()

    baseline = forward(inputs)
    repeated = forward(inputs)
    try:
        for name, module in model.named_modules():
            if name and isinstance(
                module,
                (
                    torch.nn.Linear,
                    torch.nn.LayerNorm,
                    common.SWAAtomBlock,
                    common.SWA3DRoPEAttention,
                    common.SwiGLUFFN,
                    common.ESMFold2AtomEncoder,
                ),
            ):
                hooks.append(module.register_forward_hook(observe(name)))
        observed = forward(inputs)
    finally:
        for hook in hooks:
            hook.remove()
    # Only reference positions are rounded, then restored to FP32. All learned
    # operations, weights, other inputs, and autocast policies are unchanged.
    rounded = dict(inputs, ref_pos=inputs["ref_pos"].bfloat16().float())
    counterfactual = forward(rounded)
    stored.update(
        baseline=baseline,
        repeated=repeated,
        observed=observed,
        rounded_reference=counterfactual,
    )
    if not all(np.isfinite(value).all() for value in stored.values()):
        raise ValueError("nonfinite captured output")
    _save_npz(args.output / "native.npz", stored)
    if any(_sha256(Path(path)) != digest for path, digest in bindings.items()):
        raise ValueError("bound artifact changed during capture")
    report = {
        "scope": (
            "native learned InputsEmbedder on shared captured inputs; "
            "no full-model admission"
        ),
        "full_model_admission": None,
        "bindings": bindings,
        "archive_sha256": _sha256(args.output / "native.npz"),
        "torch": str(torch.__version__),
        "torch_git": torch.version.git_version,
        "flash_attn_available": common.FLASH_ATTN_AVAILABLE,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "dtypes": dtypes,
        "repeat": difference(baseline, repeated),
        "instrumentation": difference(baseline, observed),
        "rounded_reference": difference(baseline, counterfactual),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "flash_attn_available",
                    "repeat",
                    "instrumentation",
                    "rounded_reference",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
