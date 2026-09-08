"""Native LM-shim boundary capture on an existing full LM archive.

Development-only operator evidence. Parameters and model inputs are unchanged;
intermediate slices are labelled, while the final pair output is stored whole.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from bench.esmfold2_tape import _npz, _save_npz, _sha256


def load_reference(root):
    metadata = json.loads((root / "metadata.json").read_text())
    schema = metadata["lm_schema"]
    if Path(schema["filename"]).name != schema["filename"]:
        raise ValueError("invalid LM artifact filename")
    path = root / schema["filename"]
    if _sha256(path) != schema["sha256"]:
        raise ValueError("LM artifact changed")
    arrays = _npz(path)
    if set(arrays) != {"lm_hidden_states"}:
        raise ValueError("unexpected LM archive fields")
    value = arrays["lm_hidden_states"]
    if (
        list(value.shape) != schema["shape"]
        or value.ndim != 4
        or schema["original_dtype"] != "float32"
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ValueError("this native LM-shim control requires captured finite FP32")
    return metadata, value


def boundary_slice(value):
    """Deterministic small slice; never an all-values intermediate claim."""
    return value[
        tuple(slice(0, min(size, 3)) for size in value.shape[:-1]) + (slice(None),)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-root", "reference", "weights", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--disable-bf16-reduction", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata, hidden = load_reference(args.reference)
    source = (
        args.native_root
        / "src/transformers/models/esmfold2/modeling_esmfold2_common.py"
    )
    checkpoint = args.weights / "model.safetensors"
    config_path = args.weights / "config.json"
    paths = (
        source,
        checkpoint,
        config_path,
        args.reference / "metadata.json",
        Path(__file__),
    )
    hashes = [_sha256(path) for path in paths]
    if metadata["binding"]["checkpoint"]["model.safetensors"] != hashes[1]:
        raise ValueError("checkpoint differs from LM capture")
    native_key = "transformers/models/esmfold2/modeling_esmfold2_common.py"
    if metadata["binding"]["source"]["native"][native_key] != hashes[0]:
        raise ValueError("native source differs from LM capture")
    sys.path.insert(0, str(args.native_root / "src"))
    import torch
    from safetensors import safe_open
    from transformers.models.esmfold2.modeling_esmfold2_common import LanguageModelShim

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("native shim probe requires one CUDA GPU")
    if Path(inspect.getfile(LanguageModelShim)).resolve() != source.resolve():
        raise ValueError("native shim imported from another source")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.disable_bf16_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    config = json.loads(config_path.read_text())
    if config != metadata["config"]:
        raise ValueError("configuration differs from LM capture")
    shim = LanguageModelShim(
        config["d_pair"], config["lm_d_model"], config["lm_num_layers"]
    )
    prefix = "language_model."
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        state = {
            key[len(prefix) :]: handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith(prefix)
        }
    shim.load_state_dict(state, strict=True)
    shim = shim.eval().cuda()
    inputs = torch.from_numpy(hidden).cuda()
    records, arrays, handles = {}, {}, []

    def hook(name):
        def observe(module, arguments, result):
            del module
            records[name] = {
                "input_dtype": str(arguments[0].dtype),
                "output_dtype": str(result.dtype),
                "input_shape": list(arguments[0].shape),
                "output_shape": list(result.shape),
            }
            arrays[name] = boundary_slice(result).detach().float().cpu().numpy().copy()

        return observe

    for name in ("base_z_linear.0", "base_z_linear.1", "base_z_mlp.0", "base_z_mlp.1"):
        handles.append(shim.get_submodule(name).register_forward_hook(hook(name)))
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = shim(inputs)
            combine = shim.base_z_combine.softmax(0)
    finally:
        for handle in handles:
            handle.remove()
    arrays["pair"] = output.detach().float().cpu().numpy()
    arrays["combine_softmax"] = combine.detach().float().cpu().numpy()
    _save_npz(args.output / "native.npz", arrays)
    if hashes != [_sha256(path) for path in paths]:
        raise ValueError("bound source/checkpoint/reference changed during capture")
    if metadata["lm_schema"]["sha256"] != _sha256(
        args.reference / metadata["lm_schema"]["filename"]
    ):
        raise ValueError("LM archive changed during capture")
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "source_sha256": hashes[0],
                "checkpoint_sha256": hashes[1],
                "config_sha256": hashes[2],
                "reference_sha256": hashes[3],
                "runner_sha256": hashes[4],
                "lm_sha256": metadata["lm_schema"]["sha256"],
                "torch": torch.__version__,
                "autocast": "bfloat16",
                "matmul": "highest",
                "allow_bf16_reduced_precision_reduction": (
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                ),
                "precision_control": args.disable_bf16_reduction,
                "boundaries": records,
                "combine_parameter_dtype": str(shim.base_z_combine.dtype),
                "combine_softmax_dtype": str(combine.dtype),
                "pair_dtype": str(output.dtype),
                "archive_sha256": _sha256(args.output / "native.npz"),
                "intermediate_scope": "up to three entries per non-channel axis",
                "pair_scope": "full native LM-shim pair output",
                "model_admission": None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(records, sort_keys=True))


if __name__ == "__main__":
    main()
