"""Native same-input pair transition leaves, checked against its captured update."""

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from bench.esmfold2_msa_prefix_probe import validate_transition_chunks
from bench.esmfold2_tape import _save_npz, _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    report = json.loads((root / "report.json").read_text())
    if report["engine"] != "native" or not report.get("full_msa_block"):
        raise ValueError("requires native full MSA block capture")
    bindings = dict(report["bindings"])
    bindings[str(root / "report.json")] = _sha256(root / "report.json")
    bindings[str(root / "prefix.npz")] = report["archive_sha256"]
    if bindings.get(str(args.weights.resolve())) != _sha256(args.weights):
        raise ValueError("weights differ from reference")
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("reference binding changed")
    sources = [
        Path(p)
        for p in bindings
        if p.endswith("/transformers/models/esmfold2/modeling_esmfold2.py")
    ]
    if len(sources) != 1:
        raise ValueError("requires one bound native source")
    sys.path.insert(0, str(sources[0].parents[3]))
    import torch
    from safetensors import safe_open
    from transformers.models.esmfold2.modeling_esmfold2 import PairTransition

    source = Path(inspect.getfile(PairTransition)).resolve()
    if bindings.get(str(source)) != _sha256(source):
        raise ValueError("native source differs from reference")
    if (
        torch.cuda.device_count() != 1
        or str(torch.__version__) != report["runtime"]["torch"]
        or torch.version.git_version != report["runtime"]["torch_git"]
    ):
        raise ValueError("native runtime differs")
    for path in (Path(__file__), Path(inspect.getfile(validate_transition_chunks))):
        bindings[str(path.resolve())] = _sha256(path)
    key = "blocks.0.pair_transition"
    with np.load(root / "prefix.npz") as archive:
        x = torch.as_tensor(archive[key + ".input"], device="cuda").bfloat16()
        expected = archive[key + ".output"]
    if report["dtypes"][key + ".input"] != "bfloat16":
        raise ValueError("native input dtype differs")
    module = PairTransition(x.shape[-1], expansion_ratio=4).cuda().eval()
    prefix = "msa_encoder." + key + "."
    with safe_open(args.weights, framework="pt") as handle:
        module.load_state_dict(
            {
                k.removeprefix(prefix): handle.get_tensor(k)
                for k in handle.keys()
                if k.startswith(prefix)
            },
            strict=True,
        )
    values, hooks = {}, []

    def hook_for(name):
        def hook(module, positional, output):
            for suffix, value in (("input", positional[0]), ("output", output)):
                values.setdefault(key + "." + name + "." + suffix, []).append(
                    value.clone()
                )

        return hook

    try:
        for name in ("norm", "ffn.w12", "ffn.w3"):
            hooks.append(
                module.get_submodule(name).register_forward_hook(hook_for(name))
            )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = module(x)
    finally:
        for hook in hooks:
            hook.remove()
    actual = output.float().cpu().numpy()
    if actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
        raise ValueError(
            "native isolated transition does not reproduce captured output"
        )
    geometry, arrays, dtypes = {}, {}, {}
    for name, chunks in values.items():
        geometry[name] = [list(v.shape) for v in chunks]
        validate_transition_chunks(geometry[name], x.shape[1], module._chunk_size)
        merged = torch.cat(chunks, dim=1)
        dtypes[name] = str(merged.dtype).removeprefix("torch.")
        arrays[name] = merged.float().cpu().numpy()
    arrays[key + ".output"] = actual
    dtypes[key + ".output"] = str(output.dtype).removeprefix("torch.")
    if len(arrays) != 7 or not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("incomplete or nonfinite native leaves")
    args.output.mkdir(parents=True, exist_ok=False)
    _save_npz(args.output / "prefix.npz", arrays)
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("binding changed during capture")
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "engine": "native",
                "full_pair_transition": True,
                "full_model_admission": False,
                "matches_full_block_update": True,
                "runtime": report["runtime"],
                "bindings": bindings,
                "dtypes": dtypes,
                "transition_chunk_geometry": geometry,
                "archive_sha256": _sha256(args.output / "prefix.npz"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
