"""Complete native LM encoder on captured BF16 input; no dropout/model admission."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

from bench.esmfold2_lm_encoder_candidate import (
    compiler_control,
    ffi_output_linear,
    validate_capture,
)
from bench.esmfold2_tape import _sha256


def validate_stack_input(value, mask, config):
    layers = config["lm_encoder"]["n_layers"]
    if type(layers) is not int or layers < 1 or not config["lm_encoder"]["enabled"]:
        raise ValueError("native LM encoder must be enabled with positive layer count")
    if (
        value.ndim != 4
        or min(value.shape) <= 0
        or value.shape[1] != value.shape[2]
        or value.shape[-1] != config["d_pair"]
        or value.dtype != np.float32
        or not np.isfinite(value).all()
        or np.any(value.view(np.uint32) & 0xFFFF)
    ):
        raise ValueError("requires full finite lossless native BF16 pair")
    if mask.shape != value.shape[:-1] or not np.isin(mask, [0, 1]).all():
        raise ValueError("native pair mask differs")
    return layers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("native", "jax"), required=True)
    for name in ("reference", "weights", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--native-root", type=Path)
    p.add_argument("--candidate-source", type=Path)
    p.add_argument("--ffi-library", type=Path)
    args = p.parse_args()
    if args.engine == "native" and (args.native_root is None or args.ffi_library):
        raise ValueError("native engine requires native root and no candidate FFI")
    if args.engine == "jax" and (
        args.candidate_source is None or args.ffi_library is None
    ):
        raise ValueError(
            "candidate requires an explicit source snapshot and FFI library"
        )
    reference, arrays = validate_capture(args.reference, args.weights)
    if (
        reference["precision_control"]
        or not reference["instrumentation_output_bytes_equal"]
    ):
        raise ValueError(
            "reference must preserve native default and instrumentation gate"
        )
    config_path = args.weights / "config.json"
    if reference["bindings"].get(str(config_path.resolve())) != _sha256(config_path):
        raise ValueError("configuration differs from bound native reference")
    config = json.loads(config_path.read_text())
    value, mask = arrays["block.input"], arrays["pair_mask"]
    layers = validate_stack_input(value, mask, config)
    del arrays
    paths = [
        Path(__file__),
        Path(inspect.getfile(validate_capture)),
        Path(inspect.getfile(_sha256)),
        args.reference / "report.json",
        args.reference / "native.npz",
        args.weights / "model.safetensors",
        config_path,
    ]
    bindings = {str(path.resolve()): _sha256(path) for path in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    stored = {"input": value, "mask": mask}
    runtime = {}
    if args.engine == "native":
        sys.path.insert(0, str(args.native_root / "src"))
        import torch
        from safetensors import safe_open
        from transformers.models.esmfold2.modeling_esmfold2_common import FoldingTrunk

        source = Path(inspect.getfile(FoldingTrunk)).resolve()
        expected = (
            args.native_root
            / "src/transformers/models/esmfold2/modeling_esmfold2_common.py"
        )
        if source != expected.resolve() or reference["bindings"].get(
            str(source)
        ) != _sha256(source):
            raise ValueError("native source identity differs")
        if torch.cuda.device_count() != 1 or (
            str(torch.__version__),
            torch.version.git_version,
        ) != (reference["torch"], reference["torch_git"]):
            raise ValueError("native runtime differs")
        torch.set_float32_matmul_precision(reference["matmul"])
        torch.backends.cuda.matmul.allow_tf32 = reference["allow_tf32"]
        if (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            != reference["allow_bf16_reduced_precision_reduction"]
        ):
            raise ValueError("native BF16 reduction policy differs")
        model = FoldingTrunk(
            n_layers=layers, d_pair=config["d_pair"], expansion_ratio=4
        )
        with safe_open(args.weights / "model.safetensors", framework="pt") as handle:
            state = {
                key.removeprefix("lm_encoder."): handle.get_tensor(key)
                for key in handle.keys()
                if key.startswith("lm_encoder.")
            }
        model.load_state_dict(state, strict=True)
        for block in model.blocks:
            if block._kernel_backend is not None or any(
                m._chunk_size != 64
                for m in (
                    block.tri_mul_out._engine,
                    block.tri_mul_in._engine,
                    block.pair_transition,
                )
            ):
                raise ValueError("native kernel/chunk defaults differ")
        model = model.eval().cuda()
        x = torch.from_numpy(value).cuda().to(torch.bfloat16)
        m = torch.from_numpy(mask).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            baseline = model(x, pair_attention_mask=m).float().cpu().numpy()
        hooks = []

        def observe(index):
            def hook(module, inputs, output):
                del module, inputs
                if output.dtype != torch.bfloat16:
                    raise ValueError("native block output dtype differs")
                stored[f"block.{index}"] = output.detach().float().cpu().numpy().copy()

            return hook

        try:
            for i, block in enumerate(model.blocks):
                hooks.append(block.register_forward_hook(observe(i)))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                observed = model(x, pair_attention_mask=m).float().cpu().numpy()
        finally:
            for hook in hooks:
                hook.remove()
        runtime = {
            "torch": str(torch.__version__),
            "torch_git": torch.version.git_version,
        }
    else:
        import jax
        import jax.numpy as jnp
        from safetensors import safe_open

        from bench import native_cublaslt_ffi
        from foldjax.models.esmfold2.models import trunk

        source = Path(inspect.getfile(trunk)).resolve()
        expected = args.candidate_source / "src/foldjax/models/esmfold2/models/trunk.py"
        if (
            source != expected.resolve()
            or len(jax.devices()) != 1
            or jax.devices()[0].platform != "gpu"
        ):
            raise ValueError("candidate source/device differs")
        extra = [
            *source.parent.rglob("*.py"),
            args.ffi_library,
            Path(native_cublaslt_ffi.__file__),
            Path(__file__).with_name("native_cublaslt_ffi.cc"),
            args.candidate_source
            / "src/foldjax/models/boltz2/models/primitives/native_amp_norm.py",
            args.candidate_source / "src/foldjax/models/_cp.py",
        ]
        paths.extend(extra)
        bindings.update({str(path.resolve()): _sha256(path) for path in extra})
        jax.config.update("jax_default_matmul_precision", "highest")
        target = native_cublaslt_ffi.register(args.ffi_library)
        original_linear, original_block = (
            trunk._autocast_linear,
            trunk.pair_update_block,
        )
        trunk._autocast_linear = partial(ffi_output_linear, target=target)
        with safe_open(args.weights / "model.safetensors", framework="numpy") as handle:
            params = {
                key: jnp.asarray(handle.get_tensor(key))
                for key in handle.keys()
                if key.startswith("lm_encoder.")
            }
        x, m = jnp.asarray(value, jnp.bfloat16), jnp.asarray(mask)
        options = compiler_control("native-chunks-strict-rounding")

        def run(a, w, mask):
            return trunk.folding_trunk(
                a, w, "lm_encoder", n_layers=layers, mask=mask, native_autocast=True
            )

        captured = []

        def capture_block(*a, **kw):
            result = original_block(*a, **kw)
            captured.append(result)
            return result

        def instrument(a, w, mask):
            captured.clear()
            result = run(a, w, mask)
            return result, tuple(captured)

        try:
            executable = (
                jax.jit(run, compiler_options=options).lower(x, params, m).compile()
            )
            baseline = np.asarray(executable(x, params, m).astype(jnp.float32))
            (args.output / "baseline.hlo.txt").write_text(executable.as_text())
            trunk.pair_update_block = capture_block
            observed, blocks = jax.jit(instrument, compiler_options=options)(
                x, params, m
            )
            observed = np.asarray(observed.astype(jnp.float32))
            for i, block in enumerate(blocks):
                if block.dtype != jnp.bfloat16:
                    raise ValueError("candidate block output dtype differs")
                stored[f"block.{i}"] = np.asarray(block.astype(jnp.float32))
        finally:
            trunk._autocast_linear, trunk.pair_update_block = (
                original_linear,
                original_block,
            )
        runtime = {
            "jax": jax.__version__,
            "compiler_options": options,
            "ffi_native_dispatch": True,
            "autotune_load": None,
        }
    stored["output"], stored["instrumented_output"] = baseline, observed
    expected_keys = {
        "input",
        "mask",
        "output",
        "instrumented_output",
        *[f"block.{i}" for i in range(layers)],
    }
    if set(stored) != expected_keys or not all(
        np.isfinite(a).all() for a in stored.values()
    ):
        raise ValueError("incomplete/nonfinite stack capture")
    for key in expected_keys - {"mask"}:
        if stored[key].shape != value.shape or stored[key].dtype != np.float32:
            raise ValueError("stack capture shape/storage dtype differs")
    if bindings != {str(path.resolve()): _sha256(path) for path in paths}:
        raise ValueError("bound stack sources changed")
    validate_capture(args.reference, args.weights)
    with (args.output / "arrays.npz").open("xb") as stream:
        np.savez(stream, **stored)
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "scope": "complete_native_lm_encoder_captured_input_no_dropout",
                "model_admission": None,
                "engine": args.engine,
                "layers": layers,
                "bindings": bindings,
                "runtime": runtime,
                "entry_dtype": "bfloat16",
                "block_dtype": "bfloat16",
                "chunk_size": 64,
                "instrumentation_bytes_equal": baseline.tobytes() == observed.tobytes(),
                "archive_sha256": _sha256(args.output / "arrays.npz"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
