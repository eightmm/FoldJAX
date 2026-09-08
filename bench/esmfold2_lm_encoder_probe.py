"""Development-only native first LM-encoder block, teacher-forced shim pair.

No dropout or full-model admission: this isolates one default chunked block.
Intermediates are slices of real operands, never recomputed reference formulas.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from bench.esmfold2_lm_shim_candidate import validate_native
from bench.esmfold2_lm_shim_probe import boundary_slice
from bench.esmfold2_tape import _npz, _save_npz, _sha256


def validate_inputs(pair, token_mask, config):
    if (
        pair.dtype != np.float32
        or pair.ndim != 4
        or pair.shape[1] != pair.shape[2]
        or pair.shape[-1] != config["d_pair"]
        or not np.isfinite(pair).all()
    ):
        raise ValueError("requires full finite FP32 native shim pair")
    if token_mask.shape != pair.shape[:2] or not np.isin(token_mask, [0, 1]).all():
        raise ValueError("invalid token mask")
    if config["lm_encoder"]["n_layers"] < 1:
        raise ValueError("native configuration has no first LM encoder block")


def block_state(handle):
    prefix = "lm_encoder.blocks.0."
    state = {
        key[len(prefix) :]: handle.get_tensor(key)
        for key in handle.keys()
        if key.startswith(prefix)
    }
    if not state:
        raise ValueError("missing native lm_encoder.blocks.0 weights")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "native-root",
        "native-shim",
        "reference",
        "weights",
        "features",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--disable-bf16-reduction", action="store_true")
    parser.add_argument("--capture-first-contraction", action="store_true")
    parser.add_argument("--capture-first-norm", action="store_true")
    parser.add_argument("--capture-first-projection", action="store_true")
    parser.add_argument("--capture-incoming-boundary", action="store_true")
    parser.add_argument("--capture-outgoing-boundary", action="store_true")
    parser.add_argument("--capture-transition-projection", action="store_true")
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()
    args.capture_incoming_boundary |= args.capture_outgoing_boundary
    args.capture_first_projection |= args.capture_outgoing_boundary
    args.capture_first_norm |= (
        args.capture_first_projection
        or args.capture_incoming_boundary
        or args.capture_transition_projection
    )
    args.output.mkdir(parents=True, exist_ok=False)
    source_dir = args.native_root / "src/transformers/models/esmfold2"
    paths = [
        *sorted(source_dir.rglob("*.py")),
        args.weights / "model.safetensors",
        args.weights / "config.json",
        args.reference / "metadata.json",
        args.features,
        args.native_shim / "native.npz",
        args.native_shim / "report.json",
        Path(__file__),
        Path(inspect.getfile(validate_native)),
        Path(inspect.getfile(boundary_slice)),
        Path(inspect.getfile(_npz)),
    ]
    before = {str(path.resolve()): _sha256(path) for path in paths}
    metadata = json.loads((args.reference / "metadata.json").read_text())
    report = validate_native(args.native_shim, metadata, args.reference, args.weights)
    source = source_dir / "modeling_esmfold2_common.py"
    if _sha256(source) != report["source_sha256"]:
        raise ValueError("native module source differs from captured shim")
    config_path = args.weights / "config.json"
    config = json.loads(config_path.read_text())
    if config != metadata["config"]:
        raise ValueError("native configuration differs")
    pair = _npz(args.native_shim / "native.npz")["pair"]
    if _sha256(args.features) != metadata["binding"]["input"]:
        raise ValueError("features differ from original native capture input")
    token_mask = _npz(args.features)["token_attention_mask"]
    validate_inputs(pair, token_mask, config)
    sys.path.insert(0, str(args.native_root / "src"))
    import torch
    from safetensors import safe_open
    from transformers.models.esmfold2.modeling_esmfold2_common import PairUpdateBlock

    if Path(inspect.getfile(PairUpdateBlock)).resolve() != source.resolve():
        raise ValueError("native block imported from another source")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("probe requires exactly one CUDA GPU")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.disable_bf16_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    block = PairUpdateBlock(d_pair=config["d_pair"], expansion_ratio=4)
    with safe_open(args.weights / "model.safetensors", framework="pt") as handle:
        block.load_state_dict(block_state(handle), strict=True)
    if block._kernel_backend is not None or any(
        module._chunk_size != 64
        for module in (
            block.tri_mul_out._engine,
            block.tri_mul_in._engine,
            block.pair_transition,
        )
    ):
        raise ValueError("native default kernel/chunk policy changed")
    block = block.eval().cuda()
    inputs = torch.from_numpy(pair).cuda().to(torch.bfloat16)
    mask = torch.from_numpy(token_mask).cuda().float()
    mask = mask[:, :, None] * mask[:, None, :]
    arrays, schema, handles, restorations = {}, {}, [], []

    def save(name, value, full=False):
        stored = value if full else boundary_slice(value)
        arrays[name] = stored.detach().float().cpu().numpy().copy()
        schema[name] = {
            "original_dtype": str(value.dtype),
            "shape": list(value.shape),
            "stored_shape": list(stored.shape),
            "storage_dtype": "float32",
            "scope": "full" if full else "first3_each_nonchannel_axis",
            "native_stride": list(value.stride()),
            "native_storage_offset": value.storage_offset(),
        }

    calls = {}

    def hook(name):
        def observe(module, arguments, result):
            del module
            count = calls.get(name, 0)
            calls[name] = count + 1
            save(f"{name}.{count}.input", arguments[0])
            save(f"{name}.{count}.output", result)
            if (
                args.capture_first_norm
                and name == "tri_mul_out._engine.norm_start"
                and count == 0
            ):
                save("first_norm.output", result, full=True)
            if (
                args.capture_first_projection
                and name == "tri_mul_out._engine.proj_bundle"
                and count == 0
            ):
                save("first_projection.input", arguments[0], full=True)
                save("first_projection.output", result, full=True)
            if (
                args.capture_incoming_boundary
                and name == "tri_mul_in._engine.norm_start"
                and count == 0
            ):
                save("incoming.input", arguments[0], full=True)
            if args.capture_outgoing_boundary and count == 0:
                for leaf in ("norm_mix", "proj_emit", "proj_gate"):
                    if name == f"tri_mul_out._engine.{leaf}":
                        save("outgoing." + leaf, result, full=True)
            if (
                args.capture_transition_projection
                and name == "pair_transition.ffn.w3"
                and count == 0
            ):
                save("transition_projection.input", arguments[0], full=True)
                save("transition_projection.output", result, full=True)

        return observe

    if args.capture_first_norm:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            baseline = block(inputs, pair_attention_mask=mask)
        save("block.uninstrumented_output", baseline, full=True)
        del baseline

    for name, module in block.named_modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.Linear)):
            handles.append(module.register_forward_hook(hook(name)))
    # Wrapper receives actual routed.float() halves. The chunked contraction
    # remains untouched, so its chunk order and GEMM shapes are native defaults.
    for label in ("tri_mul_out", "tri_mul_in"):
        engine = getattr(block, label)._engine
        original = engine._triangular_contract_chunked

        def wrap(left, right, chunk, *, original=original, label=label):
            save(f"{label}.contract.left", left)
            save(f"{label}.contract.right", right)
            if args.capture_incoming_boundary and label == "tri_mul_in":
                save("incoming.left", left, full=True)
                save("incoming.right", right, full=True)
            if args.capture_outgoing_boundary and label == "tri_mul_out":
                save("outgoing.left", left, full=True)
                save("outgoing.right", right, full=True)
            native_einsum = torch.einsum

            def observe_einsum(equation, lhs, rhs):
                first = "first_contraction.lhs" not in arrays
                if first:
                    if equation != "bikd,bjkd->bijd":
                        raise ValueError("unexpected first outgoing contraction")
                    save("first_contraction.lhs", lhs, full=True)
                    save("first_contraction.rhs", rhs, full=True)
                value = native_einsum(equation, lhs, rhs)
                if first:
                    save("first_contraction.output", value, full=True)
                return value

            if args.capture_first_contraction and label == "tri_mul_out":
                torch.einsum = observe_einsum
            try:
                result = original(left, right, chunk)
            finally:
                torch.einsum = native_einsum
            save(f"{label}.contract.output", result)
            if args.capture_incoming_boundary and label == "tri_mul_in":
                save("incoming.output", result, full=True)
            if args.capture_outgoing_boundary and label == "tri_mul_out":
                save("outgoing.output", result, full=True)
            return result

        engine._triangular_contract_chunked = wrap
        restorations.append((engine, original))
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = block(inputs, pair_attention_mask=mask)
        save("block.input", inputs, full=True)
        save("block.output", output, full=True)
        save("pair_mask", mask, full=True)
    finally:
        for handle in handles:
            handle.remove()
        for engine, original in restorations:
            engine._triangular_contract_chunked = original
    if args.uncompressed:
        with (args.output / "native.npz").open("xb") as stream:
            np.savez(stream, **arrays)
    else:
        _save_npz(args.output / "native.npz", arrays)
    if before != {str(path.resolve()): _sha256(path) for path in paths}:
        raise ValueError("bound inputs/source changed during capture")
    payload = {
        "scope": "teacher_forced_first_lm_encoder_block_dropout_disabled",
        "model_admission": None,
        "bindings": before,
        "checkpoint_prefix": "lm_encoder.blocks.0.",
        "input_shim_policy": report,
        "autocast": "bfloat16",
        "entry_cast": "float32_to_bfloat16",
        "chunk_size": 64,
        "kernel_backend": None,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "matmul": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "precision_control": args.disable_bf16_reduction,
        "capture_first_contraction": args.capture_first_contraction,
        "capture_first_norm": args.capture_first_norm,
        "capture_first_projection": args.capture_first_projection,
        "capture_incoming_boundary": args.capture_incoming_boundary,
        "capture_outgoing_boundary": args.capture_outgoing_boundary,
        "capture_transition_projection": args.capture_transition_projection,
        "archive_compression": "stored" if args.uncompressed else "deflate",
        "instrumentation_output_bytes_equal": (
            bool(
                np.array_equal(
                    arrays["block.uninstrumented_output"].view(np.uint8),
                    arrays["block.output"].view(np.uint8),
                )
            )
            if args.capture_first_norm
            else None
        ),
        "boundaries": schema,
        "module_call_counts": calls,
        "uncaptured": ["routed_before_float", "individual_einsum_chunks"],
        "archive_sha256": _sha256(args.output / "native.npz"),
    }
    (args.output / "report.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
