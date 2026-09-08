"""Same-input native-logits softmax diagnostic; not full-model admission."""

import argparse
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_pwa_logits_probe import load_reference, verify_bindings
from bench.boltz_pwa_production_probe import select_arrays


def rn_divide(numerator, denominator):
    """Diagnostic CUDA FP32 division with an explicit rounding instruction."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    numerator, denominator = jnp.broadcast_arrays(numerator, denominator)
    if numerator.dtype != jnp.float32 or denominator.dtype != jnp.float32:
        raise ValueError("division control requires FP32")
    if numerator.size % 32:
        raise ValueError("division control requires complete 32-element blocks")
    shape = numerator.shape
    rows = numerator.size // 32

    def kernel(a, b, out):
        out[:] = pt.elementwise_inline_asm(
            "div.rn.f32 $0, $1, $2;", args=(a[:], b[:]),
            constraints="=f,f,f", pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct((32,), jnp.float32)],
        )[0]

    spec = pl.BlockSpec((None, 32), lambda i: (i, 0))
    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((rows, 32), jnp.float32),
        grid=(rows,), in_specs=(spec, spec), out_specs=spec,
        compiler_params=pt.CompilerParams(num_warps=4),
    )(numerator.reshape(rows, 32), denominator.reshape(rows, 32)).reshape(shape)


def warp_softmax(x, *, explicit_division=False):
    """Diagnostic CUDA-warp reduction order; exp/div lowering remains JAX."""
    import jax
    import jax.numpy as jnp

    if x.shape[-1] != 437 or x.dtype != jnp.float32:
        raise ValueError("diagnostic requires 437 FP32 columns")
    padded = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, 75)],
                     constant_values=-jnp.inf)
    lanes = padded.reshape(*x.shape[:-1], 16, 32)
    maximum = lanes[..., 0, :]
    for i in range(1, 16):
        maximum = jnp.maximum(maximum, lanes[..., i, :])
    indices = jnp.arange(32)
    for offset in (16, 8, 4, 2, 1):
        maximum = jnp.maximum(maximum, maximum[..., indices ^ offset])
    exponentials = jnp.exp(lanes - maximum[..., None, :])
    total = jnp.zeros_like(maximum)
    for i in range(16):
        total = jax.lax.optimization_barrier(total + exponentials[..., i, :])
    for offset in (16, 8, 4, 2, 1):
        total = jax.lax.optimization_barrier(total + total[..., indices ^ offset])
    result = (
        rn_divide(exponentials, total[..., None, :])
        if explicit_division else exponentials / total[..., None, :]
    )
    return result.reshape(*x.shape[:-1], 512)[..., :437]


def native_probe(args, native_report, logits, weights, mask, bindings):
    import torch

    from bench.boltz_pwa_averaging_probe import bitwise_comparison, profiled_call

    if torch.__version__ != native_report["torch"]:
        raise ValueError("native Torch version differs from capture")
    if torch.cuda.get_device_name() != native_report["device"]:
        raise ValueError("native device differs from capture")
    args.out.mkdir(parents=True, exist_ok=False)
    result = {}
    with torch.inference_mode():
        mm = torch.from_numpy(mask.copy()).cuda().float()
        pair_mask = mm[:, None, :, None] * mm[:, None, None, :]
        for h in range(8):
            xx = torch.from_numpy(logits[..., h:h + 1].copy()).cuda().bfloat16()
            masked = xx.permute(0, 3, 1, 2) + (1 - pair_mask) * -1e6
            masked.softmax(-1)
            value, profile = profiled_call(torch, lambda: masked.softmax(-1))
            repeat = masked.softmax(-1)
            actual = value.float().cpu().numpy()
            result[f"head{h}/weights"] = {
                "comparison": bitwise_comparison(actual, weights[f"head{h}/weights"]),
                "repeat": bitwise_comparison(actual, repeat.float().cpu().numpy()),
                "profile": profile,
            }
    verify_bindings(bindings)
    save_new(args.out / "report.json", {
        "arm": "native",
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "device": torch.cuda.get_device_name(),
        "mask_penalty": 1e6,
        "source_sha256": bindings[Path(__file__).resolve()],
        "reference_sha256": bindings[args.reference / "report.json"],
        "bindings_unchanged": True,
        "not_model_parity_admission": True,
        "heads": result,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--warp-reduction", action="store_true")
    parser.add_argument("--explicit-division", action="store_true")
    args = parser.parse_args()
    if args.native and args.warp_reduction:
        parser.error("warp reduction is a JAX diagnostic only")
    if args.explicit_division and (args.native or not args.warp_reduction):
        parser.error("explicit division requires the JAX warp reduction control")
    native_report, _, _, logits, bindings = load_reference(args.reference)
    bindings[Path(__file__).resolve()] = sha(Path(__file__).resolve())
    bindings[args.reference / "inputs.npz"] = native_report["artifacts"]["inputs.npz"]
    verify_bindings(bindings)
    names = [f"head{h}/weights" for h in range(8)]
    weights = select_arrays(args.reference / "stages.npz", names)
    mask = select_arrays(args.reference / "inputs.npz", ["mask"])["mask"]
    if args.native:
        native_probe(args, native_report, logits, weights, mask, bindings)
        return
    import jax
    import jax.numpy as jnp

    if jax.default_backend() != "gpu":
        raise RuntimeError("requires GPU softmax, not a CPU substitute")
    args.out.mkdir(parents=True, exist_ok=False)
    result = {}
    for h, name in enumerate(names):
        x = logits[..., h:h + 1].transpose(0, 3, 1, 2)
        pair_mask = mask[:, None, :, None] * mask[:, None, None, :]
        softmax = (
            lambda x: warp_softmax(x, explicit_division=args.explicit_division)
        ) if args.warp_reduction else jax.nn.softmax
        fn = jax.jit(lambda a, m: softmax(
            a.astype(jnp.float32) + (1 - m.astype(jnp.float32)) * -1e6,
        ))
        xx, mm = jnp.asarray(x), jnp.asarray(pair_mask)
        executable = fn.lower(xx, mm).compile()
        hlo_path = args.out / f"head{h}.hlo.txt"
        with hlo_path.open("x") as stream:
            stream.write(executable.as_text())
        actual = np.asarray(executable(xx, mm), np.float32)
        repeat = np.asarray(executable(xx, mm), np.float32)
        expected = weights[name]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError("native weight shape/dtype differs")
        actual_bf16 = np.asarray(jnp.asarray(actual).astype(jnp.bfloat16), np.float32)
        expected_bf16 = np.asarray(
            jnp.asarray(expected).astype(jnp.bfloat16), np.float32
        )
        result[name] = {
            "hlo_sha256": sha(hlo_path),
            "repeat_bitwise_equal": bool(np.array_equal(
                actual.view(np.uint32), repeat.view(np.uint32)
            )),
            "unequal": int(np.count_nonzero(actual != expected)),
            "max_abs": float(np.max(np.abs(actual - expected))),
            "bf16_unequal": int(np.count_nonzero(actual_bf16 != expected_bf16)),
        }
    verify_bindings(bindings)
    save_new(args.out / "report.json", {
        "scope": "native logits to FP32 softmax weights before BF16 conversion",
        "not_model_parity_admission": True,
        "jax": jax.__version__,
        "mask_penalty": 1e6,
        "warp_reduction_control": args.warp_reduction,
        "explicit_division_control": args.explicit_division,
        "device": jax.devices()[0].device_kind,
        "source_sha256": bindings[Path(__file__).resolve()],
        "reference_sha256": bindings[args.reference / "report.json"],
        "bindings_unchanged": True,
        "heads": result,
    })


if __name__ == "__main__":
    main()
