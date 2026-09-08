"""Replay a captured native first MSA transition; not full-model admission."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from bench.boltz_closure_capture import save_new
from bench.boltz_pwa_averaging_probe import bitwise_comparison
from bench.esmfold2_lm_encoder_candidate import compiler_control
from bench.esmfold2_tape import _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-updates", action="store_true")
    parser.add_argument("--pair-transition", action="store_true")
    parser.add_argument("--ffi-library", type=Path)
    for name in ("reference", "weights", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    report = json.loads((root / "report.json").read_text())
    if args.pair_transition and args.pair_updates:
        raise ValueError("choose pair transition leaves or pair updates")
    if args.ffi_library and args.pair_updates:
        raise ValueError("FFI control requires transition leaves")
    scope = "full_pair_transition" if args.pair_transition else "full_msa_transition"
    if report["engine"] != "native" or not report.get(scope):
        raise ValueError("requires native MSA transition capture")
    if args.pair_updates and not report.get("full_msa_block"):
        raise ValueError("pair updates require native full MSA block capture")
    bindings = dict(report["bindings"])
    bindings[str(root / "prefix.npz")] = report["archive_sha256"]
    bindings[str(root / "report.json")] = _sha256(root / "report.json")
    if bindings.get(str(args.weights.resolve())) != _sha256(args.weights):
        raise ValueError("weights differ from native capture")
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("capture binding changed")
    import jax
    import jax.numpy as jnp
    from safetensors import safe_open

    from foldjax.models.boltz2.models.primitives import native_amp_norm
    from foldjax.models.esmfold2.models import primitives, trunk

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("requires one GPU")
    for path in (
        Path(__file__),
        Path(trunk.__file__),
        Path(primitives.__file__),
        Path(native_amp_norm.__file__),
        Path(inspect.getfile(compiler_control)),
    ):
        bindings[str(path.resolve())] = _sha256(path)
    key = "blocks.0." + (
        "pair_transition" if args.pair_transition else "msa_transition"
    )
    prefix = "msa_encoder." + key
    with safe_open(args.weights, framework="numpy") as handle:
        params = {
            k: jnp.asarray(handle.get_tensor(k))
            for k in handle.keys()
            if k.startswith(
                "msa_encoder.blocks.0." if args.pair_updates else prefix + "."
            )
        }
    with np.load(root / "prefix.npz") as archive:
        selected = (
            (
                "blocks.0.tri_mul_out.",
                "blocks.0.tri_mul_in.",
                "blocks.0.pair_transition.",
                "blocks.0.msa_pair_weighted_averaging.mask_input",
            )
            if args.pair_updates
            else (key + ".",)
        )
        arrays = {k: archive[k] for k in archive.files if k.startswith(selected)}
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("nonfinite transition capture")

    def operand(name):
        return jnp.asarray(arrays[name], getattr(jnp, report["dtypes"][name]))

    def chunked_linear(x, leaf):
        return jnp.concatenate(
            [
                trunk._autocast_linear(x[:, i : i + 64], params, prefix + leaf)
                for i in range(0, x.shape[1], 64)
            ],
            axis=1,
        )

    def hidden(x):
        gate, value = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16) * value

    def post_norm(x):
        outputs = []
        for start in range(0, x.shape[1], 64):
            packed = trunk._autocast_linear(
                x[:, start : start + 64], params, prefix + ".ffn.w12"
            )
            outputs.append(
                trunk._autocast_linear(hidden(packed), params, prefix + ".ffn.w3")
            )
        return jnp.concatenate(outputs, axis=1)

    def cuda_norm(x):
        return native_amp_norm._cuda_layer_norm(
            x.astype(jnp.float32),
            params[prefix + ".norm.weight"],
            params[prefix + ".norm.bias"],
            1e-5,
        )[0]

    results = {}
    arms = [
        ("cuda_norm", cuda_norm, key + ".norm.input", key + ".norm.output"),
        (
            "cuda_norm_composed",
            lambda x: post_norm(cuda_norm(x)),
            key + ".norm.input",
            key + ".output",
        ),
        (
            "native_norm_output_injected",
            post_norm,
            key + ".norm.output",
            key + ".output",
        ),
        (
            "norm",
            lambda x: trunk._autocast_norm(x, params, prefix + ".norm", 1e-5),
            key + ".norm.input",
            key + ".norm.output",
        ),
        (
            "w12",
            lambda x: chunked_linear(x, ".ffn.w12"),
            key + ".ffn.w12.input",
            key + ".ffn.w12.output",
        ),
        (
            "silu_product",
            hidden,
            key + ".ffn.w12.output",
            key + ".ffn.w3.input",
        ),
        (
            "w3",
            lambda x: chunked_linear(x, ".ffn.w3"),
            key + ".ffn.w3.input",
            key + ".ffn.w3.output",
        ),
    ]
    for native in (False, True):
        p = params if native else {k: v.astype(jnp.bfloat16) for k, v in params.items()}

        def call(x, p=p, native=native):
            return trunk.transition(
                x, p, prefix, residual=False, native_autocast=native
            )

        arms.append(
            (
                "native" if native else "legacy",
                call,
                key + ".norm.input",
                key + ".output",
            )
        )
    if args.pair_updates:
        arms = []
        mask = operand("blocks.0.msa_pair_weighted_averaging.mask_input")
        for leaf in ("tri_mul_out", "tri_mul_in", "pair_transition"):
            for native in (False, True):
                p = (
                    params
                    if native
                    else {k: v.astype(jnp.bfloat16) for k, v in params.items()}
                )

                def pair_call(x, p=p, native=native, leaf=leaf):
                    name = "msa_encoder.blocks.0." + leaf
                    if leaf == "pair_transition":
                        return trunk.transition(
                            x, p, name, residual=False, native_autocast=native
                        )
                    return trunk.triangle_multiplicative(
                        x,
                        p,
                        name,
                        outgoing=leaf == "tri_mul_out",
                        mask=mask,
                        native_autocast=native,
                    )

                name = "blocks.0." + leaf
                arms.append(
                    (
                        leaf + (".native" if native else ".legacy"),
                        pair_call,
                        name + ".input",
                        name + ".output",
                    )
                )
    if args.ffi_library:
        from bench import native_cublaslt_ffi

        for path in (
            args.ffi_library,
            Path(native_cublaslt_ffi.__file__),
            Path(native_cublaslt_ffi.__file__).with_suffix(".cc"),
        ):
            bindings[str(path.resolve())] = _sha256(path)
        ffi_target = native_cublaslt_ffi.register(args.ffi_library)

        def ffi_w3(x):
            return jnp.concatenate(
                [
                    native_cublaslt_ffi.linear(
                        x[:, i : i + 64],
                        params[prefix + ".ffn.w3.weight"],
                        target=ffi_target,
                    )
                    for i in range(0, x.shape[1], 64)
                ],
                axis=1,
            )

        arms.append(
            ("w3_native_ffi", ffi_w3, key + ".ffn.w3.input", key + ".ffn.w3.output")
        )
    compiler_options = compiler_control("native-chunks-strict-rounding")
    with jax.default_matmul_precision("highest"):
        for name, call, source, target in arms:
            fn = jax.jit(call, compiler_options=compiler_options)
            x = operand(source)
            a = np.asarray(fn(x).astype(jnp.float32))
            b = np.asarray(fn(x).astype(jnp.float32))
            results[name] = {
                "comparison": bitwise_comparison(a, arrays[target]),
                "repeat": bitwise_comparison(a, b),
            }
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("binding changed during replay")
    save_new(
        args.out,
        {
            "results": results,
            "bindings": bindings,
            "full_model_admission": False,
            "compiler_options": compiler_options,
            "pair_updates": args.pair_updates,
            "pair_transition": args.pair_transition,
        },
    )


if __name__ == "__main__":
    main()
