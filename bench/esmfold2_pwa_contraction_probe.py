"""Same-operand PWA contraction controls; no full-model admission."""

import argparse
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.boltz_closure_capture import save_new
from bench.boltz_pwa_averaging_probe import bitwise_comparison
from bench.esmfold2_lm_encoder_candidate import compiler_control
from bench.esmfold2_tape import _sha256


def spatial_softmax_control(x):
    """Hypothesis for native spatial softmax's 437-column, eight-head shape."""
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.primitives.native_pwa_weights import (
        _cuda_rn_divide,
    )

    if x.shape[-2:] != (437, 8) or x.dtype != jnp.float32:
        raise ValueError("spatial control requires FP32 437 by 8")
    rows = jnp.swapaxes(x, -1, -2)
    padded = jnp.pad(
        rows, [(0, 0)] * (rows.ndim - 1) + [(0, 75)], constant_values=-jnp.inf
    )
    exponentials = jnp.exp(padded - jnp.max(padded, axis=-1, keepdims=True))
    lanes = exponentials.reshape(*rows.shape[:-1], 4, 128)
    total = jnp.zeros_like(lanes[..., 0, :])
    for i in range(4):
        total = jax.lax.optimization_barrier(total + lanes[..., i, :])
    for offset in (64, 32, 16, 8, 4, 2, 1):
        total = jax.lax.optimization_barrier(
            total[..., :offset] + total[..., offset : 2 * offset]
        )
    output = _cuda_rn_divide(exponentials, total)[..., :437]
    return jnp.swapaxes(output, -1, -2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--native-msa-norm", action="store_true")
    args = parser.parse_args()
    compiler_options = compiler_control("native-chunks-strict-rounding")
    if args.native_msa_norm and args.weights is None:
        raise ValueError("native MSA norm control requires weights")
    root = args.reference.resolve(strict=True)
    report = json.loads((root / "report.json").read_text())
    bindings = dict(report["bindings"])
    bindings[str(root / "prefix.npz")] = report["archive_sha256"]
    bindings[str(root / "report.json")] = _sha256(root / "report.json")
    bindings[str(Path(__file__).resolve())] = _sha256(Path(__file__))
    compiler_source = Path(inspect.getfile(compiler_control)).resolve()
    bindings[str(compiler_source)] = _sha256(compiler_source)
    if report["engine"] != "native" or not report.get("full_pwa"):
        raise ValueError("requires native PWA capture")
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("capture binding changed")
    import jax
    import jax.numpy as jnp

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("requires one GPU")
    prefix = "blocks.0.msa_pair_weighted_averaging.einsum."
    for key, dtype in (
        ("input0", "float32"),
        ("input1", "bfloat16"),
        ("input2", "bfloat16"),
        ("output", "bfloat16"),
    ):
        if report["dtypes"][prefix + key] != dtype:
            raise ValueError("native contraction dtype differs")
    with np.load(root / "prefix.npz") as archive:
        inputs = [archive[prefix + f"input{i}"] for i in range(3)]
        expected = archive[prefix + "output"]
    if any(not np.isfinite(a).all() for a in [*inputs, expected]):
        raise ValueError("nonfinite native operand")
    operands = tuple(
        jnp.asarray(a, dtype)
        for a, dtype in zip(
            inputs, (jnp.float32, jnp.bfloat16, jnp.bfloat16), strict=True
        )
    )

    def mixed(a, v, g):
        return jnp.einsum("bijh,bjmhd,bimhd->bimhd", a, v, g).astype(jnp.bfloat16)

    def rounded(a, v, g):
        value = jnp.einsum("bijh,bjmhd->bimhd", a.astype(jnp.bfloat16), v)
        return (value.astype(jnp.bfloat16) * g).astype(jnp.bfloat16)

    def fp32_sum(a, v, g):
        value = jnp.einsum("bijh,bjmhd->bimhd", a, v.astype(jnp.float32))
        return (value.astype(jnp.bfloat16) * g).astype(jnp.bfloat16)

    results = {}
    with np.load(root / "prefix.npz") as archive:
        gate_logits = jnp.asarray(
            archive["blocks.0.msa_pair_weighted_averaging.Wgate.output"],
            jnp.bfloat16,
        ).reshape(inputs[2].shape)
    for promote in (False, True):

        def gate_call(x):
            return jax.nn.sigmoid(x.astype(jnp.float32) if promote else x).astype(
                jnp.bfloat16
            )

        fn = jax.jit(gate_call, compiler_options=compiler_options)
        a = np.asarray(fn(gate_logits).astype(jnp.float32))
        b = np.asarray(fn(gate_logits).astype(jnp.float32))
        results["gate_fp32" if promote else "gate_bf16"] = {
            "comparison": bitwise_comparison(a, inputs[2]),
            "repeat": bitwise_comparison(a, b),
        }
    with jax.default_matmul_precision("highest"):
        for name, call in (
            ("mixed", mixed),
            ("bf16_sum", rounded),
            ("fp32_sum_then_round", fp32_sum),
        ):
            fn = jax.jit(call, compiler_options=compiler_options)
            a = np.asarray(fn(*operands).astype(jnp.float32))
            b = np.asarray(fn(*operands).astype(jnp.float32))
            results[name] = {
                "comparison": bitwise_comparison(a, expected),
                "repeat": bitwise_comparison(a, b),
            }
    if args.weights is not None:
        from safetensors import safe_open

        from foldjax.models.boltz2.models.primitives import native_amp_norm
        from foldjax.models.esmfold2.models import native_softmax, primitives, trunk

        paths = (
            args.weights.resolve(strict=True),
            Path(native_softmax.__file__).resolve(),
            Path(trunk.__file__).resolve(),
            Path(primitives.__file__).resolve(),
            Path(native_amp_norm.__file__).resolve(),
        )
        if bindings.get(str(paths[0])) != _sha256(paths[0]):
            raise ValueError("weights differ from native capture")
        bindings.update({str(p): _sha256(p) for p in paths})
        if args.native_msa_norm:
            original_norm = trunk._autocast_norm

            def controlled_norm(x, params, prefix, eps=1e-5):
                if prefix.endswith(".msa_pair_weighted_averaging.norm_single"):
                    if x.shape[-1] != 128:
                        raise ValueError("norm control requires measured width128")
                    return native_amp_norm._cuda_layer_norm(
                        x.astype(jnp.float32),
                        params[prefix + ".weight"],
                        params[prefix + ".bias"],
                        eps,
                    )[0]
                return original_norm(x, params, prefix, eps)

            trunk._autocast_norm = controlled_norm
        prefix = "msa_encoder.blocks.0.msa_pair_weighted_averaging"
        with safe_open(args.weights, framework="numpy") as handle:
            params = {
                k: jnp.asarray(handle.get_tensor(k))
                for k in handle.keys()
                if k.startswith(prefix + ".")
            }
        capture_prefix = "blocks.0.msa_pair_weighted_averaging."
        with np.load(root / "prefix.npz") as archive:
            for leaf in (
                "norm_single",
                "compute_bias.0",
                "compute_bias.1",
                "Wv",
                "Wgate",
                "Wout",
                "softmax",
            ):
                key = capture_prefix + leaf
                dtype = getattr(jnp, report["dtypes"][key + ".input"])
                x = jnp.asarray(archive[key + ".input"], dtype)
                expected_leaf = archive[key + ".output"]
                full_prefix = prefix + "." + leaf
                if leaf in ("norm_single", "compute_bias.0"):

                    def run_leaf(x):
                        return trunk._autocast_norm(x, params, full_prefix)
                elif leaf == "softmax":

                    def run_leaf(x):
                        return jax.nn.softmax(x.astype(jnp.float32), axis=-2)
                else:

                    def run_leaf(x):
                        return trunk._autocast_linear(x, params, full_prefix)

                with jax.default_matmul_precision("highest"):
                    fn = jax.jit(run_leaf, compiler_options=compiler_options)
                    a = np.asarray(fn(x).astype(jnp.float32))
                    b = np.asarray(fn(x).astype(jnp.float32))
                results["leaf/" + leaf] = {
                    "comparison": bitwise_comparison(a, expected_leaf),
                    "repeat": bitwise_comparison(a, b),
                }
                if leaf == "softmax":
                    from foldjax.models.boltz2.models.primitives import (
                        native_pwa_weights,
                    )

                    path = Path(native_pwa_weights.__file__).resolve()
                    bindings[str(path)] = _sha256(path)
                    fn = jax.jit(
                        spatial_softmax_control, compiler_options=compiler_options
                    )
                    a = np.asarray(fn(x.astype(jnp.float32)))
                    b = np.asarray(fn(x.astype(jnp.float32)))
                    results["softmax_spatial_control"] = {
                        "comparison": bitwise_comparison(a, expected_leaf),
                        "repeat": bitwise_comparison(a, b),
                    }
        with np.load(root / "prefix.npz") as archive:
            msa = jnp.asarray(archive[capture_prefix + "msa_input"], jnp.bfloat16)
            pair = jnp.asarray(archive[capture_prefix + "pair_input"], jnp.bfloat16)
            mask = jnp.asarray(archive[capture_prefix + "mask_input"], bool)
            expected_pwa = archive[capture_prefix + "output"]
            native_attention = jnp.asarray(
                archive[capture_prefix + "softmax.output"], jnp.float32
            )
        for native in (False, True):
            selected = (
                params
                if native
                else {k: v.astype(jnp.bfloat16) for k, v in params.items()}
            )

            def run(m, z, p, mask):
                return trunk.msa_pair_weighted_averaging(
                    m, z, p, prefix, pair_mask=mask, native_autocast=native
                )

            with jax.default_matmul_precision("highest"):
                fn = jax.jit(run, compiler_options=compiler_options)
                a = np.asarray(fn(msa, pair, selected, mask).astype(jnp.float32))
                b = np.asarray(fn(msa, pair, selected, mask).astype(jnp.float32))
            results["pwa_native" if native else "pwa_legacy"] = {
                "comparison": bitwise_comparison(a, expected_pwa),
                "repeat": bitwise_comparison(a, b),
            }

        def injected(m, z, p, mask, attention):
            def softmax(x, axis=-1, **kw):
                if axis != -2 or x.shape != attention.shape or kw:
                    raise ValueError("unexpected PWA softmax invocation")
                return attention

            with patch.object(jax.nn, "softmax", softmax):
                return trunk.msa_pair_weighted_averaging(
                    m, z, p, prefix, pair_mask=mask, native_autocast=True
                )

        with jax.default_matmul_precision("highest"):
            fn = jax.jit(injected, compiler_options=compiler_options)
            a = np.asarray(
                fn(msa, pair, params, mask, native_attention).astype(jnp.float32)
            )
            b = np.asarray(
                fn(msa, pair, params, mask, native_attention).astype(jnp.float32)
            )
        results["pwa_native_attention_injected"] = {
            "comparison": bitwise_comparison(a, expected_pwa),
            "repeat": bitwise_comparison(a, b),
        }
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("binding changed during replay")
    save_new(
        args.out,
        {
            "results": results,
            "bindings": bindings,
            "compiler_options": compiler_options,
            "native_msa_norm": args.native_msa_norm,
            "full_model_admission": False,
        },
    )


if __name__ == "__main__":
    main()
