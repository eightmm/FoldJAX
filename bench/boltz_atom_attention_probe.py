"""First atom-layer same-operand leaf controls, not full attention admission."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from bench.af3_closure_capture import sha
from bench.boltz_amp_report import verify_capture
from bench.boltz_closure_capture import save_new
from bench.boltz_pwa_averaging_probe import bitwise_comparison

NAMES = (
    "adaln.a_norm", "adaln.s_norm", "adaln.s_scale", "adaln.s_bias",
    "pair_bias_attn.proj_q", "pair_bias_attn.proj_k", "pair_bias_attn.proj_v",
    "pair_bias_attn.proj_g", "pair_bias_attn.proj_o",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "weights", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    verify_capture(root, foldjax=False)
    capture_prefix = (
        "input_embedder.atom_attention_encoder.atom_encoder.diffusion_transformer.layers.0."
    )
    paths = {
        name: root / f"trunk-boundaries/{capture_prefix}{name}.npz" for name in NAMES
    }
    bindings = {str(p): sha(p) for p in (
        *paths.values(), args.weights.resolve(strict=True), Path(__file__).resolve(),
    )}
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.diffusion.diffusion_transformer import (
        _layer_norm_no_affine,
        _layer_norm_scale,
    )
    from foldjax.models.boltz2.models.primitives._common import linear
    from foldjax.models.boltz2.models.primitives.native_amp_norm import amp_layer_norm

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("requires one GPU")
    results = {}
    with safe_open(args.weights, framework="numpy") as handle:
        for name, path in paths.items():
            with np.load(path, allow_pickle=False) as archive:
                x, expected = archive["input"], archive["output"]
            if any(a.dtype != np.float32 or not np.isfinite(a).all()
                   for a in (x, expected)):
                raise ValueError("requires finite lossless FP32 native capture")
            prefix = (
                "d:trunk/d:input_embedder/d:atom_attention_encoder/d:atom_encoder/"
                "d:diffusion_transformer/d:layers/i:0/d:" + name.replace(".", "/d:")
            )
            if name.endswith("_norm"):
                scale = (np.ones(x.shape[-1], np.float32) if name == "adaln.a_norm"
                         else handle.get_tensor(prefix + "/d:scale"))
                def ordinary(x, scale):
                    return (_layer_norm_no_affine(x, 1e-5) if name == "adaln.a_norm"
                            else _layer_norm_scale(x, scale, 1e-5))

                def native_norm(x, scale):
                    return amp_layer_norm(x, scale, jnp.zeros_like(scale), 1e-5)

                calls = {"ordinary": ordinary, "native_amp": native_norm}
                operands = (jnp.asarray(x), jnp.asarray(scale))
            else:
                weight = handle.get_tensor(prefix + "/d:kernel")
                key = prefix + "/d:bias"
                bias = handle.get_tensor(key) if key in handle.keys() else None
                calls = {"ordinary": linear}
                operands = (jnp.asarray(x), jnp.asarray(weight, jnp.bfloat16),
                            None if bias is None else jnp.asarray(bias, jnp.bfloat16))
            with jax.default_matmul_precision("highest"):
                for arm, call in calls.items():
                    fn = jax.jit(
                        call, compiler_options={"xla_allow_excess_precision": False}
                    )
                    actual = np.asarray(fn(*operands).astype(jnp.float32))
                    repeat = np.asarray(fn(*operands).astype(jnp.float32))
                    results[f"{name}/{arm}"] = {
                        "comparison": bitwise_comparison(actual, expected),
                        "repeat": bitwise_comparison(actual, repeat),
                    }
            if name == "pair_bias_attn.proj_k":
                v_prefix = prefix.removesuffix("proj_k") + "proj_v"
                v_weight = handle.get_tensor(v_prefix + "/d:kernel")
                with np.load(paths["pair_bias_attn.proj_v"], allow_pickle=False) as z:
                    if not np.array_equal(x, z["input"]):
                        raise ValueError("native K/V operands differ")
                    expected_kv = np.concatenate((expected, z["output"]), axis=-1)

                def combined_kv(x, wk, wv):
                    return linear(x, jnp.concatenate((wk, wv), axis=-1))

                with jax.default_matmul_precision("highest"):
                    fn = jax.jit(
                        combined_kv,
                        compiler_options={"xla_allow_excess_precision": False},
                    )
                    kv_operands = (
                        jnp.asarray(x), jnp.asarray(weight, jnp.bfloat16),
                        jnp.asarray(v_weight, jnp.bfloat16),
                    )
                    actual = np.asarray(fn(*kv_operands).astype(jnp.float32))
                    repeat = np.asarray(fn(*kv_operands).astype(jnp.float32))
                results["pair_bias_attn.combined_kv/ordinary"] = {
                    "comparison": bitwise_comparison(actual, expected_kv),
                    "repeat": bitwise_comparison(actual, repeat),
                }
    if any(sha(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("bound inputs changed")
    save_new(args.out, {"results": results, "bindings": bindings,
                        "bindings_unchanged": True, "not_model_parity_admission": True})
    print(json.dumps(results))


if __name__ == "__main__":
    main()
