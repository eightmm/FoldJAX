"""Captured-operand attention core replay; not whole-model parity admission."""

import argparse
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new


def operands(root):
    prefix = "trunk-boundaries/input_embedder.atom_attention_encoder.atom_encoder."
    prefix += "diffusion_transformer.layers.0.pair_bias_attn."
    paths = [root / (prefix + f"proj_{name}.npz") for name in "qkv"]
    paths += [root / "trunk-boundaries/input_embedder.atom_enc_proj_z.npz",
              root / "features.npz"]
    arrays = []
    for path in paths[:3]:
        with np.load(path, allow_pickle=False) as z:
            arrays.append(z["output"].reshape(97, -1, 4, 32))
    with np.load(paths[3], allow_pickle=False) as z:
        bias = z[z.files[0]].reshape(97, 32, 128, 12)[..., :4]
    with np.load(paths[4], allow_pickle=False) as z:
        mask = z["atom_pad_mask"].reshape(-1)
    if mask.shape != (3104,):
        raise ValueError("only observed 3104-atom profile supported")
    # Native windows start three half-windows before each query window.
    indices = np.arange(97)[:, None] * 32 - 48 + np.arange(128)[None, :]
    valid = (indices >= 0) & (indices < len(mask))
    keys = mask[np.clip(indices, 0, len(mask) - 1)] * valid
    arrays += [bias.transpose(0, 3, 1, 2), keys.astype(np.float32)]
    return arrays, {str(p): sha(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("native", "jax"), required=True)
    parser.add_argument("--native-core", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    values, bindings = operands(args.reference)
    bindings[str(Path(__file__).resolve())] = sha(Path(__file__))
    if args.backend == "native":
        import torch

        torch.set_float32_matmul_precision("highest")
        q, k, v, bias, mask = [torch.from_numpy(x).cuda() for x in values]
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            score = torch.einsum("bihd,bjhd->bhij", q, k)
            logits = score / (32 ** 0.5) + bias
            logits = logits + (1 - mask[:, None, None]) * -1e6
            probability = logits.softmax(-1)
            output = torch.einsum("bhij,bjhd->bihd", probability, v)
        result = {n: x.cpu().numpy() for n, x in
                  zip(("score", "logits", "probability", "output"),
                      (score, logits, probability, output), strict=True)}
    else:
        import jax
        import jax.numpy as jnp

        def core(q, k, v, bias, mask):
            score = jnp.einsum("bihd,bjhd->bhij", q, k)
            logits = score / jnp.sqrt(jnp.float32(32)) + bias
            logits = logits + (1 - mask[:, None, None]) * -1e6
            probability = jax.nn.softmax(logits, -1)
            output = jnp.einsum("bhij,bjhd->bihd", probability, v)
            return dict(score=score, logits=logits,
                        probability=probability, output=output)

        with jax.default_matmul_precision("highest"):
            fn = jax.jit(core, compiler_options={"xla_allow_excess_precision": False})
            result = fn(*[jnp.asarray(x) for x in values])
            result = {n: np.asarray(x) for n, x in result.items()}
        if args.native_core is not None:
            reference_path = args.native_core / "arrays.npz"
            bindings[str(reference_path)] = sha(reference_path)
            with np.load(reference_path, allow_pickle=False) as z:
                native_probability = z["probability"]
                native_logits = z["logits"]
            with jax.default_matmul_precision("highest"):
                from foldjax.models.boltz2.models.primitives.native_pwa_weights import (
                    warp_softmax,
                )

                def padded_warp(logits):
                    padded = jnp.pad(logits, ((0, 0), (0, 0), (0, 0), (0, 309)),
                                     constant_values=-jnp.inf)
                    return warp_softmax(padded)[..., :128]

                warp_fn = jax.jit(
                    padded_warp,
                    compiler_options={"xla_allow_excess_precision": False},
                )
                result["warp_probability"] = np.asarray(
                    warp_fn(jnp.asarray(native_logits))
                )
                direct_fn = jax.jit(
                    warp_softmax,
                    compiler_options={"xla_allow_excess_precision": False},
                )
                result["direct_warp_probability"] = np.asarray(
                    direct_fn(jnp.asarray(native_logits))
                )
                fn = jax.jit(
                    lambda p, v: jnp.einsum("bhij,bjhd->bihd", p, v),
                    compiler_options={"xla_allow_excess_precision": False},
                )
                result["native_probability_output"] = np.asarray(fn(
                    jnp.asarray(native_probability), jnp.asarray(values[2])
                ))
                from foldjax.models.boltz2.models.diffusion import (
                    diffusion_transformer as transformer,
                )
                fn = jax.jit(
                    transformer._no_proj_qblock,
                    compiler_options={"xla_allow_excess_precision": False},
                )
                q, k, v, bias, mask = [jnp.asarray(x) for x in values]
                result["production_output"] = np.asarray(fn(
                    q, k, v, bias, mask[:, None, None, :].astype(bool),
                    jnp.sqrt(jnp.float32(32)),
                ))
    if any(sha(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("inputs changed")
    args.out.mkdir(parents=True)
    np.savez(args.out / "arrays.npz", **result)
    save_new(args.out / "report.json", {
        "bindings": bindings, "arrays_sha256": sha(args.out / "arrays.npz"),
        "backend": args.backend, "not_model_parity_admission": True,
        "scope": "first layer captured QKV; reconstructed mask; explicit core replay",
    })


if __name__ == "__main__":
    main()
