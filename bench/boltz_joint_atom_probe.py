"""Explicit library atom-boundary intervention; development-only, not admission."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new


def validate_embedding_calls(enabled, observed, recycling_steps):
    expected = recycling_steps + 1 if enabled else 0
    if observed != expected:
        raise ValueError(f"incomplete MSA embedding capture: {observed} != {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffi-library", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--capture-atom-bias", action="store_true")
    parser.add_argument("--native-atom-adaln", action="store_true")
    parser.add_argument("--native-atom-softmax", action="store_true")
    parser.add_argument("--native-atom-pooling", action="store_true")
    parser.add_argument("--capture-msa-embedding", action="store_true")
    parser.add_argument("--native-recycle-norm", action="store_true")
    parser.add_argument("--native-single-prenorm", action="store_true")
    parser.add_argument("--native-single-projections", action="store_true")
    args, remaining = parser.parse_known_args()
    library = args.ffi_library.resolve(strict=True)
    before = sha(library)
    import jax
    import jax.numpy as jnp

    from bench.boltz_foldjax_capture import main as capture
    from bench.native_cublaslt_ffi import linear, register
    from bench.protenix_foldjax_capture import save_jax_boundary
    from foldjax.models.boltz2.models.diffusion import (
        atom,
        diffusion_conditioning,
        diffusion_transformer,
    )
    from foldjax.models.boltz2.models.primitives import attention as single_attention
    from foldjax.models.boltz2.models.primitives.native_amp_norm import amp_layer_norm
    from foldjax.models.boltz2.models.trunk_blocks import (
        input_embedder,
        msa,
        pairformer,
        trunk,
    )

    fp32, bf16 = register(library, fp32=True), register(library)
    original_atom = input_embedder.atom_encoder_forward
    original_linear = diffusion_conditioning._linear
    original_embedding_linear = input_embedder._linear
    original_attention = input_embedder.atom_attention_encoder_forward
    original_softmax = diffusion_transformer.masked_softmax
    original_pool = atom.scatter_atoms_to_tokens_mean
    original_msa_embedding = msa._msa_input_embedding
    original_recycle_norm = trunk._layer_norm
    original_single_norm = pairformer._layer_norm
    original_single_attention = pairformer.attention_pair_bias_forward
    original_single_linear = single_attention._linear

    def single_linear(x, kernel, bias=None, **kwargs):
        if kernel.shape == (384, 768) and x.dtype == jnp.float32:
            if kernel.dtype != jnp.float32 or bias is not None or kwargs:
                raise ValueError("single projection requires FP32 no-bias GEMM")
            counts["single_projections"] += 1
            left, right = jnp.split(kernel, 2, axis=-1)
            return jnp.concatenate((
                linear(x, left.T, target=fp32, fp32=True),
                linear(x, right.T, target=fp32, fp32=True),
            ), axis=-1)
        return original_single_linear(x, kernel, bias, **kwargs)

    def single_attention_forward(*positional, **kwargs):
        if not args.native_single_projections:
            return original_single_attention(*positional, **kwargs)
        with patch.object(single_attention, "_linear", single_linear):
            return original_single_attention(*positional, **kwargs)

    def single_norm(x, scale, bias, eps):
        if args.native_single_prenorm:
            if x.shape[-1] not in (16, 64, 128, 256):
                raise ValueError("native single norm unavailable for this width")
            return amp_layer_norm(x, scale, bias, eps)
        return original_single_norm(x, scale, bias, eps)

    def recycle_norm(x, scale, bias, eps):
        if args.native_recycle_norm:
            return amp_layer_norm(x, scale, bias, eps)
        return original_recycle_norm(x, scale, bias, eps)
    embedding_calls = [0]

    def save_embedding(value):
        index = embedding_calls[0]
        embedding_calls[0] += 1
        save_jax_boundary(args.out_dir / f"msa-embedding-{index:02d}.npz", value)

    def msa_embedding(*a, **kw):
        result = original_msa_embedding(*a, **kw)
        if args.capture_msa_embedding:
            jax.debug.callback(save_embedding, result, ordered=True)
        return result
    counts = {"first_projection": 0, "attention_bias": 0}
    if args.native_single_projections:
        counts["single_projections"] = 0
    if args.native_atom_adaln:
        counts.update(adaln_scaled=0, adaln_unscaled=0)
    if args.native_atom_softmax:
        counts["atom_softmax"] = 0
    if args.native_atom_pooling:
        counts["atom_pooling"] = 0

    def atom_pooling(mapping, values, eps=1e-6, index=None, num_tokens=None):
        if not args.native_atom_pooling:
            return original_pool(mapping, values, eps, index, num_tokens)
        if mapping is None:
            if index is None or num_tokens is None:
                raise ValueError("pooling requires ownership")
            mapping = jax.nn.one_hot(index[0], num_tokens, dtype=jnp.float32)
            mapping = mapping * index[1][..., None]
        mapping = mapping.astype(jnp.float32)
        normalized = mapping / (mapping.sum(axis=1, keepdims=True) + eps)
        counts["atom_pooling"] += 1
        return jnp.matmul(normalized.swapaxes(1, 2), values.astype(jnp.float32))

    def atom_softmax(logits, mask):
        if not args.native_atom_softmax:
            return original_softmax(logits, mask)
        from foldjax.models.boltz2.models.primitives.native_pwa_weights import (
            warp_softmax,
        )
        if logits.shape[-1] != 128 or logits.dtype != jnp.float32:
            raise ValueError("unexpected input atom softmax profile")
        counts["atom_softmax"] += 1
        valid = jnp.asarray(mask, bool)
        nonempty = jnp.any(valid, axis=-1, keepdims=True)
        masked = jnp.where(valid, logits, -jnp.inf)
        safe = jnp.where(nonempty, masked, jnp.zeros_like(masked))
        return jnp.where(nonempty, warp_softmax(safe), jnp.zeros_like(safe))

    def scaled_norm(x, scale, eps):
        counts["adaln_scaled"] += 1
        return amp_layer_norm(x, scale, jnp.zeros_like(scale), eps)

    def unscaled_norm(x, eps):
        counts["adaln_unscaled"] += 1
        scale = jnp.ones((x.shape[-1],), dtype=jnp.float32)
        return amp_layer_norm(x, scale, jnp.zeros_like(scale), eps)

    def atom_attention(*a, **kw):
        # Scope includes attention and transition AdaLN in the input atom stack,
        # but excludes the diffusion atom stack and token transformer.
        with (
            patch.object(atom, "scatter_atoms_to_tokens_mean", atom_pooling),
            patch.object(diffusion_transformer, "masked_softmax", atom_softmax),
            patch.object(diffusion_transformer, "_layer_norm_scale",
                         scaled_norm if args.native_atom_adaln
                         else diffusion_transformer._layer_norm_scale),
            patch.object(diffusion_transformer, "_layer_norm_no_affine",
                         unscaled_norm if args.native_atom_adaln
                         else diffusion_transformer._layer_norm_no_affine),
        ):
            return original_attention(*a, **kw)

    def first_projection(x, kernel, bias=None, **kwargs):
        if x.shape == (1, 3104, 388) and kernel.shape == (388, 128):
            if bias is None or kwargs:
                raise ValueError("unexpected first projection contract")
            counts["first_projection"] += 1
            return linear(x, kernel.T, target=fp32, fp32=True) + bias
        return original_linear(x, kernel, bias, **kwargs)

    def atom_encoder(*a, **kw):
        if kw.get("structure_prediction") is not False:
            raise ValueError("intervention is input embedder only")
        with patch.object(diffusion_conditioning, "_linear", first_projection):
            return original_atom(*a, **kw)

    def attention_bias(x, kernel, bias=None, **kwargs):
        if x.shape == (1, 97, 32, 128, 16) and kernel.shape == (16, 12):
            if bias is not None or kwargs or kernel.dtype != jnp.bfloat16:
                raise ValueError("unexpected attention bias projection contract")
            counts["attention_bias"] += 1
            output = linear(x, kernel.T, target=bf16)
            if args.capture_atom_bias:
                jax.debug.callback(
                    lambda value: save_jax_boundary(
                        args.out_dir / "atom-bias.npz", value
                    ),
                    output, ordered=True,
                )
            return output
        return original_embedding_linear(x, kernel, bias, **kwargs)

    with (
        patch.object(
            pairformer, "attention_pair_bias_forward", single_attention_forward
        ),
        patch.object(pairformer, "_layer_norm", single_norm),
        patch.object(trunk, "_layer_norm", recycle_norm),
        patch.object(msa, "_msa_input_embedding", msa_embedding),
        patch.object(input_embedder, "atom_encoder_forward", atom_encoder),
        patch.object(input_embedder, "_linear", attention_bias),
        patch.object(input_embedder, "atom_attention_encoder_forward", atom_attention),
    ):
        capture(["--source-root", str(args.source_root), "--out-dir", str(args.out_dir),
                 *remaining])
    if sha(library) != before or any(n == 0 for n in counts.values()):
        raise ValueError("intervention missing or library changed")
    effective = json.loads((args.out_dir / "effective-options.json").read_text())
    validate_embedding_calls(
        args.capture_msa_embedding, embedding_calls[0], effective["recycling_steps"]
    )
    save_new(args.out_dir / "ffi-intervention.json", {
        "library_sha256": before, "wrapper_sha256": sha(Path(__file__)),
        "trace_counts_not_runtime_counts": counts,
        "native_atom_adaln": args.native_atom_adaln,
        "native_atom_softmax": args.native_atom_softmax,
        "native_atom_pooling": args.native_atom_pooling,
        "native_recycle_norm": args.native_recycle_norm,
        "native_single_prenorm": args.native_single_prenorm,
        "native_single_projections": args.native_single_projections,
        "msa_embedding_runtime_calls": embedding_calls[0],
        "msa_embedding_hashes": {
            f"msa-embedding-{i:02d}.npz": sha(
                args.out_dir / f"msa-embedding-{i:02d}.npz"
            ) for i in range(embedding_calls[0])
        },
        "atom_bias_sha256": (
            sha(args.out_dir / "atom-bias.npz") if args.capture_atom_bias else None
        ),
        "scope": (
            "input embedder projections; optional input atom-stack AdaLN, "
            "softmax and normalized-mapping pooling"
        ),
        "not_model_parity_admission": True,
    })


if __name__ == "__main__":
    main()
