"""Profile cached JAX stages on the upstream production inference workload."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("outputs/compile_cache"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def warm_measure(fn):
    jax.block_until_ready(fn())
    started = time.perf_counter()
    value = jax.block_until_ready(fn())
    return value, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.cache.resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_default_matmul_precision", "default")

    from foldjax.models.protenix.bridge.weights_io import load_native_weights
    from foldjax.models.protenix.chunking import resolve_chunk_config
    from foldjax.models.protenix.data.static_io import load_static_feature_npz
    from foldjax.models.protenix.models.diffusion.atom import (
        atom_attention_encoder_prepare_diffusion_cache,
    )
    from foldjax.models.protenix.models.diffusion.diffusion import (
        diffusion_conditioning_prepare_cache,
        inference_noise_schedule,
        sample_diffusion_with_module,
    )
    from foldjax.models.protenix.models.heads.confidence import (
        confidence_head,
        confidence_scores_from_logits,
    )
    from foldjax.models.protenix.models.heads.head import distogram_head
    from foldjax.models.protenix.models.trunk_blocks.embedders import (
        input_feature_embedder,
    )
    from foldjax.models.protenix.models.trunk_blocks.trunk import (
        pairformer_output_from_s_inputs,
    )

    features = load_static_feature_npz(args.features)
    params = load_native_weights(args.weights)
    n_token = int(features["restype"].shape[-2])
    chunks = resolve_chunk_config(
        n_token=n_token,
        n_sample=args.samples,
        policy="auto",
    )
    s_inputs, input_seconds = warm_measure(
        lambda: input_feature_embedder(
            features,
            params.input_embedder,
            n_token=n_token,
            n_heads=4,
            n_queries=32,
            n_keys=128,
            use_scan=False,
        )
    )
    trunk, trunk_seconds = warm_measure(
        lambda: pairformer_output_from_s_inputs(
            features,
            s_inputs,
            params.pairformer_output,
            n_cycle=args.cycles,
            use_pairformer_scan=False,
            single_attention_backend="xla_jit",
            triangle_attention_backend="xla_jit",
        )
    )
    _, s_trunk, z_trunk = trunk

    def prepare_cache():
        pair_z = diffusion_conditioning_prepare_cache(
            features["relp"], z_trunk, params.diffusion.conditioning
        )
        p_lm, c_l = atom_attention_encoder_prepare_diffusion_cache(
            features["atom_to_token_idx"],
            features["ref_pos"],
            features["ref_charge"],
            features["ref_mask"],
            features["ref_element"],
            features["ref_atom_name_chars"],
            features["d_lm"],
            features["v_lm"],
            features["pad_info"],
            pair_z[..., None, :, :, :],
            params.diffusion.atom_encoder,
            n_queries=32,
            n_keys=128,
        )
        return pair_z, p_lm, c_l

    (pair_z, p_lm, c_l), cache_seconds = warm_measure(prepare_cache)
    schedule = inference_noise_schedule(n_step=args.steps)
    coordinates, diffusion_seconds = warm_measure(
        lambda: sample_diffusion_with_module(
            features,
            s_inputs,
            s_trunk,
            z_trunk,
            params.diffusion,
            schedule,
            n_sample=args.samples,
            key=jax.random.PRNGKey(args.seed),
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            use_scan=False,
            use_sampler_scan=False,
            use_denoiser_jit=False,
            attention_backend="xla_jit",
            token_q_chunk_size=chunks.token_q_chunk_size,
            diffusion_chunk_size=chunks.diffusion_chunk_size,
        )
    )
    distogram, distogram_seconds = warm_measure(
        lambda: distogram_head(z_trunk, params.distogram)
    )
    logits, confidence_seconds = warm_measure(
        lambda: confidence_head(
            features,
            s_inputs,
            s_trunk,
            z_trunk,
            None,
            coordinates,
            params.confidence,
            use_embedding=True,
            use_scan=True,
            triangle_mul_chunk_size=chunks.triangle_mul_chunk_size,
            triangle_att_q_chunk_size=chunks.triangle_att_q_chunk_size,
            single_att_q_chunk_size=chunks.single_att_q_chunk_size,
        )
    )
    _, scores_seconds = warm_measure(
        lambda: confidence_scores_from_logits(
            plddt_logits=logits["plddt"],
            pae_logits=logits["pae"],
            pde_logits=logits["pde"],
            distogram_logits=distogram,
            token_has_frame=features.get("has_frame"),
            token_asym_id=features.get("asym_id"),
            atom_to_token_idx=features.get("atom_to_token_idx"),
            atom_coordinate=coordinates,
            elements_one_hot=features.get("ref_element"),
            mol_id=features.get("mol_id"),
        )
    )
    stages = {
        "input_embedder_seconds": input_seconds,
        "trunk_seconds": trunk_seconds,
        "diffusion_cache_seconds": cache_seconds,
        "diffusion_seconds": diffusion_seconds,
        "distogram_seconds": distogram_seconds,
        "confidence_head_seconds": confidence_seconds,
        "confidence_scores_seconds": scores_seconds,
    }
    metrics = {
        "backend": "jax",
        "tokens": n_token,
        "atoms": int(features["atom_to_token_idx"].shape[-1]),
        "cycles": args.cycles,
        "diffusion_steps": args.steps,
        "samples": args.samples,
        "stages": stages,
        "stage_sum_seconds": sum(stages.values()),
    }
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
