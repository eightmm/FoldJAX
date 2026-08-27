"""Profile warm Protenix JAX inference stages on static features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--diffusion-scan", action="store_true")
    parser.add_argument("--sampler-scan", action="store_true")
    parser.add_argument("--denoiser-jit", action="store_true")
    parser.add_argument(
        "--diffusion-attention-backend",
        choices=("xla", "xla_jit", "xla_sdpa", "cudnn"),
        default="xla",
    )
    parser.add_argument(
        "--trunk-single-attention-backend",
        choices=("xla", "xla_jit", "xla_sdpa", "cudnn"),
        default="xla",
    )
    parser.add_argument(
        "--trunk-triangle-attention-backend",
        choices=("xla", "xla_jit", "tokamax", "cueq", "cueq_jit"),
        default="cueq_jit",
    )
    parser.add_argument("--no-pairformer-scan", action="store_true")
    return parser.parse_args()


def ready(value):
    return jax.tree.map(jax.block_until_ready, value)


def warm_measure(fn):
    ready(fn())
    started = time.perf_counter()
    value = ready(fn())
    return value, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")

    from foldjax.models.protenix.bridge.weights_io import load_native_weights
    from foldjax.models.protenix.data.static_io import load_static_feature_npz
    from foldjax.models.protenix.models.diffusion.diffusion import (
        diffusion_conditioning_prepare_cache,
        inference_noise_schedule,
        sample_diffusion_with_module,
    )
    from foldjax.models.protenix.models.trunk_blocks.embedders import (
        input_feature_embedder,
    )
    from foldjax.models.protenix.models.trunk_blocks.msa import msa_module
    from foldjax.models.protenix.models.trunk_blocks.pairformer import pairformer_stack
    from foldjax.models.protenix.models.trunk_blocks.template import template_embedder
    from foldjax.models.protenix.models.trunk_blocks.trunk import (
        pairformer_output_from_s_inputs,
        recycle_embeddings,
        trunk_initial_embeddings,
    )

    features = load_static_feature_npz(args.features)
    params = load_native_weights(args.weights)
    with np.load(args.noise) as data:
        init_noise = jnp.asarray(data["init"].astype(np.float32))
        step_noises = tuple(
            jnp.asarray(value) for value in data["steps"].astype(np.float32)
        )

    n_token = int(features["restype"].shape[-2])
    s_inputs, input_seconds = warm_measure(
        lambda: input_feature_embedder(
            features,
            params.input_embedder,
            n_token=n_token,
            n_heads=4,
            n_queries=32,
            n_keys=128,
            use_scan=args.diffusion_scan,
        )
    )
    trunk, trunk_seconds = warm_measure(
        lambda: pairformer_output_from_s_inputs(
            features,
            s_inputs,
            params.pairformer_output,
            num_recycles=args.cycles,
            use_pairformer_scan=not args.no_pairformer_scan,
            single_attention_backend=args.trunk_single_attention_backend,
            triangle_attention_backend=args.trunk_triangle_attention_backend,
        )
    )
    _, s_trunk, z_trunk = trunk
    s_init, z_init = trunk_initial_embeddings(
        s_inputs,
        features["relp"],
        features["token_bonds"],
        params.pairformer_output.trunk.initial,
    )
    recycle, recycle_seconds = warm_measure(
        lambda: recycle_embeddings(
            s_init,
            z_init,
            jnp.zeros_like(s_init),
            jnp.zeros_like(z_init),
            params.pairformer_output.trunk.recycling,
        )
    )
    cycle_s, cycle_z = recycle
    template_update, template_seconds = warm_measure(
        lambda: template_embedder(
            features,
            cycle_z,
            None,
            params.pairformer_output.template,
            triangle_attention_backend=args.trunk_triangle_attention_backend,
        )
    )
    cycle_z = cycle_z + template_update
    cycle_z, msa_seconds = warm_measure(
        lambda: msa_module(
            features,
            cycle_z,
            s_inputs,
            None,
            params.pairformer_output.msa,
            triangle_attention_backend=args.trunk_triangle_attention_backend,
        )
    )
    _, pairformer_seconds = warm_measure(
        lambda: pairformer_stack(
            cycle_s,
            cycle_z,
            None,
            params.pairformer_output.pairformer_stack,
            use_scan=not args.no_pairformer_scan,
            single_attention_backend=args.trunk_single_attention_backend,
            triangle_attention_backend=args.trunk_triangle_attention_backend,
        )
    )
    from foldjax.models.protenix.models.diffusion.atom import (
        atom_attention_encoder_prepare_diffusion_cache,
    )

    def prepare_diffusion_cache():
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
            jnp.expand_dims(pair_z, axis=-4),
            params.diffusion.atom_encoder,
            n_queries=32,
            n_keys=128,
        )
        return pair_z, p_lm, c_l

    (pair_z, p_lm, c_l), conditioning_seconds = warm_measure(
        prepare_diffusion_cache
    )
    schedule = inference_noise_schedule(num_steps=args.steps)
    coordinate, diffusion_seconds = warm_measure(
        lambda: sample_diffusion_with_module(
            features,
            s_inputs,
            s_trunk,
            z_trunk,
            params.diffusion,
            schedule,
            num_samples=1,
            key=jax.random.PRNGKey(0),
            init_noise=init_noise,
            step_noises=step_noises,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            use_scan=args.diffusion_scan,
            use_sampler_scan=args.sampler_scan,
            use_denoiser_jit=args.denoiser_jit,
            attention_backend=args.diffusion_attention_backend,
        )
    )
    metrics = {
        "backend": jax.default_backend(),
        "tokens": n_token,
        "atoms": int(features["atom_to_token_idx"].shape[-1]),
        "msa_rows": int(features["msa"].shape[-2]),
        "cycles": args.cycles,
        "num_steps": args.steps,
        "diffusion_scan": args.diffusion_scan,
        "sampler_scan": args.sampler_scan,
        "denoiser_jit": args.denoiser_jit,
        "diffusion_attention_backend": args.diffusion_attention_backend,
        "trunk_single_attention_backend": args.trunk_single_attention_backend,
        "trunk_triangle_attention_backend": args.trunk_triangle_attention_backend,
        "pairformer_scan": not args.no_pairformer_scan,
        "input_embedder_seconds": input_seconds,
        "trunk_seconds": trunk_seconds,
        "representative_cycle": {
            "recycle_seconds": recycle_seconds,
            "template_seconds": template_seconds,
            "msa_seconds": msa_seconds,
            "pairformer_seconds": pairformer_seconds,
            "sum_seconds": (
                recycle_seconds
                + template_seconds
                + msa_seconds
                + pairformer_seconds
            ),
        },
        "conditioning_cache_seconds": conditioning_seconds,
        "diffusion_seconds": diffusion_seconds,
        "stage_sum_seconds": (
            input_seconds + trunk_seconds + conditioning_seconds + diffusion_seconds
        ),
        "coordinate_checksum": float(jnp.sum(coordinate)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
