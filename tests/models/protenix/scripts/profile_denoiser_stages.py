"""Profile one Protenix JAX denoiser step by major stack."""

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
    parser.add_argument("--denoise0", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=10)
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
    from foldjax.models.protenix.models.diffusion.atom import (
        atom_attention_decoder,
        atom_attention_encoder,
        atom_attention_encoder_prepare_diffusion_cache,
    )
    from foldjax.models.protenix.models.diffusion.diffusion import (
        diffusion_conditioning,
        diffusion_conditioning_prepare_cache,
    )
    from foldjax.models.protenix.models.diffusion.transformer import (
        diffusion_transformer_stack,
    )
    from foldjax.models.protenix.models.primitives.primitives import layer_norm, linear
    from foldjax.models.protenix.models.trunk_blocks.embedders import (
        input_feature_embedder,
    )
    from foldjax.models.protenix.models.trunk_blocks.trunk import (
        pairformer_output_from_s_inputs,
    )

    features = load_static_feature_npz(args.features)
    params = load_native_weights(args.weights)
    with np.load(args.denoise0) as data:
        x_noisy = jnp.asarray(data["x_noisy"].astype(np.float32))
        t_hat = jnp.asarray(data["t_hat"].astype(np.float32))

    n_token = int(features["restype"].shape[-2])
    s_inputs = ready(
        input_feature_embedder(
            features,
            params.input_embedder,
            n_token=n_token,
            n_heads=4,
            n_queries=32,
            n_keys=128,
            use_scan=False,
        )
    )
    _, s_trunk, z_trunk = ready(
        pairformer_output_from_s_inputs(
            features,
            s_inputs,
            params.pairformer_output,
            num_recycles=args.cycles,
            use_pairformer_scan=False,
        )
    )
    pair_z = ready(
        diffusion_conditioning_prepare_cache(
            features["relp"], z_trunk, params.diffusion.conditioning
        )
    )
    p_lm, c_l = ready(
        atom_attention_encoder_prepare_diffusion_cache(
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
    )
    conditioning_out, conditioning_seconds = warm_measure(
        lambda: diffusion_conditioning(
            t_hat,
            features["relp"],
            s_inputs,
            s_trunk,
            z_trunk,
            params.diffusion.conditioning,
            pair_z=pair_z,
        )
    )
    single_s, _ = conditioning_out
    scale = jnp.sqrt(16.0**2 + t_hat**2)[..., None, None]
    r_noisy = x_noisy / scale
    s_trunk_sample = jnp.expand_dims(s_trunk, axis=-3)
    z_pair_sample = jnp.expand_dims(pair_z, axis=-4)

    def run_atom_encoder(use_scan: bool):
        return atom_attention_encoder(
            features["atom_to_token_idx"],
            features["ref_pos"],
            features["ref_charge"],
            features["ref_mask"],
            features["ref_atom_name_chars"],
            features["ref_element"],
            features["d_lm"],
            features["v_lm"],
            features["pad_info"],
            params.diffusion.atom_encoder,
            r_l=r_noisy,
            s=s_trunk_sample,
            z=z_pair_sample,
            p_lm=p_lm,
            c_l=c_l,
            n_token=n_token,
            n_heads=4,
            n_queries=32,
            n_keys=128,
            use_scan=use_scan,
        )

    atom_outputs, atom_loop_seconds = warm_measure(lambda: run_atom_encoder(False))
    _, atom_scan_seconds = warm_measure(lambda: run_atom_encoder(True))
    a_token, q_skip, c_skip, p_skip = atom_outputs
    a_token = a_token.astype(jnp.float32) + linear(
        layer_norm(single_s, params.diffusion.layernorm_s),
        params.diffusion.linear_s,
    )

    def run_token_transformer(use_scan: bool):
        return diffusion_transformer_stack(
            a_token,
            single_s.astype(jnp.float32),
            z_pair_sample.astype(jnp.float32),
            params.diffusion.diffusion_transformer,
            num_heads=16,
            use_scan=use_scan,
        )

    token_out, token_loop_seconds = warm_measure(
        lambda: run_token_transformer(False)
    )
    _, token_scan_seconds = warm_measure(lambda: run_token_transformer(True))
    token_out = layer_norm(token_out, params.diffusion.layernorm_a)

    def run_decoder(use_scan: bool):
        return atom_attention_decoder(
            features["atom_to_token_idx"],
            token_out,
            q_skip,
            c_skip,
            p_skip,
            params.diffusion.atom_decoder,
            n_heads=4,
            n_queries=32,
            n_keys=128,
            use_scan=use_scan,
        )

    coordinate, decoder_loop_seconds = warm_measure(lambda: run_decoder(False))
    _, decoder_scan_seconds = warm_measure(lambda: run_decoder(True))
    metrics = {
        "backend": jax.default_backend(),
        "tokens": n_token,
        "atoms": int(features["atom_to_token_idx"].shape[-1]),
        "conditioning_seconds": conditioning_seconds,
        "atom_encoder_loop_seconds": atom_loop_seconds,
        "atom_encoder_scan_seconds": atom_scan_seconds,
        "token_transformer_loop_seconds": token_loop_seconds,
        "token_transformer_scan_seconds": token_scan_seconds,
        "atom_decoder_loop_seconds": decoder_loop_seconds,
        "atom_decoder_scan_seconds": decoder_scan_seconds,
        "loop_stage_sum_seconds": (
            conditioning_seconds
            + atom_loop_seconds
            + token_loop_seconds
            + decoder_loop_seconds
        ),
        "coordinate_checksum": float(jnp.sum(coordinate)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
