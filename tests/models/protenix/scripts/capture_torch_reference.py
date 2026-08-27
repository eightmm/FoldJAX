"""Capture a noise-matched upstream Protenix reference run."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def load_features(path: Path, device: torch.device) -> dict:
    features: dict = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if key.startswith("pad_info."):
                continue
            array = data[key]
            if array.dtype.kind in "USO":
                # Feature dumps carry a few string fields (chain ids and the
                # like). They are not model inputs and torch cannot hold them,
                # so a dump containing any of them used to abort the capture.
                continue
            value = torch.from_numpy(array)
            if value.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                value = value.long()
            features[key] = value.to(device)
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "PROTENIX_ROOT_DIR", str(Path(__file__).resolve().parents[2] / "protenix")
    )
    os.environ.setdefault("LAYERNORM_TYPE", "torch")

    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from ml_collections.config_dict import ConfigDict
    from protenix.config.config import parse_configs
    from protenix.model import generator
    from protenix.model.protenix import Protenix, update_input_feature_dict
    from protenix.model.utils import centre_random_augmentation

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_name = "protenix_base_default_v1.0.0"
    inference_configs["model_name"] = model_name
    configs = parse_configs(
        configs={**configs_base, **{"data": data_configs}, **inference_configs},
        fill_required_with_null=True,
    )
    configs.update(ConfigDict(model_configs[model_name]))
    configs.model.N_cycle = args.cycles
    configs.sample_diffusion.N_sample = 1
    configs.sample_diffusion.N_step = args.steps
    configs.dtype = "fp32"
    configs.use_msa = True
    configs.triangle_multiplicative = "torch"
    configs.triangle_attention = "torch"
    configs.enable_diffusion_shared_vars_cache = False
    configs.enable_efficient_fusion = False
    configs.enable_tf32 = False
    configs.use_template = False
    configs.use_rna_msa = False
    configs.sample_diffusion.guidance.enable = False
    device = torch.device("cuda:0")
    model = Protenix(configs).to(device)
    checkpoint_path = (
        Path(os.environ["PROTENIX_ROOT_DIR"]) / "checkpoint" / (model_name + ".pt")
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"]
    if next(iter(state_dict)).startswith("module."):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=configs.load_strict)
    model.eval()
    features = load_features(args.features, device)
    features.pop("d_lm", None)
    features.pop("v_lm", None)
    features.pop("pad_info", None)
    features = update_input_feature_dict(features)
    # The JAX static-feature path consumes all materialized MSA rows. Pin the
    # upstream sampler to the same full-depth, ordered contract for parity.
    msa_rows = int(features["msa"].shape[-2])
    model.msa_module.msa_configs["strategy"] = "topk"
    model.msa_module.msa_configs["test_lowerb"] = msa_rows

    pairformer_kwargs = {
        "input_feature_dict": features,
        "N_cycle": args.cycles,
        "inplace_safe": True,
        "chunk_size": None,
        "mc_dropout": False,
    }
    with torch.inference_mode():
        model.get_pairformer_output(**pairformer_kwargs)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        s_inputs, s_trunk, z_trunk = model.get_pairformer_output(**pairformer_kwargs)
    torch.cuda.synchronize(device)
    trunk_seconds = time.perf_counter() - started

    noise_schedule = model.inference_noise_scheduler(
        N_step=args.steps, device=device, dtype=s_inputs.dtype
    )
    captured_noise: list[np.ndarray] = []
    denoise0: dict[str, np.ndarray] = {}
    original_randn = torch.randn
    original_centre = generator.centre_random_augmentation
    original_forward = model.diffusion_module.forward

    def record_randn(*pos, **kwargs):
        value = original_randn(*pos, **kwargs)
        captured_noise.append(value.detach().cpu().numpy())
        return value

    def centre_only(*pos, **kwargs):
        kwargs["centre_only"] = True
        return centre_random_augmentation(*pos, **kwargs)

    def record_denoise(*pos, **kwargs):
        output = original_forward(*pos, **kwargs)
        if not denoise0:
            denoise0["x_noisy"] = kwargs["x_noisy"].detach().cpu().numpy()
            denoise0["t_hat"] = kwargs["t_hat_noise_level"].detach().cpu().numpy()
            denoise0["x_denoised"] = output.detach().cpu().numpy()
        return output

    generator.torch.randn = record_randn
    generator.centre_random_augmentation = centre_only
    model.diffusion_module.forward = record_denoise
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            coordinate = model.sample_diffusion(
                denoise_net=model.diffusion_module,
                input_feature_dict=features,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=None,
                p_lm=None,
                c_l=None,
                N_sample=1,
                noise_schedule=noise_schedule,
                inplace_safe=True,
                enable_efficient_fusion=False,
            )
        torch.cuda.synchronize(device)
    finally:
        generator.torch.randn = original_randn
        generator.centre_random_augmentation = original_centre
        model.diffusion_module.forward = original_forward
    diffusion_seconds = time.perf_counter() - started

    if len(captured_noise) != args.steps + 1:
        raise RuntimeError(
            f"expected {args.steps + 1} noise tensors, got {len(captured_noise)}"
        )
    np.savez_compressed(
        args.out_dir / "noise.npz",
        init=captured_noise[0],
        steps=np.stack(captured_noise[1:]),
    )
    np.save(args.out_dir / "coordinate.npy", coordinate.detach().cpu().numpy())
    np.savez_compressed(
        args.out_dir / "trunk.npz",
        s_inputs=s_inputs.detach().cpu().numpy(),
        s_trunk=s_trunk.detach().cpu().numpy(),
        z_trunk=z_trunk.detach().cpu().numpy(),
    )
    np.savez_compressed(args.out_dir / "denoise0.npz", **denoise0)
    metrics = {
        "backend": "torch",
        "checkpoint": "protenix_base_default_v1.0.0",
        "precision": "fp32_tf32_disabled",
        "tokens": int(features["residue_index"].shape[-1]),
        "atoms": int(features["atom_to_token_idx"].shape[-1]),
        "msa_rows": int(features["msa"].shape[-2]),
        "msa_sampling": "full_depth_topk",
        "cycles": args.cycles,
        "num_steps": args.steps,
        "trunk_seconds": trunk_seconds,
        "diffusion_seconds": diffusion_seconds,
        "warm_compute_seconds": trunk_seconds + diffusion_seconds,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
