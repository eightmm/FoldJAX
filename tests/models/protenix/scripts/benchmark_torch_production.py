"""Benchmark upstream Protenix with its production inference defaults."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--warm-iters", type=int, default=3)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--triangle-kernel",
        choices=("cuequivariance", "torch"),
        default="cuequivariance",
    )
    parser.add_argument(
        "--layer-norm", choices=("fast_layer_norm", "torch"), default="fast_layer_norm"
    )
    parser.add_argument("--full-depth-msa", action="store_true")
    parser.add_argument("--no-efficient-fusion", action="store_true")
    parser.add_argument("--no-diffusion-cache", action="store_true")
    return parser.parse_args()


def load_features(path: Path, device: torch.device) -> dict:
    features = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if key.startswith("pad_info."):
                continue
            array = data[key]
            # Feature files carry per-atom annotation strings (names, elements,
            # chain ids) used to write the output mmCIF, not to run the model.
            # torch.from_numpy rejects them outright, which used to kill the
            # benchmark on any feature file produced by the real featurizer.
            if array.dtype.kind not in "biuf":
                continue
            value = torch.from_numpy(array)
            if value.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                value = value.long()
            features[key] = value.to(device)
    features.pop("d_lm", None)
    features.pop("v_lm", None)
    features.pop("pad_info", None)
    if "is_ligand" not in features:
        token_is_ligand = features["restype"].argmax(dim=-1) == 20
        features["is_ligand"] = token_is_ligand[features["atom_to_token_idx"]].long()
    return features


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if args.warm_iters <= 0:
        raise ValueError("warm_iters must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "PROTENIX_ROOT_DIR", str(Path(__file__).resolve().parents[2] / "protenix")
    )
    os.environ["LAYERNORM_TYPE"] = args.layer_norm
    python_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")

    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from ml_collections.config_dict import ConfigDict
    from protenix.config.config import parse_configs
    from protenix.model.protenix import Protenix
    from runner.inference import update_inference_configs

    model_name = "protenix_base_default_v1.0.0"
    inference_configs["model_name"] = model_name
    configs = parse_configs(
        configs={**configs_base, **{"data": data_configs}, **inference_configs},
        fill_required_with_null=True,
    )
    configs.update(ConfigDict(model_configs[model_name]))
    configs.model.N_cycle = args.cycles
    configs.sample_diffusion.N_step = args.steps
    configs.sample_diffusion.N_sample = args.samples
    configs.dtype = args.dtype
    configs.triangle_multiplicative = args.triangle_kernel
    configs.triangle_attention = args.triangle_kernel
    configs.use_msa = True
    configs.use_template = False
    configs.use_rna_msa = False
    configs.sample_diffusion.guidance.enable = False
    configs.enable_efficient_fusion = not args.no_efficient_fusion
    configs.enable_diffusion_shared_vars_cache = not args.no_diffusion_cache

    device = torch.device("cuda:0")
    features = load_features(args.features, device)
    n_token = int(features["residue_index"].shape[-1])
    configs = update_inference_configs(configs, n_token)
    model = Protenix(configs).to(device)
    checkpoint_path = (
        Path(os.environ["PROTENIX_ROOT_DIR"]) / "checkpoint" / f"{model_name}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"]
    if next(iter(state_dict)).startswith("module."):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=configs.load_strict)
    model.eval()
    if args.full_depth_msa:
        msa_rows = int(features["msa"].shape[-2])
        model.msa_module.msa_configs["strategy"] = "topk"
        model.msa_module.msa_configs["test_lowerb"] = msa_rows
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    def run() -> tuple[dict, dict]:
        seed_all(args.seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
            prediction, _, log = model(
                input_feature_dict=copy.deepcopy(features),
                label_full_dict=None,
                label_dict=None,
                mode="inference",
                mc_dropout_apply_rate=configs.mc_dropout_apply_rate,
            )
        torch.cuda.synchronize(device)
        return prediction, log

    run()
    torch.cuda.reset_peak_memory_stats(device)
    warm_samples = []
    for _ in range(args.warm_iters):
        started = time.perf_counter()
        prediction, log = run()
        warm_samples.append(time.perf_counter() - started)
    warm_seconds = statistics.median(warm_samples)
    time_tracker = log.get("time", {})
    metrics = {
        "backend": "torch",
        "contract": "upstream_production_defaults",
        "checkpoint": model_name,
        "dtype": args.dtype,
        "tf32": bool(configs.enable_tf32),
        "triangle_multiplicative": str(configs.triangle_multiplicative),
        "triangle_attention": str(configs.triangle_attention),
        "diffusion_shared_vars_cache": bool(configs.enable_diffusion_shared_vars_cache),
        "efficient_fusion": bool(configs.enable_efficient_fusion),
        "use_msa": bool(configs.use_msa),
        "use_template": bool(configs.use_template),
        "use_rna_msa": bool(configs.use_rna_msa),
        "confidence": True,
        "seed": args.seed,
        "tokens": n_token,
        "atoms": int(features["atom_to_token_idx"].shape[-1]),
        "msa_rows_materialized": int(features["msa"].shape[-2]),
        "msa_sampling": "full_depth_topk" if args.full_depth_msa else "upstream_random",
        "cycles": args.cycles,
        "num_steps": args.steps,
        "samples": args.samples,
        "warm_seconds": warm_seconds,
        "warm_seconds_samples": warm_samples,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "model_time_tracker": time_tracker,
        "coordinate_checksum": float(prediction["coordinate"].float().sum().item()),
        "coordinate_shape": list(prediction["coordinate"].shape),
    }
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
