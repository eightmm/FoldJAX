"""Capture upstream Protenix's trunk state at each stage of one recycle.

The trunk is a chain of five stages, and every one of them is shape-preserving
in `z`. That makes a transposed composition invisible to the type system, to
the shapes, and to any test that builds its expectation out of the same pieces
in the same order -- which is how the MSA block ran its two sub-updates
backwards for the life of the port. What does catch it is the number `z`
actually carries out of each stage, next to upstream's.

Writes `stages.json`; `trunk_stage_parity.py` reads it and compares.
"""

from __future__ import annotations

import argparse
import json
import os
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
                continue
            value = torch.from_numpy(array)
            if value.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                value = value.long()
            features[key] = value.to(device)
    return features


def stats(name: str, tensor: torch.Tensor) -> dict:
    array = tensor.detach().float().cpu().numpy()
    return {
        "stage": name,
        "mean": float(array.mean()),
        "std": float(array.std()),
        "absmax": float(np.abs(array).max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="stages are recorded for the last of this many cycles",
    )
    parser.add_argument(
        "--full-depth-msa",
        action="store_true",
        help="pin the upstream MSA sampler to every materialized row, which is "
        "what the JAX static-feature path consumes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault(
        "PROTENIX_ROOT_DIR",
        str(Path(__file__).resolve().parents[5] / "protenix"),
    )
    os.environ.setdefault("LAYERNORM_TYPE", "torch")

    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from ml_collections.config_dict import ConfigDict
    from protenix.config.config import parse_configs
    from protenix.model.protenix import Protenix, update_input_feature_dict

    model_name = "protenix_base_default_v1.0.0"
    inference_configs["model_name"] = model_name
    configs = parse_configs(
        configs={**configs_base, **{"data": data_configs}, **inference_configs},
        fill_required_with_null=True,
    )
    configs.update(ConfigDict(model_configs[model_name]))
    configs.model.N_cycle = args.cycles
    configs.dtype = "fp32"
    configs.use_msa = True
    configs.triangle_multiplicative = "torch"
    configs.triangle_attention = "torch"
    configs.enable_efficient_fusion = False
    configs.enable_tf32 = False

    device = torch.device("cuda:0")
    model = Protenix(configs).to(device)
    checkpoint = torch.load(
        Path(os.environ["PROTENIX_ROOT_DIR"]) / "checkpoint" / f"{model_name}.pt",
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint["model"]
    if next(iter(state_dict)).startswith("module."):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=configs.load_strict)
    model.eval()

    features = load_features(args.features, device)
    for key in ("d_lm", "v_lm", "pad_info"):
        features.pop(key, None)
    features = update_input_feature_dict(features)
    if args.full_depth_msa:
        model.msa_module.msa_configs["strategy"] = "topk"
        model.msa_module.msa_configs["test_lowerb"] = int(features["msa"].shape[-2])

    rows = []
    with torch.inference_mode():
        s_inputs = model.input_embedder(features, inplace_safe=False, chunk_size=None)
        rows.append(stats("s_inputs", s_inputs))
        s_init = model.linear_no_bias_sinit(s_inputs)
        z_init = (
            model.linear_no_bias_zinit1(s_init)[..., None, :]
            + model.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )
        z_init = z_init + model.relative_position_encoding(features["relp"])
        z_init = z_init + model.linear_no_bias_token_bond(
            features["token_bonds"].unsqueeze(dim=-1)
        )
        rows.append(stats("z_init", z_init))

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        for cycle in range(args.cycles):
            last = cycle == args.cycles - 1
            z = z_init + model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z))
            if last:
                rows.append(stats("after_recycle_z", z))
            if model.template_embedder.n_blocks > 0:
                z = z + model.template_embedder(
                    features,
                    z,
                    triangle_multiplicative=configs.triangle_multiplicative,
                    triangle_attention=configs.triangle_attention,
                    inplace_safe=False,
                    chunk_size=None,
                )
            if last:
                rows.append(stats("after_template", z))
            z = model.msa_module(
                features,
                z,
                s_inputs,
                pair_mask=None,
                triangle_multiplicative=configs.triangle_multiplicative,
                triangle_attention=configs.triangle_attention,
                inplace_safe=False,
                chunk_size=None,
            )
            if last:
                rows.append(stats("after_msa", z))
            s = s_init + model.linear_no_bias_s(model.layernorm_s(s))
            s, z = model.pairformer_stack(
                s,
                z,
                pair_mask=None,
                triangle_multiplicative=configs.triangle_multiplicative,
                triangle_attention=configs.triangle_attention,
                inplace_safe=False,
                chunk_size=None,
            )
            if last:
                rows.append(stats("after_pairformer_z", z))
                rows.append(stats("after_pairformer_s", s))

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "impl": "torch",
        "checkpoint": model_name,
        "cycles": args.cycles,
        "template_blocks": int(model.template_embedder.n_blocks),
        "msa_rows": int(features["msa"].shape[-2]),
        "full_depth_msa": bool(args.full_depth_msa),
        "stages": rows,
    }
    (args.out / "stages.json").write_text(json.dumps(payload, indent=2))
    for row in rows:
        print(f"{row['stage']:22s} std {row['std']:12.5f} absmax {row['absmax']:12.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
