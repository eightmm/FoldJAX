"""Capture pinned OpenDDE preprocessing features for parity tests.

Run only in the isolated upstream/Torch reference environment.  This script is
not imported by the OpenDDE-JAX runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from opendde.config.inference import build_inference_config
from opendde.data.inference.infer_dataloader import InferenceDataset
from opendde.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--no-use-msa",
        action="store_true",
        help="disable the shipped use_msa=True default for a deliberate control",
    )
    parser.add_argument("--use-template", action="store_true")
    parser.add_argument("--use-rna-msa", action="store_true")
    parser.add_argument("--template-mmcif-dir", type=Path)
    parser.add_argument("--template-cache-dir", type=Path)
    parser.add_argument("--template-release-dates", type=Path)
    parser.add_argument("--template-obsolete-map", type=Path)
    parser.add_argument("--kalign-binary", type=Path)
    parser.add_argument(
        "--fetch-remote",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()

    configs = build_inference_config(fill_required_with_null=True)
    configs.input_json_path = str(args.input_json)
    configs.dump_dir = str(args.out.parent)
    configs.use_msa = not args.no_use_msa
    configs.use_template = args.use_template
    configs.use_rna_msa = args.use_rna_msa
    configs.num_workers = 0
    template_config = configs.data.template
    if args.template_mmcif_dir is not None:
        template_config.prot_template_mmcif_dir = str(args.template_mmcif_dir)
    if args.template_cache_dir is not None:
        template_config.prot_template_cache_dir = str(args.template_cache_dir)
    if args.template_release_dates is not None:
        template_config.release_dates_path = str(args.template_release_dates)
    if args.template_obsolete_map is not None:
        template_config.obsolete_pdbs_path = str(args.template_obsolete_map)
    if args.kalign_binary is not None:
        template_config.kalign_binary_path = str(args.kalign_binary)
    if args.fetch_remote is not None:
        template_config.fetch_remote = args.fetch_remote
    dataset = InferenceDataset(configs)
    seed_everything(args.seed, deterministic=True)
    data, atom_array, error = dataset[0]
    if error:
        raise RuntimeError(error)
    features = data["input_feature_dict"]
    arrays = {
        name: value.detach().cpu().numpy()
        for name, value in features.items()
        if isinstance(value, torch.Tensor)
    }
    arrays.update(
        {
            "output_atom_name": np.asarray(atom_array.atom_name, dtype="U8"),
            "output_atom_element": np.asarray(atom_array.element, dtype="U4"),
            "output_atom_res_name": np.asarray(atom_array.res_name, dtype="U8"),
            "output_atom_chain_id": np.asarray(atom_array.chain_id, dtype="U8"),
            "output_atom_res_id": np.asarray(atom_array.res_id, dtype=np.int64),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    manifest = {
        "source": {
            "version": __import__(
                "opendde.version", fromlist=["__version__"]
            ).__version__,
            "use_msa": not args.no_use_msa,
            "use_template": args.use_template,
            "use_rna_msa": args.use_rna_msa,
            "seed": args.seed,
            "template_mmcif_dir": (
                str(args.template_mmcif_dir)
                if args.template_mmcif_dir is not None
                else None
            ),
            "template_release_dates": (
                str(args.template_release_dates)
                if args.template_release_dates is not None
                else None
            ),
            "template_obsolete_map": (
                str(args.template_obsolete_map)
                if args.template_obsolete_map is not None
                else None
            ),
            "kalign_binary": (
                str(args.kalign_binary) if args.kalign_binary is not None else None
            ),
            "fetch_remote": args.fetch_remote,
        },
        "arrays": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted(arrays.items())
        },
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"features": len(arrays), "out": str(args.out)}))


if __name__ == "__main__":
    main()
