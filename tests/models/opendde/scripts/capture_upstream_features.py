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
    args = parser.parse_args()

    configs = build_inference_config(fill_required_with_null=True)
    configs.input_json_path = str(args.input_json)
    configs.dump_dir = str(args.out.parent)
    configs.use_msa = False
    configs.use_template = False
    configs.use_rna_msa = False
    configs.num_workers = 0
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
        name: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in sorted(arrays.items())
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"features": len(arrays), "out": str(args.out)}))


if __name__ == "__main__":
    main()
