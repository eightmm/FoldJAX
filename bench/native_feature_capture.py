"""Capture independent publisher preprocessing in its own Python environment."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from unittest.mock import patch


def capture_boltz2(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from boltz.data.feature import featurizerv2
    from boltz.data.module.inferencev2 import PredictionDataset
    from boltz.data.types import Manifest
    from boltz.main import check_inputs, process_inputs

    if args.asset_root is None:
        raise ValueError(
            "Boltz capture requires --asset-root pointing to canonical mols"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    work = args.out.parent / "native-processing"
    process_inputs(
        data=check_inputs(args.input),
        out_dir=work,
        ccd_path=work / "unused_ccd.pkl",
        mol_dir=args.asset_root,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        use_msa_server=False,
        boltz2=True,
    )
    processed = work / "processed"
    manifest = Manifest.load(processed / "manifest.json")
    dataset = PredictionDataset(
        manifest=manifest,
        target_dir=processed / "structures",
        msa_dir=processed / "msa",
        mol_dir=args.asset_root,
        constraints_dir=processed / "constraints",
        template_dir=processed / "templates",
        extra_mols_dir=processed / "mols",
    )
    draws = {}
    original = featurizerv2.center_random_augmentation
    randn, randn_like = torch.randn, torch.randn_like

    def record(fn, *pos, **kwargs):
        result = fn(*pos, **kwargs)
        draws[f"draw_{len(draws):06d}"] = result.detach().cpu().numpy().copy()
        return result

    def augment(*pos, **kwargs):
        with (
            patch.object(torch, "randn", lambda *a, **k: record(randn, *a, **k)),
            patch.object(
                torch, "randn_like", lambda *a, **k: record(randn_like, *a, **k)
            ),
        ):
            return original(*pos, **kwargs)

    with patch.object(featurizerv2, "center_random_augmentation", augment):
        if len(dataset) != 1:
            raise ValueError("capture requires exactly one native dataset item")
        raw = dataset[0]
    arrays = {
        key: value.detach().cpu().numpy()[None]
        for key, value in raw.items()
        if isinstance(value, torch.Tensor)
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    np.savez_compressed(
        args.out.with_name(args.out.stem + "-reference-tape.npz"), **draws
    )
    print(
        json.dumps(
            {
                "model": "boltz2",
                "features": len(arrays),
                "reference_draws": len(draws),
                "seed": args.seed,
            }
        )
    )


def capture_openfold3(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from openfold3.core.data.framework.single_datasets.inference import InferenceDataset
    from openfold3.core.data.pipelines.featurization import conformer
    from openfold3.core.data.pipelines.preprocessing.template import (
        TemplatePreprocessorSettings,
    )
    from openfold3.core.data.tools.colabfold_msa_server import (
        MsaComputationSettings,
        augment_main_msa_with_query_sequence,
    )
    from openfold3.projects.of3_all_atom.config.dataset_configs import (
        InferenceJobConfig,
        MSASettings,
        TemplateSettings,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    document = json.loads(args.input.read_text())
    query_set = augment_main_msa_with_query_sequence(
        InferenceQuerySet.model_validate(document),
        MsaComputationSettings(
            msa_output_directory=args.out.parent / "native-dummy-msa"
        ),
    )
    dataset = InferenceDataset(
        InferenceJobConfig(
            query_set=query_set,
            seeds=[args.seed],
            msa=MSASettings(subsample_main=False),
            template=TemplateSettings(take_top_k=True),
            template_preprocessor_settings=TemplatePreprocessorSettings(),
        )
    )
    draws = {}
    original_augmentation = conformer.centre_random_augmentation
    original_randn = torch.randn

    def record_randn(*pos, **kwargs):
        result = original_randn(*pos, **kwargs)
        draws[f"draw_{len(draws):06d}"] = result.detach().cpu().numpy().copy()
        return result

    def augmentation(*pos, **kwargs):
        with patch.object(torch, "randn", record_randn):
            return original_augmentation(*pos, **kwargs)

    with patch.object(conformer, "centre_random_augmentation", augmentation):
        if len(dataset) != 1:
            raise ValueError("capture requires exactly one native dataset item")
        raw = dataset[0]
    arrays = {
        key: value.detach().cpu().numpy()[None]
        for key, value in raw.items()
        if isinstance(value, torch.Tensor)
    }
    atoms = raw["atom_array"]
    for key in ("atom_name", "chain_id", "res_id", "res_name", "element"):
        arrays[f"output.{key}"] = np.asarray(getattr(atoms, key))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    np.savez_compressed(
        args.out.with_name(args.out.stem + "-reference-tape.npz"), **draws
    )
    print(
        json.dumps(
            {
                "model": "openfold3",
                "features": len(arrays),
                "reference_draws": len(draws),
                "seed": args.seed,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("opendde", "protenix-v2", "openfold3", "boltz2"),
        required=True,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if args.model == "boltz2":
        capture_boltz2(args)
        return
    if args.model == "openfold3":
        capture_openfold3(args)
        return
    if args.asset_root is not None:
        os.environ["PROTENIX_ROOT_DIR"] = str(args.asset_root.resolve())

    import numpy as np
    import torch

    if args.model == "opendde":
        from opendde.config.inference import build_inference_config
        from opendde.data.inference.infer_dataloader import InferenceDataset
        from opendde.utils.seed import seed_everything

        config = build_inference_config(fill_required_with_null=True)
    else:
        from configs.configs_base import configs
        from configs.configs_data import data_configs
        from configs.configs_inference import inference_configs
        from configs.configs_model_type import model_configs
        from ml_collections.config_dict import ConfigDict
        from protenix.config.config import parse_configs
        from protenix.data.inference.infer_dataloader import InferenceDataset
        from protenix.utils.seed import seed_everything

        config = parse_configs(
            configs={**configs, "data": data_configs, **inference_configs},
            arg_str="",
            fill_required_with_null=True,
        )
        config.update(ConfigDict(model_configs[args.model]))
        config.model_name = args.model
    config.input_json_path = str(args.input.resolve())
    config.dump_dir = str(args.out.parent.resolve())
    config.use_msa = True
    config.use_template = False
    config.use_rna_msa = False
    config.num_workers = 0
    dataset = InferenceDataset(config)
    if len(dataset) != 1:
        raise ValueError("capture requires exactly one input job")
    seed_everything(args.seed, deterministic=True)
    data, atoms, error = dataset[0]
    if error:
        raise RuntimeError(error)
    arrays = {
        key: value.detach().cpu().numpy()
        for key, value in data["input_feature_dict"].items()
        if isinstance(value, torch.Tensor)
    }
    for key, attribute in (
        ("name", "atom_name"),
        ("element", "element"),
        ("res_name", "res_name"),
        ("chain_id", "chain_id"),
        ("res_id", "res_id"),
    ):
        arrays[f"output_atom_{key}"] = np.asarray(getattr(atoms, attribute))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(
        json.dumps(
            {
                "model": args.model,
                "features": len(arrays),
                "seed": args.seed,
                "use_msa": True,
                "use_template": False,
                "use_rna_msa": False,
            }
        )
    )


if __name__ == "__main__":
    main()
