"""Capture official Torch prepared/trunk diffusion inputs before sampling."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any

import chai_lab.chai1 as chai1
import numpy as np
import torch
from reference_augmentation_tape import replay_torch


class _CaptureCompleteError(RuntimeError):
    pass


class _CaptureModule:
    def __init__(self) -> None:
        self.values: dict[str, np.ndarray] = {}

    def forward(self, crop_size: int, **kwargs: Any) -> None:
        del crop_size
        excluded = {"atom_noised_coords", "noise_sigma"}
        self.values = {
            name: value.detach().cpu().numpy()
            for name, value in kwargs.items()
            if name not in excluded
        }
        raise _CaptureCompleteError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restraint", type=Path)
    parser.add_argument("--reference-augmentation-tape", type=Path)
    parser.add_argument("--msa-directory", type=Path)
    parser.add_argument("--template-hits", type=Path)
    parser.add_argument("--use-esm", action="store_true")
    parser.add_argument(
        "--zero-trunk-component",
        choices=("msa_module", "pairformer_stack"),
    )
    parser.add_argument(
        "--pairformer-active-blocks",
        type=int,
        choices=range(49),
        help="Keep only the first K of 48 Pairformer blocks active.",
    )
    parser.add_argument(
        "--zero-pairformer-branch",
        choices=(
            "triangle_multiplication",
            "triangle_attention",
            "transition_pair",
            "attention_pair_bias",
            "transition_single",
        ),
        help="Zero one residual branch in Pairformer block 0.",
    )
    args = parser.parse_args()
    if (
        args.zero_trunk_component is not None
        and args.pairformer_active_blocks is not None
    ):
        parser.error(
            "--zero-trunk-component and --pairformer-active-blocks "
            "are mutually exclusive"
        )
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise FileExistsError(f"work directory must be empty: {args.work_dir}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    with replay_torch(args.reference_augmentation_tape):
        context = chai1.make_all_atom_feature_context(
            fasta_file=args.fasta,
            output_dir=args.work_dir,
            use_esm_embeddings=args.use_esm,
            use_msa_server=False,
            msa_directory=args.msa_directory,
            constraint_path=args.restraint,
            use_templates_server=False,
            templates_path=args.template_hits,
            esm_device=torch.device("cuda:0"),
        )
    capture = _CaptureModule()
    original = chai1._component_moved_to
    zeroed_trunk_tensor_count = 0

    @contextlib.contextmanager
    def intercept(comp_key: str, device: torch.device):
        nonlocal zeroed_trunk_tensor_count
        if comp_key == "diffusion_module.pt":
            yield capture
        else:
            with original(comp_key, device) as component:
                if comp_key == "trunk.pt" and args.zero_trunk_component is not None:
                    prefix = f"{args.zero_trunk_component}."
                    with torch.no_grad():
                        for name, value in component.jit_module.state_dict().items():
                            if name.startswith(prefix):
                                value.zero_()
                                zeroed_trunk_tensor_count += 1
                if comp_key == "trunk.pt" and args.pairformer_active_blocks is not None:
                    prefix = "pairformer_stack.blocks."
                    with torch.no_grad():
                        for name, value in component.jit_module.state_dict().items():
                            if not name.startswith(prefix):
                                continue
                            block_index = int(
                                name.removeprefix(prefix).split(".", 1)[0]
                            )
                            if block_index >= args.pairformer_active_blocks:
                                value.zero_()
                                zeroed_trunk_tensor_count += 1
                if comp_key == "trunk.pt" and args.zero_pairformer_branch is not None:
                    prefix = (
                        "pairformer_stack.blocks.0."
                        f"{args.zero_pairformer_branch}."
                    )
                    with torch.no_grad():
                        for name, value in component.jit_module.state_dict().items():
                            if name.startswith(prefix):
                                value.zero_()
                                zeroed_trunk_tensor_count += 1
                yield component

    chai1._component_moved_to = intercept
    try:
        try:
            chai1.run_folding_on_context(
                context,
                output_dir=args.work_dir,
                num_trunk_recycles=args.recycles,
                num_diffn_timesteps=2,
                num_diffn_samples=1,
                seed=args.seed,
                device=torch.device("cuda:0"),
                low_memory=True,
            )
        except _CaptureCompleteError:
            pass
    finally:
        chai1._component_moved_to = original
    required = {
        "token_single_initial_repr",
        "token_pair_initial_repr",
        "token_single_trunk_repr",
        "token_pair_trunk_repr",
        "atom_single_input_feats",
        "atom_block_pair_input_feats",
        "atom_single_mask",
        "atom_block_pair_mask",
        "token_single_mask",
        "block_indices_h",
        "block_indices_w",
        "atom_token_indices",
    }
    if set(capture.values) != required:
        raise RuntimeError(
            f"captured diffusion keys differ: {sorted(set(capture.values) ^ required)}"
        )
    digest = hashlib.sha256()
    for name, value in sorted(capture.values.items()):
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    metadata = {
        "schema_version": 1,
        "backend": "official-chai-torch",
        "fasta": str(args.fasta),
        "recycles": args.recycles,
        "seed": args.seed,
        "use_esm_embeddings": args.use_esm,
        "msa": args.msa_directory is not None,
        "template": args.template_hits is not None,
        "restraint": args.restraint is not None,
        "static_input_sha256": digest.hexdigest(),
        "zero_trunk_component": args.zero_trunk_component,
        "pairformer_active_blocks": args.pairformer_active_blocks,
        "zero_pairformer_branch": args.zero_pairformer_branch,
        "zeroed_trunk_tensor_count": zeroed_trunk_tensor_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **capture.values,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
