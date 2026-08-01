"""Benchmark the matching upstream Chai-1 Torch inference workload."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from chai_lab.chai1 import make_all_atom_feature_context, run_folding_on_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    feature_context = make_all_atom_feature_context(
        fasta_file=args.fasta,
        output_dir=args.output,
        use_esm_embeddings=False,
        use_msa_server=False,
        msa_directory=None,
        constraint_path=None,
        use_templates_server=False,
        templates_path=None,
        esm_device=torch.device("cuda:0"),
    )
    records = []
    for iteration in range(args.iterations):
        output = args.output / f"iteration_{iteration}"
        if output.exists():
            shutil.rmtree(output)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = run_folding_on_context(
            feature_context,
            output_dir=output,
            num_trunk_recycles=args.recycles,
            num_diffn_timesteps=args.timesteps,
            num_diffn_samples=args.samples,
            seed=args.seed,
            device=torch.device("cuda:0"),
            low_memory=True,
        )
        torch.cuda.synchronize()
        records.append(
            {
                "seconds": time.perf_counter() - start,
                "peak_bytes_allocated": torch.cuda.max_memory_allocated(),
                "peak_bytes_reserved": torch.cuda.max_memory_reserved(),
                "coordinate_fingerprint": float(
                    prediction.cif_paths[0].stat().st_size
                ),
            }
        )
    print(
        json.dumps(
            {"device": torch.cuda.get_device_name(), "iterations": records},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
