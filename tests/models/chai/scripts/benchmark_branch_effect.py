#!/usr/bin/env python3
"""Compare an active trunk branch with a controlled Torch/JAX baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from benchmark_sampler_parity import _load_torch_context, _real_protein_inputs


def _effect_metrics(
    torch_active: np.ndarray,
    torch_baseline: np.ndarray,
    jax_active: np.ndarray,
    jax_baseline: np.ndarray,
) -> dict[str, float]:
    torch_effect = (torch_active - torch_baseline).astype(np.float64).ravel()
    jax_effect = (jax_active - jax_baseline).astype(np.float64).ravel()
    torch_norm = float(np.linalg.norm(torch_effect))
    jax_norm = float(np.linalg.norm(jax_effect))
    denominator = max(torch_norm * jax_norm, 1e-12)
    return {
        "torch_effect_rms": float(np.sqrt(np.mean(torch_effect**2))),
        "jax_effect_rms": float(np.sqrt(np.mean(jax_effect**2))),
        "effect_cosine": float(np.dot(torch_effect, jax_effect) / denominator),
        "effect_relative_l2_error": float(
            np.linalg.norm(jax_effect - torch_effect) / max(torch_norm, 1e-12)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--active-restraint", type=Path)
    parser.add_argument("--reference-augmentation-tape", type=Path)
    parser.add_argument("--active-recycles", type=int, default=1)
    parser.add_argument("--baseline-recycles", type=int, default=1)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--conformers", type=Path, required=True)
    parser.add_argument("--torch-active", type=Path, required=True)
    parser.add_argument("--torch-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    common = {
        "fasta": args.fasta,
        "bundle": args.bundle,
        "conformers": args.conformers,
        "reference_augmentation_tape": args.reference_augmentation_tape,
    }
    jax_active, active_metadata = _real_protein_inputs(
        **common,
        recycles=args.active_recycles,
        restraint=args.active_restraint,
    )
    jax_baseline, baseline_metadata = _real_protein_inputs(
        **common, recycles=args.baseline_recycles
    )
    torch_active, torch_active_metadata = _load_torch_context(args.torch_active)
    torch_baseline, torch_baseline_metadata = _load_torch_context(args.torch_baseline)

    for index in (6, 8, 9, 10, 13):
        if not np.array_equal(torch_active[index], torch_baseline[index]):
            raise ValueError(f"Torch branch identity field {index} differs")
        if not np.array_equal(jax_active[index], jax_baseline[index]):
            raise ValueError(f"JAX branch identity field {index} differs")
        if not np.array_equal(torch_active[index], jax_active[index]):
            raise ValueError(f"Torch/JAX branch identity field {index} differs")

    token_mask = torch_active[8].astype(bool)
    pair_mask = token_mask[..., :, None] & token_mask[..., None, :]
    fields = {
        "token_single_initial_repr": (0, token_mask),
        "token_pair_initial_repr": (1, pair_mask),
        "token_single_trunk_repr": (2, token_mask),
        "token_pair_trunk_repr": (3, pair_mask),
    }
    effects = {
        name: _effect_metrics(
            torch_active[index][mask],
            torch_baseline[index][mask],
            jax_active[index][mask],
            jax_baseline[index][mask],
        )
        for name, (index, mask) in fields.items()
    }
    report = {
        "contract": args.contract,
        "fasta": str(args.fasta),
        "active_restraint": (
            str(args.active_restraint) if args.active_restraint is not None else None
        ),
        "active_recycles": args.active_recycles,
        "baseline_recycles": args.baseline_recycles,
        "reference_augmentation_tape_sha256": active_metadata[
            "reference_augmentation_tape_sha256"
        ],
        "jax_active_static_sha256": active_metadata["static_input_sha256"],
        "jax_baseline_static_sha256": baseline_metadata["static_input_sha256"],
        "torch_active_static_sha256": torch_active_metadata["static_input_sha256"],
        "torch_baseline_static_sha256": torch_baseline_metadata[
            "static_input_sha256"
        ],
        "effects": effects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
