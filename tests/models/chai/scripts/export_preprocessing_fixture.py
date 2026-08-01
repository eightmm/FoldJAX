#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Export an official Chai preprocessing batch as a differential-test fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chai-source", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--msa-directory", type=Path)
    parser.add_argument("--templates-path", type=Path)
    parser.add_argument("--constraint-path", type=Path)
    parser.add_argument("--entity-name-as-subchain", action="store_true")
    args = parser.parse_args()

    source = args.chai_source.resolve()
    if not (source / "chai_lab" / "chai1.py").is_file():
        raise SystemExit(f"invalid Chai source tree: {source}")
    sys.path.insert(0, str(source))

    import torch

    from chai_lab.chai1 import feature_factory, make_all_atom_feature_context
    from chai_lab.data.collate.collate import Collate

    work_dir = args.output.parent / f".{args.output.stem}.work"
    work_dir.mkdir(parents=True, exist_ok=True)
    context = make_all_atom_feature_context(
        args.fasta.resolve(),
        output_dir=work_dir,
        entity_name_as_subchain=args.entity_name_as_subchain,
        use_esm_embeddings=False,
        use_msa_server=False,
        msa_directory=args.msa_directory,
        constraint_path=args.constraint_path,
        use_templates_server=False,
        templates_path=args.templates_path,
        esm_device=torch.device("cpu"),
    )
    batch = Collate(
        feature_factory=feature_factory,
        num_key_atoms=128,
        num_query_atoms=32,
    )([context])

    arrays: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}
    for group in ("inputs", "features"):
        for name, value in batch[group].items():
            key = f"{group}/{name}"
            if isinstance(value, torch.Tensor):
                arrays[key] = value.detach().cpu().numpy()
            else:
                skipped[key] = type(value).__name__
    manifest = {
        "format": "chai-jax-preprocessing-fixture",
        "version": 1,
        "chai_version": __import__("chai_lab").__version__,
        "fasta_sha256": _sha256(args.fasta),
        "options": {
            "entity_name_as_subchain": args.entity_name_as_subchain,
            "use_esm_embeddings": False,
            "msa_directory": None
            if args.msa_directory is None
            else str(args.msa_directory.resolve()),
            "templates_path": None
            if args.templates_path is None
            else str(args.templates_path.resolve()),
            "constraint_path": None
            if args.constraint_path is None
            else str(args.constraint_path.resolve()),
        },
        "arrays": {
            name: {"shape": list(value.shape), "dtype": value.dtype.str}
            for name, value in sorted(arrays.items())
        },
        "skipped_non_arrays": skipped,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    arrays["__manifest__"] = np.frombuffer(manifest_bytes, dtype=np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "array_count": len(arrays) - 1,
                "skipped_count": len(skipped),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
