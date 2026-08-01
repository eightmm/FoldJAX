#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Offline Torch bridge for Chai's exact released ESM embedding model."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chai-source", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-revision", default="chai-traced-sdpa-fp16")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    chai_source = args.chai_source.resolve()
    if not (chai_source / "chai_lab" / "chai1.py").is_file():
        raise SystemExit(f"invalid Chai source tree: {chai_source}")
    sys.path[:0] = [str(repository / "src"), str(chai_source)]

    import torch

    from foldjax.models.chai.data.esm import (
        _input_esm_sequence,
        save_native_esm_embeddings,
    )
    from foldjax.models.chai.data.input import EntityType, read_inputs
    from chai_lab.data.dataset.embeddings.esm import _get_esm_contexts_for_sequences
    from chai_lab.utils.paths import downloads_path

    inputs = read_inputs(args.fasta.resolve())
    sequences = {
        _input_esm_sequence(value)
        for value in inputs
        if value.entity_type == EntityType.PROTEIN.value
    }
    if not sequences:
        raise SystemExit("FASTA contains no protein sequences to embed")
    contexts = _get_esm_contexts_for_sequences(sequences, torch.device(args.device))
    model_path = downloads_path / "esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"
    if not model_path.is_file():
        raise SystemExit(f"ESM model asset was not materialized: {model_path}")
    save_native_esm_embeddings(
        {
            sequence: context.esm_embeddings.detach().cpu().numpy()
            for sequence, context in contexts.items()
        },
        args.destination,
        model_id="esm2_t36_3B_UR50D",
        model_revision=args.model_revision,
        source_sha256=_sha256(model_path),
    )
    print(
        f"exported {len(contexts)} ESM sequence embeddings to "
        f"{args.destination.resolve()}"
    )


if __name__ == "__main__":
    main()
