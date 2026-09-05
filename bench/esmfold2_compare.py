"""ESMFold2: the JAX port against the torch model it was ported from.

Not part of the shared benchmark, and deliberately: `bench/spec.py` pins one
schedule (5 samples / 200 steps / 10 recycles) so five AlphaFold-3-shaped models
can be compared with each other. ESMFold2 has no evolutionary trunk and no
comparable schedule, and putting it on those axes would compare two different
questions. It is measured here against its own upstream instead, and at its own
released configuration -- which both sides read from the same `config.json`, so
the schedule is controlled by construction rather than by agreement.

Three things make the comparison mean something:

* **the same features.** Both sides consume one NumPy feature dictionary; the
  torch side gets the same arrays as tensors. Protein-only jobs retain the
  builder checked tensor-for-tensor against upstream's
  `prepare_protein_features`. Mixed jobs use FoldJAX's port of Biohub's
  all-biomolecule input contract, because the publisher exposes the model
  tensors but no equivalent public end-to-end preprocessing entry point. Such
  rows are model-core comparisons and report that boundary explicitly.
* **a precision difference the default run does NOT control, and says so.** Upstream's
  only autocast (`modeling_esmfold2.py:2021`) wraps the ESM-C call alone and is
  gated on ESM-C's parameters already being bfloat16, so loading it at
  `precision="bf16"` puts the language model in bfloat16 on both sides. Nothing
  else is autocast: the torch trunk runs at the checkpoint's `float32` while
  this port casts every `TRUNK_PREFIXES` sub-tree to bfloat16. Both rows report
  the dtype they actually loaded, read off the parameters rather than asserted,
  so the difference shows up in the output instead of hiding in a docstring.
  An experiment may add ``--foldjax-trunk-dtype=float32`` as an explicitly
  labelled parity control; that does not change the shipped model default.
  This file used to claim the two default sides matched here; they never did.
* **one process per row.** Both peak figures are process-lifetime high-water
  marks, so two runs in one process report the larger and the second row's
  number is unknowable.

What it cannot control is the model: ESMFold2 is stochastic at inference by
construction, so the two sides produce different structures at the same seed and
the confidence columns are distributions, not values. Agreement is reported as
the spread within each side beside the spread between them -- if the second is
not larger than the first, there is nothing to see.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

ROOT = Path(__file__).resolve().parent.parent
#: Historical harness default. The released schedule itself is read from the
#: checkpoint; callers may set only the experiment's requested sample count.
SAMPLES = 4
SEED = 101


def _job(
    case: str,
) -> tuple[
    dict,
    Path,
    list[tuple[str, str, int, int]],
    dict[int, Path],
    bool,
]:
    from bench.spec import cases

    entry = next(item for item in cases() if item.name == case)
    document = json.loads(entry.job.read_text())
    chains: list[tuple[str, str, int, int]] = []
    alignments: dict[int, Path] = {}
    all_atom = any(
        entity.get("type") != "protein" or entity.get("modifications")
        for entity in document["entities"]
    ) or bool(document.get("bonds"))
    for index, entity in enumerate(document["entities"]):
        if entity.get("type") != "protein":
            continue
        ids = entity.get("id", ["A"])
        for symmetry, chain in enumerate(ids if isinstance(ids, list) else [ids]):
            chains.append((entity["sequence"], str(chain), index, symmetry))
        if entity.get("unpaired_msa"):
            path = Path(entity["unpaired_msa"])
            alignments[index] = path if path.is_absolute() else entry.job.parent / path
    return document, entry.job.parent, chains, alignments, all_atom


def _features(
    case: str,
    *,
    seed: int,
) -> tuple[dict, str]:
    document, base, chains, alignments, all_atom = _job(case)
    if not all_atom:
        from foldjax.models.esmfold2.data import features as featurisation

        return (
            featurisation.build_features(chains, alignments),
            "shared protein feature builder; parity-tested against publisher "
            "prepare_protein_features",
        )

    from foldjax.models.esmfold2.data import all_atom as featurisation

    return (
        featurisation.build_job_features(
            document,
            base_dir=base,
            ccd_path=_weights() / "ccd.pkl",
            seed=seed,
        ),
        "shared FoldJAX NumPy port of publisher all-biomolecule tensor contract; "
        "model-core comparison, not independent native preprocessing",
    )


def _weights() -> Path:
    from foldjax.paths import weights_dir

    return weights_dir("esmfold2")


def run_foldjax(
    case: str,
    warmup: bool,
    *,
    num_samples: int = SAMPLES,
    seed: int = SEED,
    trunk_dtype: str | None = None,
) -> dict:
    import jax

    from foldjax.models.esmfold2 import inference

    model = inference.load(_weights())
    if trunk_dtype is not None:
        model = replace(
            model,
            settings=replace(model.settings, trunk_dtype=trunk_dtype),
        )
    settings = inference.structure_model.with_overrides(
        model.settings, num_samples=num_samples
    )
    built, feature_boundary = _features(case, seed=seed)

    def once() -> dict:
        return inference.predict(
            jax.random.key(seed), built, model, num_samples=num_samples
        )

    once()  # populate eligible XLA cache entries before the measured call
    if warmup:
        return {}
    start = time.perf_counter()
    output = once()
    jax.block_until_ready(output["sample_atom_coords"])
    elapsed = time.perf_counter() - start

    stats = jax.local_devices()[0].memory_stats() or {}
    return {
        "impl": "foldjax",
        "wall_s": round(elapsed, 2),
        "peak_mib": round(stats.get("peak_bytes_in_use", 0) / 2**20, 1),
        "trunk_dtype": settings.trunk_dtype,
        "coords": output["sample_atom_coords"],
        "plddt": [float(v) for v in output["complex_plddt"]],
        "ptm": [float(v) for v in output["ptm"]],
        "features": built,
        "feature_boundary": feature_boundary,
    }


def _torch_trunk_dtype(model) -> str:
    """The realized dtype of the folding trunk's parameters, as a string.

    The label this replaces was a literal. A comparison that reports an
    expected value rather than a measured one certifies agreement it never
    checked, which is worse than reporting nothing.
    """
    for name, parameter in model.named_parameters():
        if "folding_trunk" in name:
            return str(parameter.dtype).removeprefix("torch.")
    return "unknown (no folding_trunk parameter)"


def run_torch(
    case: str,
    warmup: bool,
    *,
    num_samples: int = SAMPLES,
    seed: int = SEED,
) -> dict:
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    from tests.models.esmfold2.torch_features import build_features

    weights = _weights()
    # `load_esmc=False` then an explicit path: the default resolves `esmc_id`
    # against Hugging Face and would fetch 25.4 GB that is already on disk.
    model = ESMFold2Model.from_pretrained(str(weights), load_esmc=False)
    model.load_esmc(str(weights / "esmc"), precision="bf16")
    model = model.to("cuda").eval()
    # Read, never assumed: `from_pretrained` takes no dtype here, so the trunk
    # arrives at whatever `config.json` says -- float32 for the release. The
    # row used to hardcode "bfloat16 (autocast)" and was wrong for every run.
    trunk_dtype = _torch_trunk_dtype(model)

    document, _base, chains, alignments, all_atom = _job(case)
    if all_atom:
        from foldjax.models.esmfold2.data.all_atom import OUTPUT_METADATA_FEATURES

        arrays, feature_boundary = _features(case, seed=seed)
        features = arrays
        built = {
            name: torch.from_numpy(value).to("cuda")
            for name, value in arrays.items()
            if name not in OUTPUT_METADATA_FEATURES
        }
    else:
        built = build_features(chains, alignments)
        built = {name: value.to("cuda") for name, value in built.items()}
        features = {name: value.cpu().numpy() for name, value in built.items()}
        feature_boundary = (
            "shared protein feature builder; parity-tested against publisher "
            "prepare_protein_features"
        )

    def once() -> dict:
        torch.manual_seed(seed)
        with torch.no_grad():
            return model(**built, num_diffusion_samples=num_samples)

    once()
    if warmup:
        return {}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = once()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "impl": "upstream",
        "wall_s": round(elapsed, 2),
        "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "trunk_dtype": trunk_dtype,
        "coords": output["sample_atom_coords"].float().cpu().numpy(),
        "plddt": [float(v) for v in output["complex_plddt"]],
        "ptm": [float(v) for v in output["ptm"]],
        "features": features,
        "feature_boundary": feature_boundary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("foldjax", "torch"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--coords-out", type=Path)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--num-samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--foldjax-trunk-dtype",
        choices=("bfloat16", "float32"),
        help="experiment-only FoldJAX precision control; the public model "
        "default remains bfloat16",
    )
    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.impl == "torch" and args.foldjax_trunk_dtype is not None:
        parser.error("--foldjax-trunk-dtype only applies to --impl foldjax")
    if args.impl == "foldjax":
        record = run_foldjax(
            args.case,
            args.warmup,
            num_samples=args.num_samples,
            seed=args.seed,
            trunk_dtype=args.foldjax_trunk_dtype,
        )
    else:
        record = run_torch(
            args.case,
            args.warmup,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    if not record:
        return 0

    import numpy as np

    coords = np.asarray(record.pop("coords"))
    features = record.pop("features")
    record["case"] = args.case
    record["samples"] = args.num_samples
    record["seed"] = args.seed
    print(json.dumps(record, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(record, sort_keys=True) + "\n")
    if args.coords_out:
        args.coords_out.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            name: np.asarray(features[name])
            for name in (
                "atom_to_token",
                "asym_id",
                "distogram_atom_idx",
                "mol_type",
                "ref_atom_name_chars",
                "residue_index",
                "res_type",
                "sym_id",
                "token_attention_mask",
                "token_chain_id_chars",
                "token_residue_name_chars",
            )
            if name in features
        }
        np.savez_compressed(
            args.coords_out,
            coords=coords,
            atom_mask=np.asarray(features["atom_attention_mask"]),
            **metadata,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
