"""ESMFold2: the JAX port against the torch model it was ported from.

Not part of the shared benchmark, and deliberately: `bench/spec.py` pins one
schedule (5 samples / 200 steps / 10 recycles) so five AlphaFold-3-shaped models
can be compared with each other. ESMFold2 has no evolutionary trunk and no
comparable schedule, and putting it on those axes would compare two different
questions. It is measured here against its own upstream instead, and at its own
released configuration -- which both sides read from the same `config.json`, so
the schedule is controlled by construction rather than by agreement.

Three things make the comparison mean something:

* **the same features.** Both sides consume the dictionary
  `models/esmfold2/data/features` builds; the torch side gets it as tensors. That
  builder is checked tensor-for-tensor against upstream's own
  `prepare_protein_features` for the case upstream covers, so sharing it removes
  a variable rather than hiding one.
* **a precision difference this does NOT control, and says so.** Upstream's
  only autocast (`modeling_esmfold2.py:2021`) wraps the ESM-C call alone and is
  gated on ESM-C's parameters already being bfloat16, so loading it at
  `precision="bf16"` puts the language model in bfloat16 on both sides. Nothing
  else is autocast: the torch trunk runs at the checkpoint's `float32` while
  this port casts every `TRUNK_PREFIXES` sub-tree to bfloat16. Both rows report
  the dtype they actually loaded, read off the parameters rather than asserted,
  so the difference shows up in the output instead of hiding in a docstring.
  This file used to claim the two sides matched here; they never did.
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
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

ROOT = Path(__file__).resolve().parent.parent
#: ESMFold2's own released defaults, read off the checkpoint rather than named
#: here -- except the sample count, which is cut because the confidence head
#: reruns per sample and thirty-two of them measures that head, not the model.
SAMPLES = 4
SEED = 101


def _job(case: str) -> tuple[list[tuple[str, str, int, int]], dict[int, Path]]:
    from bench.spec import cases

    entry = next(item for item in cases() if item.name == case)
    document = json.loads(entry.job.read_text())
    chains: list[tuple[str, str, int, int]] = []
    alignments: dict[int, Path] = {}
    for index, entity in enumerate(document["entities"]):
        if entity.get("type") != "protein":
            raise SystemExit(
                f"{case} needs upstream ESMFold2's all-biomolecule feature "
                "builder, which the current FoldJAX adapter has not ported"
            )
        ids = entity.get("id", ["A"])
        for symmetry, chain in enumerate(ids if isinstance(ids, list) else [ids]):
            chains.append((entity["sequence"], str(chain), index, symmetry))
        if entity.get("unpaired_msa"):
            path = Path(entity["unpaired_msa"])
            alignments[index] = path if path.is_absolute() else entry.job.parent / path
    return chains, alignments


def _weights() -> Path:
    from foldjax.paths import weights_dir

    return weights_dir("esmfold2")


def run_foldjax(case: str, warmup: bool) -> dict:
    import jax

    from foldjax.models.esmfold2 import inference

    chains, alignments = _job(case)
    model = inference.load(_weights())
    settings = inference.structure_model.with_overrides(
        model.settings, num_samples=SAMPLES
    )
    from foldjax.models.esmfold2.data import features as featurisation

    built = featurisation.build_features(chains, alignments)

    def once() -> dict:
        return inference.predict(
            jax.random.key(SEED), built, model, num_samples=SAMPLES
        )

    once()  # compile; the cold number is XLA, paid once per shape
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


def run_torch(case: str, warmup: bool) -> dict:
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    from tests.models.esmfold2.torch_features import build_features

    chains, alignments = _job(case)
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

    built = build_features(chains, alignments)
    built = {name: value.to("cuda") for name, value in built.items()}

    def once() -> dict:
        torch.manual_seed(SEED)
        with torch.no_grad():
            return model(**built, num_diffusion_samples=SAMPLES)

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
        "features": {k: v.cpu().numpy() for k, v in built.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("foldjax", "torch"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--coords-out", type=Path)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    runner = run_foldjax if args.impl == "foldjax" else run_torch
    record = runner(args.case, args.warmup)
    if not record:
        return 0

    import numpy as np

    coords = np.asarray(record.pop("coords"))
    features = record.pop("features")
    record["case"] = args.case
    record["samples"] = SAMPLES
    record["seed"] = SEED
    print(json.dumps(record, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(record, sort_keys=True) + "\n")
    if args.coords_out:
        args.coords_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.coords_out,
            coords=coords,
            atom_mask=np.asarray(features["atom_attention_mask"]),
            atom_to_token=np.asarray(features["atom_to_token"]),
            distogram_atom_idx=np.asarray(features["distogram_atom_idx"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
