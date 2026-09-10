"""Run upstream ESMFold2's own prediction for one `bench.run_upstream` row.

Every other upstream in this directory ships a command-line predictor, so
`run_upstream` can launch it directly. The ESMFold2 fork ships a `transformers`
model class and nothing above it: no job reader, no featuriser for anything
beyond a single unaligned protein chain (`prepare_protein_features`), and no
structure writer. This module is that missing layer, and only that layer -- it
builds the input, calls `ESMFold2Model.forward` once, and writes what came
back. The arithmetic between those two points is upstream's.

Three boundaries are worth naming, because they are what this row can and
cannot claim:

* **Features are FoldJAX's.** `bench/esmfold2_compare.py` already established
  this for the same reason: the publisher exposes the model tensors but no
  equivalent end-to-end preprocessing entry point, so a "native" featuriser
  would have to be invented rather than used. Both arms therefore read the
  same job document through the same NumPy builder, which is the controlled
  choice -- and for the protein-only case that builder is checked tensor for
  tensor against `prepare_protein_features`.
* **The structure writer is FoldJAX's.** The fork returns coordinate tensors;
  turning them into mmCIF is serialisation, not arithmetic, and `bench
  .structures` needs a file to read.
* **Precision is left alone.** `bench/esmfold2_tape.py` pins
  `float32_matmul_precision` and `allow_tf32` because a tape replay is a parity
  control. A benchmark row is not: it has to be upstream's released path, so
  nothing here sets a precision, a determinism flag, or a CUBLAS workspace.
  What the trunk actually did is observed through a forward pre-hook on the
  measured call and written beside the result, so the row reports a measured
  policy rather than an assumed one.

The whole pinned schedule reaches this model, which is what makes the row a
comparison rather than two settings side by side. `forward` documents
`num_loops`, `num_sampling_steps` and `num_diffusion_samples` as caller
overrides, and FoldJAX spells the first two `num_recycles` and `num_steps`
over the same contract -- both run `max(1, n + 1)` trunk loops, per
`docs/recycling-defaults.md`. Requested steps are not realised steps, though:
the sampler clips its Karras schedule at sigma 256 and prepends the cap, so
the run is shorter than it was asked for by an amount only the loaded model
knows. That number is read back rather than assumed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

#: Dense pair-shaped outputs the structure writer and the confidence summary
#: never read. Retaining them would move `num_samples x tokens^2` tensors to
#: host memory for nothing -- 11.6 GB each at 3,012 tokens and five samples.
OMITTED_OUTPUTS = (
    "distogram_logits",
    "pae",
    "pae_logits",
    "pde",
    "pde_logits",
)


def _document(path: Path) -> tuple[dict[str, Any], Path]:
    from foldjax.backends.esmfold2 import _job_document

    return _job_document(path)


def _features(document: dict[str, Any], base: Path, weights: Path, seed: int):
    """Build the job the way `backends/esmfold2.predict` builds it.

    The branch is upstream-shaped rather than a benchmark convenience: a lone
    protein chain with no alignment is the legacy builder's case, and anything
    else -- several chains, an alignment, a ligand, a bond -- is the
    all-biomolecule contract, which needs the publisher chemistry in
    `ccd.pkl`.
    """
    from foldjax.backends.esmfold2 import (
        _chains_from_document,
        _requires_all_atom_features,
    )

    if _requires_all_atom_features(document):
        from foldjax.models.esmfold2.data import all_atom

        return (
            all_atom.build_job_features(
                document,
                base_dir=base,
                ccd_path=weights / "ccd.pkl",
                seed=seed,
            ),
            "foldjax NumPy port of the publisher all-biomolecule input "
            "contract; model-core comparison, not independent native "
            "preprocessing",
        )
    chains, alignments = _chains_from_document(document, base)
    if not chains:
        raise ValueError("the job names no protein chains")
    from foldjax.models.esmfold2.data import features as protein_features

    return (
        protein_features.build_features(chains, dict(alignments)),
        "shared protein feature builder; parity-tested against publisher "
        "prepare_protein_features",
    )


def _import_upstream(root: Path):
    """Import the fork's model class, and prove that is what was imported.

    The virtualenv also carries a pip-installed `transformers`. Prepending the
    fork's `src` puts it first, but "first on the path" is a claim about the
    path rather than about the class, so read the file the class came from.
    """
    source = root / "src/transformers/models/esmfold2/modeling_esmfold2.py"
    if not source.is_file():
        raise SystemExit(f"not an upstream transformers source root: {source}")
    sys.path.insert(0, str(root / "src"))
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    if Path(inspect.getfile(ESMFold2Model)).resolve() != source.resolve():
        raise RuntimeError("upstream model imported from a different source root")
    return ESMFold2Model


def _observed_trunk_precision(torch, model) -> tuple[list[dict[str, Any]], Any]:
    """A non-perturbing record of the precision policy the trunk ran under."""

    observed: list[dict[str, Any]] = []

    def observe(_module, inputs) -> None:
        observed.append(
            {
                "input_dtype": str(inputs[0].dtype).removeprefix("torch."),
                "cuda_autocast": torch.is_autocast_enabled("cuda"),
                "autocast_dtype": str(torch.get_autocast_dtype("cuda")).removeprefix(
                    "torch."
                ),
            }
        )

    return observed, model.folding_trunk.register_forward_pre_hook(observe)


def _trunk_weight_dtype(model) -> str:
    for name, parameter in model.named_parameters():
        if "folding_trunk" in name:
            return str(parameter.dtype).removeprefix("torch.")
    return "unknown (no folding_trunk parameter)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--upstream-source-root", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--num-recycles", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    for name in ("num_samples", "num_steps", "num_recycles"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    weights = args.weights.resolve()
    model_class = _import_upstream(args.upstream_source_root.resolve())

    import torch

    from foldjax.models.esmfold2.data.all_atom import OUTPUT_METADATA_FEATURES
    from foldjax.models.esmfold2.output import write_prediction_outputs

    document, base = _document(args.job)
    built, feature_boundary = _features(document, base, weights, args.seed)

    # `load_esmc=False` then an explicit path: the default resolves `esmc_id`
    # against Hugging Face and would fetch the 25.4 GB that is already staged.
    model = model_class.from_pretrained(str(weights), load_esmc=False)
    model.load_esmc(str(weights / "esmc"), precision="bf16")
    model = model.to("cuda").eval()

    inputs = {
        name: torch.as_tensor(value, device="cuda")
        for name, value in built.items()
        if name not in OUTPUT_METADATA_FEATURES
    }
    observed, handle = _observed_trunk_precision(torch, model)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    try:
        with torch.no_grad():
            output = model(
                **inputs,
                num_diffusion_samples=args.num_samples,
                # `num_loops` is upstream's spelling of the bench's recycle
                # knob: both sides run `max(1, n + 1)` trunk loops from the
                # same n, so the two columns recycle the same number of times.
                num_loops=args.num_recycles,
                num_sampling_steps=args.num_steps,
            )
    finally:
        handle.remove()
    if not observed:
        raise RuntimeError("upstream trunk precision boundary was not observed")

    retained = {
        name: value.detach().float().cpu().numpy()
        for name, value in output.items()
        if name not in OMITTED_OUTPUTS and isinstance(value, torch.Tensor)
    }
    del output
    name = str(document.get("name") or args.job.stem)
    written = write_prediction_outputs(retained, built, args.output_dir, name=name)

    # What ran, not what was asked for. The sampler clips its schedule at
    # sigma 256 and prepends the cap, so the denoising steps it actually
    # performs are the entries that survive that cut; the trunk runs one more
    # loop than the recycle count it was given. Both are read off the loaded
    # model, and the checkpoint's own defaults ride along so the row shows how
    # far the pinned schedule moved it.
    schedule = model.structure_head.inference_noise_schedule(args.num_steps)
    released = model.structure_head.inference_noise_schedule()
    msa_knobs = (
        "msa_max_depth",
        "msa_column_mask_rate",
        "msa_subsample_at_inference",
    )
    forward_defaults = {
        key: parameter.default
        for key, parameter in inspect.signature(model.forward).parameters.items()
        if key in msa_knobs
    }
    record = {
        "model": "esmfold2",
        "impl": "upstream",
        "name": name,
        "seed": args.seed,
        "requested": {
            "num_diffusion_samples": args.num_samples,
            "num_loops": args.num_recycles,
            "num_sampling_steps": args.num_steps,
        },
        "effective": {
            "trunk_loops": max(1, args.num_recycles + 1),
            "denoising_steps": int((schedule <= 256.0).sum().item()),
        },
        "checkpoint_defaults": {
            "num_loops": int(model.config.num_loops),
            "denoising_steps": int((released <= 256.0).sum().item()),
        },
        # Left at the values `forward` itself declares, and read from its
        # signature so a moved default shows up here instead of being
        # certified by a literal written once.
        "forward_signature_defaults": forward_defaults,
        "feature_boundary": feature_boundary,
        "structure_writer": "foldjax mmCIF writer; upstream ships no structure "
        "serialiser",
        "trunk_weight_dtype": _trunk_weight_dtype(model),
        "trunk_precision_observations": observed,
        "omitted_outputs": list(OMITTED_OUTPUTS),
        "retained_outputs": sorted(retained),
        "structures": [path.name for path in written["structures"]],
        "confidence": Path(written["scores"]).name,
    }
    path = args.output_dir / "esmfold2_upstream_run.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: record[key] for key in ("requested", "effective", "seed")},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
