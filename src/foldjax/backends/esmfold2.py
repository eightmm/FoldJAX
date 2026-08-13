"""ESMFold2 adapter, driving Biohub's released torch model.

ESMFold2 is the one backend FoldJAX drives rather than reimplements, and for a
reason none of the others share: it is not an AlphaFold-3 reimplementation with
an evolutionary trunk, it is a diffusion structure head on top of a 6B protein
language model (`ESMC-6B`, loaded alongside the 1.3 GB structure weights). The
part worth porting would be the smaller half; the part doing the work is a
language model this project has no story for.

What makes it fit here anyway is its interface. `ESMFold2Model.forward` takes
the AlphaFold-3 feature contract verbatim -- `asym_id`, `sym_id`, `entity_id`,
`mol_type`, `res_type`, `token_bonds`, `ref_pos`, `ref_element`,
`ref_atom_name_chars`, `atom_to_token`, `msa`, `deletion_value` -- which is the
same dictionary the vendored ports already build, and its knobs
(`num_sampling_steps`, `num_loops`, `num_diffusion_samples`, `msa_max_depth`)
are FoldJAX's neutral vocabulary under different names.

Upstream ships `prepare_protein_features` for one chain with no alignment and
leaves the rest of that dictionary to the caller; `_esmfold2_features` is that
assembly -- several chains, their symmetry, and their alignments -- and is
checked against upstream's own builder for the case both cover. Ligands and
nucleic acids are what remains: `forward` expresses them, but their features
need reference conformers out of upstream's 400 MB CCD pickle, so
`capabilities()` reports protein until that is wired rather than accepting a
job it would fold without its ligand.

Torch is required here and only here. The prediction path for every other
model is torch-free, so this adapter is behind the `esmfold2` extra and its
imports stay inside `predict`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from foldjax.backends.base import Backend
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)

#: Upstream's own defaults, from `ESMFold2Model.forward` and the model card:
#: 20 trunk loops and 100 ODE steps for the full model. They are named here
#: because `capabilities()` reports them and the README table quotes them.
DEFAULTS = {"num_loops": 20, "num_sampling_steps": 100, "num_diffusion_samples": 1}


class ESMFold2Backend(Backend):
    name = "esmfold2"
    # The neutral names, against upstream's. `msa_max_depth` is the one that is
    # not simply a rename: upstream subsamples that many MSA rows *per trunk
    # loop* rather than cutting the alignment once, which is the same policy
    # Boltz-2 uses and the opposite of a head-of-file cut.
    sampling_options = {
        "num_samples": "num_diffusion_samples",
        "num_steps": "num_sampling_steps",
        "num_recycles": "num_loops",
        "max_msa_depth": "msa_max_depth",
    }
    # `set_kernel_backend` takes None (reference python), "fused" (upstream's
    # Triton kernels) or "cuequivariance" -- the same library FoldJAX's own
    # triangle paths use. `auto` is the fused kernel, consistent with every
    # other backend here: the fastest path this build can actually run.
    execution_options = {
        "attention_kernel": (
            "kernel_backend",
            {"auto": "fused", "cueq": "cuequivariance", "xla": "reference"},
        ),
    }
    compile_options = (
        "num_diffusion_samples",
        "num_sampling_steps",
        "num_loops",
        "msa_max_depth",
        "kernel_backend",
    )

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("foldjax",),
            # Deliberately narrow: `forward` expresses ligands and nucleic
            # acids, but this adapter does not build their features yet, and a
            # job that named them would otherwise be folded as protein alone.
            entity_types=("protein",),
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        options = self.apply_sampling(request)
        import torch
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
        from transformers.models.esmfold2.protein_utils import (
            OUTPUT_TO_PDB_FEATURE_KEYS,
        )

        from foldjax.backends._esmfold2_features import build_features

        chains, alignments = _job_chains(request.input)
        kernel = options.pop("kernel_backend", "fused")
        forward_kwargs = {
            name: int(options.pop(name))
            for name in ("num_loops", "num_sampling_steps", "num_diffusion_samples")
            if name in options
        }
        depth = options.pop("msa_max_depth", None)
        if depth is not None:
            # Upstream resubsamples this many rows every trunk loop rather than
            # cutting the alignment once, so the cap is passed through instead
            # of being applied while building the features.
            forward_kwargs["msa_max_depth"] = int(depth)
        if options:
            raise ValueError(f"unsupported ESMFold2 options: {', '.join(options)}")

        model = ESMFold2Model.from_pretrained(str(request.weights))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        model.set_kernel_backend(None if kernel == "reference" else kernel)

        features = build_features(chains, alignments)
        features = {name: value.to(device) for name, value in features.items()}
        with torch.no_grad():
            output = model(**features, **forward_kwargs)
            for key in OUTPUT_TO_PDB_FEATURE_KEYS:
                output[key] = features[key]
            structure = model.output_to_pdb(output)

        request.output_dir.mkdir(parents=True, exist_ok=True)
        name = Path(request.input).stem
        path = request.output_dir / f"{name}.pdb"
        path.write_text(structure, encoding="utf-8")
        scores = {
            key: float(output[key].mean())
            for key in ("plddt", "ptm", "iptm")
            if key in output
        }
        return PredictionResult(
            model=self.name,
            samples=(
                PredictionSample(
                    seed=request.seed, structure_path=path, scores=scores
                ),
            ),
            output_dir=request.output_dir,
            raw={"forward_kwargs": forward_kwargs, "kernel_backend": kernel},
        )


def _job_chains(
    path: Path,
) -> tuple[list[tuple[str, str, int, int]], dict[int, Path]]:
    """One entry per chain copy, and the alignment each entity pins.

    A FoldJAX entity is a sequence and the chains that carry it, which is
    exactly ESMFold2's entity/symmetry split: every copy becomes its own chain
    with its own `asym_id`, sharing an `entity_id` and counting up `sym_id`.
    Alignment paths are resolved against the job file, as everywhere else.
    """
    document: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "entities" not in document:
        raise ValueError(
            "ESMFold2 takes a FoldJAX job document; it has no native dialect"
        )
    chains: list[tuple[str, str, int, int]] = []
    alignments: dict[int, Path] = {}
    base = Path(path).parent
    for entity_index, entity in enumerate(document["entities"]):
        if entity.get("type") != "protein":
            continue
        ids = entity.get("id", ["A"])
        ids = ids if isinstance(ids, list) else [ids]
        for symmetry, chain_id in enumerate(ids):
            chains.append((entity["sequence"], str(chain_id), entity_index, symmetry))
        msa = entity.get("unpaired_msa")
        if msa:
            candidate = Path(msa)
            alignments[entity_index] = (
                candidate if candidate.is_absolute() else base / candidate
            )
    if not chains:
        raise ValueError("the job names no protein chains")
    return chains, alignments
