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

This first adapter drives the single-sequence path (`infer_protein`), which is
what upstream documents and what the released weights are evaluated on for
monomers. Complexes, ligands and nucleic acids go through the same `forward`
and need our featurizer's output mapped onto its argument names; that mapping
is the next increment, and `capabilities()` reports protein-only until it
exists rather than accepting a job it would silently fold as a monomer.

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

        sequences = _protein_sequences(request.input)
        if len(sequences) != 1:
            raise ValueError(
                "this ESMFold2 adapter folds one protein chain; the job names "
                f"{len(sequences)}. Complexes need the full feature path, which "
                "is not wired yet"
            )

        kernel = options.pop("kernel_backend", "fused")
        forward_kwargs = {
            name: int(options.pop(name))
            for name in ("num_loops", "num_sampling_steps", "num_diffusion_samples")
            if name in options
        }
        depth = options.pop("msa_max_depth", None)
        if depth is not None:
            forward_kwargs["msa_max_depth"] = int(depth)
        if options:
            raise ValueError(f"unsupported ESMFold2 options: {', '.join(options)}")

        model = ESMFold2Model.from_pretrained(str(request.weights))
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.set_kernel_backend(None if kernel == "reference" else kernel)

        with torch.no_grad():
            output = model.infer_protein(sequences[0], **forward_kwargs)
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


def _protein_sequences(path: Path) -> list[str]:
    """The protein sequences a FoldJAX job names, one entry per chain copy."""
    document: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "entities" not in document:
        raise ValueError(
            "ESMFold2 takes a FoldJAX job document; it has no native dialect"
        )
    sequences: list[str] = []
    for entity in document["entities"]:
        if entity.get("type") != "protein":
            continue
        ids = entity.get("id", ["A"])
        copies = len(ids) if isinstance(ids, list) else 1
        sequences.extend([entity["sequence"]] * copies)
    return sequences
