"""ESMFold2 adapter, driving FoldJAX's own JAX port.

ESMFold2 is not an AlphaFold-3 reimplementation with an evolutionary trunk: it
is a diffusion structure head on a linear-recurrence pair trunk, folding the
representations of `ESMC-6B`. Both halves are ported here --
`models/esmfold2/models/` for the 235M-parameter structure network and `esmc`
for the language model -- so this backend, like every other one, needs no torch
at prediction time.

Two things about it differ from the rest of the fleet and reach the interface:

* **The weights are two checkpoints.** The structure network is 940 MB; the
  language model it reads from is a separate ~12 GB download that upstream
  distributes apart from it, staged at `<weights>/esmc`. Without it the trunk
  still runs with its language-model branch absent, which is what upstream does
  when no PLM is loaded, and which is not the released model -- so it is opt-in
  through `no_language_model` rather than a silent fallback.
* **The model is stochastic by construction.** The trunk starts from a random
  pair state, the language-model pair embedding is dropped out at `p = 0.25`
  *per loop* with training-mode semantics that upstream's release path enables,
  and the sampler adds noise. Two runs at the same seed agree; two seeds do not,
  and that is the model rather than the port.

Ligands and nucleic acids are what remains: the model expresses them, but their
features need reference conformers out of upstream's 417 MB CCD pickle, so
`capabilities()` reports protein rather than accepting a job it would fold
without its ligand.
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from foldjax.backends.base import Backend
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)

#: Upstream's released `config.json`, which is what the port reads at load
#: time. These are named here because `capabilities()` reports them and the
#: README table quotes them -- and because they are *not* the dataclass
#: defaults in upstream's source, which say 20 loops and 68 steps.
DEFAULTS = {
    "num_loops": 3,
    "num_sampling_steps": 14,
    "num_diffusion_samples": 32,
    "msa_max_depth": 1024,
}


class ESMFold2Backend(Backend):
    name = "esmfold2"
    # The neutral names, against the port's. `max_msa_depth` is the one that is
    # not simply a rename: the model resubsamples that many MSA rows *per trunk
    # loop* rather than cutting the alignment once, which is the same policy
    # Boltz-2 uses and the opposite of a head-of-file cut.
    sampling_options = {
        "num_samples": "num_samples",
        "num_steps": "num_steps",
        "num_recycles": "num_loops",
        "max_msa_depth": "msa_max_depth",
    }
    # No `attention_kernel`: the port's attention is XLA's, and the fused and
    # cuEquivariance paths the torch model selected between do not exist here.
    # `no_language_model` is not a performance knob -- it changes which model
    # runs -- but it is the only way to fold without the 12 GB download, so it
    # is exposed and named for what it does.
    execution_options: dict[str, tuple[str, dict[str, Any]]] = {}
    compile_options = (
        "num_samples",
        "num_steps",
        "num_loops",
        "msa_max_depth",
        "no_language_model",
    )

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("foldjax",),
            # Deliberately narrow: the model expresses ligands and nucleic
            # acids, but this adapter does not build their features yet, and a
            # job that named them would otherwise be folded as protein alone.
            entity_types=("protein",),
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        options = self.apply_sampling(request)
        # In-package imports, kept inside `predict` for the same reason every
        # other vendored backend does it: to keep `import foldjax` off JAX's
        # import cost.
        inference = import_module("foldjax.models.esmfold2.inference")
        output_module = import_module("foldjax.models.esmfold2.output")

        chains, alignments = _job_chains(request.input)
        overrides = {
            name: int(options.pop(name))
            for name in ("num_loops", "num_steps", "num_samples", "msa_max_depth")
            if name in options
        }
        without_lm = bool(options.pop("no_language_model", False))
        esmc = options.pop("esmc_weights", None)
        if options:
            raise ValueError(f"unsupported ESMFold2 options: {', '.join(options)}")

        model = inference.load(
            request.weights, esmc=esmc, require_esmc=not without_lm
        )
        prediction, features = inference.predict_job(
            inference.seed_key(request.seed),
            chains,
            alignments,
            model,
            **overrides,
        )

        name = Path(request.input).stem
        written = output_module.write_prediction_outputs(
            prediction, features, request.output_dir, name=name
        )
        scores = {entry["sample"]: entry for entry in written["summary"]}
        return PredictionResult(
            model=self.name,
            samples=tuple(
                PredictionSample(
                    seed=request.seed,
                    structure_path=path,
                    scores={
                        key: float(value)
                        for key, value in scores.get(index, {}).items()
                        if key != "sample"
                    },
                )
                for index, path in enumerate(written["structures"])
            ),
            output_dir=request.output_dir,
            raw={
                "overrides": overrides,
                "language_model": model.has_language_model,
            },
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
