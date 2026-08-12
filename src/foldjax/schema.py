"""Public FoldJAX request and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    model: str
    input_formats: tuple[str, ...]
    entity_types: tuple[str, ...] = ("protein", "dna", "rna", "ligand")
    supports_affinity: bool = False
    supports_templates: bool = True
    supports_msa: bool = True
    # Which model-neutral sampling knobs this backend can honour, and the native
    # option each one becomes. Reported by `foldjax capabilities`.
    sampling: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    """One model-neutral prediction job.

    Only ``model`` and ``input`` are required. Weights are resolved from the
    FoldJAX weight store, the output directory is derived from the input, and
    the compile cache defaults to the shared FoldJAX root, so the common case is
    one input file plus a model name.
    """

    model: str | None = None
    input: Path | None = None
    # The plural spellings, following `seeds`: one declaration, several runs.
    # `models=("boltz2", "protenix")` with `inputs=("a.yaml", "b.yaml")` runs
    # every model on every input -- the cross product -- each into its own
    # `output_dir/<model>/<input stem>` subtree, and `predict` returns one
    # result per run. Exactly one of `model`/`models` must be set, and one of
    # `input`/`inputs`; naming both spellings of either is an error, the same
    # rule every other knob follows.
    models: tuple[str, ...] | None = None
    inputs: tuple[Path, ...] | None = None
    weights: Path | None = None
    output_dir: Path | None = None
    input_format: str = "auto"
    seed: int = 0
    # Several models take a list of seeds natively, and running one job under
    # more than one is the ordinary way to use them -- the samples from a single
    # seed are correlated. `seeds` runs the job once per entry and returns every
    # structure together; leave it unset to run the single `seed`.
    seeds: tuple[int, ...] | None = None
    # `--num-seeds 5` is the same request as `--seeds 0 1 2 3 4`, and is what
    # people actually want when they say "run it five times": the seed values
    # themselves carry no meaning, only that they differ.
    num_seeds: int | None = None
    # Model-neutral sampling knobs. None means "keep this backend's default".
    num_samples: int | None = None
    num_steps: int | None = None
    num_recycles: int | None = None
    # How many MSA rows the model may keep. This is the dominant memory knob:
    # the trunk holds an [depth, tokens, channels] representation, so an
    # alignment thousands of rows deep costs gigabytes that nothing else in the
    # graph comes close to. None keeps each backend's own default.
    max_msa_depth: int | None = None
    cache_dir: Path | None = None
    # XLA's persistent compilation cache is on by default: these graphs take
    # minutes to compile and seconds to replay, so a cold second run is almost
    # never what someone wants. Turn it off for ephemeral or benchmark runs.
    use_compile_cache: bool = True
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("input", "weights", "output_dir", "cache_dir"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))
        if (self.model is None) == (self.models is None):
            raise ValueError("set exactly one of model and models")
        if (self.input is None) == (self.inputs is None):
            raise ValueError("set exactly one of input and inputs")
        if self.models is not None:
            models = tuple(str(value) for value in self.models)
            if not models:
                raise ValueError("models must not be empty")
            if len(set(models)) != len(models):
                raise ValueError("models must be unique")
            object.__setattr__(self, "models", models)
        if self.inputs is not None:
            inputs = tuple(Path(value) for value in self.inputs)
            if not inputs:
                raise ValueError("inputs must not be empty")
            object.__setattr__(self, "inputs", inputs)
        for path in self.resolved_inputs:
            if not path.exists():
                raise FileNotFoundError(f"input does not exist: {path}")
        if self.weights is not None and not self.weights.exists():
            raise FileNotFoundError(f"weights do not exist: {self.weights}")
        if self.seeds is not None:
            seeds = tuple(int(value) for value in self.seeds)
            if not seeds:
                raise ValueError("seeds must not be empty")
            if any(value < 0 for value in seeds):
                raise ValueError("seeds must be non-negative")
            if len(set(seeds)) != len(seeds):
                raise ValueError("seeds must be unique")
            if self.seed != 0:
                # Silently preferring one would change which structures come
                # back without changing the exit code, the same rule the
                # sampling knobs follow.
                raise ValueError("seed and seeds were both set; pass one of them")
            object.__setattr__(self, "seeds", seeds)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.num_seeds is not None:
            if self.num_seeds < 1:
                raise ValueError("num_seeds must be at least 1")
            if self.seeds is not None:
                raise ValueError("seeds and num_seeds were both set; pass one of them")
        for name in ("num_samples", "num_steps", "num_recycles", "max_msa_depth"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1")

    @property
    def resolved_models(self) -> tuple[str, ...]:
        """Every model this request names, whichever spelling was used."""
        return self.models if self.models is not None else (self.model,)

    @property
    def resolved_inputs(self) -> tuple[Path, ...]:
        """Every input this request names, whichever spelling was used."""
        return self.inputs if self.inputs is not None else (self.input,)

    @property
    def resolved_seeds(self) -> tuple[int, ...]:
        """Every seed this request runs, whichever field was used to say so.

        `num_seeds` counts up from `seed`, so `--seed 7 --num-seeds 3` is seeds
        7, 8 and 9: a run stays reproducible by naming its first seed.
        """
        if self.seeds is not None:
            return self.seeds
        if self.num_seeds is not None:
            return tuple(range(self.seed, self.seed + self.num_seeds))
        return (self.seed,)

    @property
    def sampling(self) -> dict[str, int]:
        """The sampling knobs that were actually set."""
        chosen = {
            "num_samples": self.num_samples,
            "num_steps": self.num_steps,
            "num_recycles": self.num_recycles,
            "max_msa_depth": self.max_msa_depth,
        }
        return {name: value for name, value in chosen.items() if value is not None}


@dataclass(frozen=True, slots=True)
class PredictionSample:
    seed: int
    structure_path: Path | None = None
    coordinates: Any = None
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "structure_path": (
                str(self.structure_path) if self.structure_path is not None else None
            ),
            "scores": dict(self.scores),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PredictionResult:
    model: str
    samples: tuple[PredictionSample, ...] = ()
    output_dir: Path | None = None
    raw: Any = None

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "samples": [sample.summary() for sample in self.samples],
        }
