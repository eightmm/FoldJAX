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

    model: str
    input: Path
    weights: Path | None = None
    output_dir: Path | None = None
    input_format: str = "auto"
    seed: int = 0
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
        if not self.input.exists():
            raise FileNotFoundError(f"input does not exist: {self.input}")
        if self.weights is not None and not self.weights.exists():
            raise FileNotFoundError(f"weights do not exist: {self.weights}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in ("num_samples", "num_steps", "num_recycles", "max_msa_depth"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1")

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
