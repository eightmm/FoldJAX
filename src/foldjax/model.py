"""Model-neutral access to inference stages; native feature trees stay private."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from foldjax import api
from foldjax.job import Job
from foldjax.registry import capabilities, get_backend, normalize_model_name
from foldjax.sampling import get_recycle_policy
from foldjax.schema import (
    MSA_POLICIES,
    ModelCapabilities,
    PaddingConfig,
    PredictionRequest,
    PredictionResult,
    _normalize_padding,
    _strict_boolean,
    _strict_integer,
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Scientific input/inference choices, independent of execution padding.

    None retains the selected FoldJAX model/checkpoint default. ``msa_depth``
    requests the native MSA depth control, not common row identities. Pairing,
    encoding and selection remain native; identical raw MSAs need not produce
    identical selected rows. ``trunk_passes`` includes the initial evaluation.
    """

    msa_depth: int | None = None
    trunk_passes: int | None = None
    samples: int | None = None
    steps: int | None = None
    msa_search: str = "none"

    def __post_init__(self):
        for name in ("msa_depth", "trunk_passes", "samples", "steps"):
            value = getattr(self, name)
            if value is not None:
                value = _strict_integer(value, name=name, minimum=1)
                object.__setattr__(self, name, value)
        if self.msa_search not in MSA_POLICIES:
            raise ValueError(f"msa_search must be one of {MSA_POLICIES}")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Storage/compilation choices; these never select an implicit MSA depth.

    To enable padding, also set ModelConfig.msa_depth explicitly. Existing
    PredictionRequest padding presets retain their historical behavior.
    """

    padding: PaddingConfig | Mapping[str, Any] | bool | None = False
    use_compile_cache: bool = True
    cache_dir: Path | None = None
    resume: bool = False

    def __post_init__(self):
        object.__setattr__(self, "padding", _normalize_padding(self.padding))
        for name in ("use_compile_cache", "resume"):
            object.__setattr__(
                self, name, _strict_boolean(getattr(self, name), name=name)
            )
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if not self.use_compile_cache and self.cache_dir is not None:
            raise ValueError("cache_dir requires use_compile_cache=True")


def _stored_job(job: Job, base_dir: Path) -> Path:
    document = job.to_document()
    # A Python Job has no containing document directory. Resolve its public
    # asset references before moving it into the existing managed job store.
    for entity in document["entities"]:
        for name in ("unpaired_msa", "paired_msa"):
            if entity.get(name):
                entity[name] = str(
                    (base_dir / Path(entity[name]).expanduser()).resolve()
                )
        for template in entity.get("templates", ()):
            template["mmcif"] = str(
                (base_dir / Path(template["mmcif"]).expanduser()).resolve()
            )
    return Job.from_document(document).store()


@dataclass(frozen=True, slots=True)
class Model:
    """Lazy model handle, not a public container of native features/parameters.

    Methods return the existing PredictionResult. Arrays remain lazy through
    result.representations[name], so pair tensors are not silently duplicated.
    Backends own preparation, weights, execution and bounded cache lifetimes.
    """

    name: str
    config: ModelConfig = field(default_factory=ModelConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    weights: Path | None = None
    profile: str | None = None
    native_options: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "name", normalize_model_name(self.name))
        if not isinstance(self.config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        if not isinstance(self.execution, ExecutionConfig):
            raise TypeError("execution must be an ExecutionConfig")
        if self.execution.padding is not None and self.config.msa_depth is None:
            raise ValueError(
                "padding requires an explicit ModelConfig(msa_depth=...). "
                "MSA selection depth and execution capacity are separate choices"
            )
        if not isinstance(self.native_options, Mapping):
            raise TypeError("native_options must be a mapping")
        object.__setattr__(
            self, "native_options", MappingProxyType(dict(self.native_options))
        )
        if self.weights is not None:
            object.__setattr__(self, "weights", Path(self.weights))

    @property
    def capabilities(self) -> ModelCapabilities:
        return capabilities(self.name)

    def with_config(
        self,
        *,
        config: ModelConfig | None = None,
        execution: ExecutionConfig | None = None,
    ) -> Model:
        """Return a separately configured handle without mutating this one."""
        return dataclasses.replace(
            self,
            config=self.config if config is None else config,
            execution=self.execution if execution is None else execution,
        )

    def plan(
        self,
        job: Job | str | Path,
        *,
        stage: str = "full",
        outputs: Sequence[str] | str | None = None,
        seed: int = 0,
        output_dir: str | Path | None = None,
        base_dir: str | Path | None = None,
    ) -> PredictionRequest:
        """Resolve/validate without loading weights or executing a model.

        A Job object is persisted in the managed job store. Relative assets in
        that object use base_dir (default current directory); file inputs retain
        their own document-relative semantics.
        """
        supported = self.capabilities
        if stage not in {"full", "trunk", "inputs"}:
            raise ValueError("stage must be full, trunk or inputs")
        if stage == "inputs":
            if not supported.input_representations:
                raise ValueError(f"{self.name} does not support input-only extraction")
            outputs = supported.input_representations if outputs is None else outputs
        elif stage == "trunk":
            outputs = ("single",) if outputs is None else outputs
        elif outputs is None:
            outputs = ()
        source = (
            _stored_job(
                job, Path.cwd() if base_dir is None else Path(base_dir).resolve()
            )
            if isinstance(job, Job)
            else Path(job)
        )
        recycles = get_recycle_policy(self.name).from_trunk_passes(
            self.config.trunk_passes
        )
        request = PredictionRequest(
            model=self.name,
            input=source,
            weights=self.weights,
            profile=self.profile,
            output_dir=(
                Path(output_dir)
                if output_dir is not None
                else Path("foldjax-outputs") / self.name / source.stem / stage
            ),
            seed=seed,
            num_samples=self.config.samples,
            num_steps=self.config.steps,
            num_recycles=recycles,
            max_msa_depth=self.config.msa_depth,
            msa=self.config.msa_search,
            representations=outputs,
            stop_after=stage,
            padding=self.execution.padding,
            cache_dir=self.execution.cache_dir,
            use_compile_cache=self.execution.use_compile_cache,
            resume=self.execution.resume,
            options=dict(self.native_options),
        )
        return api.resolve_request(request)

    def _run(self, job, *, stage, outputs=None, **kwargs) -> PredictionResult:
        request = self.plan(job, stage=stage, outputs=outputs, **kwargs)
        result = api.predict(request)
        # A scalar handle always submits exactly one model and one input.
        if not isinstance(result, PredictionResult):
            raise TypeError("scalar model execution did not return PredictionResult")
        backend = get_backend(self.name)
        translated = backend.apply_sampling(request)
        bindings = {
            common: {
                "native_name": native,
                "value": (
                    int(translated[native]) if native in translated else None
                ),
                "source": (
                    "adapter_value" if native in translated else "backend_default"
                ),
            }
            for common, native in backend.capabilities().sampling.items()
        }
        configuration = {
            "interface": "model-v1",
            "model": self.name,
            "stage": stage,
            "requested": dataclasses.asdict(self.config),
            "sampling_bindings": bindings,
            "padding": (request.padding.summary() if request.padding else None),
            "defaults": "selected FoldJAX model/checkpoint defaults",
        }
        # This supplements the existing run manifest, which remains the resume
        # authority. Only common scalar bindings are persisted, not native trees.
        if result.output_dir is not None:
            from foldjax.input import _write_text_atomic

            _write_text_atomic(
                Path(result.output_dir) / "model_config.json",
                json.dumps(configuration, indent=2),
            )
        return dataclasses.replace(result, configuration=configuration)

    def embed(
        self, job: Job | str | Path, *, outputs=None, **kwargs
    ) -> PredictionResult:
        """Extract native input representations, before any trunk recycling."""
        return self._run(job, stage="inputs", outputs=outputs, **kwargs)

    def encode(
        self, job: Job | str | Path, *, outputs=None, **kwargs
    ) -> PredictionResult:
        """Run the trunk; default to single only, with pair explicitly opt-in."""
        return self._run(job, stage="trunk", outputs=outputs, **kwargs)

    def predict(
        self, job: Job | str | Path, *, outputs=None, **kwargs
    ) -> PredictionResult:
        """Run structure prediction, optionally retaining requested representations."""
        return self._run(job, stage="full", outputs=outputs, **kwargs)


def get_model(
    name: str,
    *,
    config: ModelConfig | None = None,
    execution: ExecutionConfig | None = None,
    weights: str | Path | None = None,
    profile: str | None = None,
    native_options: Mapping[str, Any] | None = None,
) -> Model:
    """Get a lazy model handle. No checkpoint or accelerator is loaded here."""
    return Model(
        name,
        ModelConfig() if config is None else config,
        ExecutionConfig() if execution is None else execution,
        weights,
        profile,
        {} if native_options is None else native_options,
    )
