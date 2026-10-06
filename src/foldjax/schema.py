"""Public FoldJAX request and result types."""

from __future__ import annotations

import operator
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _coordinate_shape(value: Any) -> list[int] | None:
    """Small JSON-friendly shape summary without serializing coordinate arrays."""
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return [int(dimension) for dimension in shape]
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        nested = _coordinate_shape(value[0])
        return [len(value), *(nested or [])]
    return None


#: How a job without alignments is treated. ``none``, the default, searches
#: nothing and refuses a protein chain that has no alignment, because no
#: upstream but ESMFold2 folds one from its single sequence by default:
#: Boltz-2 refuses (``boltz/main.py:581-583``) and the others search. The
#: search stays opt-in because it sends the sequence to a server. ``single``
#: folds such a chain from its sequence on purpose. ``auto`` fills empty
#: protein chains in from the shared alignment cache; ``required``
#: additionally refuses to fold a chain it could not find one for -- what a
#: batch script wants, because the silent fallback it guards against is a
#: *successful* single-sequence run.
MSA_POLICIES = ("none", "single", "auto", "required")

#: Whether to search for structural templates. ``none``, the default, uses only
#: the templates a job names. ``auto`` searches the ColabFold MMseqs2 server's
#: PDB70 hits for every protein chain that names none, fetches the hit
#: structures and applies the selected backend's released template filters and
#: date cutoff (`foldjax.template_search`). It is opt-in for the same reason as
#: ``msa='auto'``: it sends the sequence to a server. ``required`` searches the
#: same way and fails the run when the search cannot run or keeps no template
#: for a chain it searched -- ``auto`` warns and folds without, which a batch
#: script would only notice as a successful template-free run.
TEMPLATE_POLICIES = ("none", "auto", "required")

#: What a failing run does to the rest of the request.
#: Where a run may stop. ``trunk`` exists so that downstream work can take
#: the representations without paying for a structure it will discard.
#: ``inputs`` stops before the main trunk.
STOP_POINTS: tuple[str, ...] = ("full", "trunk", "inputs")

ERROR_POLICIES = ("stop", "continue")

#: Draw a seed: the backend's upstream seeds nothing by default.
RANDOM_SEED = "random"

#: Run the native job's own ``modelSeeds``; draw a seed when it has none.
JOB_SEEDS = "job"

#: The seed each backend runs when a request names none, following the
#: upstream's released CLI (``AUDIT-DEFAULTS-20260930.md``). An integer is
#: upstream's own fixed default. ``RANDOM_SEED`` is used where upstream seeds
#: nothing: a seed is drawn, printed and recorded in the manifest, so the run
#: can still be repeated with ``--seed``. ``JOB_SEEDS`` runs every entry of a
#: native job's ``modelSeeds`` and draws only when the job names none; a
#: common-schema job has no such field.
#:
#: - Protenix: ``--seeds`` defaults to 101 (``configs/configs_inference.py:24``).
#: - OpenFold3: ``seeds: [42]`` (``entry_points/validator.py:311``).
#: - Boltz-2: ``--seed`` defaults to None and nothing is seeded
#:   (``boltz/main.py:922,1102-1103``).
#: - ESMFold2: ``ESMFold2Model.forward`` takes no seed.
#: - OpenDDE: ``--seeds``, else the job's ``modelSeeds``, else
#:   ``random.randint`` (``runner/batch_inference.py:698-703``).
#: - AlphaFold 3: the job's ``modelSeeds``; the AlphaFold Server dialect
#:   draws one when they are absent (``folding_input.py:1119-1122``).
#:
#: A backend missing from the table keeps FoldJAX's historical 0.
DEFAULT_SEEDS: Mapping[str, int | str] = {
    "alphafold3": JOB_SEEDS,
    "boltz2": RANDOM_SEED,
    "esmfold2": RANDOM_SEED,
    "opendde": JOB_SEEDS,
    "openfold3": 42,
    "protenix": 101,
}

#: Where a resolved request's seed came from: the caller, the upstream's fixed
#: default, a draw, the native job's ``modelSeeds``, or FoldJAX's own 0 for a
#: backend with no table entry.
SEED_SOURCES = ("user", "upstream", "random", "job", "foldjax")

#: Drawn seeds stay below 2**31: ESMFold2 hands the seed to RDKit's
#: ``randomSeed``, a C ``int``.
RANDOM_SEED_BOUND = 2**31


class PredictionError(RuntimeError):
    """A requested prediction could not produce a usable result."""


class PredictionOutputError(PredictionError):
    """A backend returned successfully without the structure it promised."""


def _strict_integer(value: Any, *, name: str, minimum: int) -> int:
    """Return a real integer without silently truncating floats or booleans."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        normalized = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    normalized = int(normalized)
    if normalized < minimum:
        condition = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {condition}")
    return normalized


def _strict_boolean(value: Any, *, name: str) -> bool:
    """Return a real boolean without treating non-empty text as true."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _iso_date(value: Any, *, name: str) -> str:
    """Return a ``YYYY-MM-DD`` calendar date, refusing anything else."""
    import re
    from datetime import date

    if not isinstance(value, str):
        raise ValueError(f"{name} must be a YYYY-MM-DD string")
    text = value.strip()
    # `date.fromisoformat` also takes week dates (2021-W39-4) and the basic
    # form (20210930); only the calendar spelling is the documented contract.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) is None:
        raise ValueError(f"{name} must be a YYYY-MM-DD date; got {value!r}")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{name} must be a YYYY-MM-DD date; got {value!r}") from error
    return parsed.isoformat()


@dataclass(frozen=True, slots=True)
class InputRequirement:
    """Runtime and install requirements for one accepted input format.

    ``preprocessing_runtime`` uses intentionally small, user-facing labels:
    ``base`` is included in the base FoldJAX install, ``jax`` is an in-package
    NumPy/JAX path with optional chemistry dependencies, ``precomputed`` means
    the input already contains model features, and ``native`` needs a
    model-specific non-Torch runtime. ``required_extras`` names install extras
    required for that format; CUDA selection is shared by every backend and is
    therefore not repeated. ``requires_torch`` remains in the serialized
    contract for third-party backend compatibility, but every built-in backend
    reports ``False``.
    """

    prediction_runtime: str = "jax"
    preprocessing_runtime: str = "base"
    required_extras: tuple[str, ...] = ()
    requires_torch: bool = False
    notes: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "prediction_runtime": self.prediction_runtime,
            "preprocessing_runtime": self.preprocessing_runtime,
            "required_extras": list(self.required_extras),
            "requires_torch": self.requires_torch,
            "notes": self.notes,
        }


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
    # Appended after the existing fields to preserve positional construction of
    # ModelCapabilities by embedding applications. Real backends declare one
    # entry per advertised format; the empty default keeps third-party backend
    # implementations source-compatible until they add the richer contract.
    input_requirements: dict[str, InputRequirement] = field(default_factory=dict)
    # Appended to preserve positional construction by embedding applications.
    # An empty tuple keeps third-party backends source compatible.
    padding_axes: tuple[str, ...] = ()
    # Which common-schema fields this backend's dialect can carry, and which
    # scientific inputs it accepts that the common document has no field for.
    # Backends do not set these: `foldjax.capabilities` fills them in from the
    # single translation table, so the discovery surface cannot drift from the
    # writer that decides. See `foldjax.input.native_only_features`.
    common_schema_features: tuple[str, ...] = ()
    native_only_features: tuple[str, ...] = ()
    # Trunk arrays this model can hand back, in the order it builds them.
    # The tables differ by model: OpenDDE folds in a structural-token
    # space its residue space does not cover. Names retain native semantics.
    representations: tuple[str, ...] = ()
    # Arrays available before any trunk recycling. Feature ABI stays private.
    input_representations: tuple[str, ...] = ()
    # Confidence arrays a default run writes to each sample's
    # `confidence_full.npz`. Filled in by `foldjax.capabilities` from
    # `foldjax.confidence_arrays.AVAILABILITY`, like the schema fields above.
    confidence_arrays: tuple[str, ...] = ()
    # What each neutral sampling knob in `sampling` runs at when omitted, in
    # the knob's units, for the default profile's checkpoint; None only where
    # nothing can be read before a run. Filled in by `foldjax.capabilities`
    # from `Backend.sampling_resolution`, like the fields above.
    sampling_defaults: dict[str, int | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    """Preparation state for native artifacts needed by one backend.

    Most FoldJAX backends are ready as soon as the package is installed.
    Backends that need generated, ABI-specific artifacts expose the explicit
    preparation command and whether that step needs network access here.
    """

    ready: bool
    setup: str | None
    requires_network: bool
    notes: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "setup": self.setup,
            "requires_network": self.requires_network,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """User-facing readiness and capability information for one backend."""

    model: str
    capabilities: ModelCapabilities
    execution: dict[str, tuple[str, ...]]
    weights_ready: bool
    weights_path: Path
    weights_source: str
    weights_licence: str
    weights_fetchable: bool
    download_bytes: int | None
    runtime: RuntimeInfo = field(
        default_factory=lambda: RuntimeInfo(
            ready=True,
            setup=None,
            requires_network=False,
            notes="No generated runtime preparation is required.",
        )
    )
    setup: str | None = None
    notes: str = ""
    # Profile-aware models expose the alternatives without changing the
    # compatible `weights_*` fields above, which continue to describe the
    # complete released/default bundle.
    weight_profiles: tuple[dict[str, Any], ...] = ()

    def summary(self) -> dict[str, Any]:
        capabilities = self.capabilities
        return {
            "model": self.model,
            "input_formats": list(capabilities.input_formats),
            "entity_types": list(capabilities.entity_types),
            "supports_affinity": capabilities.supports_affinity,
            "supports_templates": capabilities.supports_templates,
            "supports_msa": capabilities.supports_msa,
            "padding_axes": list(capabilities.padding_axes),
            "common_schema_features": list(capabilities.common_schema_features),
            "native_only_features": list(capabilities.native_only_features),
            "representations": list(capabilities.representations),
            "input_representations": list(capabilities.input_representations),
            "confidence_arrays": list(capabilities.confidence_arrays),
            "sampling": dict(capabilities.sampling),
            "input_requirements": {
                name: requirement.summary()
                for name, requirement in capabilities.input_requirements.items()
            },
            "execution": {
                key: list(values) for key, values in self.execution.items()
            },
            "weights": {
                "ready": self.weights_ready,
                "path": str(self.weights_path),
                "source": self.weights_source,
                "licence": self.weights_licence,
                "fetchable": self.weights_fetchable,
                "download_bytes": self.download_bytes,
                "setup": self.setup,
                "notes": self.notes,
                "profiles": [dict(profile) for profile in self.weight_profiles],
            },
            "runtime": self.runtime.summary(),
        }


_PADDING_AXES = (
    "tokens",
    "atoms",
    "msa",
    "templates",
    "structural_tokens",
    "language_model_tokens",
)


@dataclass(frozen=True, slots=True)
class PaddingConfig:
    """Optional shape normalization for reusable JAX executables.

    ``None`` targets use a token bucket with derived atom/LM capacity, the
    smallest MSA bucket holding the rows the model reads, and fixed native
    template limits. Explicit axis targets override this policy; ``msa`` is a
    capacity for that axis rather than a row cap, so a target below the stored
    alignment is refused instead of truncating the input.
    Setting an integer pins that axis to an exact padded size, which is useful
    for warming one known serving profile. The default ``overflow='error'``
    refuses inputs beyond the standard grid before an unplanned large compile;
    ``'exact'`` permits an exact token size beyond that grid; derived capacity
    checks still reject oversized axes without truncation.

    Not every model owns every axis.  Backend capabilities advertise the axes
    they understand and reject an explicitly pinned unsupported axis before
    loading weights.
    """

    tokens: int | None = None
    atoms: int | None = None
    msa: int | None = None
    templates: int | None = None
    structural_tokens: int | None = None
    language_model_tokens: int | None = None
    overflow: str = "error"

    def __post_init__(self) -> None:
        for name in _PADDING_AXES:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _strict_integer(value, name=f"padding.{name}", minimum=1),
                )
        if not isinstance(self.overflow, str) or self.overflow not in {
            "exact",
            "error",
        }:
            raise ValueError("padding.overflow must be 'exact' or 'error'")

    @property
    def explicit_axes(self) -> tuple[str, ...]:
        """Axes whose target was pinned rather than backend-selected."""
        return tuple(name for name in _PADDING_AXES if getattr(self, name) is not None)

    def summary(self) -> dict[str, Any]:
        return {
            **{name: getattr(self, name) for name in _PADDING_AXES},
            "overflow": self.overflow,
        }


#: What a directory of jobs may hold when a request names the directory rather
#: than the files. Deliberately narrower than the CLI's accepted set: the CLI
#: can turn a FASTA or a deposited structure into a job document first, and a
#: request carries job documents.
JOB_DOCUMENT_SUFFIXES = frozenset({".json", ".yaml", ".yml"})


def expand_input_directories(
    paths: Sequence[Path],
    *,
    suffixes: frozenset[str] = JOB_DOCUMENT_SUFFIXES,
) -> tuple[Path, ...]:
    """Replace every directory among ``paths`` with the job files inside it.

    A large batch is usually a directory, and "run every job in here" should not
    make the caller write the glob. Writing it by hand is also how an unsorted,
    filesystem-dependent run order reaches a benchmark, so the expansion sorts.

    Anything that is not a directory passes through untouched, including a path
    that does not exist -- reporting that is the caller's existing job, and a
    ``structure:`` selector is not a path at all. Listing is one level deep: a
    directory of directories is a different intent from a directory of jobs,
    and silently guessing between them is worse than not guessing.
    """
    expanded: list[Path] = []
    for path in paths:
        if not path.is_dir():
            expanded.append(path)
            continue
        found = sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in suffixes
        )
        if not found:
            raise FileNotFoundError(
                f"no job files in directory: {path} "
                f"(looked for {', '.join(sorted(suffixes))})"
            )
        expanded.extend(found)
    return tuple(expanded)


def _normalize_padding(padding):
    """Normalize the shared execution setting without inspecting any input files."""
    if padding is True:
        padding = PaddingConfig()
    elif padding is False or padding is None:
        padding = None
    elif isinstance(padding, Mapping):
        supported = {*_PADDING_AXES, "overflow"}
        unknown = [key for key in padding if key not in supported]
        if unknown:
            raise ValueError(
                "unsupported padding fields: "
                + ", ".join(sorted(map(repr, unknown)))
            )
        padding = PaddingConfig(**dict(padding))
    elif not isinstance(padding, PaddingConfig):
        raise ValueError(
            "padding must be a boolean, mapping, PaddingConfig, or None"
        )
    return padding


@dataclass(frozen=True, slots=True)
class JobSource:
    """Where one job of a multi-job file came from.

    A ``{"jobs": [...]}`` file runs each job as its own input, through a
    generated single-job document (see ``foldjax.input.expand_jobs_files``),
    so every consumer of an input keeps reading exactly one job. This is the
    provenance that generated file cannot carry: the file the caller named and
    the job's position and name in it, for the run manifest, failure records,
    ``foldjax plan`` and error messages.

    The command line also writes a FASTA file or a deposited structure out as
    a generated job document; ``kind`` then names that format and the source
    is the one file it came from (``index`` 0), so a failure names the file
    the caller wrote rather than the store path FoldJAX generated.
    """

    path: Path
    #: The job's 0-based position in the file's ``jobs`` list.
    index: int
    name: str
    #: ``"jobs"`` for a multi-job file, else the converted format.
    kind: str = "jobs"

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise ValueError("job source index must be an integer")
        if self.index < 0:
            raise ValueError("job source index must be non-negative")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("job source name must be a non-empty string")

    def describe(self) -> str:
        """How error messages name the job: ``jobs.yaml jobs[1] ('b')``."""
        if self.kind != "jobs":
            return str(self.path)
        return f"{self.path} jobs[{self.index}] ({self.name!r})"

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "path": str(self.path),
            "index": self.index,
            "name": self.name,
        }
        # Only when set, so a multi-job file's record is unchanged.
        if self.kind != "jobs":
            summary["kind"] = self.kind
        return summary


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
    # None until `foldjax.resolve_request` applies the backend's default from
    # `DEFAULT_SEEDS`; a number here is the caller's.
    seed: int | None = None
    # Several models take a list of seeds natively, and running one job under
    # more than one is the ordinary way to use them -- the samples from a single
    # seed are correlated. `seeds` runs the job once per entry and returns every
    # structure together; leave it unset to run the single `seed`.
    seeds: tuple[int, ...] | None = None
    # `--num-seeds 5` is `--seeds s s+1 ... s+4` counting from `seed`, or from
    # the backend's default seed when `seed` is unset, and is what people
    # actually want when they say "run it five times": the seed values
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
    options: Mapping[str, Any] = field(default_factory=dict)
    # A model-specific managed bundle and, where necessary, its matching model
    # variant.  Appended to preserve the positional layout of the public
    # request dataclass while adding the same name used by ``foldjax weights``.
    profile: str | None = None
    # Appended to preserve the positional request layout. ``None``/False keeps
    # exact historical shapes; True selects a backend's standard profile.
    padding: PaddingConfig | Mapping[str, Any] | bool | None = None
    # Reuse a run only when its finished manifest proves the exact request and
    # the persisted structures/representation archive still match their
    # recorded identities. Seeds are checked individually -- a five-seed job
    # that died on the fourth must not repeat the three that finished, which is
    # hours on a real checkpoint.
    resume: bool = False
    # ``stop`` (the default, and the historical behaviour) lets the first
    # failure end the whole request. ``continue`` finishes every other run and
    # records what failed, because losing seventeen good predictions to the
    # third one's OOM is what makes people stop batching.
    on_error: str = "stop"
    # What to do about a protein chain that arrived without an alignment.
    # ``none`` searches nothing and refuses such a chain, as upstream Boltz-2
    # does (ESMFold2, whose upstream folds it from one sequence, is exempt);
    # ``single`` folds it from its sequence on purpose. ``auto`` searches and
    # caches; ``required`` additionally refuses to fall back. See
    # `MSA_POLICIES` and `foldjax.input`.
    msa: str = "none"
    # Trunk representations to hand back: the per-token single stream and the
    # token-pair state every model in this package builds before it predicts
    # coordinates. Names come from `foldjax.models._representations`; "all"
    # takes everything the chosen model has. Off by default because a pair
    # representation is quadratic in token count -- half a gigabyte at a
    # thousand tokens, nearly five at three thousand -- and asking for one
    # silently would change what a recorded command costs.
    representations: tuple[str, ...] | str | None = None
    # ``full`` runs the whole model. ``trunk`` stops once the representations
    # exist, skipping diffusion and the confidence heads: the point is to get
    # the embeddings for downstream work without paying for a structure
    # nobody asked for. A ``trunk`` run returns no samples, so it is only
    # meaningful together with `representations`.
    stop_after: str = "full"
    # One of `SEED_SOURCES`, set by `foldjax.resolve_request`: where the seed
    # came from, recorded beside it in `foldjax plan` and the run manifest.
    seed_source: str | None = None
    # Set by `foldjax.resolve_requests` when this run is one job of a
    # multi-job file: `input` is then the generated single-job document and
    # this names the file and job it came from. Provenance only -- it never
    # changes what runs, and it is not part of the resume identity.
    source: JobSource | None = None
    # Structural-template search for common-schema protein chains that name no
    # template: one of `TEMPLATE_POLICIES`. Appended to keep the positional
    # layout. Like `msa`, `auto` sends the sequence to a server.
    templates: str = "none"
    # ``YYYY-MM-DD``: keep only searched templates released by this date. None
    # applies the selected backend's released default, which differs by model
    # (`foldjax.template_search`). Meaningful only when ``templates`` searches
    # (``"auto"`` or ``"required"``).
    template_max_date: str | None = None

    def __post_init__(self) -> None:
        padding = _normalize_padding(self.padding)
        object.__setattr__(self, "padding", padding)
        object.__setattr__(
            self,
            "use_compile_cache",
            _strict_boolean(self.use_compile_cache, name="use_compile_cache"),
        )
        if self.msa not in MSA_POLICIES:
            raise ValueError(
                f"msa must be one of {', '.join(MSA_POLICIES)}; got {self.msa!r}"
            )
        if self.templates not in TEMPLATE_POLICIES:
            raise ValueError(
                f"templates must be one of {', '.join(TEMPLATE_POLICIES)}; "
                f"got {self.templates!r}"
            )
        if self.template_max_date is not None:
            object.__setattr__(
                self,
                "template_max_date",
                _iso_date(self.template_max_date, name="template_max_date"),
            )
            if self.templates == "none":
                raise ValueError(
                    "template_max_date filters searched templates; set "
                    "templates='auto' or 'required' (--templates auto) or drop it"
                )
        if self.stop_after not in STOP_POINTS:
            raise ValueError(
                f"stop_after must be one of {', '.join(STOP_POINTS)}; "
                f"got {self.stop_after!r}"
            )
        if self.stop_after in {"trunk", "inputs"} and not self.representations:
            raise ValueError(
                f"stop_after={self.stop_after!r} stops before structure prediction, "
                "so it produces nothing unless representations are requested; "
                "pass representations='all' or supported names for the selected stage"
            )
        if isinstance(self.representations, str):
            object.__setattr__(self, "representations", (self.representations,))
        elif self.representations is not None:
            object.__setattr__(self, "representations", tuple(self.representations))
        if self.on_error not in ERROR_POLICIES:
            raise ValueError(
                f"on_error must be one of {', '.join(ERROR_POLICIES)}; "
                f"got {self.on_error!r}"
            )
        object.__setattr__(
            self, "resume", _strict_boolean(self.resume, name="resume")
        )
        if not isinstance(self.options, Mapping):
            raise ValueError("options must be a mapping")
        options: dict[str, Any] = {}
        for raw_key, value in self.options.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ValueError("options keys must be non-empty strings")
            key = raw_key.strip()
            if key in options:
                raise ValueError(
                    f"option {key!r} was set more than once after trimming keys"
                )
            options[key] = value
        object.__setattr__(self, "options", options)
        if self.profile is not None:
            if not isinstance(self.profile, str) or not self.profile.strip():
                raise ValueError("profile must be a non-empty string")
            object.__setattr__(self, "profile", self.profile.strip())
        for name in ("input", "weights", "output_dir", "cache_dir"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))
        if (self.model is None) == (self.models is None):
            raise ValueError("set exactly one of model and models")
        if (self.input is None) == (self.inputs is None):
            raise ValueError("set exactly one of input and inputs")
        if self.model is not None:
            if not isinstance(self.model, str) or not self.model.strip():
                raise ValueError("model must be a non-empty string")
            object.__setattr__(self, "model", self.model.strip())
        if self.models is not None:
            if isinstance(self.models, str):
                raise ValueError("models must be a sequence; use model for one model")
            try:
                raw_models = tuple(self.models)
            except TypeError as error:
                raise ValueError("models must be a sequence of strings") from error
            if not all(isinstance(value, str) for value in raw_models):
                raise ValueError("every models entry must be a string")
            models = tuple(value.strip() for value in raw_models)
            if not models:
                raise ValueError("models must not be empty")
            if any(not value for value in models):
                raise ValueError("models entries must be non-empty strings")
            if len(set(models)) != len(models):
                raise ValueError("models must be unique")
            object.__setattr__(self, "models", models)
        if self.inputs is not None:
            if isinstance(self.inputs, (str, Path)):
                raise ValueError("inputs must be a sequence; use input for one file")
            inputs = tuple(Path(value) for value in self.inputs)
            if not inputs:
                raise ValueError("inputs must not be empty")
            # A directory here means "every job in it", which is what a batch
            # of any size actually looks like on disk.
            inputs = expand_input_directories(inputs)
            object.__setattr__(self, "inputs", inputs)
        for path in self.resolved_inputs:
            if path.is_dir():
                # Only the plural spelling expands: a directory is plural by
                # nature, and one run cannot be several.
                raise IsADirectoryError(
                    f"input names a directory: {path}; use "
                    f"inputs=({path.name!r},) to run every job inside it"
                )
            if not path.is_file():
                detail = "does not exist" if not path.exists() else "is not a file"
                raise FileNotFoundError(f"input {detail}: {path}")
        if self.weights is not None and not self.weights.exists():
            raise FileNotFoundError(f"weights do not exist: {self.weights}")
        if not self.use_compile_cache and self.cache_dir is not None:
            raise ValueError(
                "cache_dir and use_compile_cache=False were both set; remove "
                "cache_dir or enable the compile cache"
            )
        seed = (
            None
            if self.seed is None
            else _strict_integer(self.seed, name="seed", minimum=0)
        )
        object.__setattr__(self, "seed", seed)
        if self.source is not None:
            if not isinstance(self.source, JobSource):
                raise ValueError("source must be a JobSource or None")
            if self.inputs is not None:
                raise ValueError(
                    "source names the file one input came from; it cannot "
                    "describe several inputs"
                )
        if self.seed_source is not None and self.seed_source not in SEED_SOURCES:
            raise ValueError(
                f"seed_source must be one of {', '.join(SEED_SOURCES)}; "
                f"got {self.seed_source!r}"
            )
        if self.seeds is not None:
            seeds = tuple(
                _strict_integer(value, name="each seed", minimum=0)
                for value in self.seeds
            )
            if not seeds:
                raise ValueError("seeds must not be empty")
            if len(set(seeds)) != len(seeds):
                raise ValueError("seeds must be unique")
            if seed not in (None, 0):
                # Silently preferring one would change which structures come
                # back without changing the exit code, the same rule the
                # sampling knobs follow. An explicit 0 beside `seeds` is what
                # callers wrote while 0 was the default, and stays accepted.
                raise ValueError("seed and seeds were both set; pass one of them")
            object.__setattr__(self, "seeds", seeds)
        if self.num_seeds is not None:
            num_seeds = _strict_integer(
                self.num_seeds, name="num_seeds", minimum=1
            )
            object.__setattr__(self, "num_seeds", num_seeds)
            if self.seeds is not None:
                raise ValueError("seeds and num_seeds were both set; pass one of them")
        for name in ("num_samples", "num_steps", "num_recycles", "max_msa_depth"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _strict_integer(
                        value, name=name, minimum=0 if name == "num_recycles" else 1
                    ),
                )

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
        if self.seed is None:
            raise ValueError(
                "this request's seed is not resolved yet; "
                "foldjax.resolve_request() applies the model's default seed"
            )
        if self.num_seeds is not None:
            return tuple(range(self.seed, self.seed + self.num_seeds))
        return (self.seed,)

    @property
    def seed_count(self) -> int:
        """How many seeds this request runs, known before any is drawn."""
        if self.seeds is not None:
            return len(self.seeds)
        return 1 if self.num_seeds is None else self.num_seeds

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
        from foldjax.redaction import redact

        return {
            "seed": int(self.seed),
            "structure_path": (
                str(self.structure_path) if self.structure_path is not None else None
            ),
            "coordinate_shape": _coordinate_shape(self.coordinates),
            "scores": {str(key): float(value) for key, value in self.scores.items()},
            "metadata": redact(dict(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class Representations:
    """The trunk arrays a run produced, and what their axes count.

    Arrays are read from disk on access rather than held: a pair
    representation is quadratic in token count, and a five-model comparison
    that loaded every one of them eagerly would need the memory of all five
    at once for no reason. ``result.representations["pair"]`` reads one.

    ``describe`` is the part that keeps a comparison honest. Two models both
    call their pair state ``pair``, but one may index residues and another
    structural tokens, and nothing about the array says which -- so the space
    each axis counts travels with it.
    """

    model: str
    path: Path
    manifest: Mapping[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self.manifest)

    def __len__(self) -> int:
        return len(self.manifest)

    def __contains__(self, name: object) -> bool:
        return name in self.manifest

    def names(self) -> tuple[str, ...]:
        """Every representation this run wrote, in the order it built them."""
        return tuple(self.manifest)

    def describe(self, name: str) -> Mapping[str, Any]:
        """Shape, dtype, axis names and the space those axes count."""
        if name not in self.manifest:
            raise KeyError(
                f"{name!r} was not written by this run; it has: "
                + ", ".join(self.manifest)
            )
        return self.manifest[name]

    def __getitem__(self, name: str):
        import numpy as np

        description = self.describe(name)
        with np.load(self.path) as data:
            value = data[name]
        from foldjax.models._representations import restore_logical_dtype

        return restore_logical_dtype(value, description.get("dtype"))

    def load(self) -> dict[str, Any]:
        """Every array at once, for callers that want them all in memory."""
        import numpy as np

        with np.load(self.path) as data:
            values = {name: data[name] for name in data.files}
        from foldjax.models._representations import restore_logical_dtype

        return {
            name: restore_logical_dtype(
                value,
                self.manifest.get(name, {}).get("dtype")
                if isinstance(self.manifest.get(name), Mapping)
                else None,
            )
            for name, value in values.items()
        }

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "path": str(self.path),
            "names": list(self.manifest),
        }


@dataclass(frozen=True, slots=True)
class PredictionResult:
    model: str
    samples: tuple[PredictionSample, ...] = ()
    output_dir: Path | None = None
    raw: Any = None
    #: Present when the request asked for them. A ``stop_after='trunk'``
    #: run has these and no samples.
    representations: Representations | None = None
    # Present only for opt-in padding. This is the concrete profile that ran,
    # not the requested policy, so cache warm can report exactly one executable
    # shape without pretending that it populated an entire grid.
    shape_profile: dict[str, Any] | None = None
    # Populated by the model facade; unresolved checkpoint defaults are labelled.
    configuration: Mapping[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        summary = {
            "model": self.model,
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "samples": [sample.summary() for sample in self.samples],
        }
        if self.configuration is not None:
            summary["configuration"] = dict(self.configuration)
        if self.shape_profile is not None:
            summary["shape_profile"] = dict(self.shape_profile)
        if self.representations is not None:
            summary["representations"] = self.representations.summary()
        return summary


@dataclass(frozen=True, slots=True)
class PredictionFailure:
    """One model/input/seed that did not produce a structure, and why.

    A successful run leaves `foldjax_run.json` behind and can be read back
    forever. A failed one used to leave nothing at all: with ``on_error``
    set to ``continue``, seventeen predictions succeeded and the reason the
    other three did not existed only in the terminal scrollback of whoever
    started the batch. This is that record.
    """

    model: str
    input: Path
    error: str
    error_type: str
    seed: int | None = None
    output_dir: Path | None = None
    #: The multi-job file and job this input came from, when it did.
    source: JobSource | None = None

    def summary(self) -> dict[str, Any]:
        summary = {
            "model": self.model,
            "input": str(self.input),
            "seed": self.seed,
            "output_dir": (
                str(self.output_dir) if self.output_dir is not None else None
            ),
            "error_type": self.error_type,
            "error": self.error,
        }
        # Only when set, so a directory batch's failure file is unchanged.
        if self.source is not None:
            summary["source"] = self.source.summary()
        return summary


@dataclass(frozen=True, slots=True)
class BatchReport:
    """Everything one request produced: what ran, what was skipped, what failed.

    `predict` keeps its original return type -- results only -- because that is
    what every existing caller unpacks. A batch needs the other two lists to be
    actionable, so it has its own entry point rather than a return type that
    changes shape depending on a flag.
    """

    results: tuple[PredictionResult, ...] = ()
    failures: tuple[PredictionFailure, ...] = ()
    #: Runs whose finished manifest was found and reused instead of re-run.
    skipped: tuple[Path, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> dict[str, Any]:
        return {
            "results": [result.summary() for result in self.results],
            "failures": [failure.summary() for failure in self.failures],
            "skipped": [str(path) for path in self.skipped],
        }
