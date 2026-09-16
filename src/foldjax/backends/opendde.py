"""OpenDDE-JAX adapter."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from importlib import import_module
from pathlib import Path
from typing import NamedTuple

from foldjax.backends._ccd_session import ManagedCcdSession
from foldjax.backends._representations import _representations_result
from foldjax.backends._weight_session import PreparedWeightSession
from foldjax.backends.base import (
    MATMUL_PRECISION_OPTION,
    SAMPLING_OPTIONS,
    Backend,
    square_grid_cp_layout,
    validate_memory_policy_options,
)
from foldjax.execution import DETERMINISTIC_ARGV_OPTION
from foldjax.models import _representations
from foldjax.models._managed_memory import lease as managed_memory_lease
from foldjax.padding import cp_aligned_padding
from foldjax.schema import (
    InputRequirement,
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
    _strict_boolean,
)
from foldjax.scores import sample_summary_scores

# Every environment variable the native CLI assigns. Asserted against its
# source in tests/test_native_contracts.py so a new export cannot escape.
_EXPORTED_ENVIRONMENT = (
    "JAX_PLATFORMS",
    "PROTENIX_CCD_COMPONENTS_FILE",
    "PROTENIX_CCD_RDKIT_MOL_FILE",
    "PROTENIX_KALIGN_BINARY",
    "PROTENIX_TEMPLATE_MMCIF_DIR",
    "PROTENIX_TEMPLATE_OBSOLETE_FILE",
    "PROTENIX_TEMPLATE_RELEASE_DATES_FILE",
)
_CLI_OPTIONS = {
    "ccd_rdkit_cache",
    "components_cif",
    # Distribute the diffusion atom graph over CP rows. Released default on:
    # under a mesh the atom-pair cache, both atom transformer stacks and the
    # atom<->token routing are split, and a serial run ignores it. Rendered
    # like this port's other switches -- `--cp-atom-windows true|false` -- so
    # the flag loop below carries it without a negative-flag vocabulary.
    "cp_atom_windows",
    "cp_devices",
    "cp_layout",
    "deterministic_ops",
    "diffusion_attention_backend",
    "diffusion_chunk_size",
    "diffusion_dtype",
    "kalign_binary",
    "max_msa_depth",
    # Admission against this card's own reported ceiling. Members here because
    # this is the set of options the port's parser takes, and it takes them;
    # subtracted from `compile_options` below, because a refused run and a
    # warned run build the same program and must share one cache namespace.
    "memory_budget_gib",
    "memory_check",
    "token_q_chunk_size",
    "single_att_q_chunk_size",
    "triangle_att_q_chunk_size",
    "triangle_mul_chunk_size",
    "chunk_policy",
    "confidence_dtype",
    "num_recycles",
    "n_keys",
    "n_queries",
    "num_samples",
    "num_steps",
    "structural_single_attention_backend",
    "template_mmcif_dir",
    "template_obsolete_map",
    "template_release_dates",
    "trunk_dtype",
    "trunk_single_attention_backend",
    "use_rna_msa",
    "use_template",
}

#: The members of :data:`_CLI_OPTIONS` that decide whether the run starts
#: rather than what it compiles. Subtracted from `compile_options` so two runs
#: differing only in a budget share one cache namespace and one executable.
_MEMORY_OPTIONS = frozenset({"memory_budget_gib", "memory_check"})

#: Compile-relevant defaults released by the native OpenDDE prediction CLI.
#:
#: Cache-directory planning must stay lightweight, so the adapter keeps these
#: scalar copies instead of importing the model/JAX runtime. A drift test reads
#: the parser defaults directly and pins every value and exact type to that
#: native authority.
_RELEASED_COMPILE_DEFAULTS: dict[str, object] = {
    "num_samples": 5,
    "num_steps": 200,
    "num_recycles": 10,
    "max_msa_depth": 16384,
    "n_queries": 32,
    "n_keys": 128,
    "diffusion_attention_backend": "xla_jit",
    "trunk_single_attention_backend": "xla_jit",
    "structural_single_attention_backend": "xla_jit",
    "trunk_dtype": "bf16",
    "confidence_dtype": "bf16",
    "diffusion_dtype": "fp32",
    "chunk_policy": "auto",
    "cp_atom_windows": True,
    "cp_devices": 1,
    "cp_layout": "auto",
    "use_template": False,
    "use_rna_msa": False,
    "deterministic_ops": "off",
}

#: Every `PredictionConfig` field's own parser default, so a resolved request
#: becomes the configuration the run takes without parsing the argv this
#: adapter renders back into one.
#:
#: The parser remains the authority. `tests/test_native_config_equivalence.py`
#: builds the configuration both ways for a matrix of requests -- through the
#: captured parser on the rendered argv, and through `_native_invocation`
#: below -- and compares them field by field, so a bare request pins this
#: whole table against the parser's own namespace: a default that drifts here
#: fails rather than runs.
#:
#: `max_msa_depth` is the one entry that is deliberately not the parser's
#: `None`: `cli/predict.py` resolves it through `_resolve_msa_depth` before it
#: builds the configuration, because `None` there means "this port's own
#: depth" rather than "unset". `input_json`, `weights` and `out` are always
#: overridden below -- they are the request -- and are spelled anyway, because
#: this table is the parser's whole namespace and `PredictionConfig` accepts
#: nothing less than every field.
_PARSER_DEFAULTS: dict[str, object] = {
    "input_json": None,
    "weights": None,
    "out": None,
    "seed": None,
    "num_samples": 5,
    "num_steps": 200,
    "num_recycles": 10,
    "n_queries": 32,
    "n_keys": 128,
    "use_template": False,
    "use_rna_msa": False,
    "max_msa_depth": 16384,
    "deterministic_ops": "off",
    "diffusion_attention_backend": "xla_jit",
    "trunk_single_attention_backend": "xla_jit",
    "structural_single_attention_backend": "xla_jit",
    "no_graph_jit": False,
    "cp_devices": 1,
    "cp_atom_windows": True,
    "cp_layout": "auto",
    "diffusion_chunk_size": None,
    "triangle_mul_chunk_size": None,
    "triangle_att_q_chunk_size": None,
    "single_att_q_chunk_size": None,
    "token_q_chunk_size": None,
    "chunk_policy": "auto",
    "trunk_dtype": "bf16",
    "confidence_dtype": "bf16",
    "diffusion_dtype": "fp32",
    "include_raw": False,
    "representations_dir": None,
    "stop_after": "full",
    "representations": None,
    "cpu_only": False,
    "compile_cache": None,
    "components_cif": None,
    "ccd_rdkit_cache": None,
    "template_mmcif_dir": None,
    "template_release_dates": None,
    "template_obsolete_map": None,
    "kalign_binary": None,
    "memory_check": "refuse",
    "memory_budget_gib": None,
}


def _integer_option(key: str, value: object) -> int:
    """The parser's `type=int`, on the string the renderer would have passed."""
    try:
        return int(str(value))
    except ValueError as error:
        raise ValueError(f"{key} must be an integer; got {value!r}") from error


def _number_option(key: str, value: object) -> float:
    """The parser's `type=float`."""
    try:
        return float(str(value))
    except ValueError as error:
        raise ValueError(f"{key} must be a number; got {value!r}") from error


def _path_option(_key: str, value: object) -> Path:
    """The parser's `type=Path`."""
    return Path(str(value))


def _text_option(_key: str, value: object) -> str:
    """A plain string option, as `str(value)` reached the parser through argv."""
    return str(value)


def _boolean_option(key: str, value: object) -> bool:
    """The parser's own `type=_boolean`: this port spells switches as values.

    Its vocabulary rather than `_strict_boolean`'s: `cli/predict.py:_boolean`
    is what the rendered `--use-template True` went through, and it takes only
    `true` and `false`, case and surrounding space aside.
    """
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{key} is a switch; pass true or false, not {value!r}")


#: How each native option becomes its configuration field: the parser's own
#: ``type=``, and its ``choices=`` where it has them.
#:
#: Both halves, because the renderer stringified every value and the parser
#: applied both to the string. The type alone would leave the vocabulary
#: unchecked, and below the parser these values are read with ``==`` and no
#: vocabulary at all -- `runner._load_prepared_params` branches on ``bf16`` and
#: loads FP32 weights for everything else, `deterministic_ops` compares against
#: ``on`` -- so a misspelling that argparse refused would have quietly run the
#: other arm.
_OPTION_SPECS: dict[
    str, tuple[Callable[[str, object], object], tuple[str, ...] | None]
] = {
    "ccd_rdkit_cache": (_path_option, None),
    "chunk_policy": (_text_option, ("auto", "manual", "off")),
    "components_cif": (_path_option, None),
    "confidence_dtype": (_text_option, ("fp32", "bf16")),
    "cp_atom_windows": (_boolean_option, None),
    "cp_devices": (_integer_option, None),
    "cp_layout": (_text_option, ("auto", "1d", "2d")),
    "deterministic_ops": (_text_option, ("off", "on")),
    "diffusion_attention_backend": (_text_option, ("xla", "xla_jit", "xla_sdpa")),
    "diffusion_chunk_size": (_integer_option, None),
    "diffusion_dtype": (_text_option, ("fp32", "bf16")),
    "kalign_binary": (_path_option, None),
    "max_msa_depth": (_integer_option, None),
    "memory_budget_gib": (_number_option, None),
    "memory_check": (_text_option, ("refuse", "warn")),
    "n_keys": (_integer_option, None),
    "n_queries": (_integer_option, None),
    "num_recycles": (_integer_option, None),
    "num_samples": (_integer_option, None),
    "num_steps": (_integer_option, None),
    "single_att_q_chunk_size": (_integer_option, None),
    "structural_single_attention_backend": (
        _text_option,
        ("xla", "xla_jit", "xla_sdpa"),
    ),
    "template_mmcif_dir": (_path_option, None),
    "template_obsolete_map": (_path_option, None),
    "template_release_dates": (_path_option, None),
    "token_q_chunk_size": (_integer_option, None),
    "triangle_att_q_chunk_size": (_integer_option, None),
    "triangle_mul_chunk_size": (_integer_option, None),
    "trunk_dtype": (_text_option, ("bf16", "fp32")),
    "trunk_single_attention_backend": (_text_option, ("xla", "xla_jit", "xla_sdpa")),
    "use_rna_msa": (_boolean_option, None),
    "use_template": (_boolean_option, None),
}


def _option_field(key: str, value: object) -> object:
    """One native option as the parser would have produced it from argv."""
    coerce, choices = _OPTION_SPECS[key]
    field = coerce(key, value)
    if choices is not None and field not in choices:
        raise ValueError(f"{key} must be one of {choices}; got {field!r}")
    return field


class _NativeInvocation(NamedTuple):
    """One resolved request in both of the spellings the native run needs.

    ``config_fields`` is what runs: the parser's own namespace, assembled from
    the request instead of from text. ``argv`` is the canonical command this
    adapter has always rendered, kept because it is the spelling the result
    records. Both come out of one pass, so the equivalence test can treat the
    parser on ``argv`` as the oracle for the fields the run was handed.

    ``cp_devices`` and ``cp_layout`` are read while the options are still
    whole, because the mesh this run will build is what its automatic padding
    targets have to divide. The fields stay a plain mapping rather than a
    ``PredictionConfig``: the config class is read off the imported runner, and
    that import stays where it was, after every option has been checked.
    """

    argv: list[str]
    config_fields: dict[str, object]
    representations: tuple[str, ...]
    cp_devices: int
    cp_layout: str
    matmul_precision: Callable[[], AbstractContextManager[None]]


class OpenDDEBackend(ManagedCcdSession, Backend):
    name = "opendde"
    session_reuse = True
    # OpenDDE has two token spaces: the residue trunk and the expanded
    # structural diffusion branch.  Both, the atom axis, and sampled MSA rows
    # have end-to-end masks and are cropped before public output.
    padding_axes = ("tokens", "atoms", "msa", "structural_tokens")
    native_options = frozenset(_CLI_OPTIONS | {"include_raw"})
    sampling_options = SAMPLING_OPTIONS
    # OpenDDE has no triangle-kernel option of its own -- it drives Protenix's
    # trunk but exposes only the chunk sizes -- so `triangle_kernel` is absent
    # here and asking for it is an error rather than a silent no-op.
    execution_options = {
        **MATMUL_PRECISION_OPTION,
        "dtype": ("trunk_dtype", {"float32": "fp32", "bfloat16": "bf16"}),
        "attention_kernel": (
            "trunk_single_attention_backend",
            {"auto": "xla_jit", "xla": "xla_jit"},
        ),
        # Repeatable reduction orders, compiled into this run's executables
        # rather than asked for with a process-wide XLA environment variable.
        # The shared entry, because this port is driven by rendering argv for
        # its own predict parser and every argv port renders the same strings.
        **DETERMINISTIC_ARGV_OPTION,
    }
    # `_CLI_OPTIONS` plus the one generated name that is not a flag:
    # `matmul_precision` travels in a ContextVar and is popped before the flag
    # loop renders argv (`backends/base.py:123`), so deriving the cache
    # identity from the flag list alone left a `highest` run and a `high` run
    # sharing one namespace for two programs. This port calls
    # `resolved_matmul_precision` nowhere, so it pins no released value and
    # there is no spelling for `_RELEASED_COMPILE_DEFAULTS` to alias.
    compile_options = tuple(
        sorted((_CLI_OPTIONS - _MEMORY_OPTIONS) | {"matmul_precision"})
    )

    def __init__(self) -> None:
        self._weights = PreparedWeightSession(self.name)
        self._managed_memory: ExitStack | None = None
        self._ccd_memory_leased = False

    def _ccd_lease(self) -> AbstractContextManager[None]:
        """Lease shared Protenix/OpenDDE chemistry lazily."""

        from foldjax.models.protenix.data.featurize_json import (
            _release_external_ccd_cache,
        )

        return managed_memory_lease(
            "protenix_external_ccd", _release_external_ccd_cache
        )

    def validate_native_options(self, options: dict[str, object]) -> None:
        # Here rather than at the parser, and here rather than at the admission
        # check: `foldjax plan` runs this and never reaches one.
        validate_memory_policy_options(options)
        _strict_boolean(options.get("include_raw", False), name="include_raw")
        _strict_boolean(options.get("use_template", False), name="use_template")
        _strict_boolean(options.get("use_rna_msa", False), name="use_rna_msa")
        _strict_boolean(options.get("cp_atom_windows", True), name="cp_atom_windows")

    def cache_profile(self, request: PredictionRequest) -> dict[str, object]:
        """Keep explicit released defaults in the omitted cache namespace.

        The native parser supplies these exact values before chunk resolution
        and whole-model inference. Strip only exact type-and-value matches;
        non-default, malformed, conditional, and ambient graph choices retain
        their separate identities.

        ``cp_layout`` is recorded resolved for every distributed run, because
        this port's ``auto`` is the square grid on a perfect-square device
        count (`models/opendde/models/model.py:_resolve_cp_layout`): omitting
        it and asking for ``1d`` on four devices are two programs now, and
        absence has to keep one meaning. A serial run keeps the alias it had --
        the layout decides nothing without a mesh -- so every namespace
        recorded before this one is still the namespace it was.
        """

        profile = super().cache_profile(request)
        options = self.apply_sampling(request)
        self.validate_native_options(options)
        self._strip_released_defaults(profile, _RELEASED_COMPILE_DEFAULTS)
        self._strip_released_defaults(profile, {"cp_layout": "1d"})
        resolved_cp_layout = square_grid_cp_layout(options)
        if resolved_cp_layout is not None:
            profile["cp_layout"] = resolved_cp_layout
        profile["return_confidence_details"] = _strict_boolean(
            options.get("include_raw", False), name="include_raw"
        )
        return profile

    def capabilities(self) -> ModelCapabilities:
        native_requirement = InputRequirement(
            notes=(
                "NumPy/Gemmi/RDKit featurization and JAX prediction are included "
                "in the base install and do not import PyTorch. Native "
                "templatesPath and RNA unpairedMsaPath are used when the native "
                "options use_template=true and use_rna_msa=true are selected; "
                "both retain upstream's released false defaults."
            )
        )
        common_requirement = InputRequirement(
            notes=(
                "NumPy/Gemmi/RDKit featurization and JAX prediction are included "
                "in the base install and do not import PyTorch. FoldJAX common "
                "inputs can carry mapped templates and RNA unpaired MSAs when "
                "their matching native options are true; otherwise they are "
                "rejected before materialization."
            )
        )
        return ModelCapabilities(
            representations=_representations.available("opendde"),
            input_representations=("single_inputs",),
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "opendde", "foldjax"),
            input_requirements={
                "native": native_requirement,
                "opendde": native_requirement,
                "foldjax": common_requirement,
            },
            supports_templates=True,
            padding_axes=self.padding_axes,
        )

    def _native_invocation(self, request: PredictionRequest) -> _NativeInvocation:
        """Resolve one request into the native run, in both spellings at once.

        The same sequence as before, in the same order -- the sampling knobs,
        the matmul scope, the raw-output switch, then the rendered command --
        with the configuration assembled from those same values on the way
        past, rather than recovered by parsing the command back. This port has
        no escape hatch for extra native argv, so every run reaches the runner
        as a configuration.
        """

        options = self.apply_sampling(request)
        # Out before the leftover-option check below: carried by the scope, not
        # by argv.
        matmul_precision = self.matmul_precision(options)
        include_raw = options.pop("include_raw", False)
        if not isinstance(include_raw, bool):
            raise ValueError("include_raw must be a boolean")

        argv = [
            "--input-json",
            str(request.input),
            "--weights",
            str(request.weights),
            "--out",
            str(request.output_dir),
            "--seed",
            str(request.seed),
        ]
        fields: dict[str, object] = {
            **_PARSER_DEFAULTS,
            "input_json": Path(str(request.input)),
            "weights": Path(str(request.weights)),
            "out": Path(str(request.output_dir)),
            "seed": int(request.seed),
            "include_raw": include_raw,
        }
        if request.cache_dir is not None:
            argv.extend(("--compile-cache", str(request.cache_dir)))
            fields["compile_cache"] = Path(str(request.cache_dir))
        # Read before the loop below pops them into argv: the mesh this run
        # will build decides what its automatic padding targets must divide.
        # Resolved, not as spelled: an omitted layout builds the square grid on
        # a perfect-square count here, and the grid's two-row alignment is what
        # its padding has to meet -- so an omitted layout and an explicit `2d`
        # pad to the same shapes.
        cp_devices = int(options.get("cp_devices", 1))
        cp_layout = square_grid_cp_layout(options) or str(
            options.get("cp_layout", "auto")
        )
        native_options: dict[str, object] = {}
        for key in sorted(_CLI_OPTIONS):
            if key in options:
                value = options.pop(key)
                native_options[key] = value
                argv.extend((f"--{key.replace('_', '-')}", str(value)))
        if include_raw:
            argv.append("--include-raw")
        wanted = _representations.resolve(
            request.representations,
            (
                {
                    "single_inputs": _representations.specs_for("opendde")[
                        "single_inputs"
                    ]
                }
                if request.stop_after == "inputs"
                else _representations.specs_for("opendde")
            ),
        )
        if wanted:
            argv.extend(("--representations", ",".join(wanted)))
            # Pinned so that every backend puts the archive in the same place;
            # each model's own output tree is shaped differently.
            argv.extend(("--representations-dir", str(request.output_dir)))
            fields["representations"] = ",".join(wanted)
            fields["representations_dir"] = Path(str(request.output_dir))
        if request.stop_after in {"inputs", "trunk"}:
            argv.extend(("--stop-after", request.stop_after))
            fields["stop_after"] = request.stop_after
        if options:
            raise ValueError(f"unsupported OpenDDE options: {', '.join(options)}")
        # Last, which is where argparse stood: these are the parser's own
        # rejections, so they must not overtake the leftover-option check above
        # or an unknown representation name.
        for key, value in native_options.items():
            fields[key] = _option_field(key, value)
        return _NativeInvocation(
            argv=argv,
            config_fields=fields,
            representations=wanted,
            cp_devices=cp_devices,
            cp_layout=cp_layout,
            matmul_precision=matmul_precision,
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        if not request.weights.is_file():
            raise FileNotFoundError(
                f"OpenDDE-JAX weights must be a native weight file: {request.weights}"
            )
        if request.weights.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            raise ValueError(
                "OpenDDE-JAX prediction requires converted native weights; "
                "run opendde-jax-export-weights first"
            )

        invocation = self._native_invocation(request)
        argv = invocation.argv
        wanted = invocation.representations
        matmul_precision = invocation.matmul_precision

        padding_profiles: list[dict[str, object]] = []
        native = import_module("foldjax.models.opendde.cli.predict")
        runner = import_module("foldjax.models.opendde.runner")
        use_session_loader = self._weights.active and bool(
            getattr(native, "PREPARED_PARAMS_LOADER_API", False)
        )

        def session_params_loader(
            path: Path,
            trunk_dtype: str,
            cacheable: bool,
        ) -> object:
            if not cacheable:  # pragma: no cover - OpenDDE always passes True
                self._weights.invalidate()
                return native._load_prepared_params(Path(path), trunk_dtype)
            return self._weights.load(
                Path(path),
                lambda source: native._load_prepared_params(source, trunk_dtype),
                prepare_key=("trunk_dtype", trunk_dtype),
            )

        with matmul_precision(), _restored_environment(), self._ccd_memory_scope():
            # Each keyword is supplied only when this run has something to say
            # with it: the padding pair only for a padded request, and the
            # private loader only while an active FoldJAX session has
            # negotiated that capability. So an unpadded request outside a
            # session still reaches the runner with nothing but its
            # configuration.
            keywords: dict[str, object] = {}
            if request.padding is not None:
                unsupported = sorted(
                    set(request.padding.explicit_axes) - set(self.padding_axes)
                )
                if unsupported:
                    raise ValueError(
                        "opendde does not support explicit padding axes: "
                        + ", ".join(unsupported)
                    )
                keywords["padding"] = cp_aligned_padding(
                    request.padding,
                    cp_devices=invocation.cp_devices,
                    cp_layout=invocation.cp_layout,
                )
                keywords["padding_profiles"] = padding_profiles
            if use_session_loader:
                keywords["_prepared_params_loader"] = session_params_loader
            written = runner.run_prediction(
                runner.PredictionConfig(**invocation.config_fields),
                **keywords,
            )
        samples = tuple(
            PredictionSample(
                seed=request.seed,
                structure_path=path,
                scores=sample_summary_scores(path),
            )
            for path in written
            if path.suffix == ".cif"
        )
        shape_profile = _shape_profile(
            padding_profiles,
            padded=request.padding is not None,
        )
        raw: dict[str, object] = {"argv": tuple(argv)}
        if shape_profile is not None:
            raw["padding"] = shape_profile
        return PredictionResult(
            model=self.name,
            samples=samples,
            output_dir=request.output_dir,
            raw=raw,
            shape_profile=shape_profile,
            representations=_representations_result(
                self.name, request.output_dir, wanted
            ),
        )


@contextmanager
def _restored_environment() -> Iterator[None]:
    """Undo the asset environment variables the native CLI exports.

    OpenDDE's CLI hands asset paths to the Protenix featurizer through
    ``os.environ``, which is process-scoped and therefore harmless for its own
    entry point. FoldJAX runs that CLI in-process, so without this a job that
    passes ``components_cif`` would silently leave it applied to every later
    prediction in the same session, including ones for other backends.
    """
    saved = {name: os.environ.get(name) for name in _EXPORTED_ENVIRONMENT}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _shape_profile(
    profiles: list[dict[str, object]],
    *,
    padded: bool,
) -> dict[str, object] | None:
    """Collapse identical native-job profiles without hiding heterogeneous runs."""

    if not padded:
        return None
    if not profiles:
        raise RuntimeError(
            "OpenDDE padding completed without reporting a concrete shape profile"
        )
    first = profiles[0]
    if all(profile == first for profile in profiles[1:]):
        return dict(first)
    return {"per_run": [dict(profile) for profile in profiles]}
