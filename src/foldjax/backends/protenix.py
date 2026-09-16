"""Protenix-JAX adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

from foldjax.backends._ccd_session import ManagedCcdSession
from foldjax.backends._representations import _representations_result
from foldjax.backends._weight_session import PreparedWeightSession
from foldjax.backends.base import (
    GLU_BACKENDS,
    MATMUL_PRECISION_OPTION,
    SAMPLING_OPTIONS,
    Backend,
    square_grid_cp_layout,
    validate_memory_policy_options,
)
from foldjax.execution import DETERMINISTIC_ARGV_OPTION
from foldjax.models import _representations
from foldjax.models._managed_memory import lease as managed_memory_lease
from foldjax.models.protenix.runtime_policy import MODEL_INFERENCE_DEFAULTS
from foldjax.schema import (
    InputRequirement,
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)
from foldjax.scores import sample_summary_scores

_CLI_OPTIONS = {
    # Distribute the diffusion atom graph over CP rows. Released default on:
    # under a mesh the atom-pair cache, both atom transformer stacks and the
    # atom<->token routing are split, and a serial run ignores it.
    "cp_atom_windows",
    "cp_devices",
    "cp_layout",
    "num_samples",
    "num_steps",
    "num_recycles",
    "model_name",
    "strict_token_limit",
    # Admission against this card's own ceiling, rather than against the
    # 2,560-token constant `strict_token_limit` restores. Neither is a compile
    # option: they decide whether the run starts, never what it compiles.
    "memory_check",
    "memory_budget_gib",
    "esm_checkpoint_dir",
    "trunk_dtype",
    # Which stages run under the BF16 autocast. Released default `auto`
    # narrows the confidence head at every size and keeps upstream's 3,840
    # gate on the diffusion sampler, so it is resolved from the job rather
    # than from this table; the table only carries the request. `upstream`
    # is the spelling that reproduces the native gate on both stages.
    "amp_policy",
    "max_msa_depth",
    # The seed of the per-cycle random MSA row draw, separately from the
    # diffusion seed the request already carries. Not a compile option: `seed`
    # is not one either, and the draw's padded row count already forks the
    # executable through XLA's own HLO hash inside one cache namespace.
    "msa_seed",
    "diffusion_attention_backend",
    "trunk_single_attention_backend",
    "trunk_triangle_attention_backend",
    # Unset follows the trunk. It is separately settable because the head and
    # the trunk do not have to fit at the same moment, but it must not default
    # to a different kernel than the trunk -- it used to, and that was a 39 GiB
    # temp arena at 2030 tokens.
    "confidence_triangle_attention_backend",
    # Which gated-linear-unit implementation the transitions run. Protenix
    # only: OpenDDE reaches the same primitives, so the value must not be
    # reachable from its own option set -- argparse does not validate a
    # default against `choices`, and a shared default would run an unmeasured
    # kernel there under a name nobody chose.
    "glu_backend",
    "chunk_policy",
    "triangle_mul_chunk_size",
    "triangle_att_q_chunk_size",
    "single_att_q_chunk_size",
    "token_q_chunk_size",
    "opm_chunk_size",
    "diffusion_chunk_size",
    "deterministic_ops",
}
#: Switches whose released value is on, so the *negative* flag is the one that
#: has to be rendered. `_render_switch`'s vocabulary cannot express these: it
#: drops a falsey value, and dropping a falsey value here would select the
#: parser's default, which is the opposite of what was asked for.
_NEGATED_FLAG_OPTIONS = frozenset({"cp_atom_windows"})
_RESERVED_CLI_FLAGS = frozenset(
    {
        "--features",
        "--input-json",
        "--weights",
        "--out",
        "--output-format",
        "--seed",
        "--seeds",
        "--compile-cache",
        "--no-compile-cache",
        "--padding",
        "--pad-tokens",
        "--pad-atoms",
        "--pad-msa",
        "--pad-templates",
        "--pad-language-model-tokens",
        "--padding-overflow",
    }
    | {f"--{name.replace('_', '-')}" for name in _CLI_OPTIONS}
    | {f"--no-{name.replace('_', '-')}" for name in _NEGATED_FLAG_OPTIONS}
)
_PROFILE_MODEL_NAMES = {
    "released": "protenix_base_default_v1.0.0",
    "v2": "protenix-v2",
    "base-20250630": "protenix_base_20250630_v1.0.0",
    "mini-esm-v0.5.0": "protenix_mini_esm_v0.5.0",
    "mini-ism-v0.5.0": "protenix_mini_ism_v0.5.0",
}
#: Options the native CLI takes as a bare switch rather than a value. Passing
#: `--strict-token-limit true` makes argparse reject the whole command, and
#: the usage dump that comes back says nothing about which argument was wrong.
_FLAG_OPTIONS = frozenset({"strict_token_limit"})
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})

#: Taken from the adapters' shared copy rather than imported from
#: :data:`foldjax.models._glu.GLU_BACKENDS`, which cannot be read without
#: importing JAX -- and cache planning runs before any model runtime is loaded,
#: which is the reason this module carries its own copies of the native defaults
#: at all. `tests/models/protenix/test_glu_backend.py` asserts this tuple and
#: `_glu`'s are equal, so a value added there and not to the shared copy fails
#: rather than drifts. The name stays because the check and its message below
#: are this port's own.
_GLU_BACKENDS = GLU_BACKENDS


def _strict_cp_devices(value: Any) -> int:
    """The requested device count, or 1 when it is not a number at all.

    Malformed values are somebody else's error -- the native parser's -- so
    this only has to avoid crashing the cross-check in `validate_native_options`
    on the way there.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _render_switch(key: str, value: Any) -> list[str]:
    """One switch option as the native CLI wants it: the flag, or nothing."""
    flag = f"--{key.replace('_', '-')}"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return [flag]
        if lowered in _FALSE:
            return []
        raise ValueError(f"{key} is a switch; pass true or false, not {value!r}")
    return [flag] if value else []


def _render_negated_switch(key: str, value: Any) -> list[str]:
    """A default-on switch: nothing when on, the negative flag when off."""
    flag = f"--no-{key.replace('_', '-')}"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return []
        if lowered in _FALSE:
            return [flag]
        raise ValueError(f"{key} is a switch; pass true or false, not {value!r}")
    return [] if value else [flag]


#: The profiles whose checkpoint is staged beside a matching ESM/ISM encoder.
#: The base profiles have no language-model conditioning, so pointing
#: `esm_checkpoint_dir` at their weight directory would name an encoder that is
#: not there.
_ESM_STAGED_PROFILES = frozenset({"mini-esm-v0.5.0", "mini-ism-v0.5.0"})
_MANAGED_ASSET_PROFILES = {
    model_name: profile
    for profile, model_name in _PROFILE_MODEL_NAMES.items()
}

#: Model-independent compile defaults supplied by the native prediction CLI.
#:
#: The sampler schedule is deliberately absent: the parser leaves its step and
#: recycle counts unset, then :data:`MODEL_INFERENCE_DEFAULTS` resolves them
#: from the explicit model name.  Keeping these lightweight scalar values here
#: lets cache planning stay free of the JAX model runtime.  A drift test reads
#: the parser defaults and pins every value and exact type; ``matmul_precision``
#: is the one entry with no flag behind it, and that test pins it to the model
#: function's signature instead.
_RELEASED_COMPILE_DEFAULTS: dict[str, object] = {
    "num_samples": 5,
    "max_msa_depth": 16384,
    # The parser's own default, and the value it leaves alone: `auto` asks the
    # native CLI to read the model name off the weight filename
    # (`cli/predict.py:553`). Naming it here is what keeps
    # `--option model_name=auto` in the namespace an omitted option selects.
    #
    # The literal rather than what it resolves to. Two runs that both say
    # `auto` and mean two models are two namespaces already: `api.py:1342`,
    # inside `resolve_cache_dir`, digests `weight_identity(request.weights)`
    # beside this profile, and the resolution reads that same path. A model
    # named explicitly keeps its own namespace -- one alias not taken rather
    # than a collision.
    "model_name": "auto",
    # `protenix_predict_static`'s own default, in the neutral vocabulary:
    # `models/predict.py:103`, read at :153 through
    # `resolved_matmul_precision`, and overridden by no caller -- the adapter
    # never renders it, because it is not a flag. So `high` is what an omitted
    # knob runs, and naming it here is what keeps the two in one namespace
    # while `highest` keeps its own.
    "matmul_precision": "high",
    "trunk_dtype": "bf16",
    "amp_policy": "auto",
    # The released denoiser attention is tokamax's fused kernel; the numbers
    # are in `models/protenix/models/predict.py`. The trunk's single attention
    # stays on this port's blocked XLA path, where the fused kernel measured
    # slower.
    "diffusion_attention_backend": "tokamax",
    "trunk_single_attention_backend": "xla_jit",
    # The shipped 2-D ring body. Named here so an explicit `xla` stays in the
    # namespace an omitted option selects, and only `tokamax` forks.
    "triangle_attention_ring_kernel": "xla",
    "chunk_policy": "auto",
    "cp_atom_windows": True,
    "cp_devices": 1,
    "cp_layout": "auto",
    "deterministic_ops": "off",
    "glu_backend": "xla",
}

#: What an omitted `diffusion_attention_backend` runs when a context-parallel
#: mesh is active: this port's own blocked attention, traced, which is what the
#: guard in `models/primitives/attention.py` names as the supported arm and
#: what the trunk single attention already defaults to. Spelled rather than
#: left absent so the resolved value reaches the cache profile too -- the
#: namespace has to name the program that ran.
_CP_DIFFUSION_ATTENTION_BACKEND = "xla_jit"

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
_PARSER_DEFAULTS: dict[str, Any] = {
    "features": None,
    "input_json": None,
    "weights": None,
    "out": None,
    "seed": None,
    "seeds": None,
    "msa_seed": None,
    "output_format": "npz",
    "num_samples": 5,
    "num_steps": None,
    "s_max": 160.0,
    "s_min": 0.0004,
    "rho": 7.0,
    "sigma_data": 16.0,
    "num_recycles": None,
    "gamma0": None,
    "eta": None,
    "n_queries": 32,
    "n_keys": 128,
    "max_msa_depth": 16384,
    "msa_search": "off",
    "msa_cache_dir": Path("outputs/msa_cache"),
    "msa_search_version": None,
    "msa_local_command": None,
    "msa_remote_url": None,
    "rna_msa_local_command": None,
    "rna_msa_search_version": None,
    "rna_msa_cache_dir": Path("outputs/rna_msa_cache"),
    "template_search_command": None,
    "template_search_version": None,
    "template_search_cache_dir": Path("outputs/template_cache"),
    "template_mmcif_dir": None,
    "strict_token_limit": False,
    "memory_check": "refuse",
    "memory_budget_gib": None,
    "full_depth_msa": True,
    "msa_row_alignment": 64,
    "max_msa_padding_rows": 8,
    "input_atom_heads": 4,
    "atom_encoder_heads": 4,
    "token_heads": 16,
    "atom_decoder_heads": 4,
    "triangle_mul_chunk_size": None,
    "triangle_att_q_chunk_size": None,
    "single_att_q_chunk_size": None,
    "token_q_chunk_size": None,
    "opm_chunk_size": None,
    "diffusion_chunk_size": None,
    "trunk_dtype": "bf16",
    "amp_policy": "auto",
    "chunk_policy": "auto",
    "use_pairformer_scan": False,
    "diffusion_scan": False,
    "sampler_scan": True,
    "denoiser_jit": False,
    "deterministic_ops": "off",
    "diffusion_attention_backend": "tokamax",
    "trunk_single_attention_backend": "xla_jit",
    "trunk_triangle_attention_backend": None,
    "confidence_triangle_attention_backend": None,
    "glu_backend": "xla",
    "confidence_scan": False,
    "no_confidence": False,
    "no_confidence_scores": False,
    "no_graph_jit": False,
    "cp_devices": 1,
    "cp_atom_windows": True,
    "cp_layout": "auto",
    "include_trunk": False,
    "representations_dir": None,
    "stop_after": "full",
    "representations": None,
    "cpu_only": False,
    "compile_cache": Path("outputs/compile_cache"),
    "no_compile_cache": False,
    "prewarm_only": False,
    "model_name": "auto",
    "esm_checkpoint_dir": None,
    "guidance_config": None,
    "padding": False,
    "pad_tokens": None,
    "pad_atoms": None,
    "pad_msa": None,
    "pad_templates": None,
    "pad_language_model_tokens": None,
    "padding_overflow": "error",
}


def _integer_option(key: str, value: Any) -> int:
    """The parser's `type=int`, on the string the renderer would have passed."""
    try:
        return int(str(value))
    except ValueError as error:
        raise ValueError(f"{key} must be an integer; got {value!r}") from error


def _number_option(key: str, value: Any) -> float:
    """The parser's `type=float`."""
    try:
        return float(str(value))
    except ValueError as error:
        raise ValueError(f"{key} must be a number; got {value!r}") from error


def _path_option(_key: str, value: Any) -> Path:
    """The parser's `type=Path`."""
    return Path(str(value))


def _text_option(_key: str, value: Any) -> str:
    """A plain string option, as `str(value)` reached the parser through argv."""
    return str(value)


def _switch_option(key: str, value: Any) -> bool:
    """A bare switch: whether the renderer would have emitted its flag.

    Read off the renderer rather than re-deciding, so the two spellings share
    one truth vocabulary and one error message.
    """
    return bool(_render_switch(key, value))


def _negated_switch_option(key: str, value: Any) -> bool:
    """A default-on switch: the negative flag is the one that turns it off."""
    return not _render_negated_switch(key, value)


#: How each native option becomes its configuration field: the parser's own
#: ``type=``, and its ``choices=`` where it has them.
#:
#: Both halves, because the renderer stringified every value and the parser
#: applied both to the string. The type alone would leave the vocabulary
#: unchecked, and below the parser these values are read with ``==`` and no
#: vocabulary at all -- `runner._load_prepared_params` branches on ``bf16`` and
#: loads FP32 weights for everything else, `deterministic_ops` compares against
#: ``on`` -- so a misspelling that argparse refused would have quietly run the
#: other arm. Refused here, at prediction time, where argparse refused it:
#: `validate_native_options` is also what `foldjax plan` runs, and planning
#: never rendered these values.
#: The 2-D ring bodies a request may ask for. Spelled here rather than
#: imported from `models/_cp_attention.py`, which imports JAX, because this
#: module must stay import-time JAX-free for `foldjax plan`.
#: `tests/test_cp_option_surface.py` asserts the two spellings are the same
#: tuple, so the copy cannot drift.
#:
#: Deliberately NOT an `_OPTION_SPECS` entry: that table is the *parser's*
#: flag specs, pinned equal to `_CLI_OPTIONS` by
#: `tests/test_native_config_equivalence.py`, and this option is not a flag --
#: it is popped before the flag loop and carried by a scope. Its values are
#: checked in `validate_native_options` instead.
_RING_TILE_KERNELS: tuple[str, ...] = ("xla", "tokamax")


_OPTION_SPECS: dict[str, tuple[Callable[[str, Any], Any], tuple[str, ...] | None]] = {
    "amp_policy": (_text_option, ("auto", "upstream", "fp32", "bf16")),
    "chunk_policy": (_text_option, ("auto", "manual", "off")),
    "confidence_triangle_attention_backend": (
        _text_option,
        ("xla", "xla_jit", "tokamax", "cueq", "cueq_jit"),
    ),
    "cp_atom_windows": (_negated_switch_option, None),
    "cp_devices": (_integer_option, None),
    "cp_layout": (_text_option, ("auto", "1d", "2d")),
    "deterministic_ops": (_text_option, ("off", "on")),
    "diffusion_attention_backend": (
        _text_option,
        ("xla", "xla_jit", "xla_sdpa", "tokamax"),
    ),
    "diffusion_chunk_size": (_integer_option, None),
    "esm_checkpoint_dir": (_path_option, None),
    "glu_backend": (_text_option, _GLU_BACKENDS),
    "max_msa_depth": (_integer_option, None),
    "memory_budget_gib": (_number_option, None),
    "memory_check": (_text_option, ("refuse", "warn")),
    "model_name": (_text_option, None),
    "msa_seed": (_integer_option, None),
    "num_recycles": (_integer_option, None),
    "num_samples": (_integer_option, None),
    "num_steps": (_integer_option, None),
    "opm_chunk_size": (_integer_option, None),
    "single_att_q_chunk_size": (_integer_option, None),
    "strict_token_limit": (_switch_option, None),
    "token_q_chunk_size": (_integer_option, None),
    "triangle_att_q_chunk_size": (_integer_option, None),
    "triangle_mul_chunk_size": (_integer_option, None),
    "trunk_dtype": (_text_option, ("bf16", "fp32")),
    "trunk_single_attention_backend": (
        _text_option,
        ("xla", "xla_jit", "xla_sdpa", "tokamax"),
    ),
    "trunk_triangle_attention_backend": (
        _text_option,
        ("xla", "xla_jit", "tokamax", "cueq", "cueq_jit"),
    ),
}


def _ring_tile_kernel_scope(kernel: str | None):
    """Enter the 2-D ring's tile-kernel scope for one prediction."""

    from foldjax.models._cp_attention import ring_tile_kernel_scope

    return ring_tile_kernel_scope(kernel)


def _option_field(key: str, value: Any) -> Any:
    """One native option as the parser would have produced it from argv."""
    coerce, choices = _OPTION_SPECS[key]
    field = coerce(key, value)
    if choices is not None and field not in choices:
        raise ValueError(f"{key} must be one of {choices}; got {field!r}")
    return field


def managed_asset_profile(options: dict[str, Any]) -> str:
    """Select managed weights for a released Protenix model variant."""
    model_name = options.get("model_name", "auto")
    if not isinstance(model_name, str):
        raise ValueError("model_name must be a string")
    return _MANAGED_ASSET_PROFILES.get(model_name, "released")


def apply_managed_profile(
    options: dict[str, Any],
    profile: str,
    *,
    weights: Path | None = None,
) -> dict[str, Any]:
    """Apply one public profile without silently overriding native choices."""
    try:
        expected_model_name = _PROFILE_MODEL_NAMES[profile]
    except KeyError as error:
        choices = ", ".join(_PROFILE_MODEL_NAMES)
        raise ValueError(
            f"unsupported asset profile {profile!r} for protenix; "
            f"choose one of {choices}"
        ) from error

    merged = dict(options)
    model_name = merged.get("model_name", "auto")
    if not isinstance(model_name, str):
        raise ValueError("model_name must be a string")
    if model_name not in {"auto", expected_model_name}:
        raise ValueError(
            f"profile {profile!r} selects model_name {expected_model_name!r}, "
            f"which conflicts with {model_name!r}"
        )
    merged["model_name"] = expected_model_name

    if profile not in _ESM_STAGED_PROFILES or weights is None:
        return merged

    managed_dir = Path(weights).parent
    checkpoint_dir = merged.get("esm_checkpoint_dir")
    if checkpoint_dir is not None:
        if not isinstance(checkpoint_dir, (str, Path)):
            raise ValueError("esm_checkpoint_dir must be a path")
        if Path(checkpoint_dir).resolve() != managed_dir.resolve():
            raise ValueError(
                f"profile {profile!r} stages its ESM/ISM checkpoint beside "
                f"the structure weights at {managed_dir}; this conflicts with "
                f"esm_checkpoint_dir={checkpoint_dir}"
            )
    merged["esm_checkpoint_dir"] = managed_dir
    return merged


def _extra_cli_args(value: Any) -> tuple[str, ...]:
    """Validate native escape-hatch arguments without reopening owned fields."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError("Protenix cli_args must be a non-string sequence of strings")
    arguments = tuple(value)
    if not all(isinstance(argument, str) for argument in arguments):
        raise ValueError("Protenix cli_args must be a non-string sequence of strings")
    for argument in arguments:
        flag = argument.split("=", 1)[0]
        if not flag.startswith("--"):
            continue
        # argparse accepts unambiguous long-option prefixes. Reject those too,
        # otherwise ``--weig other.ckpt`` still overrides ``--weights``.
        owned = next(
            (reserved for reserved in _RESERVED_CLI_FLAGS if reserved.startswith(flag)),
            None,
        )
        if owned is not None:
            raise ValueError(
                f"Protenix cli_args cannot set adapter-owned flag {owned!r}"
            )
    return arguments


class _NativeInvocation(NamedTuple):
    """One resolved request in both of the spellings the native run needs.

    ``config_fields`` is what runs: the parser's own namespace, assembled from
    the request instead of from text. ``argv`` is the canonical command this
    adapter has always rendered, kept because it is the spelling the result
    records. Both come out of one pass, so the equivalence test can treat the
    parser on ``argv`` as the oracle for the fields the run was handed.

    The fields stay a plain mapping rather than a ``PredictionConfig``: the
    config class is read off the imported runner, and that import stays where
    it was, after every option has been checked.
    """

    argv: list[str]
    config_fields: dict[str, Any]
    cli_args: tuple[str, ...]
    representations: tuple[str, ...]
    matmul_precision: Callable[[], AbstractContextManager[None]]
    #: Which body of the 2-D triangle-attention ring this run asks for. A
    #: field rather than a rendered flag because no `PredictionConfig` field
    #: and no native flag carries it: like `matmul_precision`, it travels in a
    #: scope. `None` is the shipped body.
    ring_tile_kernel: str | None


class ProtenixBackend(ManagedCcdSession, Backend):
    name = "protenix"
    session_reuse = True
    padding_axes = (
        "tokens",
        "atoms",
        "msa",
        "templates",
        "language_model_tokens",
    )
    native_options = frozenset(
        _CLI_OPTIONS
        | {"cli_args", "output_format", "triangle_attention_ring_kernel"}
    )
    sampling_options = SAMPLING_OPTIONS
    # Protenix spells both the names and the values its own way: `bf16` for the
    # dtype, and `_jit` suffixes on the kernels for the traced variants.
    execution_options = {
        **MATMUL_PRECISION_OPTION,
        "dtype": ("trunk_dtype", {"float32": "fp32", "bfloat16": "bf16"}),
        "triangle_kernel": (
            "trunk_triangle_attention_backend",
            {"auto": "cueq_jit", "cueq": "cueq_jit", "xla": "xla_jit"},
        ),
        "attention_kernel": (
            "trunk_single_attention_backend",
            {"auto": "xla_jit", "xla": "xla_jit", "tokamax": "tokamax"},
        ),
        # Repeatable reduction orders, compiled into this run's executables
        # rather than asked for with a process-wide XLA environment variable.
        # The shared entry, because this port is driven by rendering argv for
        # its own predict parser and every argv port renders the same strings.
        **DETERMINISTIC_ARGV_OPTION,
    }
    compile_options = (
        "cp_atom_windows",
        "cp_devices",
        "cp_layout",
        "num_samples",
        "num_steps",
        "num_recycles",
        "model_name",
        "trunk_dtype",
        "amp_policy",
        "max_msa_depth",
        "diffusion_attention_backend",
        "trunk_single_attention_backend",
        "trunk_triangle_attention_backend",
        "confidence_triangle_attention_backend",
        "glu_backend",
        "chunk_policy",
        "triangle_mul_chunk_size",
        "triangle_att_q_chunk_size",
        # A different ring body and different arithmetic, so a different
        # program. Not a parser flag: it reaches the ring through a scope, for
        # the reason the field on `_NativeInvocation` records.
        "triangle_attention_ring_kernel",
        "single_att_q_chunk_size",
        "token_q_chunk_size",
        "opm_chunk_size",
        "diffusion_chunk_size",
        "deterministic_ops",
        "cli_args",
        # Two policies, two programs: the value becomes the `precision`
        # attribute on every float32 dot XLA lowers, it selects the
        # cuEquivariance triangle FFI's float32 mode (`models/_cueq.py:69`),
        # and it decides which dot-algorithm preset Tokamax picks. The strip
        # table above neutralises the pinned value, so only a departure forks.
        "matmul_precision",
    )

    def __init__(self) -> None:
        self._weights = PreparedWeightSession(self.name)
        self._managed_memory: ExitStack | None = None
        self._ccd_memory_leased = False

    def managed_asset_profile(
        self,
        options: Mapping[str, Any],
        *,
        weights: Path | None = None,
        requested: str | None = None,
    ) -> str | None:
        """Read the managed variant out of the model name, when one is owed.

        Managed mini ESM/ISM checkpoints live with their matching encoder, so a
        request that lets FoldJAX resolve weights, or that names a first-class
        profile, gets one derived from ``model_name``. The legacy options-only
        external-weight path -- explicit weights and no profile -- stays
        untouched.
        """
        if weights is not None and requested is None:
            return requested
        return managed_asset_profile(options)

    def apply_managed_profile(
        self,
        options: dict[str, Any],
        profile: str,
        *,
        weights: Path | None = None,
    ) -> dict[str, Any]:
        """Apply the profile twice: the architecture first, its encoder after.

        The profile owns the architecture name, which is knowable before any
        weight path is; the matching ESM/ISM directory is added on the second
        call, once the structure-weight path is known.
        """
        return apply_managed_profile(options, profile, weights=weights)

    def _ccd_lease(self) -> AbstractContextManager[None]:
        """Lease shared Protenix/OpenDDE chemistry lazily."""

        from foldjax.models.protenix.data.featurize_json import (
            _release_external_ccd_cache,
        )

        return managed_memory_lease(
            "protenix_external_ccd", _release_external_ccd_cache
        )

    def validate_native_options(self, options: dict[str, Any]) -> None:
        _extra_cli_args(options.get("cli_args", ()))
        if "model_name" in options and not isinstance(options["model_name"], str):
            raise ValueError("model_name must be a string")
        msa_seed = options.get("msa_seed")
        # `bool` is an `int`, and `--msa-seed True` is not a number argparse
        # accepts -- it answers with the whole usage dump, which does not say
        # which argument was wrong. Rejected here for the same reason
        # `glu_backend` is.
        if msa_seed is not None and (
            isinstance(msa_seed, bool) or not isinstance(msa_seed, int)
        ):
            raise ValueError("msa_seed must be an integer")
        checkpoint_dir = options.get("esm_checkpoint_dir")
        if checkpoint_dir is not None and not isinstance(checkpoint_dir, (str, Path)):
            raise ValueError("esm_checkpoint_dir must be a path")
        # Here rather than at the parser for the reason `glu_backend` gives,
        # and here rather than at the admission check because `foldjax plan`
        # runs this and never reaches one.
        validate_memory_policy_options(options)
        output_format = options.get("output_format", "protenix")
        if output_format not in {"npz", "protenix", "both"}:
            raise ValueError(
                "output_format must be one of 'npz', 'protenix', or 'both'"
            )
        ring_kernel = options.get("triangle_attention_ring_kernel")
        if ring_kernel is not None:
            if ring_kernel not in _RING_TILE_KERNELS:
                raise ValueError(
                    "triangle_attention_ring_kernel must be one of "
                    f"{_RING_TILE_KERNELS}"
                )
            # Whether this card can run the fused tile is settled at trace
            # time (`_cp_attention.resolve_ring_tile_kernel`); asking here
            # would initialise a JAX backend inside `foldjax plan`. What is
            # settled here is that there is a ring to pick a body of at all.
            if ring_kernel != "xla" and square_grid_cp_layout(options) != "2d":
                raise ValueError(
                    "triangle_attention_ring_kernel selects a body of the 2-D "
                    "context-parallel triangle-attention ring; it needs "
                    "cp_layout=2d on a perfect-square cp_devices"
                )
        glu_backend = options.get("glu_backend", "xla")
        if glu_backend not in _GLU_BACKENDS:
            # Rejected here rather than left to the parser: argparse answers a
            # bad choice with the whole usage dump, which does not say which
            # argument was wrong.
            raise ValueError(
                f"glu_backend must be one of {_GLU_BACKENDS}; got {glu_backend!r}"
            )
        atom_windows = options.get("cp_atom_windows", True)
        if isinstance(atom_windows, str):
            if atom_windows.strip().lower() not in (_TRUE | _FALSE):
                raise ValueError(
                    "cp_atom_windows is a switch; pass true or false, not "
                    f"{atom_windows!r}"
                )
        elif not isinstance(atom_windows, bool):
            raise ValueError("cp_atom_windows must be a boolean")
        cp_devices = options.get("cp_devices", 1)
        if glu_backend != "xla" and _strict_cp_devices(cp_devices) > 1:
            raise ValueError(
                "context parallelism requires glu_backend='xla'; a fused GLU "
                "cannot be partitioned"
            )

    def apply_sampling(self, request: PredictionRequest) -> dict[str, Any]:
        """Resolve the fused denoiser attention away under context parallelism.

        The released `diffusion_attention_backend` is tokamax's fused kernel,
        and the two sites it reaches have no ``shard_map`` wrapper -- they see
        the pair representation already sharded, so
        `models/primitives/attention.py:28` refuses the kernel under a mesh.
        Left alone, that refusal rejected the shipped configuration: `--option
        cp_devices=2` with no attention option at all failed at start.

        An omitted knob is the caller taking the release, so it resolves to the
        blocked XLA path here, before validation and before either the rendered
        argv or the cache profile is built. A spelled `tokamax` is a request
        this run cannot honour and keeps the refusal above; this is the one
        layer where the two are still distinguishable, because argv and the
        native parser default read the same string either way. Same division as
        Boltz-2's (`models/boltz2/api.py:834` resolves, `backends/boltz2.py`
        refuses the named value).

        `trunk_single_attention_backend` needs none of this: it ships `xla_jit`
        at the parser, the wrapper signature and the table below, so nothing
        resolves it into the kernel the guard rejects.
        """

        options = super().apply_sampling(request)
        if (
            "diffusion_attention_backend" not in options
            and _RELEASED_COMPILE_DEFAULTS["diffusion_attention_backend"] == "tokamax"
            and _strict_cp_devices(options.get("cp_devices", 1)) > 1
        ):
            options["diffusion_attention_backend"] = _CP_DIFFUSION_ATTENTION_BACKEND
        return options

    def cache_profile(self, request: PredictionRequest) -> dict[str, Any]:
        """Keep proven released-default aliases in one cache namespace.

        The native CLI resolves the fixed defaults below before featurization
        or inference, and resolves step/recycle counts from an explicit known
        model name.  Its compiled wrapper also maps both ``cp_layout=auto`` and
        ``cp_layout=1d`` to the same 1-D mesh.  Strip only exact type-and-value
        matches and an actually empty validated ``cli_args`` sequence; ambient,
        conditional, malformed, and custom routes remain distinct.
        """

        profile = super().cache_profile(request)
        options = self.apply_sampling(request)
        self.validate_native_options(options)
        self._strip_released_defaults(profile, _RELEASED_COMPILE_DEFAULTS)

        model_name = options.get("model_name")
        schedule = (
            MODEL_INFERENCE_DEFAULTS.get(model_name)
            if isinstance(model_name, str) and model_name not in {"auto", "unknown"}
            else None
        )
        if schedule is not None:
            self._strip_released_defaults(
                profile,
                {name: schedule[name] for name in ("num_steps", "num_recycles")},
            )

        if not _extra_cli_args(options.get("cli_args", ())):
            profile.pop("cli_args", None)
        self._strip_released_defaults(profile, {"cp_layout": "1d"})
        profile["return_confidence_details"] = (
            options.get("output_format", "protenix") != "protenix"
        )
        return profile

    def capabilities(self) -> ModelCapabilities:
        requirement = InputRequirement(
            notes=(
                "The default path and Protenix ESM/ISM embedding variants use "
                "only FoldJAX's NumPy/JAX runtime."
            )
        )
        return ModelCapabilities(
            representations=_representations.available("protenix"),
            input_representations=("single_inputs",),
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "protenix", "foldjax"),
            input_requirements={
                name: requirement for name in ("native", "protenix", "foldjax")
            },
            padding_axes=self.padding_axes,
        )

    def _native_invocation(self, request: PredictionRequest) -> _NativeInvocation:
        """Resolve one request into the native run, in both spellings at once.

        The same sequence as before, in the same order -- the sampling knobs,
        the native-option check, the matmul scope, then the rendered command --
        with the configuration assembled from those same values on the way
        past, rather than recovered by parsing the command back.

        One exception, and it is why the parser is still here: a non-empty
        ``cli_args`` is arbitrary native argv, and only that parser can turn it
        into option values. Those runs keep going through it (`predict` calls
        ``main(argv)``), so an override is never silently dropped; the fields
        below are then the configuration those overrides would have been
        applied on top of.
        """

        options = self.apply_sampling(request)
        self.validate_native_options(options)
        # Taken out before the leftover-option check below, which is what makes
        # a misspelling an error: this one is carried by the scope, not by argv.
        matmul_precision = self.matmul_precision(options)
        cli_args = _extra_cli_args(options.pop("cli_args", ()))
        output_format = str(options.pop("output_format", "protenix"))
        # Out here for the same reason `matmul_precision` is: a scope carries
        # it, so the leftover-option check below must not see it and the flag
        # loop has no flag to render for it.
        ring_tile_kernel = options.pop("triangle_attention_ring_kernel", None)
        if ring_tile_kernel is not None:
            ring_tile_kernel = str(ring_tile_kernel)
        argv = [
            "--input-json",
            str(request.input),
            "--weights",
            str(request.weights),
            "--out",
            str(request.output_dir),
            "--output-format",
            output_format,
            "--seed",
            str(request.seed),
        ]
        fields: dict[str, Any] = {
            **_PARSER_DEFAULTS,
            "input_json": Path(str(request.input)),
            "weights": Path(str(request.weights)),
            "out": Path(str(request.output_dir)),
            "output_format": output_format,
            "seed": int(request.seed),
        }
        if request.cache_dir is not None:
            argv.extend(("--compile-cache", str(request.cache_dir)))
            fields["compile_cache"] = Path(str(request.cache_dir))
        else:
            # Protenix's native CLI defaults --compile-cache to
            # `outputs/compile_cache`, so leaving the flag off does not turn
            # the cache off -- it relocates it, to a path relative to whatever
            # the working directory happens to be. `--no-cache` has to be said
            # out loud, or a run asked to write nothing still seeds a cache
            # directory next to wherever it was launched. The configuration
            # says the same thing the same way: the refusal is the field that
            # moves, and `compile_cache` keeps the relative default nothing
            # reads once it is set.
            argv.append("--no-compile-cache")
            fields["no_compile_cache"] = True
        if request.padding is not None:
            argv.append("--padding")
            fields["padding"] = True
            for axis in self.padding_axes:
                target = getattr(request.padding, axis)
                if target is not None:
                    # Dashes, like every other rendered flag. Four of the five
                    # axes are one word, so this went unnoticed:
                    # `--pad-language_model_tokens` is not a flag this parser
                    # declares and not a prefix of one either, so pinning that
                    # axis explicitly ended in the usage dump rather than a
                    # prediction. The configuration below names the parser's
                    # destination, so the two agree now.
                    flag = f"--pad-{axis.replace('_', '-')}"
                    argv.extend((flag, str(target)))
                    fields[f"pad_{axis}"] = int(target)
            argv.extend(("--padding-overflow", request.padding.overflow))
            fields["padding_overflow"] = str(request.padding.overflow)
        native: dict[str, Any] = {}
        for key in sorted(_CLI_OPTIONS):
            if key not in options:
                continue
            value = options.pop(key)
            native[key] = value
            flag = f"--{key.replace('_', '-')}"
            if key in _NEGATED_FLAG_OPTIONS:
                argv.extend(_render_negated_switch(key, value))
                continue
            if key not in _FLAG_OPTIONS:
                argv.extend((flag, str(value)))
                continue
            argv.extend(_render_switch(key, value))
        argv.extend(cli_args)
        if options:
            raise ValueError(f"unsupported Protenix options: {', '.join(options)}")
        wanted = _representations.resolve(
            request.representations,
            (
                {
                    "single_inputs": _representations.specs_for("protenix")[
                        "single_inputs"
                    ]
                }
                if request.stop_after == "inputs"
                else _representations.specs_for("protenix")
            ),
        )
        if wanted:
            argv.extend(("--representations", ",".join(wanted)))
            # Pinned so every backend puts the archive in the same place.
            argv.extend(("--representations-dir", str(request.output_dir)))
            fields["representations"] = ",".join(wanted)
            fields["representations_dir"] = Path(str(request.output_dir))
        if request.stop_after in {"inputs", "trunk"}:
            argv.extend(("--stop-after", request.stop_after))
            fields["stop_after"] = request.stop_after
        # Last, which is where argparse stood: these are the parser's own
        # rejections, so they must not overtake the leftover-option check above
        # or an unknown representation name.
        for key, value in native.items():
            fields[key] = _option_field(key, value)
        return _NativeInvocation(
            argv=argv,
            config_fields=fields,
            cli_args=cli_args,
            representations=wanted,
            matmul_precision=matmul_precision,
            ring_tile_kernel=ring_tile_kernel,
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        invocation = self._native_invocation(request)
        argv = invocation.argv
        wanted = invocation.representations
        padding_plans: list[dict[str, Any]] = []
        module = import_module("foldjax.models.protenix.cli.predict")
        runner = import_module("foldjax.models.protenix.runner")
        use_session_loader = self._weights.active and bool(
            getattr(module, "PREPARED_PARAMS_LOADER_API", False)
        )

        def session_params_loader(
            path: Path,
            trunk_dtype: str,
            cacheable: bool,
        ) -> Any:
            if not cacheable:
                # The native runner just released a per-call mini ESM/ISM
                # provider. Do not retain structure params into the next call,
                # where the provider would be reconstructed beside them.
                self._weights.invalidate()
                return module._load_prepared_params(Path(path), trunk_dtype)
            return self._weights.load(
                Path(path),
                lambda source: module._load_prepared_params(source, trunk_dtype),
                prepare_key=("trunk_dtype", trunk_dtype),
            )

        def on_padding_plan(plan: Any, static: Any = None) -> None:
            padding_plans.append(
                {
                    **plan.summary(),
                    **({"static": dict(static)} if static is not None else {}),
                }
            )

        # Each keyword is supplied only when this run has something to say with
        # it: the padding callback only for a padded request, and the private
        # loader only while an active FoldJAX session has negotiated that
        # capability. So an unpadded request outside a session still reaches
        # the runner with nothing but its configuration.
        keywords: dict[str, Any] = {}
        if request.padding is not None:
            keywords["on_padding_plan"] = on_padding_plan
        if use_session_loader:
            keywords["_prepared_params_loader"] = session_params_loader
        with (
            invocation.matmul_precision(),
            self._ccd_memory_scope(),
            _ring_tile_kernel_scope(invocation.ring_tile_kernel),
        ):
            if invocation.cli_args:
                written = module.main(argv, **keywords)
            else:
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
        shape_profile = None
        if padding_plans:
            shape_profile = (
                padding_plans[0]
                if len(padding_plans) == 1
                else {"jobs": tuple(padding_plans)}
            )
        raw: dict[str, Any] = {"argv": tuple(argv)}
        if padding_plans:
            raw["padding_plans"] = tuple(padding_plans)
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
