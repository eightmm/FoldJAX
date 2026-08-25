"""Protenix-JAX adapter."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from foldjax.backends._representations import _representations_result
from foldjax.backends.base import Backend
from foldjax.models import _representations
from foldjax.schema import (
    InputRequirement,
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)
from foldjax.scores import scalar_scores

_CLI_OPTIONS = {
    "cp_devices",
    "cp_layout",
    "n_sample",
    "n_step",
    "n_cycle",
    "model_name",
    "strict_token_limit",
    "esm_checkpoint_dir",
    "trunk_dtype",
    "max_msa_rows",
    "diffusion_attention_backend",
    "trunk_single_attention_backend",
    "trunk_triangle_attention_backend",
    # Unset follows the trunk. It is separately settable because the head and
    # the trunk do not have to fit at the same moment, but it must not default
    # to a different kernel than the trunk -- it used to, and that was a 39 GiB
    # temp arena at 2030 tokens.
    "confidence_triangle_attention_backend",
    "chunk_policy",
    "triangle_mul_chunk_size",
    "triangle_att_q_chunk_size",
    "single_att_q_chunk_size",
    "token_q_chunk_size",
    "opm_chunk_size",
    "diffusion_chunk_size",
}
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
)
# Protenix writes "<name>_sample_<rank>.cif" next to
# "<name>_summary_confidence_sample_<rank>.json" in one predictions directory.
_CONFIDENCE_INFIX = "_summary_confidence_sample_"

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


#: The profiles whose checkpoint is staged beside a matching ESM/ISM encoder.
#: The base profiles have no language-model conditioning, so pointing
#: `esm_checkpoint_dir` at their weight directory would name an encoder that is
#: not there.
_ESM_STAGED_PROFILES = frozenset({"mini-esm-v0.5.0", "mini-ism-v0.5.0"})
_MANAGED_ASSET_PROFILES = {
    model_name: profile
    for profile, model_name in _PROFILE_MODEL_NAMES.items()
}


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


class ProtenixBackend(Backend):
    name = "protenix"
    padding_axes = (
        "tokens",
        "atoms",
        "msa",
        "templates",
        "language_model_tokens",
    )
    native_options = frozenset(_CLI_OPTIONS | {"cli_args", "output_format"})
    sampling_options = {
        "num_samples": "n_sample",
        "num_steps": "n_step",
        "num_recycles": "n_cycle",
        "max_msa_depth": "max_msa_rows",
    }
    # Protenix spells both the names and the values its own way: `bf16` for the
    # dtype, and `_jit` suffixes on the kernels for the traced variants.
    execution_options = {
        "dtype": ("trunk_dtype", {"float32": "fp32", "bfloat16": "bf16"}),
        "triangle_kernel": (
            "trunk_triangle_attention_backend",
            {"auto": "cueq_jit", "cueq": "cueq_jit", "xla": "xla_jit"},
        ),
        "attention_kernel": (
            "trunk_single_attention_backend",
            {"auto": "xla_jit", "xla": "xla_jit"},
        ),
    }
    compile_options = (
        "n_sample",
        "n_step",
        "n_cycle",
        "model_name",
        "trunk_dtype",
        "max_msa_rows",
        "diffusion_attention_backend",
        "trunk_single_attention_backend",
        "trunk_triangle_attention_backend",
        "confidence_triangle_attention_backend",
        "chunk_policy",
        "triangle_mul_chunk_size",
        "triangle_att_q_chunk_size",
        "single_att_q_chunk_size",
        "token_q_chunk_size",
        "opm_chunk_size",
        "diffusion_chunk_size",
        "cli_args",
    )

    def validate_native_options(self, options: dict[str, Any]) -> None:
        _extra_cli_args(options.get("cli_args", ()))
        if "model_name" in options and not isinstance(options["model_name"], str):
            raise ValueError("model_name must be a string")
        checkpoint_dir = options.get("esm_checkpoint_dir")
        if checkpoint_dir is not None and not isinstance(checkpoint_dir, (str, Path)):
            raise ValueError("esm_checkpoint_dir must be a path")
        output_format = options.get("output_format", "protenix")
        if output_format not in {"npz", "protenix", "both"}:
            raise ValueError(
                "output_format must be one of 'npz', 'protenix', or 'both'"
            )

    def capabilities(self) -> ModelCapabilities:
        requirement = InputRequirement(
            notes=(
                "The default path and Protenix ESM/ISM embedding variants use "
                "only FoldJAX's NumPy/JAX runtime."
            )
        )
        return ModelCapabilities(
            representations=_representations.available("protenix"),
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "protenix", "foldjax"),
            input_requirements={
                name: requirement for name in ("native", "protenix", "foldjax")
            },
            padding_axes=self.padding_axes,
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        options = self.apply_sampling(request)
        self.validate_native_options(options)
        cli_args = _extra_cli_args(options.pop("cli_args", ()))
        argv = [
            "--input-json",
            str(request.input),
            "--weights",
            str(request.weights),
            "--out",
            str(request.output_dir),
            "--output-format",
            str(options.pop("output_format", "protenix")),
            "--seed",
            str(request.seed),
        ]
        if request.cache_dir is not None:
            argv.extend(("--compile-cache", str(request.cache_dir)))
        else:
            # Protenix's native CLI defaults --compile-cache to
            # `outputs/compile_cache`, so leaving the flag off does not turn
            # the cache off -- it relocates it, to a path relative to whatever
            # the working directory happens to be. `--no-cache` has to be said
            # out loud, or a run asked to write nothing still seeds a cache
            # directory next to wherever it was launched.
            argv.append("--no-compile-cache")
        if request.padding is not None:
            argv.append("--padding")
            for axis in self.padding_axes:
                target = getattr(request.padding, axis)
                if target is not None:
                    argv.extend((f"--pad-{axis}", str(target)))
            argv.extend(("--padding-overflow", request.padding.overflow))
        for key in sorted(_CLI_OPTIONS):
            if key not in options:
                continue
            value = options.pop(key)
            flag = f"--{key.replace('_', '-')}"
            if key not in _FLAG_OPTIONS:
                argv.extend((flag, str(value)))
                continue
            argv.extend(_render_switch(key, value))
        argv.extend(cli_args)
        if options:
            raise ValueError(f"unsupported Protenix options: {', '.join(options)}")
        wanted = _representations.resolve(
            request.representations, _representations.specs_for("protenix")
        )
        if wanted:
            argv.extend(("--representations", ",".join(wanted)))
            # Pinned so every backend puts the archive in the same place.
            argv.extend(("--representations-dir", str(request.output_dir)))
        if request.stop_after == "trunk":
            argv.extend(("--stop-after", "trunk"))
        padding_plans: list[dict[str, Any]] = []
        module = import_module("foldjax.models.protenix.cli.predict")
        if request.padding is None:
            # Keep the default adapter/native callable contract byte-for-byte:
            # third-party wrappers and older test doubles commonly accept only
            # ``argv``. The callback exists solely for the opt-in padding path.
            written = module.main(argv)
        else:
            written = module.main(
                argv,
                on_padding_plan=lambda plan, static=None: padding_plans.append(
                    {
                        **plan.summary(),
                        **({"static": dict(static)} if static is not None else {}),
                    }
                ),
            )
        samples = tuple(
            PredictionSample(
                seed=request.seed,
                structure_path=path,
                scores=_scores(path),
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


def _scores(structure_path: Path) -> dict[str, float]:
    """Read the summary confidence JSON Protenix writes beside ``structure_path``."""
    name, separator, rank = structure_path.stem.rpartition("_sample_")
    if not separator:
        return {}
    return scalar_scores(
        structure_path.with_name(f"{name}{_CONFIDENCE_INFIX}{rank}.json")
    )
