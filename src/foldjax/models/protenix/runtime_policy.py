"""Runtime policy checks shared by Protenix JAX CLIs."""

from __future__ import annotations

from pathlib import Path

PROTENIX_V2_MAX_TOKENS = 2560
KNOWN_MODEL_NAMES = (
    "protenix-v2",
    "protenix_base_default_v1.0.0",
    "protenix_base_20250630_v1.0.0",
    "protenix_base_default_v0.5.0",
    "protenix_base_constraint_v0.5.0",
    "protenix_mini_default_v0.5.0",
    "protenix_mini_esm_v0.5.0",
    "protenix_mini_ism_v0.5.0",
    "protenix_tiny_default_v0.5.0",
)

_BASE_INFERENCE_DEFAULTS = {
    "n_cycle": 10,
    "n_step": 200,
    "gamma0": 0.8,
    "step_scale_eta": 1.5,
}
_SMALL_INFERENCE_DEFAULTS = {
    "n_cycle": 4,
    "n_step": 5,
    "gamma0": 0.0,
    "step_scale_eta": 1.0,
}
MODEL_INFERENCE_DEFAULTS = {
    "protenix-v2": _BASE_INFERENCE_DEFAULTS,
    "protenix_base_default_v1.0.0": _BASE_INFERENCE_DEFAULTS,
    "protenix_base_20250630_v1.0.0": _BASE_INFERENCE_DEFAULTS,
    "protenix_base_default_v0.5.0": _BASE_INFERENCE_DEFAULTS,
    "protenix_base_constraint_v0.5.0": _BASE_INFERENCE_DEFAULTS,
    "protenix_mini_default_v0.5.0": _SMALL_INFERENCE_DEFAULTS,
    "protenix_mini_esm_v0.5.0": _SMALL_INFERENCE_DEFAULTS,
    "protenix_mini_ism_v0.5.0": _SMALL_INFERENCE_DEFAULTS,
    "protenix_tiny_default_v0.5.0": _SMALL_INFERENCE_DEFAULTS,
}


def infer_model_name_from_path(path: str | Path | None) -> str | None:
    """Infer a known Protenix model name from a native weight path."""

    if path is None:
        return None
    name = Path(path).name
    for model_name in KNOWN_MODEL_NAMES:
        if model_name in name:
            return model_name
    return None


def model_inference_defaults(model_name: str) -> dict[str, int | float]:
    """Return the original Protenix sampler defaults for a model variant."""

    try:
        defaults = MODEL_INFERENCE_DEFAULTS[model_name]
    except KeyError as exc:
        raise ValueError(f"unknown Protenix model: {model_name}") from exc
    return dict(defaults)


def validate_inference_limits(*, model_name: str | None, n_token: int) -> None:
    """Apply model-specific inference limits from the Protenix runtime."""

    if n_token <= 0:
        raise ValueError("n_token must be positive")
    if model_name == "protenix-v2" and n_token > PROTENIX_V2_MAX_TOKENS:
        raise ValueError(
            "protenix-v2 model does not support n_token > 2560. "
            "It might cause OOM."
        )
