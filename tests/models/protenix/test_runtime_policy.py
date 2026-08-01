from __future__ import annotations

import pytest

from foldjax.models.protenix.runtime_policy import (
    infer_model_name_from_path,
    model_inference_defaults,
    validate_inference_limits,
)


@pytest.mark.parametrize(
    ("model_name", "n_cycle", "n_step", "gamma0", "step_scale_eta"),
    [
        ("protenix_base_default_v1.0.0", 10, 200, 0.8, 1.5),
        ("protenix_base_constraint_v0.5.0", 10, 200, 0.8, 1.5),
        ("protenix_mini_default_v0.5.0", 4, 5, 0.0, 1.0),
        ("protenix_mini_esm_v0.5.0", 4, 5, 0.0, 1.0),
        ("protenix_mini_ism_v0.5.0", 4, 5, 0.0, 1.0),
        ("protenix_tiny_default_v0.5.0", 4, 5, 0.0, 1.0),
    ],
)
def test_original_model_inference_defaults(
    model_name: str,
    n_cycle: int,
    n_step: int,
    gamma0: float,
    step_scale_eta: float,
) -> None:
    assert model_inference_defaults(model_name) == {
        "n_cycle": n_cycle,
        "n_step": n_step,
        "gamma0": gamma0,
        "step_scale_eta": step_scale_eta,
    }


@pytest.mark.parametrize(
    "model_name",
    [
        "protenix_base_constraint_v0.5.0",
        "protenix_mini_default_v0.5.0",
        "protenix_mini_esm_v0.5.0",
        "protenix_mini_ism_v0.5.0",
        "protenix_tiny_default_v0.5.0",
    ],
)
def test_infer_all_original_model_names(model_name: str) -> None:
    assert infer_model_name_from_path(f"/weights/{model_name}.pkl") == model_name


def test_unknown_model_defaults_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="unknown Protenix model"):
        model_inference_defaults("made-up")


def test_v2_token_limit_is_preserved() -> None:
    with pytest.raises(ValueError, match="2560"):
        validate_inference_limits(model_name="protenix-v2", n_token=2561)
