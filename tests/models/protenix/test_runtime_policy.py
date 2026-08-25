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


def test_v2_token_limit_is_preserved_under_strict() -> None:
    """Upstream's exact refusal is still reachable, and still says 2560.

    It stopped being the default when the port was measured running past it,
    but a caller who wants upstream's behaviour rather than this one's has to
    be able to get it back unchanged.
    """
    with pytest.raises(ValueError, match="2560"):
        validate_inference_limits(
            model_name="protenix-v2", n_token=2561, strict_token_limit=True
        )


def test_v2_runs_past_the_upstream_limit_and_says_what_it_costs() -> None:
    """Upstream's 2,560-token refusal is memory, not architecture.

    Its own assertion says "It might cause OOM", and nothing in the model is
    bounded by token count: relative position encoding buckets separation with
    `clip(delta, 0, 2 * r_max)` at r_max=32, so every pair more than 32
    residues apart already shares a bucket. A budget calibrated on one
    implementation does not transfer to another that measurably uses less --
    3,012 tokens ran here at 78.2 GiB -- so the port runs past it and warns
    with the size instead of refusing.
    """
    import warnings

    from foldjax.models.protenix.runtime_policy import (
        PROTENIX_V2_MAX_TOKENS,
        validate_inference_limits,
    )

    validate_inference_limits(
        model_name="protenix-v2", n_token=PROTENIX_V2_MAX_TOKENS
    )

    with pytest.raises(ValueError, match="strict_token_limit"):
        validate_inference_limits(
            model_name="protenix-v2",
            n_token=PROTENIX_V2_MAX_TOKENS + 1,
            strict_token_limit=True,
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_inference_limits(
            model_name="protenix-v2", n_token=PROTENIX_V2_MAX_TOKENS + 1
        )
    assert len(caught) == 1, "running past it must still say so"
    assert issubclass(caught[0].category, RuntimeWarning)
    message = str(caught[0].message)
    assert "memory budget" in message
    assert "validates the model at this length" in message, (
        "fitting is not the same as being right, and the warning has to say so"
    )


def test_lifting_the_v2_limit_does_not_touch_the_other_models() -> None:
    """The guard names one model, and only that model gets the escape."""
    import warnings

    from foldjax.models.protenix.runtime_policy import validate_inference_limits

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_inference_limits(
            model_name="protenix_base_default_v1.0.0",
            n_token=100_000,
        )
    assert caught == [], "the release has no such limit to lift"

    with pytest.raises(ValueError, match="must be positive"):
        validate_inference_limits(model_name="protenix-v2", n_token=0)
