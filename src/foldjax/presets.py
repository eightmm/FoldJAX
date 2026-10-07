"""Named sampling presets: a schedule a model's publisher released, by name.

``preset="fast"`` (``--preset fast``) sets the diffusion steps and trunk
recycles to the reduced values a model's own publisher documents for the
checkpoint being run -- never to numbers FoldJAX chose. Where no publisher
documents one, the preset is refused with the reason, and with the
publisher's own faster *checkpoint* when that is what it offers instead.

As of this table (upstream sources read 2026-10-06), exactly one reduced
schedule is published for a checkpoint FoldJAX carries: Protenix's Mini and
Tiny models, ``N_cycle`` 4 and ``sample_diffusion.N_step`` 5
(Protenix ``docs/supported_models.md``, "Mini & Tiny Models"), which FoldJAX
runs as the ``mini-default-v0.5.0``, ``mini-esm-v0.5.0``, ``mini-ism-v0.5.0``
and ``tiny-default-v0.5.0`` profiles. That is
also those checkpoints' released default, so ``fast`` there records the
choice rather than changing the run. Every other model publishes only its
full schedule, or its fast option is a different model.

The resolved preset is recorded in the run manifest (``preset``) beside the
sampling it set.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from foldjax.portspec import (
    PROTENIX_MINI_DEFAULT_PROFILE,
    PROTENIX_MINI_ESM_PROFILE,
    PROTENIX_MINI_ISM_PROFILE,
    PROTENIX_TINY_DEFAULT_PROFILE,
)

_PROTENIX_MINI_SOURCE = (
    "Protenix docs/supported_models.md, 'Mini & Tiny Models': N_cycle 4, "
    "sample_diffusion.N_step 5"
)

#: ``(preset, model, profile) -> (sampling, source)``.
PUBLISHED: Mapping[tuple[str, str, str], tuple[Mapping[str, int], str]] = {
    ("fast", "protenix", profile): (
        {"num_steps": 5, "num_recycles": 4},
        _PROTENIX_MINI_SOURCE,
    )
    for profile in (
        PROTENIX_MINI_DEFAULT_PROFILE,
        PROTENIX_MINI_ESM_PROFILE,
        PROTENIX_MINI_ISM_PROFILE,
        PROTENIX_TINY_DEFAULT_PROFILE,
    )
}

#: Why each model has no ``fast`` schedule for what it runs.
UNPUBLISHED: Mapping[str, str] = {
    "alphafold3": (
        "AlphaFold 3 publishes no reduced diffusion or recycling schedule "
        "(its docs/performance.md covers hardware and the genetic search only)"
    ),
    "boltz2": (
        "Boltz-2 publishes only its default schedule (--recycling_steps 3, "
        "--sampling_steps 200; docs/prediction.md) and no reduced one"
    ),
    "esmfold2": (
        "Biohub's faster option is ESMFold2-Fast, a separate single-sequence "
        "checkpoint (huggingface.co/biohub/ESMFold2), not a reduced schedule "
        "for this one, and FoldJAX does not carry it"
    ),
    "opendde": (
        "OpenDDE publishes one schedule, model.N_cycle 10 and "
        "sample_diffusion.N_step 200 (docs/supported_models.md, 'Recommended "
        "inference defaults')"
    ),
    "openfold3": (
        "OpenFold3 v0.5.0's model presets are train, predict, low_mem and mps "
        "(config/model_setting_presets.yml); none reduces steps or recycles"
    ),
    "protenix": (
        "Protenix publishes its reduced schedule (N_cycle 4, N_step 5) only "
        "for its Mini/Tiny checkpoints (docs/supported_models.md); the base "
        "and v2 checkpoints publish 10 cycles and 200 steps. Run a published "
        f"fast model with --profile {PROTENIX_MINI_DEFAULT_PROFILE}, "
        f"{PROTENIX_MINI_ESM_PROFILE}, {PROTENIX_MINI_ISM_PROFILE} or "
        f"{PROTENIX_TINY_DEFAULT_PROFILE}"
    ),
}


def resolve_preset(
    name: str, model: str, profile: str | None
) -> tuple[dict[str, int], str]:
    """The sampling ``name`` stands for on this model and profile, and its source."""
    published = PUBLISHED.get((name, model, profile or ""))
    if published is not None:
        sampling, source = published
        return dict(sampling), source
    reason = UNPUBLISHED.get(model, f"{model} publishes no {name!r} schedule")
    if model == "protenix" and profile is None:
        reason += " (and an explicit --weights names no managed profile)"
    raise ValueError(f"preset {name!r} is not available for {model}: {reason}")


def preset_updates(request: Any, *, model: str, profile: str | None) -> dict[str, int]:
    """The request fields a preset sets; refuses a knob set to something else."""
    if request.preset is None:
        return {}
    sampling, _source = resolve_preset(request.preset, model, profile)
    updates = {}
    for knob, value in sampling.items():
        given = getattr(request, knob)
        if given is not None and given != value:
            flag = "--" + knob.replace("_", "-")
            raise ValueError(
                f"preset {request.preset!r} sets {knob}={value} for {model}; "
                f"{flag} {given} asks for another value. Pass one of them"
            )
        updates[knob] = value
    return updates


def preset_record(request: Any) -> dict[str, Any] | None:
    """The manifest's ``preset`` block, or None when no preset was asked for."""
    if getattr(request, "preset", None) is None:
        return None
    sampling, source = resolve_preset(
        request.preset, str(request.model), request.profile
    )
    return {"name": request.preset, "sampling": sampling, "source": source}
