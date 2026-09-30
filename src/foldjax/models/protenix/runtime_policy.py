"""Runtime policy checks shared by Protenix JAX CLIs."""

from __future__ import annotations

import warnings
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

#: The models upstream lets read an RNA alignment: `--use_rna_msa` with any
#: other name fails an assertion (Protenix 2.0.0
#: `runner/batch_inference.py:893-898`).
RNA_MSA_MODEL_NAMES = frozenset(
    {
        "protenix_base_default_v1.0.0",
        "protenix_base_20250630_v1.0.0",
        "protenix-v2",
    }
)

_BASE_INFERENCE_DEFAULTS = {
    "num_recycles": 10,
    "num_steps": 200,
    "gamma0": 0.8,
    "step_scale_eta": 1.5,
}
_SMALL_INFERENCE_DEFAULTS = {
    "num_recycles": 4,
    "num_steps": 5,
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

#: Which Pallas pair kernels an omitted setting realises on a GPU process:
#: ``"trimul"`` for the triangle multiplication
#: (``PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND``), ``"glu"`` for the pair
#: transitions (``glu_backend``). Boltz-2 and OpenFold3 default to both on a
#: GPU; Protenix defaults to the multiplication only.
#:
#: At 4,100 tokens (1GTE, ``foldjax-bench/x53-pallas-4k-20260925``) sample 2's
#: deposited CA RMSD was 1.222/1.224 A released (jobs 2500/2505), 1.222 with
#: the multiplication alone (2513), 1.222 with the transitions alone (2514),
#: and 1.636/1.642 with both (2501/2506), against a 0.002 A rerun floor; every
#: other sample stayed within 0.02 A in every arm. Only the combination moves
#: the structure, and the multiplication alone takes most of the time: warm
#: 823 -> 762 s (-7.4%) at 57,916 -> 57,940 MiB, where the transitions alone
#: gave 783 s and both 723 s. So ``"glu"`` stays out and ``glu_backend=pallas``
#: stays opt-in.
#:
#: Flipping a component is adding its name here; the explicit spellings work
#: either way. Read at call time, and JAX-free so the adapter can read it
#: while it plans a cache namespace.
PALLAS_DEFAULT: frozenset[str] = frozenset({"trimul"})


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


def estimate_protenix_v2_peak_mib(n_token: int) -> float:
    """Roughly what protenix-v2 needs on one device at this token count.

    Fitted on this port's own measurements at 2,096 and 3,012 tokens (39,888
    and 80,034 MiB) as ``a + k * n^2``: the trunk's dominant tensors are pair
    tensors of shape [n, n, c_z], so above about a thousand tokens the square
    term is the whole story. It reproduces 1,003 tokens to within 5%. Below
    that the weights and fixed costs dominate and this reads low, which is the
    harmless direction for a warning.
    """
    return 2_188.0 + 0.008581 * float(n_token) ** 2


def validate_inference_limits(
    *,
    model_name: str | None,
    n_token: int,
    strict_token_limit: bool = False,
) -> None:
    """Apply model-specific inference limits from the Protenix runtime.

    Upstream refuses protenix-v2 above 2,560 tokens, and its own reason is
    memory: the assertion reads "It might cause OOM." Nothing in the
    architecture is bounded by token count -- the relative position encoding
    buckets residue separation with ``clip(delta, 0, 2 * r_max)`` at r_max=32,
    so every pair beyond 32 residues apart already shares a bucket -- which
    makes the guard a budget rather than a capability.

    A budget calibrated on one implementation's memory does not transfer to
    another that uses less of it. Measured here, this port runs 3,012 tokens of
    protenix-v2 to five completed samples at a peak of 78.2 GiB, inside a 96 GB
    card, where upstream refuses at 2,561. So the refusal becomes a warning
    carrying the size, which is what this repository already does for a job
    that may not fit: OpenDDE says what it estimates and lets the allocator
    answer.

    What the warning does not say is that a longer run is a good one. Neither
    implementation validates protenix-v2 at that length. ``strict_token_limit``
    restores upstream's refusal exactly, for a caller who wants upstream's
    behaviour rather than this one's.
    """

    if n_token <= 0:
        raise ValueError("n_token must be positive")
    if model_name != "protenix-v2" or n_token <= PROTENIX_V2_MAX_TOKENS:
        return
    if strict_token_limit:
        raise ValueError(
            "protenix-v2 model does not support n_token > 2560. "
            "It might cause OOM. This is upstream's limit, restored by "
            "strict_token_limit; FoldJAX runs past it by default."
        )
    estimate = estimate_protenix_v2_peak_mib(n_token)
    warnings.warn(
        f"protenix-v2 at {n_token} tokens is past upstream's 2,560-token "
        f"limit, which is a memory budget rather than an architectural bound. "
        f"Expect roughly {estimate / 1024:.0f} GiB of device memory; measured "
        f"here, 3,012 tokens peaked at 78.2 GiB. Neither implementation "
        f"validates the model at this length.",
        RuntimeWarning,
        stacklevel=2,
    )
