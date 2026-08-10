from __future__ import annotations

import inspect

from foldjax.models.boltz2.api import COMPUTE_DTYPES, predict
from foldjax.models.boltz2.models.diffusion.atom import diffusion_transformer_forward
from foldjax.models.boltz2.models.trunk_blocks.pairformer import (
    pairformer_module_forward,
)
from foldjax.models.boltz2.models.trunk_blocks.trunk import boltz2_sample_forward


def test_production_layer_stacks_default_to_memory_stable_scan() -> None:
    assert inspect.signature(pairformer_module_forward).parameters["use_scan"].default
    diffusion_scan = inspect.signature(diffusion_transformer_forward).parameters[
        "use_scan"
    ]
    assert diffusion_scan.default
    assert inspect.signature(boltz2_sample_forward).parameters["use_scan"].default


def test_prediction_defaults_to_the_precision_upstream_ships() -> None:
    """Boltz-2's own predict path is `Trainer(..., precision="bf16-mixed")`.

    A port that defaults to float32 is not being conservative, it is running a
    configuration its reference implementation does not ship -- which is the
    mistake this repository has made before. The trunk takes this dtype; the
    diffusion and confidence modules stay float32 whatever it says, which is
    the same split upstream's autocast draws.
    """
    assert inspect.signature(predict).parameters["compute_dtype"].default == "bfloat16"


def test_the_float32_trunk_is_still_reachable() -> None:
    """The old behaviour has to remain one argument away.

    bf16 costs ~0.002 of reported pLDDT and moves atoms by about three times
    the noise floor of running the same program twice. That is small, but a
    parity harness comparing against a float32 tape needs the exact previous
    program, not something close to it.
    """
    assert {"float32", "bfloat16"} <= set(COMPUTE_DTYPES)


def test_the_matmul_precision_pin_is_a_scope_not_a_latch() -> None:
    """Boltz-2's `"highest"` must not follow the caller out of `predict`.

    JAX's matmul precision is process-global. This was a `jax.config.update`
    inside `predict`, so a process that ran Boltz-2 and then anything else --
    another port, a notebook cell, a test collected later -- left that other
    thing in float32. It went unnoticed while every port pinned the same value;
    Protenix and OpenFold3 now pin TF32, which is what their upstreams run, so
    the difference is observable.
    """
    import jax

    from foldjax.models.boltz2.api import MATMUL_PRECISION, _pinned_matmul_precision

    # Boltz-2 is the one port here whose upstream asks for true float32.
    assert MATMUL_PRECISION == "highest"

    seen = []

    @_pinned_matmul_precision
    def record():
        seen.append(jax.config.jax_default_matmul_precision)

    before = jax.config.jax_default_matmul_precision
    record()

    assert seen == [MATMUL_PRECISION]
    assert jax.config.jax_default_matmul_precision == before
