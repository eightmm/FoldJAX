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


def test_the_compiled_path_hands_the_graph_unplaced_features() -> None:
    """Placing the feature dict costs device memory the program never asks for.

    The featurizer emits 78 arrays and the inference graph reads 31. Building
    the dict with `jnp.asarray` put all 78 on the device and kept them there
    for the whole run; `disto_target` -- a training label, f32[N, N, 1, 64] --
    was 246 MiB of that at 1,003 tokens on its own, and the term is quadratic
    in token count. Handing `jax.jit` the NumPy arrays instead lets it drop the
    unread ones before anything is transferred.
    """
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models.boltz2.api import _graph_features

    feats = {"msa": np.zeros((1, 4, 3), np.int64)}

    compiled = _graph_features(feats, place=False)
    assert type(compiled["msa"]) is np.ndarray

    placed = _graph_features(feats, place=True)
    assert isinstance(placed["msa"], jnp.ndarray)

    # Both spellings reach the graph as the same argument, so the traced
    # program -- and therefore every coordinate it produces -- is unchanged.
    assert jnp.asarray(compiled["msa"]).dtype == placed["msa"].dtype
    assert compiled["msa"].shape == placed["msa"].shape


def test_jit_drops_arguments_the_graph_never_reads() -> None:
    """The saving above is JAX's argument pruning, so pin that it still prunes.

    `jax.jit` DCEs unread arguments out of the lowered program and filters them
    out before the rest are placed. If an upgrade ever stopped doing that, the
    unread features would silently be transferred again and only a memory
    benchmark would notice.
    """
    import jax
    import numpy as np

    def graph(read, never_read):
        return read * 2.0

    lowered = jax.jit(graph).lower(
        np.zeros((3,), np.float32), np.zeros((1024,), np.float32)
    )
    signature = lowered.as_text().split("func.func public @main")[1].split("\n")[0]
    assert "1024" not in signature
    assert "3xf32" in signature
