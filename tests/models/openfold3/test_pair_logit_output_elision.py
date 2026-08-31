"""Managed pair-logit retention and the compiler boundary it controls."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3 import inference
from foldjax.models.openfold3.inference import released_config
from foldjax.models.openfold3.models.heads import (
    PairHeadParams,
    distogram_head,
    predicted_aligned_error_head,
    predicted_distance_error_head,
)
from foldjax.models.openfold3.models.primitives import LayerNormParams, LinearParams

SAMPLES = 5
TOKENS = 32
CHANNELS = 16
BINS = 64


def _head(offset: int, *, normalize: bool = True) -> PairHeadParams:
    weight = (
        jnp.arange(BINS * CHANNELS, dtype=jnp.float32).reshape(BINS, CHANNELS) + offset
    ) / 10_000.0
    norm = (
        LayerNormParams(
            weight=jnp.ones(CHANNELS, dtype=jnp.float32),
            bias=jnp.zeros(CHANNELS, dtype=jnp.float32),
        )
        if normalize
        else None
    )
    return PairHeadParams(
        linear=LinearParams(weight=weight, bias=None), layer_norm=norm
    )


PAE_HEAD = _head(1)
PDE_HEAD = _head(3)
DISTOGRAM_HEAD = _head(5, normalize=False)


def _pair_output_probe(z_conf, z_trunk, *, returned: tuple[str, ...]):
    """Small real-head graph with the same static output decision as prediction."""

    pae = predicted_aligned_error_head(z_conf, PAE_HEAD)
    bins = jnp.arange(BINS, dtype=pae.dtype)
    metric = jnp.sum(jax.nn.softmax(pae, axis=-1) * bins)
    pde = predicted_distance_error_head(z_conf, PDE_HEAD)
    distogram = distogram_head(z_trunk, DISTOGRAM_HEAD)
    kept = set(returned)
    return (
        metric,
        pae if "pae_logits" in kept else None,
        pde if "pde_logits" in kept else None,
        distogram if "distogram_logits" in kept else None,
    )


def _lower(returned: tuple[str, ...]):
    function = jax.jit(functools.partial(_pair_output_probe, returned=returned))
    return function.lower(
        jax.ShapeDtypeStruct((SAMPLES, TOKENS, TOKENS, CHANNELS), jnp.float32),
        jax.ShapeDtypeStruct((1, TOKENS, TOKENS, CHANNELS), jnp.float32),
    )


def test_unreturned_pair_heads_are_removed_from_compiled_hlo() -> None:
    """A Python-side conditional alone is unnecessary: XLA already performs DCE."""

    dropped = _lower(())
    retained = _lower(("pae_logits", "pde_logits", "distogram_logits"))

    # PAE remains because the scalar metric consumes it. PDE and distogram have
    # no consumers in the managed graph and their real linear projections vanish.
    assert dropped.as_text().count("stablehlo.dot_general") == 1
    assert retained.as_text().count("stablehlo.dot_general") == 3


def test_pair_logit_plan_removes_the_exact_compiled_output_payload() -> None:
    dropped = _lower(()).compile().memory_analysis()
    retained = (
        _lower(("pae_logits", "pde_logits", "distogram_logits"))
        .compile()
        .memory_analysis()
    )

    pair_values = 2 * SAMPLES * TOKENS * TOKENS * BINS + TOKENS * TOKENS * BINS
    pair_bytes = pair_values * np.dtype(np.float32).itemsize
    removed = retained.output_size_in_bytes - dropped.output_size_in_bytes
    # CPU's tuple table adds a few pointer-sized bytes; the tensor payload itself
    # is exact and must account for all but that small fixed container overhead.
    assert pair_bytes <= removed < pair_bytes + 128


def test_n32_output_retention_does_not_change_the_shared_metric() -> None:
    """DCE alone preserves the shared metric independently of scheduling."""

    rng = np.random.default_rng(20260829)
    z_conf = rng.normal(size=(SAMPLES, TOKENS, TOKENS, CHANNELS)).astype(np.float32)
    z_trunk = rng.normal(size=(1, TOKENS, TOKENS, CHANNELS)).astype(np.float32)
    dropped = jax.jit(functools.partial(_pair_output_probe, returned=()))(
        z_conf, z_trunk
    )
    retained = jax.jit(
        functools.partial(
            _pair_output_probe,
            returned=("pae_logits", "pde_logits", "distogram_logits"),
        )
    )(z_conf, z_trunk)

    np.testing.assert_array_equal(np.asarray(dropped[0]), np.asarray(retained[0]))
    assert dropped[1:] == (None, None, None)


def test_direct_released_config_keeps_its_native_pair_outputs() -> None:
    """The managed backend override must not narrow the direct/raw API default."""

    config = released_config(n_token=132, n_atom=1)
    assert config.returned_pair_logits == (
        "pae_logits",
        "pde_logits",
        "distogram_logits",
    )


@pytest.mark.parametrize(
    ("n_token", "default_pair_logits", "default_sink", "managed_sink"),
    [
        (750, ("pae_logits", "pde_logits", "distogram_logits"), False, True),
        (751, ("pae_logits", "pde_logits", "distogram_logits"), False, True),
        (1225, ("pae_logits", "pde_logits", "distogram_logits"), False, True),
        (1226, ("pde_logits", "distogram_logits"), True, True),
    ],
)
def test_managed_budget_sink_boundaries(
    n_token: int,
    default_pair_logits: tuple[str, ...],
    default_sink: bool,
    managed_sink: bool,
) -> None:
    """The managed default sinks PAE metrics for every multi-sample size."""

    direct = released_config(n_token=n_token, n_atom=1)
    managed = released_config(n_token=n_token, n_atom=1, max_array_bytes=0)

    assert direct.returned_pair_logits == default_pair_logits
    assert managed.returned_pair_logits == ()
    assert inference._sink_pae_metrics(direct) is default_sink
    assert inference._sink_pae_metrics(managed) is managed_sink
