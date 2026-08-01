"""Recycling as a scan must equal recycling unrolled.

``pairformer_output_from_s_inputs`` emits its recycling body once and iterates. At
Protenix's released depth that is ten cycles over a 48-block Pairformer, so the
unrolled graph is ten times larger and compiles accordingly -- but a scan is only
worth having if it computes the same thing, and the failure mode is a plausible
structure rather than an error.

The projection is exercised directly rather than through the whole trunk so the test
needs no checkpoint and no torch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
)
from foldjax.models.protenix.models.trunk_blocks.trunk import (
    RecyclingProjectionParams,
    _stacked_cycle_msa,
    recycle_embeddings,
)

CYCLES = 4


def _layer_norm(channels: int) -> LayerNormParams:
    return LayerNormParams(
        weight=jnp.ones(channels, dtype=jnp.float32),
        bias=jnp.zeros(channels, dtype=jnp.float32),
    )


def _linear(key: jax.Array, fan_in: int, fan_out: int) -> LinearParams:
    # Scaled down so four cycles of a residual projection stay in a range where
    # float32 comparisons mean something.
    return LinearParams(
        weight=jax.random.normal(key, (fan_out, fan_in), dtype=jnp.float32) * 0.05,
        bias=None,
    )


def _setup():
    n_token, c_s, c_z = 5, 8, 6
    keys = jax.random.split(jax.random.key(0), 4)
    s_init = jax.random.normal(keys[0], (n_token, c_s), dtype=jnp.float32)
    z_init = jax.random.normal(keys[1], (n_token, n_token, c_z), dtype=jnp.float32)
    params = RecyclingProjectionParams(
        layernorm_z=_layer_norm(c_z),
        linear_z=_linear(keys[2], c_z, c_z),
        layernorm_s=_layer_norm(c_s),
        linear_s=_linear(keys[3], c_s, c_s),
    )
    return s_init, z_init, params


def test_recycling_scan_matches_the_unrolled_loop() -> None:
    s_init, z_init, params = _setup()

    def body(carry, _):
        s, z = carry
        return recycle_embeddings(s_init, z_init, s, z, params), None

    start = (jnp.zeros_like(s_init), jnp.zeros_like(z_init))
    scanned, _ = jax.lax.scan(body, start, xs=None, length=CYCLES)
    unrolled = start
    for _ in range(CYCLES):
        unrolled, _ = body(unrolled, None)

    for name, left, right in zip(("s", "z"), unrolled, scanned, strict=True):
        np.testing.assert_allclose(
            np.asarray(right, dtype=np.float64),
            np.asarray(left, dtype=np.float64),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"{name} differs between scanned and unrolled recycling",
        )


def test_recycling_is_not_a_no_op() -> None:
    """Guard the comparison above from passing because nothing happens.

    If the projection returned its input, scanned and unrolled would agree for a
    reason that proves nothing about the scan.
    """
    s_init, z_init, params = _setup()
    start = (jnp.zeros_like(s_init), jnp.zeros_like(z_init))
    once = recycle_embeddings(s_init, z_init, *start, params)
    twice = recycle_embeddings(s_init, z_init, *once, params)
    assert not np.allclose(
        np.asarray(once[1], dtype=np.float64),
        np.asarray(twice[1], dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    ), "a second cycle changes nothing; recycling is inert and the scan test is void"


def test_stacked_cycle_msa_accepts_padded_cycles() -> None:
    """``sample_msa_cycle_features`` pads to a common bucket so this can stack."""
    cycles = tuple({"msa": np.zeros((2, 4), np.float32)} for _ in range(3))
    stacked = _stacked_cycle_msa(cycles)
    assert stacked is not None
    assert stacked["msa"].shape == (3, 2, 4)


def test_stacked_cycle_msa_refuses_ragged_cycles() -> None:
    """Ragged cycles must fall back to the unrolled loop, not raise from jnp.stack.

    A caller may build the per-cycle subsets itself. ``jnp.stack`` on mismatched
    shapes raises from inside ``tree_map``, naming neither the cycle nor the field.
    """
    assert _stacked_cycle_msa(None) is None
    assert _stacked_cycle_msa(()) is None
    assert (
        _stacked_cycle_msa(
            (
                {"msa": np.zeros((2, 4), np.float32)},
                {"msa": np.zeros((3, 4), np.float32)},
            )
        )
        is None
    )
    assert (
        _stacked_cycle_msa(
            (
                {"msa": np.zeros((2, 4), np.float32)},
                {"msa": np.zeros((2, 4), np.float32), "extra": np.zeros(1, np.float32)},
            )
        )
        is None
    )


def test_the_scan_body_gets_the_feature_dict_not_none(monkeypatch) -> None:
    """With no per-cycle alignment, every cycle must see the feature dict.

    ``lax.scan`` with ``xs=None`` calls the body with ``None``. Forwarding that to
    ``msa_module`` reaches it as a missing feature dict and raises
    ``TypeError: argument of type 'NoneType' is not iterable`` -- which is what
    happened on a real 2030-token target after the recycling loop became a scan.

    The existing trunk test calls this same function with ``n_cycle=2`` and did not
    catch it: its fixture has no MSA blocks, and ``msa_module`` returns early on
    ``not params.blocks`` before the dict is ever indexed. So the check here is on
    what ``msa_module`` is *handed*, which holds whether or not it would use it.
    """
    from foldjax.models.protenix.models.trunk_blocks import trunk as trunk_module

    from .test_trunk import _pairformer_output_params

    seen = []

    def spy(msa_features, z, *args, **kwargs):
        seen.append(msa_features)
        return z

    monkeypatch.setattr(trunk_module, "msa_module", spy)
    monkeypatch.setattr(
        trunk_module, "template_embedder", lambda *a, **k: jnp.zeros_like(a[1])
    )

    features = {
        "relp": jnp.zeros((2, 2, 2), dtype=jnp.float32),
        "token_bonds": jnp.zeros((2, 2), dtype=jnp.float32),
    }
    trunk_module.pairformer_output_from_s_inputs(
        features,
        jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        _pairformer_output_params(),
        n_cycle=3,
    )

    assert seen, "msa_module was never called"
    for received in seen:
        assert received is not None, "the scan body forwarded lax.scan's None"
        assert "relp" in received
