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
import pytest

from foldjax.models.protenix.bridge.torch_mapping import (
    map_pairformer_output_state_dict,
)
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
)
from foldjax.models.protenix.models.trunk_blocks.msa import (
    sample_msa_cycle_features,
    sample_msa_cycle_index_tape,
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

    The existing trunk test calls this same function with ``num_recycles=2`` and did not
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
        num_recycles=3,
    )

    assert seen, "msa_module was never called"
    for received in seen:
        assert received is not None, "the scan body forwarded lax.scan's None"
        assert "relp" in received


@pytest.mark.parametrize("use_cycle_scan", [False, True])
def test_cycle_msa_index_tape_matches_materialized_trunk_cycles(
    use_cycle_scan: bool,
) -> None:
    """Gathering inside recycling preserves the complete MSA/trunk result."""
    from foldjax.models.protenix.models.trunk_blocks.trunk import (
        pairformer_output_from_s_inputs,
    )

    from .test_trunk import _pairformer_output_state

    rng = np.random.default_rng(51)
    n_token = 4
    features = {
        "relp": np.zeros((n_token, n_token, 139), dtype=np.float32),
        "token_bonds": np.zeros((n_token, n_token), dtype=np.float32),
        "msa": rng.integers(0, 32, size=(7, n_token), dtype=np.int32),
        "has_deletion": rng.integers(0, 2, size=(7, n_token)).astype(np.float32),
        "deletion_value": rng.normal(size=(7, n_token)).astype(np.float32),
    }
    num_recycles = 3
    materialized = sample_msa_cycle_features(
        features, num_recycles=num_recycles, seed=29, bucket_size=4
    )
    tape = sample_msa_cycle_index_tape(
        features, num_recycles=num_recycles, seed=29, bucket_size=4
    )
    assert tape is not None
    params = map_pairformer_output_state_dict(_pairformer_output_state())
    s_inputs = rng.normal(size=(n_token, 2)).astype(np.float32)

    def run(**cycle_kwargs):
        return pairformer_output_from_s_inputs(
            features,
            s_inputs,
            params,
            num_recycles=num_recycles,
            use_cycle_scan=use_cycle_scan,
            single_attention_backend="xla",
            triangle_attention_backend="xla",
            **cycle_kwargs,
        )

    expected = run(cycle_msa_features=materialized)
    actual = run(cycle_msa_index_tape=tape)

    for name, expected_value, actual_value in zip(
        ("single_inputs", "single", "pair"), expected, actual, strict=True
    ):
        expected_host = np.asarray(expected_value)
        actual_host = np.asarray(actual_value)
        assert actual_host.dtype == expected_host.dtype
        np.testing.assert_array_equal(
            actual_host.reshape(-1).view(np.uint8),
            expected_host.reshape(-1).view(np.uint8),
            err_msg=f"{name} changed on the compact cycle-MSA path",
        )


def test_cycle_msa_index_tape_rejects_ambiguous_or_malformed_inputs() -> None:
    from foldjax.models.protenix.models.trunk_blocks.trunk import (
        pairformer_output_from_s_inputs,
    )

    from .test_trunk import _pairformer_output_params

    features = {
        "relp": jnp.zeros((2, 2, 2), dtype=jnp.float32),
        "token_bonds": jnp.zeros((2, 2), dtype=jnp.float32),
    }
    s_inputs = jnp.zeros((2, 2), dtype=jnp.float32)
    cycle = {"msa": jnp.zeros((1, 2), dtype=jnp.int32)}
    from foldjax.models.protenix.models.trunk_blocks.msa import MSACycleIndexTape

    tape = MSACycleIndexTape(
        row_indices=jnp.zeros((1, 1), dtype=jnp.int32),
        row_mask=jnp.ones((1, 1), dtype=bool),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        pairformer_output_from_s_inputs(
            features,
            s_inputs,
            _pairformer_output_params(),
            num_recycles=1,
            cycle_msa_features=(cycle,),
            cycle_msa_index_tape=tape,
        )

    bad_tape = tape._replace(row_mask=jnp.ones((1, 2), dtype=bool))
    with pytest.raises(ValueError, match="row mask"):
        pairformer_output_from_s_inputs(
            features,
            s_inputs,
            _pairformer_output_params(),
            num_recycles=1,
            cycle_msa_index_tape=bad_tape,
        )
