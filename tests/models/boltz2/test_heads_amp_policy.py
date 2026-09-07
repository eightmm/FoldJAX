"""CPU checks for the released Boltz confidence and distogram AMP boundaries."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.heads import bfactor, confidence, distogram


@pytest.mark.parametrize("compiled", [False, True])
def test_bf16_distogram_symmetrizes_before_single_linear_rounding(compiled):
    rng = np.random.default_rng(21)
    z = jnp.asarray(rng.normal(size=(1, 3, 3, 4)), dtype=jnp.float32)
    kernel = jnp.asarray(rng.normal(size=(4, 6)), dtype=jnp.bfloat16)
    bias = jnp.asarray(rng.normal(size=6), dtype=jnp.float32)
    params = {"distogram": {"kernel": kernel, "bias": bias}}
    sym = (z + jnp.swapaxes(z, 1, 2)).astype(jnp.bfloat16)
    expected = (
        (
            jnp.matmul(sym, kernel, preferred_element_type=jnp.float32)
            + bias.astype(jnp.bfloat16).astype(jnp.float32)
        )
        .astype(jnp.bfloat16)
        .reshape(1, 3, 3, 2, 3)
    )
    def fn(p, x):
        return distogram.distogram_forward(p, x, num_distograms=2, num_bins=3)

    actual = (jax.jit(fn) if compiled else fn)(params, z)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, expected)


def test_fp32_distogram_retains_historical_linear_then_symmetry_order():
    rng = np.random.default_rng(2)
    z = jnp.asarray(rng.normal(size=(1, 3, 3, 4)), dtype=jnp.float32)
    kernel = jnp.asarray(rng.normal(size=(4, 3)), dtype=jnp.float32)
    bias = jnp.asarray(rng.normal(size=3), dtype=jnp.float32)
    projected = z @ kernel
    expected = (projected + jnp.swapaxes(projected, 1, 2) + bias)[..., None, :]
    actual = distogram.distogram_forward(
        {"distogram": {"kernel": kernel, "bias": bias}}, z, num_bins=3
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_bfactor_uses_selected_linear_precision(dtype, compiled):
    rng = np.random.default_rng(51)
    s = jnp.asarray(rng.normal(size=(1, 3, 4)), dtype=jnp.float32)
    kernel = jnp.asarray(rng.normal(size=(4, 7)), dtype=dtype)
    bias = jnp.asarray(rng.normal(size=7), dtype=jnp.float32)
    params = {"bfactor": {"kernel": kernel, "bias": bias}}
    if dtype == jnp.bfloat16:
        expected = (
            jnp.matmul(s.astype(dtype), kernel, preferred_element_type=jnp.float32)
            + bias.astype(dtype).astype(jnp.float32)
        ).astype(dtype)
    else:
        expected = s @ kernel + bias
    fn = jax.jit(bfactor.bfactor_forward) if compiled else bfactor.bfactor_forward
    actual = fn(params, s)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected)


def test_confidence_linear_casts_only_at_projection():
    x = jnp.asarray([[1.0137, -0.8311, 0.9133]], dtype=jnp.float32)
    kernel = jnp.asarray([[0.3], [0.7], [-0.2]], dtype=jnp.bfloat16)
    actual = confidence._linear(x, kernel)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, x.astype(jnp.bfloat16) @ kernel)


def test_bf16_confidence_logits_use_native_fp32_softmax():
    logits = jnp.asarray([[0.191, -0.937, 1.271, 0.603]], dtype=jnp.bfloat16)
    expected = jnp.sum(
        jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        * jnp.asarray([0.125, 0.375, 0.625, 0.875]),
        axis=-1,
    )
    actual = confidence.compute_aggregated_metric(logits)
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("multiplicity,packed", [(1, False), (5, False), (5, True)])
def test_confidence_rounds_only_representative_coords_before_cdist(
    monkeypatch, dtype, multiplicity, packed
):
    norm = {"scale": jnp.ones(2), "bias": jnp.zeros(2)}
    params = {
        "s_inputs_norm": norm,
        "s_norm": norm,
        "z_norm": norm,
        "token_bonds": {"kernel": jnp.zeros((1, 2), dtype=dtype)},
        "token_bonds_type": jnp.zeros((1, 2)),
        "rel_pos": {},
        "contact_conditioning": {},
        "pairformer_stack": {},
        "confidence_heads": {},
        "boundaries": jnp.asarray([2.0]),
        "dist_bin_pairwise_embed": jnp.asarray([[0.1371, 0.3311], [1.0137, 1.2299]]),
    }
    params.update(
        {
            name: {"kernel": jnp.zeros((2, 2), dtype=dtype)}
            for name in (
                "s_input_to_s",
                "s_to_z",
                "s_to_z_transpose",
                "s_to_z_prod_in1",
                "s_to_z_prod_in2",
                "s_to_z_prod_out",
            )
        }
    )
    feats = {
        "token_bonds": jnp.zeros((1, 2, 2, 1)),
        "type_bonds": jnp.zeros((1, 2, 2), dtype=jnp.int32),
        "token_to_rep_atom": jnp.asarray([[[1, 0, 0], [0, 0, 1]]], dtype=jnp.float32),
        "token_pad_mask": jnp.ones((1, 2)),
    }
    raw = jnp.repeat(
        jnp.asarray([[[0, 0, 0], [0.9173, 0.5311, 0.2331], [2.001, 0, 0]]]),
        multiplicity,
        axis=0,
    )
    x_pred = raw[None] if packed else raw
    seen = []
    original_cdist = confidence._cdist

    def cdist(a, b):
        seen.append((a, b))
        return original_cdist(a, b)

    monkeypatch.setattr(confidence, "_cdist", cdist)
    monkeypatch.setattr(
        confidence, "relative_position_forward", lambda *a, **k: jnp.zeros((1, 2, 2, 2))
    )
    monkeypatch.setattr(
        confidence,
        "contact_conditioning_forward",
        lambda *a, **k: jnp.zeros((1, 2, 2, 2)),
    )
    monkeypatch.setattr(
        confidence, "pairformer_module_forward", lambda p, s, z, **k: (s, z)
    )
    monkeypatch.setattr(confidence, "_confidence_heads_forward", lambda p, **k: k)
    result = confidence.confidence_module_forward(
        params,
        jnp.zeros((1, 2, 2)),
        jnp.zeros((1, 2, 2)),
        jnp.zeros((1, 2, 2, 2)),
        x_pred,
        feats,
        jnp.zeros((1, 2, 2, 64)),
        multiplicity=multiplicity,
    )
    expected_coords = raw[:, [0, 2]].astype(dtype).astype(jnp.float32)
    assert len(seen) == 1
    assert seen[0][0].dtype == jnp.float32
    np.testing.assert_array_equal(seen[0][0], expected_coords)
    np.testing.assert_array_equal(result["x_pred"], raw)
    assert result["s"].dtype == jnp.float32
    assert result["z"].dtype == jnp.float32
    assert result["d"].dtype == jnp.float32
    expected_bin = int(dtype == jnp.float32)
    np.testing.assert_array_equal(
        result["z"][:, 0, 1],
        jnp.broadcast_to(
            params["dist_bin_pairwise_embed"][expected_bin], (multiplicity, 2)
        ),
    )


def test_native_head_output_dtypes_and_contact_probabilities(monkeypatch):
    rng = np.random.default_rng(81)
    params = {
        name: {"kernel": jnp.asarray(rng.normal(size=(2, bins)), dtype=jnp.bfloat16)}
        for name, bins in {
            "to_pae_intra_logits": 64,
            "to_pae_inter_logits": 64,
            "to_pde_intra_logits": 64,
            "to_pde_inter_logits": 64,
            "to_resolved_logits": 2,
            "to_plddt_logits": 50,
        }.items()
    }
    s = jnp.asarray(rng.normal(size=(1, 3, 2)), dtype=jnp.float32)
    z = jnp.asarray(rng.normal(size=(1, 3, 3, 2)), dtype=jnp.float32)
    logits = jnp.asarray(rng.normal(size=(1, 3, 3, 64)), dtype=jnp.bfloat16)
    feats = {
        "asym_id": jnp.asarray([[0, 0, 1]]),
        "mol_type": jnp.asarray([[0, 0, 3]]),
        "token_pad_mask": jnp.ones((1, 3)),
    }
    monkeypatch.setattr(
        confidence, "_compute_ptms", lambda *a, **k: (jnp.zeros(1),) * 4 + ({},)
    )
    result = confidence._confidence_heads_forward(
        params,
        s=s,
        z=z,
        x_pred=jnp.zeros((1, 3, 3)),
        d=jnp.ones((1, 3, 3)),
        feats=feats,
        pred_distogram_logits=logits,
        multiplicity=1,
        rep_atom_index=(jnp.arange(3)[None], jnp.ones((1, 3), dtype=bool)),
        return_pair_chains_iptm=True,
        recompute_nonpolymer_frames=True,
    )
    for name in ("plddt_logits", "resolved_logits"):
        assert result[name].dtype == jnp.bfloat16
    for name in ("pae_logits", "pde_logits", "plddt", "pde", "pae", "complex_pde"):
        assert result[name].dtype == jnp.float32
    probabilities = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    contacts = jnp.zeros((1, 1, 1, 64)).at[..., :20].set(1)
    weights = jnp.sum(probabilities * contacts, axis=-1) * (1 - jnp.eye(3)[None])
    expected = jnp.sum(result["pde"] * weights, axis=(1, 2)) / jnp.sum(
        weights, axis=(1, 2)
    )
    np.testing.assert_array_equal(result["complex_pde"], expected)
