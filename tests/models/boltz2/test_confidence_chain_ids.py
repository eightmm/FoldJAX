"""Static chain labels preserve the complete confidence tree under JIT."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.heads import confidence
from tests.models.boltz2.test_confidence_frames import _valid_case

CHAIN_IDS = (7, 19, 42, 99)


def _inputs():
    coords, frames, feats = _valid_case(batch=1, multiplicity=5)
    feats["frames_idx"] = frames
    feats["asym_id"] = jnp.asarray([[7, 7, 42, 42, 19, 99]], jnp.int32)
    feats["token_pad_mask"] = feats["token_pad_mask"].at[0, -1].set(0)
    rng = np.random.default_rng(84)
    logits = jnp.asarray(rng.normal(size=(5, 6, 6, 8)), jnp.float32)
    return logits, coords, feats


@pytest.mark.parametrize("sequential", [False, True])
def test_jitted_ptms_n5_match_eager_with_noncontiguous_and_padding_ids(sequential):
    logits, coords, feats = _inputs()
    expected = confidence._compute_ptms(
        logits, coords, feats, 5, recompute_nonpolymer_frames=False
    )

    def run(logits, coords, feats):
        def evaluate(logits, coords, multiplicity):
            return confidence._compute_ptms(
                logits,
                coords,
                feats,
                multiplicity,
                confidence_chain_ids=CHAIN_IDS,
                recompute_nonpolymer_frames=False,
            )

        if sequential:
            result = jax.lax.map(
                lambda value: evaluate(value[0][None], value[1][None], 1),
                (logits, coords),
            )
            return jax.tree.map(lambda value: value[:, 0], result)
        return evaluate(logits, coords, 5)

    actual = jax.jit(run)(logits, coords, feats)
    assert tuple(actual[-1]) == CHAIN_IDS
    for row in actual[-1].values():
        assert tuple(row) == CHAIN_IDS
        assert all(value.shape == (5,) for value in row.values())
    # Padding labels are native dictionary keys, even though their metrics are zero.
    for chain_id in CHAIN_IDS:
        np.testing.assert_array_equal(actual[-1][99][chain_id], jnp.zeros(5))
        np.testing.assert_array_equal(actual[-1][chain_id][99], jnp.zeros(5))
    for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(result, reference, rtol=2e-6, atol=2e-7)


def test_eager_static_metadata_is_numerically_inert():
    logits, coords, feats = _inputs()
    automatic = confidence._compute_ptms(
        logits, coords, feats, 5, recompute_nonpolymer_frames=False
    )
    supplied = confidence._compute_ptms(
        logits,
        coords,
        feats,
        5,
        confidence_chain_ids=CHAIN_IDS,
        recompute_nonpolymer_frames=False,
    )
    for actual, expected in zip(jax.tree.leaves(supplied), jax.tree.leaves(automatic)):
        np.testing.assert_array_equal(actual, expected)


def test_jitted_pair_output_requires_explicit_static_labels():
    logits, coords, feats = _inputs()
    with pytest.raises(ValueError, match="confidence_chain_ids must be a static tuple"):
        jax.jit(
            lambda scores, x, f: confidence._compute_ptms(
                scores, x, f, 5, recompute_nonpolymer_frames=False
            )
        )(logits, coords, feats)


def test_existing_serving_mode_does_not_need_labels():
    logits, coords, feats = _inputs()
    result = jax.jit(
        lambda scores, x, f: confidence._compute_ptms(
            scores,
            x,
            f,
            5,
            return_pair_chains_iptm=False,
            recompute_nonpolymer_frames=False,
        )
    )(logits, coords, feats)
    assert result[-1] == {}


@pytest.mark.parametrize(
    "chain_ids", [(), (7, 19, 42), (7, 19, 42, 98), (7, 19, 42, 99, 100)]
)
def test_eager_metadata_must_match_all_input_labels(chain_ids):
    with pytest.raises(ValueError, match="match all unique"):
        confidence._resolve_confidence_chain_ids(_inputs()[2]["asym_id"], chain_ids)


@pytest.mark.parametrize("chain_ids", [[7, 19], (7.0, 19), (True, 19), (7, 7)])
def test_malformed_or_duplicate_static_labels_fail_clearly(chain_ids):
    with pytest.raises(ValueError, match="static tuple|duplicate"):
        confidence._resolve_confidence_chain_ids(_inputs()[2]["asym_id"], chain_ids)


def test_capture_options_bind_all_host_chain_ids_without_relabeling():
    from bench.boltz_foldjax_capture import prediction_options
    from tests.test_boltz_foldjax_capture import _effective, _meta

    options = prediction_options(
        _meta(), _effective(), trunk_only=False, features=_inputs()[2]
    )
    assert options["confidence_chain_ids"] == CHAIN_IDS
    assert options["return_pair_chains_iptm"] is True


@pytest.mark.parametrize("jit", [False, True])
def test_predict_full_confidence_lax_map_preserves_chain_tree(monkeypatch, jit):
    predict = importlib.import_module("foldjax.models.boltz2.models.predict")
    _, coords, feats = _inputs()
    feats.update(
        token_bonds=jnp.zeros((1, 6, 6, 1)),
        type_bonds=jnp.zeros((1, 6, 6), dtype=jnp.int32),
        sample_coords=coords,
    )
    norm = {"scale": jnp.ones(2), "bias": jnp.zeros(2)}
    params = {
        "s_inputs_norm": norm,
        "s_norm": norm,
        "z_norm": norm,
        "token_bonds": {"kernel": jnp.zeros((1, 2))},
        "token_bonds_type": jnp.zeros((1, 2)),
        "rel_pos": {},
        "contact_conditioning": {},
        "pairformer_stack": {},
        "boundaries": jnp.asarray([2.0]),
        "dist_bin_pairwise_embed": jnp.asarray([[0.1, 0.3], [1.1, 1.2]]),
    }
    for name in (
        "s_input_to_s", "s_to_z", "s_to_z_transpose", "s_to_z_prod_in1",
        "s_to_z_prod_in2", "s_to_z_prod_out",
    ):
        params[name] = {"kernel": jnp.zeros((2, 2))}
    rng = np.random.default_rng(28)
    params["confidence_heads"] = {
        name: {"kernel": jnp.asarray(rng.normal(size=(2, bins)), jnp.float32)}
        for name, bins in {
            "to_pae_intra_logits": 64,
            "to_pae_inter_logits": 64,
            "to_pde_intra_logits": 64,
            "to_pde_inter_logits": 64,
            "to_resolved_logits": 2,
            "to_plddt_logits": 50,
        }.items()
    }
    monkeypatch.setattr(
        predict,
        "boltz2_trunk_forward",
        lambda *_a, **_k: {
            "s_inputs": jnp.zeros((1, 6, 2)),
            "s": jnp.zeros((1, 6, 2)),
            "z": jnp.zeros((1, 6, 6, 2)),
        },
    )
    monkeypatch.setattr(
        predict,
        "boltz2_sample_forward",
        lambda _p, f, *_a, **_k: {"sample_atom_coords": f["sample_coords"]},
    )
    monkeypatch.setattr(
        predict, "distogram_forward", lambda *_a, **_k: jnp.zeros((1, 6, 6, 1, 64))
    )
    for name in ("relative_position_forward", "contact_conditioning_forward"):
        monkeypatch.setattr(
            confidence, name, lambda *_a, **_k: jnp.zeros((1, 6, 6, 2))
        )
    monkeypatch.setattr(
        confidence, "pairformer_module_forward", lambda p, s, z, **k: (s, z)
    )

    def run(features):
        return predict.boltz2_predict(
            {"trunk": {}, "confidence": params},
            features,
            jax.random.key(0),
            multiplicity=5,
            confidence_sequentially=True,
            confidence_chain_ids=CHAIN_IDS if jit else None,
            recompute_nonpolymer_frames=False,
        )

    output = (jax.jit(run) if jit else run)(feats)
    assert tuple(output["pair_chains_iptm"]) == CHAIN_IDS
    for row in output["pair_chains_iptm"].values():
        assert tuple(row) == CHAIN_IDS
        assert all(value.shape == (5,) for value in row.values())
    for name in ("ptm", "iptm", "ligand_iptm", "protein_iptm", "complex_plddt"):
        assert output[name].shape == (5,)
    assert output["pae_logits"].shape == (5, 6, 6, 64)
