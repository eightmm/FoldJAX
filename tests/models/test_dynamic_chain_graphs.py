"""Chain cardinality crosses the real compiled model boundary as runtime data."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models._chain_profile import (
    chain_capacity,
    chain_mask,
    crop_chain_scores,
    resolve_chain_count,
)
from foldjax.models._jit_pool import BoundedJitPool
from foldjax.models.protenix.data.output import _sample_summary
from foldjax.models.protenix.models.heads.confidence import (
    confidence_scores_from_logits,
)


def _features(chain_ids):
    valid = jnp.asarray([True] * 6 + [False] * 2)
    indices = jnp.arange(8, dtype=jnp.int32)
    features = {
        "restype": jnp.zeros((8, 32), dtype=jnp.float32),
        "ref_pos": jnp.zeros((8, 3), dtype=jnp.float32),
        "ref_element": jnp.zeros((8, 128), dtype=jnp.float32),
        "ref_atom_name_chars": jnp.zeros((8, 4, 64), dtype=jnp.float32),
        "asym_id": jnp.asarray([*chain_ids, 99, 99], dtype=jnp.int32),
        "token_index": indices,
        "residue_index": indices,
        "entity_id": jnp.zeros((8,), dtype=jnp.int32),
        "sym_id": jnp.zeros((8,), dtype=jnp.int32),
        "relp": jnp.zeros((8, 8, 2), dtype=jnp.float32),
        "atom_to_token_idx": indices,
        "atom_to_tokatom_idx": jnp.zeros((8,), dtype=jnp.int32),
        "has_frame": valid,
        "frame_atom_index": jnp.zeros((8, 3), dtype=jnp.int32),
        "pae_rep_atom_mask": valid,
        "distogram_rep_atom_mask": valid,
        "token_padding_mask": valid,
        "atom_padding_mask": valid,
        "token_is_ligand": jnp.asarray([False] * 5 + [True, False, False]),
        "is_protein": jnp.asarray([True] * 5 + [False] * 3),
        "parent_residue_idx": indices,
        "subtoken_role_id": jnp.ones((8,), dtype=jnp.int32),
        "structural_token_index": indices,
        "atom_to_structural_token_idx": indices,
        "atom_to_structural_tokatom_idx": jnp.zeros((8,), dtype=jnp.int32),
        "structural_distogram_rep_atom_mask": valid,
        "structural_pae_rep_atom_mask": valid,
        "structural_has_frame": valid,
        "structural_frame_atom_index": jnp.zeros((8, 3), dtype=jnp.int32),
        "structural_token_padding_mask": valid,
    }
    return features


def _raw_output():
    rng = np.random.default_rng(17)
    return {
        "coordinate": jnp.asarray(rng.normal(size=(2, 8, 3)), dtype=jnp.float32),
        "plddt": jnp.asarray(rng.normal(size=(2, 8, 50)), dtype=jnp.float32),
        "pae": jnp.asarray(rng.normal(size=(2, 8, 8, 64)), dtype=jnp.float32),
        "pde": jnp.asarray(rng.normal(size=(2, 8, 8, 64)), dtype=jnp.float32),
        "distogram_logits": jnp.asarray(rng.normal(size=(8, 8, 96)), dtype=jnp.float32),
    }


def _score(model, features, raw, **chain_options):
    if model == "opendde":
        from foldjax.models.opendde.postprocess import opendde_confidence_scores

        return opendde_confidence_scores(
            raw,
            features,
            num_recycles=1,
            include_shape_complementarity=False,
            **chain_options,
        )
    return confidence_scores_from_logits(
        plddt_logits=raw["plddt"],
        pae_logits=raw["pae"],
        pde_logits=raw["pde"],
        distogram_logits=raw["distogram_logits"],
        token_has_frame=features["has_frame"],
        token_asym_id=features["asym_id"],
        atom_to_token_idx=features["atom_to_token_idx"],
        atom_coordinate=raw["coordinate"],
        atom_is_polymer=features["is_protein"],
        token_is_ligand=features["token_is_ligand"],
        token_mask=features["token_padding_mask"],
        atom_mask=features["atom_padding_mask"],
        num_recycles=1,
        **chain_options,
    )


class _RecordingPool(BoundedJitPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return super().__call__(*args, **kwargs)


@pytest.mark.parametrize("model", ["protenix", "opendde"])
def test_compiled_model_reuses_one_program_and_crops_chain_scores(monkeypatch, model):
    if model == "protenix":
        from foldjax.models.protenix.models import model as implementation
    else:
        from foldjax.models.opendde.models import model as implementation

    raw = _raw_output()
    traces = []

    def score_only(features, params, schedule, **kwargs):
        traces.append(kwargs)
        assert params == ()
        assert schedule.shape == (2,)
        return {
            "coordinate": raw["coordinate"],
            **_score(model, features, raw, chain_mask=kwargs["chain_mask"]),
        }

    monkeypatch.setattr(implementation, f"{model}_infer_static", score_only)
    pool = _RecordingPool(
        getattr(implementation, f"_{model}_infer_graph"),
        static_argnames=(
            *implementation.GRAPH_STATIC_ARGNAMES,
            "params_treedef",
            "params_flags",
        ),
    )
    monkeypatch.setattr(implementation, f"_compiled_{model}_infer", pool)
    infer = getattr(implementation, f"{model}_infer_compiled")
    schedule = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    cases = (
        ([0, 0, 0, 0, 0, 0], None, 1),
        ([0, 0, 0, 1, 1, 1], None, 2),
        ([0, 0, 1, 1, 2, 2], None, 3),
        ([0, 0, 0, 2, 2, 2], None, 3),
        ([0, 0, 1, 1, 2, 2], 1, 1),
    )
    try:
        for ids, explicit, count in cases:
            features = _features(ids)
            options = {} if explicit is None else {"n_chain": explicit}
            actual = infer(
                features, (), schedule, run_confidence_scores=True, **options
            )
            expected = {
                "coordinate": raw["coordinate"],
                **_score(model, features, raw, n_chain=count),
            }
            jax.block_until_ready(actual)
            assert actual.keys() == expected.keys()
            for key in expected:
                assert actual[key].shape == expected[key].shape, key
                np.testing.assert_allclose(
                    actual[key], expected[key], rtol=1e-4, atol=1e-4, err_msg=key
                )
            assert actual["chain_ptm"].shape == (2, count)
            assert actual["chain_pair_iptm"].shape == (2, count, count)
            summary = json.loads(json.dumps(_sample_summary(actual, 0, 2)))
            assert len(summary["chain_ptm"]) == count
            assert len(summary["chain_pair_iptm"]) == count
            assert all(len(row) == count for row in summary["chain_pair_iptm"])

        assert len(traces) == 1
        assert pool._entry_count() == pool._cache_size() == 1
        assert "n_chain" not in implementation.GRAPH_STATIC_ARGNAMES
        lowerings = []
        for (args, kwargs), (_, _, count) in zip(pool.calls, cases, strict=True):
            assert "n_chain" not in kwargs
            np.testing.assert_array_equal(kwargs["chain_mask"], np.arange(4) < count)
            lowerings.append(pool.lower(*args, **kwargs).as_text())
        assert all(program == lowerings[0] for program in lowerings[1:])
    finally:
        pool.clear_cache()


@pytest.mark.parametrize("count,capacity", [(1, 4), (2, 4), (3, 4), (4, 4), (5, 8)])
def test_chain_capacity_and_host_crop_keep_only_actual_slots(count, capacity):
    assert chain_capacity(count) == capacity
    np.testing.assert_array_equal(chain_mask(count), np.arange(capacity) < count)
    output = {
        "chain_ptm": np.arange(2 * capacity).reshape(2, capacity),
        "chain_pair_iptm": np.arange(2 * capacity**2).reshape(2, capacity, capacity),
        "coordinate": np.zeros((2, 8, 3)),
    }
    cropped = crop_chain_scores(output, count)
    assert cropped["chain_ptm"].shape == (2, count)
    assert cropped["chain_pair_iptm"].shape == (2, count, count)
    assert cropped["coordinate"] is output["coordinate"]
    assert output["chain_ptm"].shape == (2, capacity)


def test_chain_extent_ignores_padded_ids_and_preserves_sparse_or_explicit_extent():
    features = _features([0, 0, 0, 2, 2, 2])
    assert resolve_chain_count(features) == 3
    assert resolve_chain_count(features, 1) == 1
    features["token_padding_mask"] = jnp.zeros((8,), dtype=bool)
    assert resolve_chain_count(features) == 0


def test_tiny_native_protenix_uses_one_body_for_one_two_and_three_chains(monkeypatch):
    from foldjax.models.protenix.models import model as implementation
    from tests.models.protenix.test_model import _toy_features, _toy_params

    features = _toy_features()
    features.update(
        restype=jnp.zeros((3, 32), dtype=jnp.float32),
        profile=jnp.zeros((3, 32), dtype=jnp.float32),
        deletion_mean=jnp.zeros((3,), dtype=jnp.float32),
        relp=jnp.zeros((3, 3, 2), dtype=jnp.float32),
        token_bonds=jnp.zeros((3, 3), dtype=jnp.float32),
        has_frame=jnp.ones((3,), dtype=jnp.int32),
        atom_to_token_idx=jnp.arange(3, dtype=jnp.int32),
        atom_to_tokatom_idx=jnp.zeros((3,), dtype=jnp.int32),
        distogram_rep_atom_mask=jnp.ones((3,), dtype=jnp.int32),
        token_is_ligand=jnp.zeros((3,), dtype=bool),
    )
    params = _toy_params()
    schedule = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    noise = jnp.ones((1, 3, 3), dtype=jnp.float32)
    options = {
        "key": None,
        "num_samples": 1,
        "init_noise": noise,
        "step_noises": (jnp.zeros_like(noise),),
        "num_recycles": 1,
        "input_atom_heads": 1,
        "atom_encoder_heads": 1,
        "token_heads": 1,
        "atom_decoder_heads": 1,
        "n_queries": 2,
        "n_keys": 4,
        "sigma_data": 4.0,
        "centre_each_step": False,
        "diffusion_attention_backend": "xla_jit",
    }
    pool = _RecordingPool(
        implementation._protenix_infer_graph,
        static_argnames=(
            *implementation.GRAPH_STATIC_ARGNAMES,
            "params_treedef",
            "params_flags",
        ),
    )
    monkeypatch.setattr(implementation, "_compiled_protenix_infer", pool)
    try:
        for count, chain_ids in enumerate(([0, 0, 0], [0, 1, 1], [0, 1, 2]), start=1):
            features["asym_id"] = jnp.asarray(chain_ids, dtype=jnp.int32)
            expected = implementation.protenix_infer_static(
                features, params, schedule, n_chain=count, **options
            )
            actual = implementation.protenix_infer_compiled(
                features, params, schedule, **options
            )
            jax.block_until_ready(actual)
            assert actual.keys() == expected.keys()
            for key in expected:
                assert actual[key].shape == expected[key].shape, key
                np.testing.assert_allclose(
                    actual[key], expected[key], rtol=1e-4, atol=1e-4, err_msg=key
                )
            assert actual["chain_ptm"].shape == (1, count)
            assert actual["chain_pair_iptm"].shape == (1, count, count)
        assert pool._entry_count() == pool._cache_size() == 1
        programs = [
            pool.lower(*args, **kwargs).as_text() for args, kwargs in pool.calls
        ]
        assert programs[0] == programs[1] == programs[2]
    finally:
        pool.clear_cache()
