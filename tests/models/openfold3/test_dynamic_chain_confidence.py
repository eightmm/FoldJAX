"""Chain counts are runtime data; score storage is a small padded axis."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3 import inference, streaming
from foldjax.models.openfold3.data import prepare_msa_cycle_features
from foldjax.models.openfold3.models import confidence
from tests.models.openfold3.test_pae_metric_sinking import (
    _install_tiny_predict,
    _tiny_inference_config,
)
from tests.models.openfold3.test_stable_compile import _table

PTM = {"bin_min": 0.0, "bin_max": 32.0, "no_bins": 8}


@pytest.mark.parametrize("count", [None, 1, 2])
def test_dynamic_count_preserves_undefined_monomer_and_unknown_semantics(count):
    logits = jnp.full((1, 4, 4, 8), jnp.nan)
    frames = jnp.ones((1, 4), bool)
    mask = jnp.ones(4, bool)
    asym = jnp.zeros(4, jnp.int32)
    active = jnp.asarray(-1 if count is None else count)
    actual = inference._compact_confidence_metrics_from_pae(
        logits,
        frames,
        mask,
        asym,
        n_chain=inference._PaddedChains(active, 4),
        return_chain_ptm=True,
        **PTM,
    )
    expected = inference._compact_confidence_metrics_from_pae(
        logits,
        frames,
        mask,
        asym,
        n_chain=count,
        return_chain_ptm=True,
        **PTM,
    )
    for left, right in zip(actual[:2], expected[:2], strict=True):
        np.testing.assert_array_equal(left, right)
    if count is not None:
        np.testing.assert_array_equal(actual[3][..., :count], expected[3])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize(
    "count,ids",
    [(1, [0] * 7), (2, [0, 0, 1, 1, 1, 0, 77]), (3, [0, 0, 2, 2, 2, 0, 77])],
)
def test_padded_scores_match_independent_legacy_loops(dtype, count, ids):
    atol = 1e-5 if dtype == jnp.bfloat16 else 2e-6
    rng = np.random.default_rng(107)
    logits = jnp.asarray(rng.normal(size=(3, 7, 7, 8)), dtype=dtype)
    frames = jnp.asarray([[1, 1, 0, 1, 1, 0, 1], [0] * 7, [0, 0, 1, 1, 1, 1, 0]], bool)
    mask = jnp.asarray([1, 1, 1, 1, 1, 1, 0], bool)
    asym = jnp.asarray(ids)
    chains = inference._PaddedChains(jnp.asarray(count), 4)
    ptm, iptm, pair, chain_ptm = jax.jit(
        lambda values, valid, active: inference._compact_confidence_metrics_from_pae(
            values,
            valid,
            mask,
            asym,
            n_chain=inference._PaddedChains(active, 4),
            return_chain_ptm=True,
            **PTM,
        )
    )(logits, frames, chains.count)
    expected_ptm = confidence.compute_ptm(logits, frames, mask, **PTM)
    expected_iptm = confidence.compute_ptm(
        logits, frames, mask, asym_id=asym, interface=True, **PTM
    )
    expected_chain = confidence.compute_chain_ptm(
        logits, frames, mask, asym, n_chain=count, **PTM
    )
    expected_pair = confidence.compute_chain_pair_iptm(
        logits, frames, mask, asym, n_chain=count, **PTM
    )
    for actual, expected in (
        (ptm, expected_ptm),
        (iptm, expected_iptm),
        (pair[..., :count, :count], expected_pair),
        (chain_ptm[..., :count], expected_chain),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)
    ligand = jnp.asarray([0, 0, 1, 1, 1, 0, 0])
    expected_bespoke = inference._bespoke_iptm(
        expected_pair, frames, mask, asym, ligand, n_chain=count
    )
    actual_bespoke = inference._bespoke_iptm(
        pair, frames, mask, asym, ligand, n_chain=chains
    )
    np.testing.assert_allclose(
        actual_bespoke[..., :count, :count], expected_bespoke, rtol=0, atol=atol
    )
    assert not np.any(np.asarray(pair[..., count:, :]))


@pytest.mark.parametrize("returned,cutoff", [((), 0), (("pae_logits",), 0), ((), None)])
def test_real_predict_boundary_reuses_one_lowering_and_crops_scores(
    monkeypatch, returned, cutoff
):
    batch, _ = _install_tiny_predict(monkeypatch, n_token=12)
    batch["token_mask"] = jnp.ones((1, 12))
    batch["is_ligand"] = jnp.asarray([[0, 0, 1, 0] * 3])
    params = inference.InferenceParams(*([None] * 8))
    config = _tiny_inference_config(
        per_sample_token_cutoff=cutoff, returned_pair_logits=returned
    )._replace(n_token=12, n_atom=12)
    lowered = []
    inference._compiled_predict.clear_cache()
    try:
        for count, ids in (
            (1, [0] * 12),
            (5, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4]),
            (12, list(range(12))),
        ):
            current = {**batch, "asym_id": jnp.asarray([ids])}
            run = inference.compile_predict(
                config, _table(), n_chain=count, triangle_kernel="xla"
            )
            args = (jax.random.key(0), current, params)
            lowered.append(run.lower(*args))
            expected = inference.predict(*args, config, _table(), n_chain=count)
            actual = run(*args)
            assert isinstance(actual.chain_ptm, jax.Array)
            if count > 1:
                assert isinstance(actual.chain_pair_iptm, jax.Array)
                assert isinstance(actual.bespoke_iptm, jax.Array)
            for name in (
                "coordinates",
                "ptm",
                "iptm",
                "chain_ptm",
                "chain_pair_iptm",
                "bespoke_iptm",
            ):
                left, right = getattr(actual, name), getattr(expected, name)
                if right is None:
                    assert left is None
                else:
                    np.testing.assert_allclose(
                        left, right, rtol=0, atol=2e-6, err_msg=name
                    )
            if count == 12:
                compiled = lowered[-1].compile()(*args)
                assert isinstance(compiled.chain_ptm, jax.Array)
                assert compiled.chain_ptm.shape == (3, 12)
                np.testing.assert_allclose(
                    compiled.chain_pair_iptm, actual.chain_pair_iptm, atol=2e-6
                )
        assert len({item.as_text() for item in lowered}) == 1
        assert inference._compiled_predict._cache_size() == 1
        if not returned and cutoff == 0:
            assert "tensor<3x12x12x64xf32>" not in lowered[0].as_text()
    finally:
        inference._compiled_predict.clear_cache()


def test_streamed_real_confidence_tail_reuses_chain_counts(monkeypatch):
    batch, _ = _install_tiny_predict(monkeypatch, n_token=12)
    batch["token_mask"] = jnp.ones((1, 12))
    batch["is_ligand"] = jnp.asarray([[0, 0, 1, 0] * 3])
    for name in ("msa", "has_deletion", "deletion_value"):
        batch[name] = np.zeros((1, 1, 12))
    batch = prepare_msa_cycle_features(
        batch, 1, num_recycles=1, rng=np.random.default_rng(1)
    )
    params = inference.InferenceParams(*([None] * 8))
    config = _tiny_inference_config(
        per_sample_token_cutoff=0, returned_pair_logits=()
    )._replace(msa_depth=1, n_token=12, n_atom=12)
    traces = []
    trunk = inference.trunk

    def initialize(batch, params, **kwargs):
        traces.append("initialize")
        return trunk(batch, params)

    def cycle(batch, msa, params, initial, carry, **kwargs):
        traces.append("cycle")
        return initial[1:]

    monkeypatch.setattr(streaming, "initialize_trunk", initialize)
    monkeypatch.setattr(streaming, "trunk_cycle", cycle)
    streaming._compiled_streams.clear_cache()
    try:
        for count, ids in (
            (1, [0] * 12),
            (5, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4]),
            (12, list(range(12))),
        ):
            current = {**batch, "asym_id": np.asarray([ids], np.int32)}
            value = streaming.compile_streamed_predict(
                config, _table(), n_chain=count, triangle_kernel="xla"
            )(jax.random.key(0), current, params)
            assert isinstance(value.chain_ptm, jax.Array)
            expected = inference.predict(
                jax.random.key(0), current, params, config, _table(), n_chain=count
            )
            np.testing.assert_allclose(value.chain_ptm, expected.chain_ptm, atol=2e-6)
            if count > 1:
                np.testing.assert_allclose(
                    value.bespoke_iptm, expected.bespoke_iptm, atol=2e-6
                )
        assert traces == ["initialize", "cycle"]
        assert streaming._compiled_streams._cache_size() == 3
    finally:
        streaming._compiled_streams.clear_cache()
